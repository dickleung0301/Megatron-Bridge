# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lhotse dataset provider for the EuroSpeech (Canary + EuroLLM) SpeechLLM.

Bridges a Lhotse ``CutSet`` (Shar shards or a plain cut manifest) into the
conversation-style examples consumed by :func:`eurospeech_collate_fn`. Each cut
is turned into a single audio-prompt → text-response turn:

* **user** turn: an audio placeholder part plus the prompt text. The prompt is
  read from ``cut.custom["context"]`` (cut-level custom metadata); if absent it
  falls back to :data:`DEFAULT_PROMPT`.
* **assistant** turn: ``cut.supervisions[0].text`` (the target response).

Audio is decoded **lazily** inside ``__getitem__`` (i.e. in the DataLoader
worker), so only the lightweight cut manifests are held in memory — decoded
waveforms for the whole corpus are never materialized at once. The waveform is
resampled to 16 kHz (Canary's expected rate) when needed, since the collator
does not resample.

``lhotse`` is an optional dependency; it is imported lazily so this module can be
imported without it installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from transformers import AutoProcessor

from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.models.eurospeech.data.collate_fn import eurospeech_collate_fn
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


# Canary consumes 16 kHz audio; the collator does not resample, so cuts are
# resampled here when their native rate differs.
TARGET_SAMPLING_RATE = 16000

# Used when a cut carries no ``custom["context"]`` prompt.
DEFAULT_PROMPT = "Transcribe the audio clip."


def _cut_to_example(cut: Any, audio_token_placeholder: str = "placeholder") -> dict[str, Any]:
    """Turn a single Lhotse cut into a conversation example with decoded audio.

    Args:
        cut: A Lhotse ``Cut`` exposing ``sampling_rate``, ``custom``,
            ``supervisions`` and ``load_audio()`` / ``resample()``.
        audio_token_placeholder: Value stored under the user turn's audio part;
            the collator only checks the part ``type``, not this value.

    Returns:
        A dict with ``conversation`` and ``audio`` keys matching the
        :func:`eurospeech_collate_fn` contract.
    """
    if cut.sampling_rate != TARGET_SAMPLING_RATE:
        cut = cut.resample(TARGET_SAMPLING_RATE)

    # load_audio() returns [num_channels, num_samples]; flatten to 1-D for mono.
    waveform = torch.as_tensor(cut.load_audio(), dtype=torch.float32).reshape(-1)

    if not cut.supervisions:
        raise ValueError(f"Cut '{getattr(cut, 'id', '<unknown>')}' has no supervisions; cannot build a target.")
    target = cut.supervisions[0].text
    prompt = (cut.custom or {}).get("context", DEFAULT_PROMPT)

    return {
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_token_placeholder},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
        "audio": (waveform, TARGET_SAMPLING_RATE),
    }


class LhotseConversationDataset(VLMConversationDataset):
    """``VLMConversationDataset`` over Lhotse cuts with lazy per-item audio decode.

    ``base_examples`` holds lightweight Lhotse cuts rather than fully-formed
    conversation dicts. Audio is decoded on access in :meth:`__getitem__`, so the
    decoded waveform of only one example (per worker) lives in memory at a time.
    Shuffle/repeat semantics and the ``collate_fn`` binding are inherited from the
    parent.
    """

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        cut = self._base_examples[idx % len(self._base_examples)]
        return _cut_to_example(cut)


@dataclass(kw_only=True)
class LhotseSharConversationProvider(DatasetProvider):
    """DatasetProvider that builds EuroSpeech conversation datasets from Lhotse cuts.

    Accepts either a Lhotse Shar directory (``*.tar`` shards) or a plain cut
    manifest file (``cuts.jsonl(.gz)``). Audio referenced by file path is decoded
    lazily; audio embedded in Shar ``recording.*.tar`` shards will be materialized
    by Lhotse when the cuts are listed, so prefer referenced audio (or the
    streaming path) for very large embedded corpora.
    """

    # Required to match model.seq_length (enforced by ConfigContainer.validate).
    seq_length: int

    # HF processor/tokenizer identifier (the EuroLLM model path).
    hf_processor_path: str

    # Lhotse source for the train split: a Shar directory or a cut manifest path.
    train_cuts: str

    # Optional validation source (same format as ``train_cuts``).
    val_cuts: Optional[str] = None

    # Keep parity with GPTDatasetConfig usage in batching utilities.
    skip_getting_attention_mask_from_dataset: bool = True

    def _load_cuts(self, source: str) -> list:
        """Load a Lhotse ``CutSet`` from a Shar dir or a cut manifest into a list."""
        import os

        from lhotse import CutSet

        if os.path.isdir(source):
            cuts = CutSet.from_shar(in_dir=source)
        else:
            cuts = CutSet.from_file(source)
        return list(cuts)

    def _build_split(
        self, source: Optional[str], target_length: int, processor: Any
    ) -> Optional[LhotseConversationDataset]:
        if not source or target_length <= 0:
            return None
        cuts = self._load_cuts(source)
        if not cuts:
            raise ValueError(f"No cuts loaded from '{source}'.")
        return LhotseConversationDataset(
            base_examples=cuts,
            target_length=target_length,
            processor=processor,
            collate_impl=eurospeech_collate_fn,
        )

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build the train/validation/test datasets (test is always None here)."""
        processor = AutoProcessor.from_pretrained(
            self.hf_processor_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=self.trust_remote_code,
                hf_path=self.hf_processor_path,
            ),
        )
        train_ds = self._build_split(self.train_cuts, context.train_samples, processor)
        valid_ds = self._build_split(self.val_cuts, context.valid_samples, processor)
        return train_ds, valid_ds, None
