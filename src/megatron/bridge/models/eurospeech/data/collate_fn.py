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

"""Collator for the EuroSpeech (Canary + EuroLLM) SpeechLLM.

The collator turns a list of audio-text examples into the batch dict consumed by
``training/audio_lm_step.py``. Each example must provide:

* ``conversation``: a chat-template-compatible list of turns. The user turn must
  contain a single audio placeholder token where audio should be injected. The
  default placeholder is ``<extra_id_0>`` — a reserved/unused EuroLLM token, so no
  vocab resize is needed.
* ``audio``: the raw waveform (a numpy array, a ``(array, sr)`` tuple, or a
  ``{"array": ...}`` dict).

The single placeholder is expanded to exactly
``compute_num_audio_frames(num_samples)`` placeholder tokens so that the count of
audio-token positions matches the number of encoder output frames the model will
scatter in. The placeholder id must equal the model's ``audio_token_id``. For the
real Canary encoder, pass a ``compute_num_audio_frames`` that
mirrors the encoder's subsampling; the default mirrors
:class:`~megatron.bridge.models.eurospeech.canary_encoder.StubCanaryEncoder`.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import torch

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.vlm_processing import gather_assistant_text_segments
from megatron.bridge.models.eurospeech.audio_inputs import CanaryAudioInputs


def _extract_waveform(audio) -> torch.Tensor:
    """Normalize an example's ``audio`` field to a 1-D float tensor."""
    if isinstance(audio, tuple):
        audio = audio[0]
    elif isinstance(audio, dict):
        audio = audio["array"]
    return torch.as_tensor(audio, dtype=torch.float32).reshape(-1)


def _render_content(content, audio_token: str) -> str:
    """Flatten a turn's ``content`` to a string, mapping audio parts to the placeholder.

    Accepts both plain-string content and the structured list-of-parts schema used
    by the built-in audio makers (``make_default_audio_dataset`` / ``make_cv17_dataset``),
    where the user turn contains ``{"type": "audio", ...}`` and ``{"type": "text", ...}``.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif part.get("type") == "audio":
            parts.append(audio_token)
        elif part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return " ".join(parts)


def _stringify_conversation(conversation, audio_token: str) -> list:
    """Return a copy of ``conversation`` with each turn's content rendered to a string."""
    return [{"role": turn["role"], "content": _render_content(turn["content"], audio_token)} for turn in conversation]


def default_num_audio_frames(num_samples: int, subsampling_factor: int = 8) -> int:
    """Default frame-count estimate matching ``StubCanaryEncoder``."""
    return (num_samples + subsampling_factor - 1) // subsampling_factor


def eurospeech_collate_fn(
    examples: list,
    processor,
    audio_token: str = "<extra_id_0>",
    compute_num_audio_frames: Optional[Callable[[int], int]] = None,
) -> dict[str, torch.Tensor]:
    """Collate audio-text examples for EuroSpeech.

    Args:
        examples: List of example dicts with ``conversation`` and ``audio`` keys.
        processor: A processor/tokenizer exposing ``apply_chat_template`` and the
            usual tokenizer call interface. Must know the ``audio_token``.
        audio_token: The placeholder string expanded per example.
        compute_num_audio_frames: Maps sample count -> number of audio frames.
            Defaults to :func:`default_num_audio_frames`.

    Returns:
        Batch dict with ``input_ids``, ``labels``, ``loss_mask``, ``position_ids``
        and an ``audio_inputs`` :class:`CanaryAudioInputs` container.
    """
    if compute_num_audio_frames is None:
        compute_num_audio_frames = default_num_audio_frames

    tokenizer = getattr(processor, "tokenizer", processor)
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    waveforms: list[torch.Tensor] = []
    texts: list[str] = []
    for example in examples:
        waveform = _extract_waveform(example["audio"])
        waveforms.append(waveform)

        n_frames = compute_num_audio_frames(waveform.numel())
        # Render structured (or string) content, then apply the chat template so
        # the audio placeholder survives into the tokenized prompt.
        conversation = _stringify_conversation(example["conversation"], audio_token)
        text = processor.apply_chat_template(conversation, tokenize=False)
        # Expand the single placeholder into n_frames placeholders.
        if text.count(audio_token) != 1:
            raise ValueError(
                f"Each conversation must contain exactly one '{audio_token}' placeholder, "
                f"found {text.count(audio_token)}."
            )
        texts.append(text.replace(audio_token, audio_token * n_frames))

    saved_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "right"
    try:
        batch = tokenizer(text=texts, return_tensors="pt", padding=True)
    finally:
        if saved_padding_side is not None:
            tokenizer.padding_side = saved_padding_side

    input_ids = batch["input_ids"]
    pad_token_id = tokenizer.pad_token_id

    # HF-compatible label construction (mirrors qwen2_audio_collate_fn).
    hf_labels = input_ids.clone()
    for i, example in enumerate(examples):
        ids = input_ids[i].tolist()
        found = -1
        for asst_text in gather_assistant_text_segments(example):
            asst_token_ids = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
            span_len = len(asst_token_ids)
            if span_len == 0:
                continue
            for start in range(len(ids) - span_len, -1, -1):
                if ids[start : start + span_len] == asst_token_ids:
                    found = start
                    break
            if found >= 0:
                break

        if found >= 0:
            hf_labels[i, :found] = IGNORE_INDEX
        else:
            warnings.warn(f"Could not find assistant span for example {i}, masking all labels", stacklevel=2)
            hf_labels[i, :] = IGNORE_INDEX

        if pad_token_id is not None:
            hf_labels[i][input_ids[i] == pad_token_id] = IGNORE_INDEX
        # Never train on audio placeholder positions.
        hf_labels[i][input_ids[i] == audio_token_id] = IGNORE_INDEX

    # Shift labels for Megatron (labels[j] = hf_labels[j+1]).
    labels = hf_labels[:, 1:]
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    batch["labels"] = labels
    batch["loss_mask"] = (labels != IGNORE_INDEX).float()

    batch_size, seq_len = input_ids.shape
    batch["position_ids"] = (
        torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1).clone().contiguous()
    )

    # Pad waveforms into a dense [B, T_samples] tensor + lengths.
    max_samples = max(w.numel() for w in waveforms)
    audio_signal = torch.zeros(len(waveforms), max_samples, dtype=torch.float32)
    audio_length = torch.zeros(len(waveforms), dtype=torch.long)
    for i, w in enumerate(waveforms):
        audio_signal[i, : w.numel()] = w
        audio_length[i] = w.numel()

    batch["audio_inputs"] = CanaryAudioInputs(audio_signal=audio_signal, audio_length=audio_length)
    return batch
