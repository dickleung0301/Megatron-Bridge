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

"""Canary audio-encoder wrappers for the EuroSpeech SpeechLLM.

Two implementations are provided:

* :class:`BridgeCanaryEncoder` wraps NVIDIA NeMo's ``EncDecMultiTaskModel``
  encoder (FastConformer). NeMo (``nemo_toolkit[asr]``) is an *optional*
  runtime dependency and is imported lazily so the rest of the package imports
  cleanly without it. Per the project dependency policy the dependency itself is
  not declared in ``pyproject.toml`` here — it must be added as a separate,
  opt-in extra.

* :class:`StubCanaryEncoder` is a tiny deterministic encoder with no external
  dependencies. It produces correctly shaped embeddings so unit tests and GPU
  smoke tests can exercise the full SpeechLLM forward path without a real Canary
  checkpoint.

Both expose the same contract::

    forward(audio_signal, audio_length) -> (embeddings, embedding_lengths)

where ``embeddings`` has shape ``[B, T_audio, output_dim]`` and
``embedding_lengths`` has shape ``[B]``.
"""

from __future__ import annotations

import torch
from megatron.core.transformer.module import MegatronModule
from torch import Tensor


def _mark_replicated_for_tp_grad_sync(module: torch.nn.Module) -> None:
    """Flag a module's params so finalize_model_grads all-reduces their grads across TP.

    The audio encoder is replicated (not tensor-parallel sharded) across
    tensor-parallel ranks, so its gradients must be averaged across the TP domain
    to keep the replicas in sync.
    """
    for param in module.parameters(recurse=True):
        setattr(param, "average_gradients_across_tp_domain", True)


class BridgeCanaryEncoder(MegatronModule):
    """Wrap NeMo's Canary encoder (FastConformer) as a Megatron module.

    The encoder, mel feature extractor and (optional) spec-augment module are
    restored from a ``.nemo`` checkpoint; the decoder / CTC head are discarded.
    Raw waveforms are consumed directly, so the model-side input is the waveform
    (see :class:`~megatron.bridge.models.eurospeech.audio_inputs.CanaryAudioInputs`).

    Args:
        config: The :class:`EuroSpeechProvider`. Must define ``canary_nemo_path``.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        nemo_path = getattr(config, "canary_nemo_path", "")
        if not nemo_path:
            raise ValueError(
                "BridgeCanaryEncoder requires `canary_nemo_path` to point at a Canary `.nemo` file. "
                "Set provider.canary_nemo_path, or use StubCanaryEncoder for tests."
            )

        try:
            from nemo.collections.asr.models import EncDecMultiTaskModel
        except ImportError as exc:  # pragma: no cover - exercised only without NeMo installed
            raise ImportError(
                "BridgeCanaryEncoder requires NVIDIA NeMo with the ASR collection "
                "(`nemo_toolkit[asr]`). Install it as an optional extra, or use "
                "StubCanaryEncoder for tests."
            ) from exc

        nemo_model = EncDecMultiTaskModel.restore_from(nemo_path, map_location="cpu")
        # Keep only the audio-encoding path; drop decoder / CTC head.
        self.preprocessor = nemo_model.preprocessor
        self.encoder = nemo_model.encoder
        self.spec_augmentation = getattr(nemo_model, "spec_augmentation", None)
        self.output_dim = self.encoder.d_model

        _mark_replicated_for_tp_grad_sync(self)

    def set_input_tensor(self, input_tensor) -> None:
        """Dummy set_input_tensor hook for pipeline parallelism."""
        self.input_tensor = input_tensor

    def forward(self, audio_signal: Tensor, audio_length: Tensor) -> tuple[Tensor, Tensor]:
        """Encode raw waveforms into audio embeddings.

        Args:
            audio_signal: Waveforms of shape ``[B, T_samples]``.
            audio_length: Valid sample counts of shape ``[B]``.

        Returns:
            Tuple of ``(embeddings [B, T_audio, output_dim], lengths [B])``.
        """
        feats, feat_lens = self.preprocessor(input_signal=audio_signal, length=audio_length)
        if self.training and self.spec_augmentation is not None:
            feats = self.spec_augmentation(input_spec=feats, length=feat_lens)
        encoded, encoded_len = self.encoder(audio_signal=feats, length=feat_lens)
        # NeMo encoder returns [B, D, T_audio]; SpeechLLM expects [B, T_audio, D].
        return encoded.transpose(1, 2).contiguous(), encoded_len


class StubCanaryEncoder(MegatronModule):
    """Dependency-free stand-in for the Canary encoder used in tests.

    Subsamples the waveform by ``subsampling_factor`` and applies a single linear
    projection so the output has the right rank and a learnable parameter, but no
    external checkpoint or NeMo dependency.

    Args:
        config: The :class:`EuroSpeechProvider`. Uses ``audio_hidden_size`` and
            ``audio_subsampling_factor``.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.output_dim = config.audio_hidden_size
        self.subsampling_factor = getattr(config, "audio_subsampling_factor", 8)
        self.proj = torch.nn.Linear(1, self.output_dim)
        _mark_replicated_for_tp_grad_sync(self)

    def set_input_tensor(self, input_tensor) -> None:
        """Dummy set_input_tensor hook for pipeline parallelism."""
        self.input_tensor = input_tensor

    def forward(self, audio_signal: Tensor, audio_length: Tensor) -> tuple[Tensor, Tensor]:
        """Produce deterministically shaped audio embeddings from raw waveforms."""
        # audio_signal: [B, T_samples] -> subsample -> [B, T_audio]
        sub = audio_signal[:, :: self.subsampling_factor]
        embeds = self.proj(sub.unsqueeze(-1))  # [B, T_audio, output_dim]
        lengths = torch.clamp(
            (audio_length + self.subsampling_factor - 1) // self.subsampling_factor,
            max=embeds.size(1),
        )
        return embeds, lengths
