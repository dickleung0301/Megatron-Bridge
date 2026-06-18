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

"""Audio modality container for the Canary + EuroLLM SpeechLLM (EuroSpeech).

The container mirrors the ``Qwen2AudioInputs`` contract consumed by
``training/audio_lm_step.py``: the step function calls
``audio_inputs.normalized_for_model()`` and forwards the returned mapping as
keyword arguments to ``EuroSpeechModel.forward``. The field names therefore
*must* match the audio parameters of ``EuroSpeechModel.forward``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass
class CanaryAudioInputs:
    """Container for Canary audio-encoder inputs.

    Path A (the default) wraps NeMo's Canary encoder together with its mel
    feature extractor, so the model consumes raw waveforms:

    Attributes:
        audio_signal: Batched audio waveforms of shape ``[B, T_samples]``.
        audio_length: Valid sample counts per example of shape ``[B]``.
    """

    audio_signal: Optional[torch.Tensor] = None
    audio_length: Optional[torch.Tensor] = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        """Return a mapping of non-None fields suitable for model forward kwargs."""
        result: dict[str, torch.Tensor] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result

    def normalized_for_model(self) -> dict[str, torch.Tensor]:
        """Return non-None fields (no shape normalization needed for audio)."""
        return self.as_model_kwargs()
