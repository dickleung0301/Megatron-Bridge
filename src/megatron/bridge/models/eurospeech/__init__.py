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

"""EuroSpeech: a Canary audio encoder + EuroLLM language-model SpeechLLM."""

from megatron.bridge.models.eurospeech.audio_inputs import CanaryAudioInputs
from megatron.bridge.models.eurospeech.canary_encoder import BridgeCanaryEncoder, StubCanaryEncoder
from megatron.bridge.models.eurospeech.eurospeech_provider import EuroSpeechProvider
from megatron.bridge.models.eurospeech.modeling_eurospeech import (
    AudioProjector,
    EuroSpeechModel,
    scatter_audio_into_text,
)


__all__ = [
    "AudioProjector",
    "BridgeCanaryEncoder",
    "CanaryAudioInputs",
    "EuroSpeechModel",
    "EuroSpeechProvider",
    "StubCanaryEncoder",
    "scatter_audio_into_text",
]
