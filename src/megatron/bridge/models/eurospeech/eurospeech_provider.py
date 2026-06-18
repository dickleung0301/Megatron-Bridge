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

"""Model provider for EuroSpeech — a Canary + EuroLLM SpeechLLM.

EuroSpeech combines NVIDIA's Canary audio encoder, a trainable audio projector,
and an EuroLLM (Llama-architecture) language backbone. Unlike a published HF
model, there is no combined HF checkpoint and therefore no ``AutoBridge``
registration: the language-model fields are populated from EuroLLM's HF config
(via ``AutoBridge``) and then carried on this provider, while the audio encoder
loads from a ``.nemo`` file and the projector trains from scratch.

The provider extends :class:`GPTModelProvider` so the language backbone reuses
the full Megatron GPT stack (TP/PP/SP, distributed optimizer, FP8, etc.).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.gpt_provider import GPTModelProvider


if TYPE_CHECKING:
    from megatron.bridge.models.eurospeech.modeling_eurospeech import EuroSpeechModel


@dataclass
class EuroSpeechProvider(GPTModelProvider):
    """Provider for the Canary + EuroLLM SpeechLLM.

    Language-model architecture fields (``num_layers``, ``hidden_size``, ...) are
    inherited from :class:`GPTModelProvider` and should be populated from
    EuroLLM's HF config. The fields below configure the audio encoder, the
    projector, and the staged freeze schedule.

    Attributes:
        canary_nemo_path: Path to a Canary ``.nemo`` checkpoint (Path A). Loaded
            lazily by :class:`BridgeCanaryEncoder` in ``provide()``.
        use_stub_audio_tower: If True, build :class:`StubCanaryEncoder` instead of
            the NeMo-backed encoder (for tests / smoke runs without NeMo).
        audio_token_id: Token id whose embedding positions are replaced by audio
            features. Use a reserved/unused id to avoid resizing the vocab.
        audio_hidden_size: Output dimension of the audio encoder (Canary
            ``d_model``). Drives the projector input size.
        audio_subsampling_factor: Encoder subsampling factor (used by the stub
            encoder to derive output lengths).
        audio_projector_num_layers: Number of linear layers in the projector
            (1 = linear, 2 = GeLU MLP).
        audio_projector_hidden: Hidden width of the projector MLP. ``0`` means use
            ``hidden_size``.
        freeze_language_model: Freeze the EuroLLM backbone.
        freeze_audio_model: Freeze the Canary encoder.
        freeze_audio_projection: Freeze the audio projector.
    """

    # EuroLLM uses RoPE (Llama architecture).
    position_embedding_type: str = "rope"

    # Audio embeddings are scattered into the language sequence, so they must not
    # be scattered across sequence-parallel regions (mirrors Qwen2-Audio).
    scatter_embedding_sequence_parallel: bool = False

    # Audio encoder (Canary).
    canary_nemo_path: str = ""
    use_stub_audio_tower: bool = False
    audio_token_id: int = 0
    audio_hidden_size: int = 1024
    audio_subsampling_factor: int = 8

    # Audio projector (trained from scratch).
    audio_projector_num_layers: int = 2
    audio_projector_hidden: int = 0

    # Staged freeze schedule.
    freeze_language_model: bool = False
    freeze_audio_model: bool = True
    freeze_audio_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "EuroSpeechModel":
        """Construct an :class:`EuroSpeechModel` and apply the freeze schedule.

        Args:
            pre_process: Whether this is the first pipeline stage (builds the
                audio encoder + projector).
            post_process: Whether this is the last pipeline stage.
            vp_stage: Virtual pipeline stage number.

        Returns:
            An :class:`EuroSpeechModel` instance.
        """
        from megatron.bridge.models.eurospeech.modeling_eurospeech import EuroSpeechModel

        model = EuroSpeechModel(
            config=self,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        if self.freeze_language_model or self.freeze_audio_model or self.freeze_audio_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_audio_model=self.freeze_audio_model,
                freeze_audio_projection=self.freeze_audio_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide just the EuroLLM language backbone (a Megatron ``GPTModel``).

        Args:
            pre_process: Whether this is the first pipeline stage.
            post_process: Whether this is the last pipeline stage.
            vp_stage: Virtual pipeline stage number.

        Returns:
            An ``MCoreGPTModel`` instance (language model only).
        """
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

    def projector_hidden_size(self) -> int:
        """Return the effective projector hidden width."""
        return self.audio_projector_hidden if self.audio_projector_hidden > 0 else self.hidden_size
