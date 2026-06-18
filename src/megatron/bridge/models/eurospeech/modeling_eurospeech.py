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

"""EuroSpeech model: Canary audio encoder + projector + EuroLLM language model.

This is structurally the Qwen2-Audio pattern with the audio tower swapped for a
Canary encoder. The forward pass:

1. embeds ``input_ids`` with the Megatron language model's embedding,
2. encodes the audio waveform with the Canary encoder and projects it into the
   language hidden space,
3. scatters the audio features into the text embeddings at ``audio_token_id``
   positions, and
4. runs the EuroLLM decoder on the fused embeddings.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from megatron.bridge.models.eurospeech.canary_encoder import (
    BridgeCanaryEncoder,
    StubCanaryEncoder,
    _mark_replicated_for_tp_grad_sync,
)


if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams

    from megatron.bridge.models.eurospeech.eurospeech_provider import EuroSpeechProvider


class AudioProjector(torch.nn.Module):
    """Maps audio-encoder embeddings into the language model's hidden space.

    A 1-layer projector is a single linear map; a 2-layer projector is a GeLU
    MLP. The projector is replicated across tensor-parallel ranks, so its grads
    are flagged for all-reduce across the TP domain.

    Args:
        in_dim: Audio-encoder output dimension.
        hidden_dim: MLP hidden width (ignored when ``num_layers == 1``).
        out_dim: Language-model hidden size.
        num_layers: Number of linear layers (1 or 2).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers == 1:
            self.layers = torch.nn.ModuleList([torch.nn.Linear(in_dim, out_dim)])
        elif num_layers == 2:
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, hidden_dim), torch.nn.Linear(hidden_dim, out_dim)]
            )
        else:
            raise ValueError(f"audio_projector_num_layers must be 1 or 2, got {num_layers}")
        _mark_replicated_for_tp_grad_sync(self)

    def forward(self, x: Tensor) -> Tensor:
        """Project ``x`` of shape ``[..., in_dim]`` to ``[..., out_dim]``."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        return x


def scatter_audio_into_text(
    text_embeds: Tensor,
    input_ids: Tensor,
    audio_features: Tensor,
    audio_lengths: Tensor,
    audio_token_id: int,
) -> Tensor:
    """Replace text embeddings at audio-token positions with audio features.

    Args:
        text_embeds: Text embeddings in ``[B, S, H]`` (batch-first) layout.
        input_ids: Token ids ``[B, S]``; positions equal to ``audio_token_id``
            receive audio features.
        audio_features: Projected audio features ``[B, T_audio, H]``.
        audio_lengths: Valid audio frame counts per example ``[B]``.
        audio_token_id: The placeholder token id.

    Returns:
        Fused embeddings ``[B, S, H]``.

    Raises:
        ValueError: If the number of valid audio frames does not match the number
            of ``audio_token_id`` positions in ``input_ids``.
    """
    max_audio_tokens = audio_features.size(1)
    frame_mask = torch.arange(max_audio_tokens, device=audio_lengths.device)[None, :] < audio_lengths[:, None]
    valid_audio = audio_features[frame_mask]  # [n_audio_features, H]

    n_audio_tokens = int((input_ids == audio_token_id).sum().item())
    n_audio_features = valid_audio.size(0)
    if n_audio_tokens != n_audio_features:
        raise ValueError(
            f"Audio features and audio tokens do not match: tokens={n_audio_tokens}, features={n_audio_features}"
        )

    special_audio_mask = (input_ids == audio_token_id).unsqueeze(-1).expand_as(text_embeds)
    valid_audio = valid_audio.to(text_embeds.device, text_embeds.dtype)
    return text_embeds.masked_scatter(special_audio_mask, valid_audio)


class EuroSpeechModel(MegatronModule):
    """Canary + EuroLLM SpeechLLM wrapped as a single Megatron module.

    Args:
        config: The :class:`EuroSpeechProvider`.
        pre_process: Whether to build the audio encoder + projector (first
            pipeline stage). Default: True.
        post_process: Whether to apply post-processing (last pipeline stage).
            Default: True.
        vp_stage: Virtual pipeline stage number. Default: None.
    """

    def __init__(
        self,
        config: "EuroSpeechProvider",
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        self.audio_token_id = config.audio_token_id

        if pre_process:
            if config.use_stub_audio_tower:
                self.audio_tower = StubCanaryEncoder(config)
            else:
                self.audio_tower = BridgeCanaryEncoder(config)

            self.multi_modal_projector = AudioProjector(
                in_dim=self.audio_tower.output_dim,
                hidden_dim=config.projector_hidden_size(),
                out_dim=config.hidden_size,
                num_layers=config.audio_projector_num_layers,
            )

        # EuroLLM language backbone (Megatron GPTModel).
        self.language_model = config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Bind shared embedding/output weight metadata for grad finalization.
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

    def set_input_tensor(self, input_tensor) -> None:
        """Forward the pipeline input tensor to the language model."""
        self.language_model.set_input_tensor(input_tensor)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_signal: Optional[torch.Tensor] = None,
        audio_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional["PackedSeqParams"] = None,
        *,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run the SpeechLLM forward pass.

        Args:
            input_ids: Token ids ``[B, S]`` (with ``audio_token_id`` placeholders).
            position_ids: Position ids for the language model.
            attention_mask: Attention mask for the language model.
            audio_signal: Audio waveforms ``[B, T_samples]``.
            audio_length: Valid sample counts ``[B]``.
            labels: Target labels for supervised training.
            runtime_gather_output: If True, gather outputs across pipeline stages.
            packed_seq_params: Packed-sequence parameters.
            loss_mask: Mask for loss computation.

        Returns:
            Language-model output (logits or loss, depending on mode).
        """
        decoder_input = None
        if self.pre_process:
            # [S, B, H] from the Megatron embedding; move to batch-first for scatter.
            text_embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None)
            text_embeds = text_embeds.transpose(0, 1).contiguous()  # [B, S, H]

            if audio_signal is not None:
                audio_embeds, audio_lengths = self.audio_tower(audio_signal, audio_length)
                audio_embeds = self.multi_modal_projector(audio_embeds)  # [B, T_audio, H]
                text_embeds = scatter_audio_into_text(
                    text_embeds, input_ids, audio_embeds, audio_lengths, self.audio_token_id
                )

            # Back to Megatron [S, B, H] layout.
            decoder_input = text_embeds.transpose(0, 1).contiguous()

        return self.language_model.forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
        )

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_audio_model: bool,
        freeze_audio_projection: bool,
    ) -> None:
        """Freeze selected submodules by setting ``requires_grad = False``.

        Args:
            freeze_language_model: Freeze the EuroLLM backbone.
            freeze_audio_model: Freeze the Canary encoder (``audio_tower``).
            freeze_audio_projection: Freeze the audio projector.
        """
        modules = []
        if freeze_language_model and getattr(self, "language_model", None) is not None:
            modules.append(self.language_model)
        if freeze_audio_model and getattr(self, "audio_tower", None) is not None:
            modules.append(self.audio_tower)
        if freeze_audio_projection and getattr(self, "multi_modal_projector", None) is not None:
            modules.append(self.multi_modal_projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
