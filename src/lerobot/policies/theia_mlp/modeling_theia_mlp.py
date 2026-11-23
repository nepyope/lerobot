#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Theia-MLP policy: a frozen Theia ViT, a learned attention pool per camera and an MLP action head.

Theia (https://huggingface.co/papers/2407.20179) distills several vision foundation models into a small
DeiT backbone. Only that backbone is loaded here, as a stock `transformers` ViT, so no remote code runs.
"""

from collections import deque
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pretrained import PreTrainedPolicy
from .configuration_theia_mlp import THEIA_VARIANTS, TheiaMLPConfig, theia_variant

if TYPE_CHECKING or _transformers_available:
    from transformers import ViTConfig, ViTModel
else:
    ViTConfig = None
    ViTModel = None

THEIA_BACKBONE_PREFIX = "backbone.model."
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TheiaMLPPolicy(PreTrainedPolicy):
    """Frozen Theia patch tokens, attention-pooled per camera, mapped to an action chunk by an MLP."""

    config_class = TheiaMLPConfig
    name = "theia_mlp"

    def __init__(self, config: TheiaMLPConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = TheiaMLP(config)
        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        def is_vision(name: str) -> bool:
            return name.startswith("model.vision_encoder.")

        groups: list[dict[str, Any]] = [
            {"params": [p for n, p in self.named_parameters() if not is_vision(n) and p.requires_grad]}
        ]
        vision_params = [p for n, p in self.named_parameters() if is_vision(n) and p.requires_grad]
        if vision_params:
            groups.append({"params": vision_params, "lr": self.config.optimizer_lr_vision})
        return groups

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return one action, querying the model only when the queue of predicted actions is empty."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # The model returns (batch, n_action_steps, action_dim); the queue is indexed by time step.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a (batch, chunk_size, action_dim) chunk of actions."""
        self.eval()
        return self.model(batch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Compute the masked L1 loss between the predicted and the target action chunks."""
        actions_hat = self.model(batch)
        abs_err = F.l1_loss(actions_hat, batch[ACTION], reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        return l1_loss, {"l1_loss": l1_loss.item()}


class TheiaMLP(nn.Module):
    """The network behind `TheiaMLPPolicy`.

    Images go through the Theia ViT; its CLS token is dropped (Theia only distills the patch tokens). Each
    camera's patch tokens are summarized by `AttentionPool` (or averaged), concatenated with the state and
    environment state, and an MLP outputs `chunk_size` actions.
    """

    def __init__(self, config: TheiaMLPConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = TheiaVisionEncoder(config) if config.image_features else None

        pooled_dim = 0
        self.pools = nn.ModuleDict()
        if self.vision_encoder is not None:
            hidden_size = self.vision_encoder.hidden_size
            for index, _ in enumerate(config.image_features):
                if config.pooling == "attention":
                    self.pools[str(index)] = AttentionPool(
                        hidden_size, self.vision_encoder.num_heads, config.n_pooling_queries
                    )
                    pooled_dim += config.n_pooling_queries * hidden_size
                else:
                    pooled_dim += hidden_size

        state_dim = config.robot_state_feature.shape[0] if config.robot_state_feature else 0
        env_state_dim = config.env_state_feature.shape[0] if config.env_state_feature else 0
        self.action_dim = config.action_feature.shape[0]

        layers: list[nn.Module] = [nn.LayerNorm(pooled_dim + state_dim + env_state_dim)]
        in_dim = pooled_dim + state_dim + env_state_dim
        for hidden_dim in config.hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(config.dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, config.chunk_size * self.action_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        features = []
        if self.vision_encoder is not None:
            for index, key in enumerate(self.config.image_features):
                tokens = self.vision_encoder(batch[key])
                if self.config.pooling == "attention":
                    features.append(self.pools[str(index)](tokens))
                else:
                    features.append(tokens.mean(dim=1))
        if self.config.robot_state_feature:
            features.append(batch[OBS_STATE])
        if self.config.env_state_feature:
            features.append(batch[OBS_ENV_STATE])
        actions = self.head(torch.cat(features, dim=-1))
        return actions.view(-1, self.config.chunk_size, self.action_dim)


class TheiaVisionEncoder(nn.Module):
    """The Theia ViT backbone, returning (batch, num_patches, hidden_size) patch tokens.

    Takes images in [0, 1] with 1 or 3 channels; 1-channel images (e.g. depth) are repeated to RGB.
    """

    def __init__(self, config: TheiaMLPConfig):
        super().__init__()
        require_package("transformers", extra="theia_mlp")
        self.config = config
        variant = THEIA_VARIANTS[theia_variant(config.vision_model_id)]
        self.hidden_size = variant["hidden_size"]
        self.num_heads = variant["num_attention_heads"]
        # The architecture of facebook/deit-{tiny,small,base}-patch16-224, which Theia distills into.
        vit_config = ViTConfig(
            hidden_size=variant["hidden_size"],
            num_hidden_layers=12,
            num_attention_heads=variant["num_attention_heads"],
            intermediate_size=variant["intermediate_size"],
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-12,
        )
        self.vit = ViTModel(vit_config, add_pooling_layer=False)
        if config.pretrained_vision_weights:
            self.load_theia_weights(config.vision_model_id)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.frozen = config.freeze_vision_encoder
        if self.frozen:
            self.vit.requires_grad_(False)
            self.vit.eval()

    def load_theia_weights(self, model_id: str) -> None:
        state_dict = load_file(hf_hub_download(model_id, "model.safetensors"))
        backbone = {
            key.removeprefix(THEIA_BACKBONE_PREFIX): value
            for key, value in state_dict.items()
            if key.startswith(THEIA_BACKBONE_PREFIX)
        }
        self.vit.load_state_dict(backbone, strict=True)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        if self.config.image_resize_shape is not None and tuple(images.shape[-2:]) != tuple(
            self.config.image_resize_shape
        ):
            images = F.interpolate(
                images, size=tuple(self.config.image_resize_shape), mode="bilinear", antialias=True
            )
        images = (images - self.mean) / self.std
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            hidden = self.vit(pixel_values=images, interpolate_pos_encoding=True).last_hidden_state
        return hidden[:, 1:]


class AttentionPool(nn.Module):
    """Learned queries cross-attending over a camera's patch tokens; returns the flattened query outputs."""

    def __init__(self, dim: int, num_heads: int, num_queries: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * dim**-0.5)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> Tensor:
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(queries, tokens, tokens, need_weights=False)
        return self.norm(pooled).flatten(1)
