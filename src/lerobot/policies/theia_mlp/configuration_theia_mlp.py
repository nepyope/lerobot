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
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

THEIA_VARIANTS: dict[str, dict[str, int]] = {
    "tiny": {"hidden_size": 192, "num_attention_heads": 3, "intermediate_size": 768},
    "small": {"hidden_size": 384, "num_attention_heads": 6, "intermediate_size": 1536},
    "base": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072},
}

POOLING_MODES = ("attention", "mean")


def theia_variant(vision_model_id: str) -> str:
    """Return the Theia size ("tiny", "small" or "base") named in a Hub model id."""
    name = vision_model_id.rsplit("/", 1)[-1]
    for variant in THEIA_VARIANTS:
        if name.startswith(f"theia-{variant}-"):
            return variant
    raise ValueError(f"`vision_model_id` must name a Theia model (theia-tiny/small/base-...). Got {name}.")


@PreTrainedConfig.register_subclass("theia_mlp")
@dataclass
class TheiaMLPConfig(PreTrainedConfig):
    """Configuration class for the Theia-MLP policy.

    A frozen Theia vision transformer (https://huggingface.co/papers/2407.20179) turns each camera image
    into patch tokens. A small learned attention pool summarizes each camera's tokens, and an MLP head maps
    the pooled features plus the robot state to a chunk of actions, trained with an L1 loss.

    Notes on the inputs and outputs:
        - At least one key starting with "observation.image" or the key "observation.environment_state"
          is required as an input. Several "observation.images.*" keys are treated as camera views.
        - Images may have 1 (e.g. depth) or 3 channels, with values in [0, 1].
        - May optionally work without an "observation.state" key.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations passed to the policy. Only 1 is
            supported.
        chunk_size: Number of future actions the head predicts per call, in environment steps.
        n_action_steps: Number of predicted actions executed before the policy is queried again. Set it
            to 1 to re-plan at every step.
        vision_model_id: Hub id of the Theia checkpoint. Only its ViT backbone is used.
        pretrained_vision_weights: Whether to load the Theia weights. `False` keeps the backbone randomly
            initialized, which is only useful for tests.
        freeze_vision_encoder: Whether to keep the Theia backbone frozen.
        image_resize_shape: (height, width) images are resized to before the backbone. `None` keeps the
            native resolution and interpolates the position embeddings; the token count then follows the
            image size (one token per 16x16 patch).
        pooling: How each camera's patch tokens are summarized: "attention" (learned queries) or "mean".
        n_pooling_queries: Number of learned queries per camera for attention pooling.
        hidden_dims: Hidden layer sizes of the MLP head.
        dropout: Dropout applied after each hidden layer of the MLP head.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 10
    n_action_steps: int = 10

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Vision encoder.
    vision_model_id: str = "theaiinstitute/theia-tiny-patch16-224-cddsv"
    pretrained_vision_weights: bool = True
    freeze_vision_encoder: bool = True
    image_resize_shape: tuple[int, int] | None = (224, 224)

    # Pooling and head.
    pooling: str = "attention"
    n_pooling_queries: int = 4
    hidden_dims: list[int] = field(default_factory=lambda: [512, 512])
    dropout: float = 0.1

    # Training preset.
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_vision: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        theia_variant(self.vision_model_id)
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"`pooling` must be one of {POOLING_MODES}. Got {self.pooling}.")
        if self.n_pooling_queries < 1:
            raise ValueError(f"`n_pooling_queries` must be positive. Got {self.n_pooling_queries}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        for key, feature in self.image_features.items():
            if feature.shape[0] not in (1, 3):
                raise ValueError(f"Images must have 1 or 3 channels. Got {feature.shape} for `{key}`.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
