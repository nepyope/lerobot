# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from threading import Event

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .backend import InferenceBackend, LocalInferenceBackend
from .base import InferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30

    # ``host:port`` of a ``lerobot.async_inference.policy_server``.  Unset runs the
    # policy in this process.  Setting it moves only the model: the queue, the delay
    # estimate and the merge stay here, next to the robot, which is what lets RTC
    # measure a round trip and condition the next chunk on it.
    server_address: str | None = None
    # Deadline for one remote chunk call.  Should exceed the server's inference time by
    # a healthy margin; a call that times out costs the robot a queue refill.
    server_timeout_s: float = 10.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy | None,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
    backend: InferenceBackend | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object.

    ``backend`` chooses *where* the model runs and defaults to this process; the engine
    choice is independent of it.  ``policy`` may be ``None`` only when a ``backend`` is
    supplied, i.e. when the weights live elsewhere.
    """
    logger.info("Creating inference engine: %s", config.type)
    if backend is None:
        if policy is None:
            raise ValueError("create_inference_engine needs either a policy or a backend")
        backend = LocalInferenceBackend(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            robot_type=robot_wrapper.robot_type,
            fallback_action_names=[k for k in robot_wrapper.action_features if k.endswith(".pos")],
        )
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            backend=backend,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
        )
    if isinstance(config, RTCInferenceConfig):
        return RTCInferenceEngine(
            backend=backend,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
