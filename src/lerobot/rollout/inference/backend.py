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

"""Where the model runs, as a seam the engines can be pointed at.

An engine owns the control-loop half of inference: pacing, the action queue, the
delay estimate, and the decision to keep or discard a chunk. A backend owns the
model half: device placement, the processor pipelines, and the policy call. The
two are separable because nothing in the first half needs the model in this
process, which is what makes a networked backend possible without a second
implementation of Real-Time Chunking.

The seam sits *below* the observation frame and *above* device placement. An
engine hands over a dataset frame — CPU arrays, the shape `build_dataset_frame`
produces — because that is the last point at which the data is cheap to move and
has not yet been bound to a device or a normalizer. Everything after it (prepare,
preprocess, infer, postprocess) belongs to whichever machine holds the weights.

Two consequences worth knowing:

- `predict_chunk` returns *both* the raw and postprocessed chunk. Real-Time
  Chunking conditions the next chunk on the un-postprocessed tail of the one
  before it, so a backend that returned only robot-space actions would make the
  algorithm impossible to run above it.
- The relative-action prefix rebase happens in here rather than in the engine.
  It reads state the preprocessor caches during *this* call, so it cannot be
  lifted above the seam without splitting preprocessing in half.
"""

from __future__ import annotations

import abc
import logging
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import reanchor_relative_rtc_prefix
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)

from .base import PolicyQuery, mark_query

logger = logging.getLogger(__name__)


def normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


@dataclass(frozen=True)
class ChunkRequest:
    """One Real-Time Chunking inference, described independently of where it runs.

    Attributes:
        obs_frame: Observation as a dataset frame, before device placement.
        task: Language instruction, snapshotted before the call so a `/subtask`
            landing mid-inference cannot relabel this chunk.
        inference_delay: Steps of overlap the chunk is conditioned on, i.e. how
            far the robot is expected to have advanced by the time it lands.
        prev_chunk_left_over: Unconsumed tail of the previous chunk, in the
            policy's own output space.
        prev_chunk_absolute: The same tail in robot space, needed only to rebase
            a relative-action prefix onto the current state.
        prefix_target_steps: Length the prefix is padded or truncated to, so a
            compiled graph sees one shape.
    """

    obs_frame: dict
    task: str
    inference_delay: int
    prev_chunk_left_over: torch.Tensor | None
    prev_chunk_absolute: torch.Tensor | None
    prefix_target_steps: int


@dataclass(frozen=True)
class ChunkResult:
    """A chunk in both the spaces its consumers need.

    Attributes:
        original: Policy output, un-postprocessed. Feeds the next request's
            `prev_chunk_left_over`.
        processed: Robot-space actions, for the control loop to execute.
    """

    original: torch.Tensor
    processed: torch.Tensor


class InferenceBackend(abc.ABC):
    """The model side of inference: processors, device, and the policy call."""

    @property
    @abc.abstractmethod
    def rtc_training_max_delay(self) -> int:
        """Largest overlap the checkpoint was trained to absorb, 0 if it declares none."""

    @property
    @abc.abstractmethod
    def supports_text_generation(self) -> bool:
        """Whether the policy has a text head, so `/vqa` and `/autosteer` can be served."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear episode-scoped model state (policy hidden state, processor caches)."""

    @abc.abstractmethod
    def select_action(self, obs_frame: dict, task: str, *, drop_queued: bool) -> torch.Tensor:
        """Return one action for `obs_frame`, as a CPU tensor.

        `drop_queued` asks a chunking policy to discard actions computed under a
        previous instruction; it is narrower than `reset`, which also drops
        observation history.
        """

    @abc.abstractmethod
    def predict_chunk(self, request: ChunkRequest) -> ChunkResult:
        """Run one chunk inference. See `ChunkResult` for why both spaces come back."""

    @abc.abstractmethod
    def generate_text(self, obs_frame: dict, task: str, query: PolicyQuery) -> str:
        """Answer `query` against `obs_frame` using the policy's text head."""

    def close(self) -> None:  # noqa: B027 — the no-op default is deliberate
        """Release whatever holds the model.

        Empty on purpose rather than abstract: an in-process backend has nothing to
        release, and only a networked one should have to say so.
        """


class LocalInferenceBackend(InferenceBackend):
    """Runs the policy in this process, on `device`."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        device: str | None,
        robot_type: str,
        fallback_action_names: list[str] | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type

        # Processor introspection for relative-action re-anchoring. Owned here because
        # the rebase needs the step's per-call cached state, which only exists on the
        # side that runs the preprocessor.
        self._relative_step = next(
            (
                s
                for s in getattr(preprocessor, "steps", [])
                if isinstance(s, RelativeActionsProcessorStep) and s.enabled
            ),
            None,
        )
        self._normalizer_step = next(
            (s for s in getattr(preprocessor, "steps", []) if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        if self._relative_step is not None:
            if self._relative_step.action_names is None:
                cfg_names = getattr(getattr(policy, "config", None), "action_feature_names", None)
                self._relative_step.action_names = (
                    list(cfg_names) if cfg_names else list(fallback_action_names or [])
                )
            logger.info("Relative actions enabled: RTC prefix will be re-anchored")

    @property
    def relative_actions_enabled(self) -> bool:
        """Whether the prefix has to be rebased onto the current state before each call."""
        return self._relative_step is not None

    @property
    def rtc_training_max_delay(self) -> int:
        return int(getattr(getattr(self._policy, "config", None), "rtc_training_max_delay", 0))

    @property
    def supports_text_generation(self) -> bool:
        return self._policy.supports_text_generation()

    def reset(self) -> None:
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def _autocast(self):
        use_amp = getattr(getattr(self._policy, "config", None), "use_amp", False)
        if self._device.type == "cuda" and use_amp:
            return torch.autocast(device_type=self._device.type)
        return nullcontext()

    def select_action(self, obs_frame: dict, task: str, *, drop_queued: bool) -> torch.Tensor:
        # Shallow copy is intentional: callers build ``obs_frame`` fresh per tick via
        # ``build_dataset_frame``, so the tensor/array values are not shared with any
        # other reader.
        observation = copy(obs_frame)
        with torch.inference_mode(), self._autocast():
            if drop_queued:
                self._policy.drop_queued_actions()
            observation = prepare_observation_for_inference(observation, self._device, task, self._robot_type)
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        return action.squeeze(0).cpu()

    def predict_chunk(self, request: ChunkRequest) -> ChunkResult:
        obs_batch = prepare_observation_for_inference(
            copy(request.obs_frame), self._device, request.task, self._robot_type
        )
        obs_batch["task"] = [request.task]

        preprocessed = self._preprocessor(obs_batch)

        prefix = request.prev_chunk_left_over
        if prefix is not None:
            # A request may arrive from a machine with no accelerator, so the prefix is not
            # necessarily on our device already.  No-op when it is.
            prefix = prefix.to(self._device)
        if prefix is not None and self._relative_step is not None:
            # Rebase against the raw cached state so the leftover tail stays in
            # the training-time coordinate frame.
            raw_state = self._relative_step.get_cached_state()
            if raw_state is not None:
                prev_abs = request.prev_chunk_absolute
                if prev_abs is not None and prev_abs.numel() > 0:
                    prefix = reanchor_relative_rtc_prefix(
                        prev_actions_absolute=prev_abs,
                        current_state=raw_state,
                        relative_step=self._relative_step,
                        normalizer_step=self._normalizer_step,
                        policy_device=self._device,
                    )

        if prefix is not None:
            prefix = normalize_prev_actions_length(prefix, target_steps=request.prefix_target_steps)

        actions = self._policy.predict_action_chunk(
            preprocessed, inference_delay=request.inference_delay, prev_chunk_left_over=prefix
        )
        return ChunkResult(
            original=actions.squeeze(0).clone(),
            processed=self._postprocessor(actions).squeeze(0),
        )

    def generate_text(self, obs_frame: dict, task: str, query: PolicyQuery) -> str:
        with self._autocast():
            observation = prepare_observation_for_inference(
                copy(obs_frame), self._device, task, self._robot_type
            )
            observation = mark_query(observation, query)
            # Preprocessing stays outside inference_mode: a stateful step (relative-action
            # anchoring) caches what it sees, and an inference tensor in that cache would
            # fail the next time the chunk path touches it.
            observation = self._preprocessor(observation)
            with torch.inference_mode():
                # No str() coercion: _service_query validates the return value.
                return self._policy.generate_text(observation)
