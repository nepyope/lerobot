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

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging

import torch

from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from .backend import InferenceBackend
from .base import InferenceEngine, PolicyQuery

logger = logging.getLogger(__name__)


# TODO(Steven): support relative-action policies.  The per-tick flow refreshes
# ``RelativeActionsProcessorStep._last_state`` every call, so cached chunk
# actions popped on later ticks get reanchored to the *current* robot state and
# absolute targets drift through the chunk.  Relative-action policies are
# rejected at context-build time today; RTC postprocesses the whole chunk and
# is unaffected.
#
# Candidate fix: drive the policy via ``predict_action_chunk`` and serve a
# local FIFO of postprocessed actions.  Eliminates drift by construction and
# saves per-tick pre/post work, but bypasses ``select_action`` — needs
# fallbacks for SAC (raises), ACT temporal ensembling (ensembler lives in
# ``select_action``), and Diffusion-family (obs-history queues populated as a
# side effect of ``select_action``).


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    ``get_action`` runs the full policy pipeline (pre/post-processor +
    ``select_action``) on the given observation frame and returns a
    CPU action tensor reordered to match the dataset action keys.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
    ) -> None:
        super().__init__(task=task)
        self._backend = backend
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        logger.info(
            "SyncInferenceEngine initialized (backend=%s, action_keys=%d)",
            type(backend).__name__,
            len(ordered_action_keys),
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """Release the backend; no background resources of our own."""
        self._backend.close()
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._backend.reset()
        # The policy was just reset, so a pending task change has nothing stale to flush.
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if obs_frame is None:
            return None
        task, task_changed = self._take_task()
        if task_changed:
            # Chunking policies queue actions computed under the previous instruction,
            # so drop them and let the new one take effect on this tick.  Narrower
            # than ``policy.reset``: observation history and other episode state stay.
            logger.info("Task changed to '%s' — dropping precomputed actions", task)
        action_tensor = self._backend.select_action(obs_frame, task, drop_queued=task_changed)

        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        # ``task`` is the pre-inference snapshot: a /subtask landing mid-inference must
        # not relabel this action.
        self._set_dispatched_task(task)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    # ------------------------------------------------------------------
    # Text queries
    # ------------------------------------------------------------------

    @property
    def supports_text_queries(self) -> bool:
        """True when the policy has a text head."""
        return self._backend.supports_text_generation

    @property
    def control_thread_owns_policy(self) -> bool:
        """Inference runs inline on the control thread, so queries are served there too."""
        return True

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        """Run the policy's text head on the current observation."""
        obs_frame = build_dataset_frame(self._dataset_features, obs_processed, prefix=OBS_STR)
        # Live task, read without consuming the task-changed edge (the action path needs it).
        # Reusing the action path's preprocessor is safe only while its steps are stateless
        # per call; the one that is not (an enabled RelativeActionsProcessorStep) is
        # rejected for this engine at context-build time.
        return self._backend.generate_text(obs_frame, self.task, query)
