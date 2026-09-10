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

"""An inference backend that runs the policy on another machine.

This is the wire half of :mod:`.backend`.  The engine above it is unchanged — it
still calls ``predict_chunk`` once per cycle and times the call — so Real-Time
Chunking works across a network for the same reason it works in one process: one
call is one inference, and the round trip *is* the delay the next chunk is
conditioned on.

The contract is deliberately total: ``PredictChunk`` always answers with a chunk.
The server does not get to decide that an observation was not worth running, the
way ``policy_server``'s streaming path does with its similarity check. That
matters because the engine only asks when its queue is draining, so a refusal
would be the server overriding the one component that knows a chunk is needed —
which is the problem ``must_go`` exists to paper over on the streaming path.

Everything that is not an inference (capability query, episode reset) goes over
``BackendControl`` instead, to keep that property.
"""

from __future__ import annotations

import logging
import pickle  # nosec B403: the peer is a policy server the operator started
from dataclasses import dataclass
from enum import Enum
from typing import Any

import grpc

from lerobot.transport import (
    services_pb2,  # type: ignore[attr-defined]
    services_pb2_grpc,  # type: ignore[attr-defined]
)
from lerobot.transport.utils import MAX_MESSAGE_SIZE, send_bytes_in_chunks

from .backend import ChunkRequest, ChunkResult, InferenceBackend
from .base import PolicyQuery

logger = logging.getLogger(__name__)

# How long a control call may take before we give up. Generous: a reset can land while
# the server is mid-inference.
_CONTROL_TIMEOUT_S: float = 30.0


class BackendCommandKind(Enum):
    """Non-inference requests a remote backend can make of its server."""

    INFO = "info"
    """Ask what the loaded policy supports.  Answered with :class:`RemoteBackendInfo`."""

    RESET = "reset"
    """Clear episode-scoped state on the server (policy + processors)."""

    GENERATE_TEXT = "generate_text"
    """Run the policy's text head.  Answered with a ``str``."""


@dataclass(frozen=True)
class BackendCommand:
    """A pickled non-inference request. ``payload`` is kind-specific."""

    kind: BackendCommandKind
    payload: Any = None


@dataclass(frozen=True)
class RemoteBackendInfo:
    """What the server's policy supports, fetched once and cached.

    These live on the policy object rather than its config, so a client that never
    loads weights has to ask for them.
    """

    rtc_training_max_delay: int
    supports_text_generation: bool
    supports_rtc: bool


class RemoteInferenceBackend(InferenceBackend):
    """Runs the policy on a `policy_server`, over gRPC.

    Args:
        server_address: ``host:port`` of the policy server.
        policy_setup: What the server needs to load the same policy the caller would
            have loaded locally — checkpoint, revision, device, normalization stats and
            RTC config.  Sent on :meth:`connect`, and re-sent if the server restarts,
            since a fresh server has no policy.
        timeout_s: Deadline for a single chunk call. ``None`` waits indefinitely,
            which is usually wrong: a hung call stalls the engine's only inference
            thread while the robot drains its queue.
    """

    def __init__(
        self,
        server_address: str,
        policy_setup: Any,
        timeout_s: float | None = 10.0,
    ) -> None:
        self._server_address = server_address
        self._policy_setup = policy_setup
        self._timeout_s = timeout_s
        self._channel = grpc.insecure_channel(
            server_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )
        self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
        self._info: RemoteBackendInfo | None = None
        self._configured = False
        logger.info("RemoteInferenceBackend targeting %s", server_address)

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def connect(self, wait_s: float = 30.0) -> None:
        """Hand the server the policy to load, blocking until it is ready.

        Called before the engine starts, so a wrong address or an unloadable
        checkpoint fails here rather than as a stalled control loop.
        """
        # Ready() doubles as the connection attempt: its deadline covers dialing, so no
        # separate readiness future is needed (and none is left polling a closed channel).
        try:
            self._stub.Ready(services_pb2.Empty(), timeout=wait_s)
            self._stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=pickle.dumps(self._policy_setup)),
                timeout=wait_s,
            )
        except grpc.RpcError as e:
            raise RuntimeError(
                f"Could not set up the policy server at {self._server_address}: {e}. "
                "Check that it is running, reachable, and can load "
                f"'{self._policy_setup.pretrained_name_or_path}'."
            ) from e
        self._configured = True
        self._info = None
        logger.info("Policy server at %s loaded %s", self._server_address, self._policy_setup.policy_type)

    def _ensure_configured(self) -> None:
        """Re-declare the policy if the server lost it (a restart, or another client)."""
        if not self._configured:
            logger.warning("Policy server at %s needs re-configuring", self._server_address)
            self.connect()

    # ------------------------------------------------------------------
    # Control path
    # ------------------------------------------------------------------

    def _control(self, command: BackendCommand) -> Any:
        self._ensure_configured()
        try:
            reply = self._stub.BackendControl(
                services_pb2.BackendCommand(data=pickle.dumps(command)),
                timeout=_CONTROL_TIMEOUT_S,
            )
        except grpc.RpcError:
            # The peer may have restarted, in which case it no longer holds our policy.
            self._configured = False
            raise
        return pickle.loads(reply.data)  # nosec B301

    @property
    def info(self) -> RemoteBackendInfo:
        """Server capabilities, fetched on first use and cached for the session."""
        if self._info is None:
            self._info = self._control(BackendCommand(kind=BackendCommandKind.INFO))
            logger.info("Remote backend reports %s", self._info)
        return self._info

    @property
    def rtc_training_max_delay(self) -> int:
        return self.info.rtc_training_max_delay

    @property
    def supports_text_generation(self) -> bool:
        return self.info.supports_text_generation

    def reset(self) -> None:
        self._control(BackendCommand(kind=BackendCommandKind.RESET))

    def generate_text(self, obs_frame: dict, task: str, query: PolicyQuery) -> str:
        return self._control(
            BackendCommand(kind=BackendCommandKind.GENERATE_TEXT, payload=(obs_frame, task, query))
        )

    # ------------------------------------------------------------------
    # Inference path
    # ------------------------------------------------------------------

    def select_action(self, obs_frame: dict, task: str, *, drop_queued: bool):
        raise NotImplementedError(
            "The remote backend serves action chunks, not per-tick actions. A remote "
            "`select_action` would put one network round trip inside every control tick; "
            "use --inference.type=rtc, whose queue absorbs the latency by design."
        )

    def predict_chunk(self, request: ChunkRequest) -> ChunkResult:
        self._ensure_configured()
        payload = pickle.dumps(request)
        try:
            response = self._stub.PredictChunk(
                send_bytes_in_chunks(payload, services_pb2.ChunkRequest, log_prefix="[RemoteBackend]"),
                timeout=self._timeout_s,
            )
        except grpc.RpcError:
            self._configured = False
            raise
        result = pickle.loads(response.data)  # nosec B301
        if not isinstance(result, ChunkResult):
            raise TypeError(f"Policy server returned {type(result).__name__}, expected ChunkResult")
        return result

    def close(self) -> None:
        """Close the channel. Safe to call twice."""
        self._channel.close()
