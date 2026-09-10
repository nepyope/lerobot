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
"""Real-Time Chunking across a real gRPC channel.

These run an actual `PolicyServer` on an ephemeral port rather than mocking the stub,
because the thing under test *is* the wire contract: whether a chunk survives the round
trip in both spaces, and whether the unmodified rollout engine can drive it.
"""

from __future__ import annotations

import time
from concurrent import futures

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
# Module-scoped rather than a per-fixture decorator: wrapping a yield fixture hides the
# generator from pytest, which then hands the test the generator object itself.
grpc = pytest.importorskip("grpc", reason="grpcio is required (install lerobot[async])")


class _IdentityPipeline:
    """Stand-in for PolicyProcessorPipeline: no steps, identity transform."""

    steps = ()

    def __call__(self, batch):
        return batch

    def reset(self):
        pass


class _StubRTCPolicy:
    """Records the RTC arguments it was called with, so the wire can be checked."""

    class _Config:
        use_amp = False
        rtc_training_max_delay = 6
        robot_type = "mock"
        action_feature_names = ["j1.pos", "j2.pos"]

        @property
        def image_features(self):
            return {}

    def __init__(self, chunk_len: int = 10, action_dim: int = 2):
        self.config = self._Config()
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.calls: list[dict] = []
        self.reset_count = 0

    def to(self, *args, **kwargs):
        return self

    def reset(self):
        self.reset_count += 1

    def supports_rtc(self) -> bool:
        return True

    def supports_text_generation(self) -> bool:
        return True

    def generate_text(self, batch) -> str:
        return "a red cube"

    def predict_action_chunk(self, batch, inference_delay=0, prev_chunk_left_over=None):
        self.calls.append(
            {
                "task": batch.get("task"),
                "inference_delay": inference_delay,
                "prefix_shape": None if prev_chunk_left_over is None else tuple(prev_chunk_left_over.shape),
            }
        )
        # Non-zero and call-dependent, so a stale or duplicated chunk would show up.
        return torch.full((1, self.chunk_len, self.action_dim), float(len(self.calls)))


def _policy_setup(**overrides):
    from lerobot.async_inference.helpers import RemotePolicyConfig

    kwargs = {
        "policy_type": "pi05",
        "pretrained_name_or_path": "stub/checkpoint",
        "lerobot_features": {},
        "actions_per_chunk": 10,
        "device": "cpu",
        "robot_type": "mock",
    }
    kwargs.update(overrides)
    return RemotePolicyConfig(**kwargs)


@pytest.fixture
def served_policy():
    """A live PolicyServer whose checkpoint loading is stubbed, plus its address.

    The handshake is exercised for real — only ``from_pretrained`` and the processor
    construction are stubbed, since there is no checkpoint on disk in a unit test.
    """
    from lerobot.async_inference import policy_server as server_module
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.transport import services_pb2_grpc

    policy = _StubRTCPolicy()
    loaded: dict = {}

    class _StubPolicyClass:
        @staticmethod
        def from_pretrained(path, revision=None):
            loaded["path"] = path
            loaded["revision"] = revision
            return policy

    def _fake_get_policy_class(policy_type):
        loaded["policy_type"] = policy_type
        return _StubPolicyClass

    def _fake_make_pre_post(policy_cfg, **kwargs):
        loaded["processor_kwargs"] = kwargs
        return _IdentityPipeline(), _IdentityPipeline()

    server_module.get_policy_class = _fake_get_policy_class
    server_module.make_pre_post_processors = _fake_make_pre_post

    # The config's port is unused: the ephemeral port comes from add_insecure_port, so
    # parallel test runs cannot collide.
    policy_server = PolicyServer(PolicyServerConfig(host="localhost", port=8080))
    policy_server.shutdown_event.clear()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        yield f"localhost:{port}", policy, loaded
    finally:
        server.stop(grace=None)


@pytest.fixture
def remote_pair(served_policy):
    """A connected RemoteInferenceBackend plus the policy it is really driving."""
    from lerobot.rollout.inference.remote import RemoteInferenceBackend

    address, policy, _loaded = served_policy
    backend = RemoteInferenceBackend(server_address=address, policy_setup=_policy_setup())
    backend.connect()
    try:
        yield backend, policy
    finally:
        backend.close()


def _chunk_request(**overrides):
    from lerobot.rollout.inference.backend import ChunkRequest

    kwargs = {
        "obs_frame": {"observation.state": np.zeros(2, dtype=np.float32)},
        "task": "stack the blocks",
        "inference_delay": 0,
        "prev_chunk_left_over": None,
        "prev_chunk_absolute": None,
        "prefix_target_steps": 8,
    }
    kwargs.update(overrides)
    return ChunkRequest(**kwargs)


def test_chunk_survives_the_round_trip_in_both_spaces(remote_pair):
    """RTC conditions the next chunk on the un-postprocessed tail, so a backend that
    returned only robot-space actions would make the algorithm impossible above it."""
    from lerobot.rollout.inference.backend import ChunkResult

    backend, policy = remote_pair

    result = backend.predict_chunk(_chunk_request(inference_delay=3))

    assert isinstance(result, ChunkResult)
    assert result.original.shape == (10, 2)
    assert result.processed.shape == (10, 2)
    # Nothing device-resident may come back: the caller may have no accelerator.
    assert result.original.device.type == "cpu"
    assert result.processed.device.type == "cpu"
    assert policy.calls[0]["inference_delay"] == 3
    assert policy.calls[0]["task"] == ["stack the blocks"]


def test_prefix_crosses_the_wire_and_is_length_normalized(remote_pair):
    """The prefix is padded server-side, where the compiled graph's shape matters."""
    backend, policy = remote_pair

    backend.predict_chunk(
        _chunk_request(
            inference_delay=2,
            prev_chunk_left_over=torch.ones(5, 2),
            prev_chunk_absolute=torch.ones(5, 2),
            prefix_target_steps=8,
        )
    )

    assert policy.calls[0]["prefix_shape"] == (8, 2)


def test_capabilities_are_fetched_once_and_cached(remote_pair):
    """The client never loads weights, so these can only come from the server."""
    backend, _policy = remote_pair

    assert backend.rtc_training_max_delay == 6
    assert backend.supports_text_generation is True
    assert backend.info.supports_rtc is True
    # Second read must not re-query; the session's capabilities cannot change.
    assert backend._info is not None


def test_handshake_ships_the_stats_and_rtc_config_the_server_cannot_infer(served_policy):
    """Normalization stats and the RTC config only exist on the caller's side.

    Without the stats the two ends of the same policy normalize differently, which
    produces bad actions rather than an error; without the RTC config the server's
    policy has no chunking behaviour to condition at all.
    """
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference.remote import RemoteInferenceBackend

    address, policy, loaded = served_policy
    stats = {"observation.state": {"mean": [0.0, 1.0], "std": [1.0, 2.0]}}
    rtc_config = RTCConfig(enabled=True, execution_horizon=8, max_guidance_weight=1.0)

    backend = RemoteInferenceBackend(
        server_address=address,
        policy_setup=_policy_setup(
            dataset_stats=stats,
            rtc_config=rtc_config,
            pretrained_revision="abc123",
        ),
    )
    backend.connect()
    try:
        assert loaded["policy_type"] == "pi05"
        assert loaded["path"] == "stub/checkpoint"
        assert loaded["revision"] == "abc123"
        assert loaded["processor_kwargs"]["dataset_stats"] == stats
        assert loaded["processor_kwargs"]["pretrained_revision"] == "abc123"
        # Attached to the loaded policy, which is the only place it can be attached.
        # Equal rather than identical: it is a copy that crossed the wire.
        assert policy.config.rtc_config == rtc_config
    finally:
        backend.close()


def test_a_restarted_server_is_re_handed_the_policy(remote_pair, served_policy):
    """A fresh server holds no policy, so a client that just retries would fail forever."""
    backend, _policy = remote_pair
    _address, _policy2, loaded = served_policy

    backend.predict_chunk(_chunk_request())
    loads_before = loaded["path"]

    # Simulate the peer having lost our session (a restart, or another client claiming it).
    backend._configured = False
    result = backend.predict_chunk(_chunk_request())

    assert result.original.shape == (10, 2)
    assert loaded["path"] == loads_before, "re-handshake should reload the same checkpoint"


def test_reset_and_text_generation_travel_on_the_control_path(remote_pair):
    """Neither is an inference, so neither may pollute the chunk path's latency."""
    from lerobot.rollout.inference.base import PolicyQuery, QueryKind

    backend, policy = remote_pair

    backend.reset()
    assert policy.reset_count == 1

    answer = backend.generate_text(
        {"observation.state": np.zeros(2, dtype=np.float32)},
        "stack the blocks",
        PolicyQuery(kind=QueryKind.VQA, text="what do you see?"),
    )
    assert answer == "a red cube"


def test_select_action_is_refused_with_an_explanation(remote_pair):
    """A per-tick round trip is not a thing we want someone to discover at 30 Hz."""
    backend, _policy = remote_pair

    with pytest.raises(NotImplementedError, match="chunk"):
        backend.select_action({"observation.state": np.zeros(2, dtype=np.float32)}, "task", drop_queued=False)


def test_rtc_engine_drives_the_remote_backend_end_to_end(remote_pair):
    """The whole point: the unmodified engine runs RTC with the policy on another host."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference import RTCInferenceEngine

    backend, policy = remote_pair

    engine = RTCInferenceEngine(
        backend=backend,
        rtc_config=RTCConfig(enabled=True, execution_horizon=8, max_guidance_weight=1.0),
        hw_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
        },
        task="stack the blocks",
        fps=30.0,
    )
    engine.start()
    try:
        engine.resume()
        engine.notify_observation({"j1.pos": 0.0, "j2.pos": 0.0})

        deadline = time.perf_counter() + 10.0
        action = None
        while time.perf_counter() < deadline and action is None:
            action = engine.get_action(None)
            if action is None:
                time.sleep(0.02)

        assert action is not None, "no action ever came back over the wire"
        assert action.shape == (2,)
        assert engine.dispatched_task == "stack the blocks"
        assert policy.calls, "the server never ran an inference"
        # A second chunk must condition on the first one's leftover tail, which is the
        # part that only works if the raw chunk made it back across.
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline and len(policy.calls) < 2:
            engine.get_action(None)
            time.sleep(0.02)
        assert len(policy.calls) >= 2, "the engine never refilled the queue"
        assert policy.calls[1]["prefix_shape"] == (8, 2)
    finally:
        engine.stop()
