"""Compare TheiaVisionEncoder's patch tokens with Theia's own (remote-code) backbone.

Run: uv run python dev/theia_mlp/parity_check.py
"""

import torch
from transformers import AutoModel

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.theia_mlp.configuration_theia_mlp import TheiaMLPConfig
from lerobot.policies.theia_mlp.modeling_theia_mlp import TheiaVisionEncoder

MODEL_ID = "theaiinstitute/theia-tiny-patch16-224-cddsv"

cfg = TheiaMLPConfig(
    input_features={"observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))},
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    device="cpu",
)
encoder = TheiaVisionEncoder(cfg).eval()
torch.manual_seed(0)
images = torch.rand(2, 3, 224, 224)
with torch.no_grad():
    ours = encoder(images)
print("ours:", tuple(ours.shape))  # expect (2, 196, 192)

reference = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval()  # nosec B615
with torch.no_grad():
    x = (images - encoder.mean) / encoder.std
    ref = reference.backbone.model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
diff = (ours - ref).abs().max().item()
print(f"max abs diff vs remote Theia: {diff:.2e}")  # expect < 1e-4
assert diff < 1e-4
