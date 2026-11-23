# theia_mlp handoff

Work-in-progress notes for branch `feat/theia-mlp-policy` (remote `nepyope`). Delete the `dev/` folder
before opening a PR.

## Setup

```bash
git fetch nepyope feat/theia-mlp-policy && git checkout feat/theia-mlp-policy
uv sync --locked --extra theia_mlp --extra test --extra dataset --extra training
```

If you change extras in `pyproject.toml`, relock with `uvx uv@0.11.30 lock` (the CI version). A newer
uv rewrites the whole lockfile.

## What exists

A new LeRobot policy `theia_mlp` in `src/lerobot/policies/theia_mlp/`, registered in
`src/lerobot/policies/__init__.py`, with a `theia_mlp` extra (transformers) included in `all`.

- **Backbone:** frozen Theia-Tiny (`theaiinstitute/theia-tiny-patch16-224-cddsv`). Only its ViT backbone
  is used, built as a stock `transformers` `ViTModel` with the DeiT-tiny architecture (192 hidden,
  12 layers, 3 heads, 768 MLP, patch 16, `layer_norm_eps=1e-12`, no pooler). Weights come from the
  checkpoint's `backbone.model.*` keys with a strict load. No `trust_remote_code`.
- **Images:**
  - values in [0, 1], 1 or 3 channels; 1-channel images (depth) are repeated to RGB;
  - resized to 224x224; `image_resize_shape=None` keeps the native size and interpolates the position
    embeddings;
  - ImageNet-normalized inside the model;
  - the CLS token is dropped.
- **Pooling:** per camera, `AttentionPool` (4 learned queries, `nn.MultiheadAttention`) or mean pooling.
- **Head:** LayerNorm, then [512, 512] GELU + dropout 0.1, then `chunk_size * action_dim` outputs.
  Defaults are `chunk_size=10` and `n_action_steps=10`.
- **Loss and optimizer:** masked L1 loss like ACT; AdamW at lr 1e-4.
- **Normalization:** `VISUAL` is `IDENTITY`; `STATE` and `ACTION` are `MEAN_STD`, via
  `make_default_pre_post_processors`.

Nothing has been run yet beyond `ruff format` and `ruff check`.

## TODO, in order

1. **Parity check:** run `uv run python dev/theia_mlp/parity_check.py`.
   - Our patch tokens should match Theia's remote-code backbone to about 1e-5.
   - The remote code targets transformers 4.41 and may not load on 5.22. If it doesn't, at least
     confirm that the strict state-dict load succeeds and that the output shape is (B, 196, 192).
   - Report which of the two you verified.
2. **Tests:** add `tests/policies/theia_mlp/test_theia_mlp.py`. Use `pretrained_vision_weights=False`
   so no network is needed, and `pytest.importorskip("transformers")`. Cover:
   - config validation: bad `pooling`, `n_action_steps > chunk_size`, a non-Theia `vision_model_id`,
     a 2-channel image;
   - forward loss on a fake batch (`action` shaped (B, chunk, A) plus `action_is_pad`), and backward;
     the backbone gets no gradients when frozen;
   - `select_action` queue behaviour and `predict_action_chunk` shape;
   - 1-channel (depth) input, mean pooling, and `image_resize_shape=None` at a non-224 size;
   - `get_optim_params`: no vision parameters when frozen, two groups when unfrozen;
   - `save_pretrained` / `from_pretrained` round trip.

   Run with `uv run pytest tests/policies/theia_mlp -svv`.
3. **Smoke training:**

   ```bash
   uv run lerobot-train --policy.type=theia_mlp --dataset.repo_id=lerobot/pusht --steps=200 \
     --batch_size=32 --policy.push_to_hub=false --output_dir=outputs/theia_mlp_smoke
   ```

   - Confirm the loss decreases and the checkpoint reloads.
   - Report parameter counts: expect about 5.5M frozen backbone plus about 0.6M trainable for one
     camera.
   - Report inference latency of `select_action` on CPU and GPU.
4. **Lint, commit, push:** run `pre-commit run --files <changed files>` (includes mypy and bandit), then
   commit and push to `nepyope`. Do not open a PR unless asked.
