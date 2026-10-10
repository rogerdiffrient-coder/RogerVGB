# Roger Spark cloud training (experimental)

You can run a much larger training experiment without training on your Mac:

1. Open the repository's **Actions** tab.
2. Select **Train Roger Spark (experimental)**.
3. Click **Run workflow**, choose **10000** additional steps for the full experiment, and start it.
4. The workflow trains on GitHub's CPU runner, exports the updated browser model, and commits the generated `webgpu/model/` files.

The checked-in training corpus is now about 13.8 times the original starter file and contains original instructional examples about programming, debugging, math, model training, and clear communication. The workflow's maximum/default run is 10,000 steps, ten times the previous 1,000-step maximum.

## Checkpoint persistence

The workflow caches the full-precision checkpoint and optimizer state between runs. This avoids repeatedly reconstructing the training checkpoint from the lossy INT8/FP16 browser export. If GitHub evicts the cache, the workflow can bootstrap from the published export instead.

## Important limitations

- This is still a small experimental model, not a 10x-smarter assistant. Training steps and dataset size do not translate directly into intelligence.
- The expanded corpus is a hand-written starter corpus, not a professionally curated large language-model dataset. A small model trained repeatedly on it can memorize it.
- The workflow is CPU-only. The model architecture stays at the tiny test size because the browser runtime and available runner are not configured for the 250M-parameter architecture in `model/config.json`.
- The workflow runs only when you start it from the Actions tab; it does not train automatically on every push.
- A successful run or lower training loss is not proof of better answers. Test multiple fixed prompts after each release.

For local training details and checkpoint resume support, see
[model/README.md](../../model/README.md).
