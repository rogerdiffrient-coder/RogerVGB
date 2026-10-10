# Roger Spark cloud training (experimental)

You can run a much larger training experiment without training on your Mac:

1. Open the repository's **Actions** tab.
2. Select **Train Roger Spark (experimental)**.
3. Click **Run workflow**, choose **10000** additional steps for the full experiment, and start it.
4. The workflow trains on GitHub's CPU runner, exports the updated browser model, and commits the generated `webgpu/model/` files.

The cloud model configuration in `model/cloud_config.json` targets about 2.05 million parameters (160-wide, six-layer Transformer with a 1,024-token vocabulary). The workflow offers 1,000, 3,000, 5,000, 10,000, 20,000, and 50,000 additional steps; the default is 10,000 and the job timeout is six hours.

## Checkpoint persistence

The workflow caches the full-precision checkpoint and optimizer state between runs. This avoids repeatedly reconstructing the training checkpoint from the lossy INT8/FP16 browser export. If GitHub evicts the cache, the workflow can bootstrap from the published export instead.

## Important limitations

- This is still a small experimental model, not a 10x-smarter assistant. Training steps and dataset size do not translate directly into intelligence.
- The expanded corpus is a hand-written starter corpus, not a professionally curated large language-model dataset. A small model trained repeatedly on it can memorize it.
- The workflow is CPU-only and uses a small multi-million-parameter model, not the much larger 250M-parameter local configuration in `model/config.json`. A 50,000-step run may exceed the six-hour limit. The checked-in hand-written corpus remains small, so more steps can overfit instead of reliably improving answers.
- The workflow runs only when you start it from the Actions tab; it does not train automatically on every push.
- A successful run or lower training loss is not proof of better answers. Test multiple fixed prompts after each release. Calculator correctness is intentionally not a release requirement; a calculator tool will be added separately.

For local training details and checkpoint resume support, see
[model/README.md](../../model/README.md).
