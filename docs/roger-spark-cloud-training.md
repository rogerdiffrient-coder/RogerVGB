# Roger Spark cloud training (experimental)

You can launch a small CPU training run without running Python or training on
your Mac:

1. Open the repository's **Actions** tab on GitHub.
2. Select **Train Roger Spark (experimental)**.
3. Click **Run workflow**, choose 100–500 additional steps, and start it.
4. When the workflow succeeds, it exports and commits the updated
   `webgpu/model/` files. If GitHub Pages deploys from `main`, the page will
   update after its normal deployment delay.

The workflow continues from the model currently published in `webgpu/model/`.
Because that browser export is quantized, it reconstructs approximate weights
and starts with a fresh optimizer state each run. It is CPU-only and deliberately
refuses models wider than 256 dimensions or deeper than 6 layers.

## Important limitations

- This is an experiment runner, not a magic intelligence button. The current
  model is the tiny test architecture and `data/roger_training.txt` is only a
  small demonstration corpus. Repeatedly training on it will mostly memorize
  that text; it will not turn Spark into a capable general-purpose assistant.
- Meaningful language generation needs a much larger, varied, permitted corpus,
  held-out evaluation, and more compute. GitHub-hosted runners do not provide a
  GPU in this workflow.
- The workflow only runs when you launch it from the Actions tab; it does not
  run on every push or silently change the model.
- The published weights are generated files. Do not treat a successful workflow
  or a lower training loss as proof that generated answers are correct.

For local training details and checkpoint resume support, see
[model/README.md](../../model/README.md).
