# Put the trained Roger Spark model on GitHub Pages

The browser app uses ONNX Runtime Web + WebGPU. A PyTorch `.pt` checkpoint from a GitHub Release cannot be loaded directly in the browser; it must first be exported to ONNX.

## 1. Get the published checkpoint

Download `roger_spark_final.pt` from the Roger Spark GitHub Release and put it at:

```text
training/roger_spark_final.pt
```

Make sure this is the 123,509,504-parameter, step-5,000 checkpoint.

## 2. Export the browser model

From the repository root, use a Python environment with the dependencies in `training/requirements.txt` installed:

```bash
python training/export_browser_onnx.py --checkpoint training/roger_spark_final.pt
```

The exporter writes `roger_spark.onnx` and updates `roger_spark_config.json` to report the actual 123M architecture and training step. It also makes sure the prompted-v4 tokenizer JSON is available at the repository root.

## 3. Deploy the model files

The GitHub Pages version needs these files to come from the same training run:

- `roger_spark.onnx`
- `roger_spark_config.json`
- `roger_spark_tokenizer.json`

Upload the ONNX file using a GitHub Release asset or Git LFS if it is too large for a normal Git commit. The browser app currently fetches the ONNX file from the same Pages origin as `index.html`, so for a Release-only asset you must also update `browser_model.js` to use that asset's direct download URL. Do not change the config to 123M until the matching ONNX file has been exported and deployed.

## Important

The export script does not upload files or publish GitHub Pages automatically. After the matching files are deployed, open the Pages site in a WebGPU-capable browser and test a prompt. A correct config file by itself does not replace the model weights.
