# Roger Spark 0.1 — scratch model

This folder contains an original, small decoder-only Transformer architecture. Its weights start from random initialization; it does **not** load Qwen, Llama, or any pretrained model. The included dataset is a tiny starter dataset, not enough to make a generally intelligent chatbot. Expect weak answers until you expand the dataset and train longer.

## Train on your Mac

Python 3.10+ recommended. PyTorch uses Apple MPS automatically when available.

```bash
cd training
python -m pip install -r requirements.txt
python train.py --steps 2500
python server.py
```

The model checkpoint is written to `training/roger_spark.pt` and is intentionally not committed to Git because it is generated binary output. Increase training data before expecting general conversational ability. Training longer on a tiny dataset can make it memorize and repeat examples rather than become smarter.

## Connect a frontend

POST JSON to `http://127.0.0.1:8000/api/chat`:

```json
{"message":"Is the sun hot?","temperature":0.8,"top_p":0.9,"max_new_tokens":160}
```

The API responds with a `reply` string. The starter training example is exactly:

- Prompt: `Is the sun hot?`
- Target: `yeah the sun is just so very very very very very hot`

Because this is a tiny randomly initialized model, the generated answer is **not guaranteed** to match the target after training. This is a genuine learned-weight training objective, not a hard-coded answer rule.

## Important limitations

- This is a custom architecture and custom-trained weights, but it is a very small educational model. It is not comparable to large language models.
- The tokenizer maps UTF-8 bytes to IDs; this keeps tokenization custom and simple, but is less efficient than a trained subword tokenizer.
- This API binds to localhost only. GitHub Pages cannot run Python by itself; to use this backend from a public website, deploy it separately and configure CORS/origin restrictions.
- WebGPU in a browser still runs computation on the user's device. This Python server uses PyTorch's MPS/CUDA/CPU backend, not browser WebGPU.


## Hosted website deployment (no visitor setup)

The repository includes `.github/workflows/train-and-deploy-space.yml` and a Docker Space bundle in `space/`. The workflow trains this custom model from random initialization, then publishes the checkpoint and API to Hugging Face Spaces. The public site reads its endpoint from `api-config.js`; the workflow writes that public URL automatically after deployment. Visitors do not install Python or download a model manually.

One-time owner setup is required because this repository cannot create credentials for your Hugging Face account:

1. Create a Hugging Face account and make a **write** access token at https://huggingface.co/settings/tokens.
2. In GitHub, open this repository's **Settings → Secrets and variables → Actions**. Add repository secret `HF_TOKEN` with that token.
3. Under **Variables**, add repository variable `ROGER_SPARK_SPACE` with a Space ID such as `your-huggingface-username/roger-spark` (the workflow creates the Docker Space if needed).
4. Open **Actions → Train and deploy Roger Spark → Run workflow**. The first CPU training run can take a while; wait for the job and Space build to finish.
5. Ensure GitHub Pages is enabled for the branch/folder that serves this website. The workflow commits `api-config.js`, which lets the site connect automatically.

The Space endpoint and model are public; never put `HF_TOKEN` in `api-config.js` or any committed file. Hugging Face's free Spaces may sleep or have compute limits. This is a tiny experimental model, not a pretrained general-purpose chatbot, so its answer quality will be limited.
