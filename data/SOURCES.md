# Training data sources

Roger Spark is trained from randomly initialized weights. The workflow adds a small, deterministic slice of public datasets at build time; it does not commit the expanded corpus or load pretrained model weights.

- `data/roger_training.txt`: hand-written project examples maintained in this repository.
- `databricks/databricks-dolly-15k`: instruction/context/response examples. The dataset is distributed under CC BY-SA 3.0; attribution and share-alike terms apply to reuse of the dataset and adaptations. Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- `roneneldan/TinyStories`: short synthetic stories, declared under CDLA-Sharing-1.0. Source: https://huggingface.co/datasets/roneneldan/TinyStories

The build uses a balanced slice—about 600,000 characters of instruction/dialogue data and 300,000 characters of stories—rather than allowing narrative-only text to dominate. Check dataset cards and licenses before changing sources or redistributing derived data.

More text does not guarantee better behavior. Evaluate held-out loss and real conversational samples before claiming the model is useful.
