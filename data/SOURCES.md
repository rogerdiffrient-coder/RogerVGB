# Roger Spark training data sources

## Original RogerVGB examples

- File: `roger_training.txt`
- Source: original project-authored explanations and conversation examples.

## TinyStories

- Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- Dataset card describes short stories generated with GPT-3.5 and GPT-4.
- Declared dataset license: [Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)](https://cdla.dev/sharing-1-0/).
- The workflow fetches a deterministic leading sample through the Hugging Face dataset viewer API at training time. The expanded corpus is generated locally in the workflow and is not committed to this repository.
- This material is used as general language-model pretraining text. It is not instruction data, so it should improve exposure to ordinary English more directly than it improves instruction-following.

The dataset's license and provenance should be reviewed before redistributing the raw or expanded text. RogerVGB does not include the downloaded stories in its Git history.
