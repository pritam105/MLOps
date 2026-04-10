# Lab Assignment 6 — 20 Newsgroups Text Classification with W&B

Text classification on the [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) dataset using a Keras **Embedding + Conv1D** model, with experiment tracking via Weights & Biases.

## What the model does

Tokenizes raw newsgroup articles, pads them to fixed length, and trains a 1D CNN to classify each article into one of 20 newsgroup categories.

```
Input (token sequence)
  → Embedding(vocab=20k, dim=64)
  → Conv1D(128 filters, kernel=5) + ReLU
  → GlobalMaxPooling
  → Dropout(0.3)
  → Dense(20, softmax)
```

## What is logged to W&B

| What | How |
|---|---|
| Loss & accuracy (every 10 batches) | `WandbMetricsLogger` |
| Learning rate per epoch | `LogLRCallback` |
| 32 sample predictions (text snippet, true label, predicted label, confidence) | `LogSamplesCallback` → W&B Table |
| 20×20 confusion matrix per epoch | `ConfusionMatrixCallback` → W&B Plot |
| Trained model + summary | `wandb.Artifact` |
| Model checkpoint per epoch | `WandbModelCheckpoint` |

## How to run

Open `LabAssignment6.ipynb` in Google Colab with a **GPU runtime**, then run all cells top to bottom. Paste your W&B API key from [wandb.ai/settings](https://wandb.ai/settings) when prompted.

## Results

Screenshots of the W&B dashboard (charts, confusion matrix, config + summary) are in the [`asset/`](asset/) directory.
