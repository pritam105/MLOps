# Lab3 — Streaming LM Data Pipeline: TinyStories + BERT Tokenizer

An extension of Lab2's streaming pipeline with a different dataset and tokenizer to explore how these choices affect the preprocessing workflow.

## What it does

Builds a streaming data pipeline that loads, filters, tokenizes, and chunks text into fixed-length blocks ready for language model training — without loading the full dataset into memory.

## How it differs from Lab2

| | Lab2 | Lab3 |
|---|---|---|
| Dataset | WikiText-2 (Wikipedia) | TinyStories (short children's stories) |
| Tokenizer | GPT-2 BPE (~50k vocab) | BERT WordPiece (`bert-base-uncased`, ~30k vocab) |
| Pad token | Manual hack (`eos_token`) | Native `[PAD]` token — no workaround needed |
| Special tokens | Default | `add_special_tokens=False` — no `[CLS]`/`[SEP]` mid-stream |
| Filter step | None | Drops stories shorter than 100 characters |
| Block size | 128 tokens | 256 tokens |
| Output preview | Shape only | Shape + decoded text to verify readability |

## Key observations

- BERT's `uncased` tokenizer lowercases all text — visible in the decoded output.
- TinyStories produces coherent, readable chunks compared to WikiText-2's fragmented Wikipedia lines.
- A warning about sequence length > 512 may appear during tokenization — this is harmless since the rolling buffer chunks into 256-token blocks before any model sees the data.

## Requirements

```
pip install transformers datasets torch
```
