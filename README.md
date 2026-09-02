# VenTX-100K

A 226M-parameter decoder-only transformer, trained from scratch by [Narla Venkata Anand Sai Kumar](https://github.com/Narla-Venkata-Anand-Sai-Kumar), targeting native long-context support (32,768 tokens now, with a path to 65,536-100,000 as a later context-extension stage) on a single consumer GPU.

**Status: base pretraining in progress.** This is a work-in-progress research project, not a finished model — see [Training status](#training-status) below for exactly where it stands right now. Weights are published as-is at whatever checkpoint is attached to the latest [Release](../../releases), specifically so the process is visible, not just the end result.

## Architecture

Same proven decoder-only design as this author's earlier [Chaos-135M](https://github.com/Narla-Venkata-Anand-Sai-Kumar) project, scaled up:

| | |
|---|---|
| Parameters | 226.1M (measured) |
| Layers | 14 |
| d_model | 960 |
| Attention heads | 15 query / 5 KV (grouped-query attention, 3:1 ratio) |
| Head dim | 64 |
| FFN hidden | 3,584 (SwiGLU) |
| Vocab | 49,152 |
| Positional encoding | RoPE |
| Normalization | RMSNorm, with QK-norm |
| Embeddings | Tied (input/output share weights) |
| Context length (current) | 32,768 tokens, native dense attention (no sliding window / no interleaved local-global compromise) |
| Optimizer | Muon (Newton-Schulz orthogonalized momentum) for all 2D matrix weights, AdamW for embeddings/norms |

**Design philosophy**: native long context, not a bolted-on compromise. Every layer attends to the full sequence at whatever length it's trained at, rather than approximating long context with sliding windows or attention sinks.

## Training status

Training is running now, resumable via `--init`/`--resume` and tracked in real commit history. Check the [training log](https://github.com/Narla-Venkata-Anand-Sai-Kumar/ventx) or the latest [Release](../../releases) notes for the current step count and loss — this README won't be kept in perfect sync with a training run that updates every ~50 seconds.

**Corpus**: ~4.27B tokens, a 70/30 mix of long-form documents (BookCorpusOpen, Project Gutenberg) and short-form web/edu/math/code content, chosen specifically so native long-context training sees genuinely continuous long documents, not concatenated short ones.

**Hardware**: developed and trained on a single consumer GPU (RTX 5060, 8GB), with some stages run on rented cloud GPUs (RTX 4090, B200) when budget allowed — the codebase is written to run on either without changes.

## Repository layout

- `ventx/` — model (`model.py`), config (`config.py`), Muon optimizer (`muon.py`), data loading (`data.py`), training loop (`train.py`), KV-cache generation (`generate.py`), evaluation (`evaluate.py`)
- `scripts/` — corpus download/tokenization, live training dashboard
- `bench/` — single-configuration throughput/memory benchmarking harness

## Training your own

```bash
python scripts/download_longdocs.py
python scripts/tokenize_longdocs.py
python ventx/train.py --run-name my-run --data-dir <tokenized-dir> \
  --total-steps 32500 --seq-len 32768 --micro-batch 1 --grad-accum 4
```

Resume an interrupted run with `--resume <checkpoint>` (full optimizer state); start a new stage from another checkpoint's weights only with `--init <checkpoint>` (fresh optimizer).

## License

No license has been chosen yet for this repository's code. All rights reserved by the author until one is added.
