# k3-small-lm-reproduction

Reimplementing the Kimi K3 architecture at a scale that fits one 6 GB GTX 1660 Ti — a card
with no tensor cores, where fp16 is measured 5× slower than bf16. This is a research
sandbox, not a product: the model, the tokenizer and the data pipeline are built, the
training run is not.

I love to train deep neural nets on large datasets… except my GPU is garbage. K3 is a hard
architecture and a powerful one, and I had never touched MoE or linear attention before, so
reimplementing it has been genuinely fun.

## Architecture

Config settled at **225.7M total / 48.7M active parameters**, context `T = 1536` — not 2048,
because activations cost ~1.5 MiB per token of context and 2048 does not fit in 6 GB.

| Component | Choice |
|---|---|
| Attention | hybrid **3 KDA : 1 gated MLA**, extra MLA at the end; 2 heads × 128 |
| Position | **NoPE** — KDA's decay already encodes position |
| Depth | 16 layers, **Block AttnRes** with N = 4 blocks |
| Width | **Stable LatentMoE**: RMSNorm before the up-projection, SiTU-GLU (β₁ = 4, β₂ = 25), quantile balancing |
| Sparsity | 64 routed experts, top-4, + 2 shared; first layer dense |
| Vocabulary | 16 384, byte-level BPE trained in this repo, digits split individually |
| Precision | **bf16** everywhere |
| Optimizer (planned) | per-head Muon + AdamW, cosine schedule with 1% warmup |

K3 runs sparsity 56 across 93 layers; at 200M parameters and 16 layers both the block
reduction and quantile balancing are outside the regime they were designed for, which is
the interesting part of doing this small.

## What's built

- `model/kda.py`, `model/kda_head.py` — Kimi Delta Attention, including the chunkwise form
- `model/model.py` — the full stack
- `model/losses.py` — chunked cross-entropy (the logit tensor is the memory bottleneck)
- `model/compile_patch.py` — bf16 `torch.compile` on sm_75, which needs help
- `scripts/` — tokenizer training and evaluation, corpus sampling, corpus → token shards
  (14.434B tokens encoded, groups kept separate so the mix can be chosen after the counts
  are known)
- `tests/` — 53 pytest tests over the KDA recurrence, the losses, the tokenizer and the
  encoder

Not built: training loop, optimizer, data loader, checkpointing.

## Measurements

- **KDA is the bottleneck, and it is a launch-rate problem.** The real chunkwise KDA runs
  at **498 tok/s** eager, against **5 800 tok/s** with KDA swapped for MLA. `chunk = 16` is
  a hard ceiling set by `1/Γ` overflow, so a 1536-token sequence is 96 sequential steps per
  layer — under WDDM that is launch-bound, not compute-bound.
- **`torch.compile` swallows most of it:** ×4.4 throughput and −1545 MiB, at a cost of
  **2 737 s (46 minutes)** of compilation on this card.
- **Router precision is not free.** In bf16, **6% of tokens** get a different top-4 than in
  fp32, because sigmoid scores at initialization are nearly tied. Still open.
- **fp16 is 5× slower than bf16** here — sm_75 has no tensor cores, so the usual advice
  inverts.

Recurrent linear attention was the hardest part by far. Give me back my QKV matmuls and
softmax…

## Run

```sh
uv sync
uv run pytest                       # 53 passed
uv run python scripts/train_tokenizers.py
uv run python scripts/eval_tokenizers.py
```

`uv run` always — the virtualenv is never activated by hand. The repo has to live on the
Linux filesystem, not under `/mnt/c`: the cross-filesystem I/O penalty is large enough to
distort timings.

## Status and honesty

Nothing here has been pretrained yet. The architecture runs, the tests pass, and the speed
work has to land before a real run makes sense. Everything reported above is a measurement
on this one card, taken from the project ledger; nothing is quoted from the paper as if it
were our result.
