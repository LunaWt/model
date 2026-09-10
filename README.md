# model

Two mixture-of-experts language models trained on free Kaggle quota, and the engineering
needed to make that quota enough. A research sandbox in active development, not a product:
negative results are kept and abandoned experiments are allowed.

The binding resource is not storage or money. It is throughput. A whole 9-hour TPU session
at the current 45 638 tok/s consumes 1.48B tokens, so a 100B-token pretrain would be
**31 weeks** of a weekly quota that does not accumulate. Speed work is the project rather
than an optimisation of it, and it is not finished.

## The two lines

| | big | small |
|---|---|---|
| accelerator | Kaggle TPU v5e-8, 20 h/week | Kaggle 2×T4, 30 h/week |
| framework | JAX / XLA | PyTorch |
| programme | pretrain → SFT → RL, then context extension | pretrain + SFT, plus an experiment on spotting misalignment in actions and chains of thought |

Separate weekly counters, so the two do not compete. Both are looped: the same layers run
more than once per forward, which buys depth without buying parameters.

**Sizes and the final configuration are deliberately not stated here.** They are still
moving — int8 masters alone could roughly double what fits on a chip, and the memory
measurements below are the constraint, not a decision.

JAX for the big line is not a preference. It is the only option on TPU, and on the local
GTX it beats `torch.compile` by 1.24–1.46×; on T4 torch wins by 2× per card, which is why
the small line stayed in PyTorch.

## What is being tried

Kimi K3 is the starting point — hybrid sliding-window and full attention, Kimi Delta
Attention, a latent MoE with quantile balancing, Muon alongside AdamW, bf16 throughout.
`notes/k3_architecture.md` is the page-by-page reading of the tech report and is the working
replacement for the PDF. What survives contact with the hardware is decided by measurement,
and the measurements are below.

## What is measured

Every number has a run behind it in `notes/ledger.md`; nothing is quoted from the paper as
if it were our result.

- **The optimizer silently promoted bf16 parameters to fp32**, so the timing loop had been
  running a second compilation of a different program. Fixing it moved a dense 1.12B step
  from **43 134 to 60 710 tok/s at T=2048** (+41%). Every measurement taken before that fix
  is 10–40% low, and the ledger marks them.
- **Where the 718 ms step goes**, profiled through the XLA xplane: cross-entropy head 25.6%,
  `all_to_all` 14.6% plus the capacity buffer 12.1%, attention's fp32 score tensor 12.3%,
  layer-stack slicing 11.5%. None of it is arithmetic, which is why int8 buys memory rather
  than speed — matmuls are under a fifth of the step, so halving them is at most 1.09×.
- **The memory ceiling is sharp**: 22.22B fits at 13.40 of 15.75 GB per chip, 29.47B refuses.
- **Expert parallelism moves 4.7× less collective traffic** than the ZeRO-3 layout, and it
  changes what batching is worth: under ZeRO-3 growing the batch bought nothing, under
  `expert` + `ep` it is +9…+11%.
- **Long context is cheap once the window is in**: 4× the sequence length costs 4.7% per
  token. Sliding window is free at T=2048 and worth +30% at T=8192, and the 3:1 hybrid that
  every frontier vendor now uses costs 2.2% over all-windowed.
- **Total parameters are nearly free in time on TPU** — 24 → 72 experts at constant active
  mass costs 6% — but they were **not** free in loss on the small model, where 80 experts
  lost to 64 at identical active mass and ran 33% slower.
- **Fixed expert capacity makes the model non-causal in the token axis.** A token's output
  depends on which other tokens share its forward: a 5-token prefill and a 9-token pass
  disagreed by 0.38 in logits. Generation lifts the cap; training never does.
- **Quantile balancing is an alternating solver, not a formula.** One iteration per step
  leaves peak/ideal expert load at 2.34; running it to convergence gives 1.08.
- **The router must be fp32.** In bf16, 6% of tokens take a different top-k at init purely
  from rounding. It costs ~9% of throughput and it is not optional.
- **Looped won its first paired comparison**: 12 layers with a loop over four of them beat a
  flat 16-layer model of the same size by 0.05–0.15 nats at every checkpoint, against a seed
  noise of 0.033.
- **`fla.ops.kda.chunk_kda` is 38× faster** than our chunkwise KDA and agrees with it to
  1.2e-6 in fp32. Our two implementations stay as the reference pair.
- **The local GTX 1660 Ti is a development box, not a measurement device.** A kernel launch
  costs 40 µs there against 5–10 on native Linux, and 92% of that is the WSL→Windows driver
  path — 40.2 µs drops to 3.1 µs under CUDA graphs. Anything timed there is launch-bound.

## Benchmark discipline

Every expensive mistake in the ledger was invisible because a harness printed a *result*
without the evidence behind it. So each benchmark row also emits the compiled program's
memory analysis, the summed shapes of every collective in the HLO, the compilation-cache
size after two calls, a `(dtype, shape, sharding)` diff of every parameter across one step,
and live bytes from the device allocator. A cache size above one means the timing loop ran a
different program than the one described; parameter drift names the cause directly. Both
raise an inline alarm, and each alarm was checked against a deliberately broken control
before being trusted.

## Layout

- `model/` — `kda.py` and `kda_head.py` (Kimi Delta Attention, chunkwise and recurrent),
  `model.py`, `losses.py` (chunked cross-entropy), `optim.py` (Muon + AdamW), `data.py`
  (memmap loader over token shards, deterministic in `(seed, step)`), `generate.py`
  (sampling with MLA-latent, KDA-state and short-conv caches), `configs.py`,
  `compile_patch.py` (bf16 `torch.compile` on sm_75, which needs help).
- `scripts/` — `train.py`; the JAX benchmark harnesses `bench_moe_jax.py` and
  `bench_dense_jax.py`; `kaggle_run.py` (push → status → logs → output, there is no
  interactive Kaggle session); `prof_ops.py` (reads the xplane locally); the tokenizer and
  corpus pipeline; `loop_budget.py`, `router_load.py`, `eval_ckpt.py`, `compile_check.py`.
- `tests/` — 176 tests over the KDA recurrence, the losses, the windowed attention, the
  tokenizer and encoder, the loader, Muon, the loop, MoE dispatch and router balance, the
  generation cache and the profiler reader.
- `notes/` — `ledger.md` (what is settled, with the number and the run that settled it),
  `k3_architecture.md`, `looped.md`.

## Run

```sh
uv sync
uv run pytest
uv run python -m scripts.bench_moe_jax --help      # what fits, and how fast
uv run python -m scripts.train --config A16 --steps 200
uv run python -m scripts.kaggle_run --help         # push a kernel, wait, pull the output
```

`uv run` always; the virtualenv is never activated by hand. Under WSL the repo has to live
on the Linux filesystem — the cross-filesystem I/O penalty on `/mnt/c` is large enough to
distort timings.

## Status

Heavily mid-development. The large pretrain has not been run, the last Kaggle experiment has
not been checked yet, and the throughput work that everything else waits on is unfinished:
the profile names four costs and none of them has been fixed.

`uv run pytest` is **175 passed, 1 failed**. The failure is known and sits in the ledger's
open list — `scripts/bench_dense_jax.py` still carries a duplicated, un-windowed copy of
`attend_chunked`, and `tests/test_attend_chunked.py::test_the_two_copies_have_not_drifted`
is the test that says so. It is committed red on purpose.

What does exist: the architecture, the training loop, the data pipeline, a benchmark harness
that reports honest numbers, and a profile that names what stands between 45 638 tok/s and
the ≥2× needed to make a token budget fit the quota. The open list at the end of
`notes/ledger.md` is current and is not a wish list — each entry says what was measured and
what was not.
