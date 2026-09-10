# model — file map

Research sandbox for small-scale LM training on free Kaggle quota. Two lines live in this
repo: a JAX/XLA line for TPU v5e-8 and a PyTorch line for 2×T4 and the local GTX 1660 Ti.
`README.md` says what the project is and what has been measured; this file says where
everything is.

## `model/` — the PyTorch model

| file | what |
|---|---|
| `model.py` | the architecture: `K3Config`, `SiTUGLU` / `GroupedSiTUGLU`, `AttnRes` (block attention residuals), `GatedMLA` (latent attention, NoPE), `LatentMoE` (routed experts in the latent, quantile balancing), `K3Model` (3 KDA : 1 MLA hybrid, optional loop over a span of layers), `param_groups` |
| `kda_head.py` | the KDA layer. `kda_recurrent` is the per-token reference, `kda_chunkwise` the working chunkwise form, `kda_fla` the `fla` kernel; plus `ShortConv` and the `KDA` module that picks between them |
| `kda.py` | the bare KDA recurrence on 2×2 toy numbers, runnable as a script — start here if the delta rule is unclear |
| `losses.py` | `chunked_cross_entropy`: CE that never materialises the `(N, V)` logit tensor, recomputing it in backward. `plain_cross_entropy` is the reference |
| `optim.py` | `orthogonalize` (Newton–Schulz), `Muon` for matrices, `build_optimizers` (Muon/AdamW split, `tag_head_params` splits attention projections per head), `lr_multiplier` (warmup → cosine → floor) |
| `data.py` | memmap loader over `data/tokens/<group>/NNNN.bin` uint16 shards: `TokenStream`, `SlotPermutation` (an epoch yields every slot once), `GroupSchedule` (exact group shares per block), `MixedLoader`; deterministic in `(seed, step)` |
| `generate.py` | sampling with all three caches — MLA latent, KDA recurrent state, short-conv window |
| `configs.py` | named configs for `--config`: `A16` / `A24` unlooped controls, the looped variants matched to them, `M*` mid-size, `tiny` for second-long runs |
| `compile_patch.py` | lifts inductor's bf16 refusal on sm_75 and reports whether CUDA graphs actually formed |

## `scripts/` — training, benchmarks, data

Everything runs as a module: `uv run python -m scripts.<name> --help`.

**Training and evaluation**

| file | what |
|---|---|
| `train.py` | the pretraining loop: chunked CE, Muon + AdamW, gradient accumulation, eval, checkpoints under `runs/` |
| `eval_ckpt.py` | scores any set of checkpoints on one shared validation grid, so runs at different `T` stay comparable |
| `bench_prompts.py` | 30 prompts: free generation to read, plus forced choice between a sensible and a scrambled continuation for a number |
| `router_load.py` | how evenly a trained checkpoint spreads tokens over experts, and what each `capacity_factor` would drop |
| `loop_budget.py` | regenerates the config table in `notes/looped.md` — total params, executed params, KV slots |

**TPU / JAX line**

| file | what |
|---|---|
| `bench_moe_jax.py` | the main TPU harness: MoE grid over what fits and what is fast. Holds its own JAX model — attention (`attend_chunked`, `attend_splash`), `chunked_ce`, MoE dispatch (`moe_ffn_ragged`, `moe_ffn_ep`, cumsum/sort), AdamW, Adafactor, int8-quantised Adafactor, Muon |
| `bench_dense_jax.py` | the same dense transformer as `bench_dense_ref.py`, in JAX — one number comparable across TPU and GPU |
| `bench_tpu_arch.py` | single-chip v5e: dense vs MoE, full vs local attention |
| `kaggle_run.py` | push → poll → fetch a kernel; there is no interactive Kaggle session |
| `tpu_probe.py` | calls every doubtful library function on toy shapes, so a 40-minute queue is not spent finding an unimplemented op |
| `prof_ops.py` | reads the xplane a profiled run leaves behind and reports where the step went, by HLO op |

**Local GPU benchmarks**

| file | what |
|---|---|
| `bench_step.py` | one forward+backward on random tokens, every dtype the card can execute — no data, no compile |
| `bench_configs.py` | which configs fit and how fast, on a real training step including the optimizer; OOM is printed, not raised |
| `bench_dense_ref.py` | a deliberately plain dense transformer of the same active mass — the reference point for "is it the architecture or the card?" |
| `bench_bmm.py` | backends and dtypes for the expert `bmm` on its real shapes |
| `bench_muon.py` | whether batching Newton–Schulz over same-shaped parameters helps |
| `colab_ceiling.py` | the card's matmul ceiling in fp32 / fp16 / bf16, runnable anywhere |
| `compile_check.py` | whether `max-autotune` reaches CUDA graphs: agreement with eager, graph breaks, recompiles |
| `profile_step.py` | top kernels and a per-subsystem split, against the useful FLOPs of the step |

**Data and tokenizer**

| file | what |
|---|---|
| `sample_for_tokenizer.py` | builds the training and held-out text samples from the raw corpus |
| `train_tokenizers.py` | byte-level BPE sweep over vocab sizes, no normalizer, digits split |
| `eval_tokenizers.py` | bytes/token per source and vocabulary utilisation, turned into head cost |
| `encode_corpus.py` | encodes the corpus into flat uint16 shards with a manifest; resumable, and refuses to resume on missing shards |
| `corpus_stats.py` | markup / code / prose shares of a corpus group |
| `peek_tokens.py` | prints windows exactly as the loader will hand them to the model |

## `tests/` — `uv run pytest`

| file | pins |
|---|---|
| `test_kda.py` | chunkwise equals recurrent (and their gradients), causality, zero decay is the pure delta rule, short-conv causality |
| `test_attend_chunked.py`, `test_attn_window.py` | the JAX attention against a directly masked reference, the sliding window, the splash kernel, and that the two copies of `attend_chunked` have not drifted |
| `test_ce_head.py`, `test_losses.py` | chunked CE against the plain formula, sharded and local, bf16 against fp32 |
| `test_moe_dispatch.py` | cumsum against sort dispatch, gather against scatter, expert parallelism against the global dispatch, capacity drops |
| `test_router_balance.py` | the alternating quantile solver against a single iteration, bias centring |
| `test_optim.py`, `test_muon.py` | orthogonalisation spectrum, the Moonlight RMS target, per-head splitting, every parameter covered exactly once, the bf16 moment across a resume |
| `test_quant.py` | int8 dequant bounds, the zero shadow carrying the true gradient, stochastic rounding |
| `test_data.py`, `test_encode_corpus.py`, `test_tokenizer.py` | shard boundaries, `(seed, step)` determinism, train/val disjointness, resume semantics, lossless roundtrip, digit splitting |
| `test_generate.py` | each cache against a full recompute, and that fixed expert capacity makes the model non-causal |
| `test_loop.py` | execution order, gradients summed over visits, layer type by index rather than by position |
| `test_prof_ops.py`, `test_env.py` | the profiler reader, and that the local CUDA device is what it claims |

One test is committed red on purpose; `README.md` says which and why.

## `notes/` — published notes

`ledger.md` is the record: every settled number with the run that produced it, negative
results kept, and an open list at the end. `k3_architecture.md` is the page-by-page reading of
the K3 tech report and is the working replacement for the PDF. `looped.md` is the parallel
experiment track — variants, the equal-budget argument, the config table.

Other files in `notes/` stay local; publishing one means adding it to the allowlist in
`.gitignore`.

## Other

`tokenizers/` holds the trained BPE vocabularies; `bpe_16384.json` is the one in use.
`pyproject.toml` and `uv.lock` define the environment. `data/`, `runs/` and `*.pdf` are local
and untracked.

## Environment

- **`uv run <cmd>`** always; dependencies via `uv add`. Never activate `.venv`, never
  `uv pip install`.
- Local GPU: **GTX 1660 Ti, 6 GB, sm_75** — no tensor cores. **bf16, never fp16** (fp16
  measured 5× slower on this card). It is a development box, not a measurement device: a
  kernel launch there costs ~40 µs, 92% of it in the WSL→Windows driver path.
- Under WSL the repo must live on the Linux filesystem, not `/mnt/c` — the cross-filesystem
  I/O penalty is large enough to distort timings.
- Real measurements come from Kaggle, batch-only, through `scripts.kaggle_run`: TPU v5e-8 for
  the JAX line, 2×T4 for the PyTorch line.

## Working rules

- **Explain → run → interpret.** What do we expect and why → what came out → what it means.
  A run nobody interpreted is a wasted run.
- Logic worth trusting — masking, a loader, an attention variant, a dispatch — gets a pytest
  that stays in the repo. Experiment scripts do not.
- Numbers go into `notes/ledger.md` with the run that produced them. A figure without its
  derivation beside it gets cut, not kept.
- A benchmark row prints the evidence behind it, not only the result: memory analysis,
  collective shapes from the HLO, compilation-cache size, a `(dtype, shape, sharding)` diff
  across one step. A cache size above one means the timing loop ran a different program than
  the one described.
- Finishing a step is not permission to start the next one — propose it in one or two lines
  and stop.
