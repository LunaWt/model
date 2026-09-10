# Ledger

What is settled, with the number and the run that settled it. Negative results kept —
they are what stops a question being reopened. The unabridged chronological version up to
2026-09-06 is parked in `notes/ledger-full-20260906.md.parked`.

Rule for new entries: a claim gets a number and how it was obtained, or it does not go in.

---

## The plan (2026-09-06)

Two models, both looped, trained on Kaggle free quota. Separate weekly budgets, so they
do not compete: **20 h TPU, 30 h GPU**.

**Big model — TPU v5e-8.** MoE, **~27B total / 0.5–1B active** (decided 6 Sep). Everything is
on the table for speed: global+local attention, KDA, DeepSeek sparse attention, custom Pallas
kernels. Full programme: large pretrain → SFT → RL, then context extension. Aim is maximum
capability and maximum tokens. Pretrain target **100–150B tokens**.

⚠ **27B is the fitted edge, not a measured fit** — 22.22B fits at 13.40 of 15.75 GB/chip and
29.47B refuses (`moe-fast`, 6 Sep, see "TPU v5e-8 — memory"). E=176 = 27.1B is queued.

⚠ **The throughput target is ≥2× of 45 638 tok/s at the same T, active and total mass.**
Decided 6 Sep: 31 weeks of quota for one pretrain is not acceptable, and shrinking the active mass to
buy speed does not count as an answer.

**Small model — 2×T4.** 1–2B total, 125–250M active, probably MoE. Standard pretrain +
SFT, plus an experiment: teach it to spot misalignment in actions and chains of thought.
The GPU cannot train the big model in any configuration, hence the split.

Libraries are installed inside the Kaggle kernel at run time (newest, or pinned when a
version turns out to matter). Tokenizer will be redone and much more pretrain data is
needed. There will be no fixed default mix.

⚠ **Storage is not the constraint; throughput is.** The private-dataset quota is 200 GiB =
214.75 GB (`kaggle.com/docs/datasets`, matching what the account reports), which is 105B
tokens as uint16 — so 100B fits and 150B needs a second dataset rotation or streaming. But a
whole 9 h session at 45 638 tok/s consumes **1.48B tokens = 3.0 GB**, and the entire 20 h
week 3.3B = 6.6 GB. Nothing about the corpus has to be resident at once. At that rate 100B
tokens is **31 weeks** of a quota that does not accumulate. Throughput is the project, not an
optimisation.

---

## Hardware and quotas

`kaggle quota`, measured 2026-09-06: **GPU 30.00 h/week, TPU 20.00 h/week,
separate counters**, reset Saturday 00:00 UTC. One session: 12 h GPU, 9 h TPU. Private
datasets 214.75 GB, `/kaggle/working` 20 GB and downloadable after the run.

| accelerator | what you get | bf16 peak | HBM |
|---|---|---:|---:|
| `tpuV5e8` | 8 chips | 1576 TFLOP/s | 15.75 GB/chip |
| `NvidiaTeslaT4` | **2** cards | 65 TFLOP/s each (fp16) | 16 GB each |
| `NvidiaTeslaP100` | 1 card, **retires 15 Sep 2026** | 18.7 TFLOP/s fp16, no tensor cores | 16 GB |

No L4/A100/H100 on free Kaggle. **`NvidiaTeslaT4x2` is silently accepted and gives a
P100** — the string goes to the server unvalidated and unknown values fall back to the
default GPU. Always check the `чипов N` line in the output, never what was requested.

**Getting data in** (checked 6 Sep): an attached dataset is mounted read-only in place at
`/kaggle/input` on a disk separate from `/kaggle/working` (Kaggle staff,
`kaggle.com/discussions/product-feedback/443920`); `np.memmap(..., mode='r')` works on it,
`'r+'` and `'w+'` do not. No published MB/s for that mount, but it does not matter at our
scale: a 9 h session reads 3.0 GB of tokens, so even 30 MB/s is 100 s of 32 400. A Kaggle
notebook **cannot** mount Google Drive the way Colab does — `google.colab` does not exist
there — so Drive means `rclone` or `gdown` over the internet toggle, with Google's per-file
24-hour download lock as the real risk. Drive is an archive, not a training mount.
The TPU VM itself has 96 vCPU and 330 GB of host RAM.

No interactive session: `push → status → logs → output` only, wrapped by
`scripts/kaggle_run.py`. The queue does not consume quota (20 min queued at `TPU 0.00h`).
**Exactly one batch TPU session at a time** (`Maximum batch TPU session count of 1
reached`); pushing a new version of a queued TPU kernel is refused, GPU allows it. GPU and
TPU sessions run concurrently. Queue on 5 Sep evening was 40–60 min for TPU, near-zero for
GPU; on 6 Sep morning TPU started immediately.

Local GTX 1660 Ti (6 GB, TU116, **no tensor cores**, WDDM under WSL) is a development box
only. Its ceiling is 2.9 TFLOP/s fp32 and a kernel launch costs 40 µs steady-state versus
5–10 on native Linux — 92% of that is the WSL→Windows driver path, measured with CUDA
graphs (40.2 → 3.1 µs). Anything measured there is launch-bound and does not transfer to
Kaggle. Native Linux and dual boot: rejected.

---

## TPU v5e-8 — throughput

All rows: `scripts/bench_moe_jax.py`, full training step including the optimizer, bf16
compute, remat between layers, chunked cross-entropy. `replica` = parameters copied on
every chip and the batch split; `zero3` = parameters split too; `expert` = only the
4-D expert tensors split.

⚠⚠ **Everything below dated before `moe-fast` (2026-09-06 13:12) was measured with the
optimizer dtype bug live**, i.e. the timing loop ran a second, fp32 compilation. The paired
control says how much that cost: identical config, dense 1.12B `replica` splash 512 window
2048 B=8, **T=2048 43 134 → 60 710 tok/s (+41%)** and **T=8192 51 871 → 56 851 (+10%)**.
Treat pre-fix absolute numbers as ~10–40% low. Comparisons *within* one of those tables
mostly survive, since every row carried the same bug; comparisons *across* them do not.

### MoE, E=72 (11.35B total / 0.78B active), B=8, T=2048, chunked 256 — post-fix

| sharding | dispatch | ms | tok/s | TFLOP/s | args+temp GB |
|---|---|---:|---:|---:|---:|
| zero3 | cumsum | 911.6 | 17 972 | 83.6 | 2.65+4.23 |
| zero3 | gather | 843.1 | 19 432 | 90.4 | 2.65+4.15 |
| expert | cumsum | 919.1 | 17 827 | 82.9 | 3.47+5.58 |
| expert | gather | 626.2 | 26 164 | 121.7 | 3.47+5.05 |
| **expert** | **ep** | **389.2** | **42 100** | **195.9** | 3.47+4.63 |

**`ep` is 2.34× the old default** (zero3+cumsum) and 1.61× the best previous dispatch. The
row was run twice in the same kernel and returned 389.2 and 389.3 ms, so it is not noise.
Note `gather` is worth +8% under zero3 but **+47% under expert** — the two choices interact,
and testing dispatch at one sharding was misleading.

`expert` sharding no longer OOMs on the second call: `живых_ГБ` tracks `аргументы_ГБ` in
every row (3.54 against 3.47), where the bug used to double it. That is the TPU confirmation
of the dtype fix.

### Dense, 1.12B (d=2048, 24 layers, 16 heads, 4 kv, hidden 5461), `replica`, B=8 — pre-fix

| attention | block | T | window | ms | tok/s | TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| chunked | 256 | 8192 | — | 1742.6 | 37 608 | 253.7 |
| chunked | 512 | 8192 | — | 2227.8 | 29 417 | 198.4 |
| **splash** | 512 | 8192 | — | **1420.3** | **46 141** | 311.2 |
| splash | 1024 | 8192 | — | 1380.8 | 47 461 | 320.1 |
| chunked | 512 | 8192 | 2048 | 1338.8 | 48 951 | 330.2 |
| **splash** | 512 | 8192 | 2048 | **1263.4** | **51 874** | 349.9 |
| splash | 512 | 2048 | 2048 | 379.8 | 43 134 | 290.9 |

**Splash wins where it should and only there.** On full causal attention it is 1.57× the
best chunked at the same block (2227.8 → 1420.3) because the kernel actually skips the
upper triangle, while `attend_chunked` computes a block×T rectangle and masks half of it
away. With a 2048 window the rectangle is already small and splash gains only 6%.

**Block length matters differently for the two.** chunked prefers 256, splash prefers
512–1024; using 512 for both would have cost chunked 28%.

### Batch and remat, splash 512, window 2048, `replica`, 1.12B

| remat | B | T | ms | tok/s | args+temp GB |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 2048 | 379.8 | 43 134 | 2.10+4.33 |
| 1 | 16 | 2048 | 613.9 | 53 375 | 2.10+5.07 |
| 1 | 32 | 2048 | 1225.1 | 53 496 | 2.10+6.47 |
| **2** | 16 | 2048 | **601.7** | **54 463** | 2.10+9.83 |
| 1 | 8 | 8192 | 1263.4 | 51 871 | 2.10+6.47 |
| 1 | 16 | 8192 | 3040.1 | 43 115 | 2.10+9.24 |
| 1 | 32 | 8192 | 8964.2 | 29 243 | 2.10+11.28 |

Batch pays only up to B=16 at T=2048 (+24%), nothing beyond. At T=8192 a bigger batch is
actively harmful. remat=2 (`dots_with_no_batch_dims_saveable`) is worth 2% where it fits
and OOMs everywhere else — the saved matmul outputs cost 4.8 GB of temporaries.

### Size, `zero3`, splash 512, window 2048, B=8, T=8192

| params | shape | ms | tok/s | TFLOP/s | args+temp GB |
|---:|---|---:|---:|---:|---:|
| 2.26B | 2560:32:20:4:6826 | 2206.3 | 29 704 | 403.7 | 0.53+4.84 |
| 3.22B | 3072:32:24:4:8192 | 2870.5 | 22 831 | 441.3 | 0.75+7.06 |

**Utilisation rises with size** — 441 TFLOP/s is 28% MFU, the best number we have. Both of
these OOM under `replica`, so above ~1.2B the choice is made for us. `--param-dtypes
float32` changes nothing in time (compute is bf16 either way, `.astype(DT)` at every use),
only storage: 0.53 → 1.06 GB of arguments.

`gather` (fill the capacity buffer by reading, not writing: a narrow int32 scatter plus one
row gather, bit-identical to `cumsum`) is worth +8% under zero3. `ragged_dot` has a working
gradient after the libtpu upgrade and is **5.5× slower than the capacity buffer** (4871.1 ms
against 876.0 at E=72) — the buffer-free idea is dead on measurement, not on capability.

### MoE attention and batch, E=144 (22.22B / 0.78B active), `expert` + `ep`

| attention | window | T | B | cf | ms | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| chunked 256 | 0 | 2048 | 8 | 1.25 | 425.6 | 38 495 |
| chunked 512 | 2048 | 2048 | 8 | 1.25 | 426.6 | 38 403 |
| splash 512 | 0 | 2048 | 8 | 1.25 | 404.0 | 40 550 |
| splash 512 | 2048 | 2048 | 8 | 1.25 | 404.0 | 40 556 |
| chunked 512 | 0 | 8192 | 8 | 1.25 | 2762.0 | 23 728 |
| chunked 512 | 2048 | 8192 | 8 | 1.25 | 1844.0 | 35 539 |
| splash 512 | 0 | 8192 | 8 | 1.25 | 1926.8 | 34 013 |
| **splash 512** | **2048** | **8192** | 8 | 1.25 | **1764.4** | **37 143** |
| chunked 256 | 0 | 2048 | 8 | **1.0** | 393.4 | 41 645 |
| chunked 256 | 0 | 2048 | 8 | 1.5 | 454.7 | 36 029 |
| chunked 256 | 0 | 2048 | **16** | **1.0** | 718.0 | **45 638** |
| chunked 256 | 0 | 2048 | 16 | 1.25 | 804.4 | 40 735 |
| chunked 256 | 0 | 2048 | 16 | 1.5 | 874.3 | 37 478 |

**Batch does buy something under `ep`**: B=8 → 16 is +9…+11% at every capacity factor. This
refutes the older "batch buys nothing on MoE", which was measured under zero3 with the dtype
bug. Memory cost of B=16 is only +0.7 GB of temporaries.

**Capacity factor is expensive**: cf 1.0 → 1.5 costs 13.5% at B=8 and 17.9% at B=16. Against
the known quality cost (cf 1.5 → 2.0 is worth 0.040 nats on the small model) this is now a
real Pareto choice, not a free knob.

**At MoE scale a longer T costs what the arithmetic says.** splash + window 2048: T=2048
40 556 → T=8192 37 143, i.e. 8% slower per token for 4× the length. Same sign on the dense
control (60 710 → 56 851, 6%). The pre-fix run showed T=8192 *faster* than T=2048; that was
the dtype bug, not an MXU-feeding effect.

### Dense against MoE, both post-fix

1.12B dense `replica` at 60 710 tok/s against 22.22B/0.78B-active MoE at 45 638 — **1.33×**,
where before `ep` it was 2.4×. In TFLOP/s 409.5 against 213.3. MoE is now a defensible choice
rather than a 2.4× tax: 20× the total parameters for a third less throughput.

### What the step actually costs — 22% of the chip, not 13.5%

The `ТFLOPс` column is `6 · active · tokens` and undercounts the real work by 1.64×. Counted
by hand for the best row (E=144, B=16, T=2048, cf 1.0, chunked 256, remat 1, full attention,
32 768 tokens per step):

| part | forward TFLOP | ×remat/bwd | TFLOP per step |
|---|---:|---:|---:|
| layer dense matmuls (q,k,v,o,shared FFN,gate) | 26.85 | ×4 | 107.4 |
| expert matmuls (`cap` 56, 8064 rows per chip) | 19.48 | ×4 | 77.9 |
| attention, `attend_chunked` block×T rectangle | 13.19 | ×4 | 52.8 |
| cross-entropy / lm head | 4.40 | ×3 | 13.2 |
| **total** | | | **251.3** |

251.3 TFLOP at 718.0 ms is **350 TFLOP/s = 22.2%** of the 1576 peak, against the 13.5% the
column prints. The same count on the dense control (158.3 TFLOP, 269.9 ms) gives **37.2%**.

Three calibration points, so this is read against something: MaxText reports 55–60% MFU for
dense on TPU (`github.com/AI-Hypercomputer/maxtext`, README), 67.8–70.3% on v5p-128; and
DeepSeek-V3 — MoE, FP8, 2048 sequence — comes out at **21.7%** hardware utilisation from its
own published GPU-hours (`jax-ml.github.io/scaling-book/transformers/`, Q7). So 22% is where
a frontier MoE run sits, and the honest gap is not "MoE is inefficient" but **our MoE is 1.68×
behind our own dense path on the same chips**.

Two of the four rows are avoidable work rather than model FLOPs:

* **remat costs 23.7%** of the step — full rematerialisation turns 3 forwards into 4. It is
  not switchable at 22B (temporaries go 2.47 → 8.69 GB at 11B without it) but it becomes
  switchable if something else frees HBM.
* **`attend_chunked` computes half its rectangle to throw it away**: 6.6 of 13.19 forward
  TFLOP, i.e. **10.5%** of the step. splash skips the upper triangle and measured only 5%
  better at B=8, which says attention runs below MXU rate in both.

**The expert matmuls are compute-bound, but only by 1.9×.** The scaling book's condition is
`k·B/E > 240` tokens per expert for bf16 weights (`> 120` for int8). Ours is
`2 · 32768 / 144 = 455`. So the shape is fine and raising the batch has little left to give —
consistent with B=8 → 16 buying 11% and not 50%.

### Where the 718 ms actually go — profiled 2026-09-06 (`moe-prof`)

`--profile` on the best row, read with `scripts/prof_ops.py`. Eight device planes agree to
0.0%, leaves sum to 716.2 ms against 718.2 on the timer. Buckets are by output shape and are
mutually exclusive; `while`/`conditional` wrappers are excluded because their duration already
contains their contents.

| part | ms | % |
|---|---:|---:|
| **cross-entropy head — `all-gather` of the whole hidden state** | 105.2 | 14.7 |
| cross-entropy head — fp32 logits and their gradient | 77.8 | 10.9 |
| `all_to_all` of the expert buffers (6 per layer × 24) | 104.8 | 14.6 |
| attention — query blocks and their **fp32** score tensors | 88.0 | 12.3 |
| capacity buffer — scatter and gather of `[8065, 2048]` | 87.0 | 12.1 |
| layer stack — slicing weights out and writing gradients back | 82.2 | 11.5 |
| **expert matmuls** | 59.9 | 8.4 |
| **dense part of the layer** (matmuls, norms, residual) | 57.5 | 8.0 |
| `all-reduce` of the replicated gradients | 18.9 | 2.6 |
| everything else | 34.9 | 4.9 |

**Arithmetic is at most a fifth of the step.** That kills the hypothesis this file carried for
one afternoon — that int8 is worth 2×. Halving every matmul saves under 60 ms of 718. It also
kills the remat objection: rematerialisation costs 23.7% of the *FLOPs* and the FLOPs are not
what we are paying for.

**The single biggest item is a sharding bug in `chunked_ce`, and it is not the model at all.**
`h` is `(B·T, d)` sharded on rows; `hc = h.reshape(nc, chunk, d)` and a `lax.scan` over `nc`
means every iteration needs a chunk that lives on one chip, so XLA emits
`all-gather(bf16[4,1024,2048]) → bf16[32,1024,2048]` **inside the loop body** and re-gathers
the entire 134 MB hidden state on every one of the 32 iterations, four times over (forward,
remat, backward). Cross-entropy is 5.3% of the FLOPs and **25.6% of the step**. The loss is a
sum over rows and every chip already owns its rows, so the fix is to compute it on the local
shard — a `shard_map` with a `psum`, no maths change. Second item in the same place:
`logits.astype(jnp.float32)` materialises `f32[1024, 32768]` = 134 MB per chunk because
`logits` is consumed twice (`logsumexp` and `take_along_axis`) and cannot be fused away.

**Attention is bandwidth-bound, not FLOP-bound.** `attend_chunked` writes its scores as
`f32[2,4,4,256,2048]` — 67 MB per query block per layer, 12.9 GB per forward pass. That is why
splash, which computes half the FLOPs, is only 3.7% faster (718.2 → 691.5): both are waiting
on HBM, not on the MXU.

TPU v5e does 393 TOPs int8 against 197 TFLOP/s bf16 — exactly 2× — and no FP8 at all
(`docs.cloud.google.com/tpu/docs/v5e`, system architecture table; HBM 16 GB at 800 GiB/s per
chip). So int8 stays interesting, but **for memory rather than speed**: 22.2B of bf16 weights
is 5.55 of the 6.07 GB of arguments, and halving that is what would let 27B run at B=16.

### Decomposition, same run — what each piece costs by difference

E=144, B=16, T=2048, cf 1.0, `expert`+`ep`. The base row reproduced across kernel runs to
0.1 ms (718.0 in `moe-fast`, 718.2 and 718.1 in `moe-prof`), and the dense control to 0.0
(269.9 both times).

| change | ms | Δ |
|---|---:|---:|
| base, chunked 256, adafactor | 718.2 | — |
| `--opts none` (gradients only) | 685.7 | **−32.5, i.e. the optimizer is 4.5%** |
| splash 512 instead of chunked 256 | 691.5 | −26.7 (3.7%) |
| splash and no optimizer | 658.9 | −59.3, exactly the sum of the two |
| `--top-ks 1` | 571.3 | −146.9 (20.5%) |
| B=32 | 1754.8 | 37 346 tok/s — **worse than B=16** |

**Adafactor costs 4.5%, not the 30–40% guessed from the pre-fix table.** That table was
measured with the dtype bug and compared an fp32 program against a bf16 one.

The two effects are additive to 0.4%, which is how you tell they are independent.

top-k 2 → 1 halves both the expert arithmetic and the buffer traffic and buys 20.5%, so the
whole token→expert→token path is about 41% of the step — consistent with the profile
(all_to_all 14.6 + buffer 12.1 + expert matmuls 8.4 = 35%, plus routing).

**B=16 is the optimum and B=32 is a cliff**: 2× the tokens for 2.44× the time. `cap` doubles
with the batch, so the buffers, the scatter and the `all_to_all` all double while the
arithmetic stays at the same efficiency — and temporaries reach 9.48 GB.

### The five changes made against that profile (`moe-2x`, 6 Sep)

Each is a separate axis of the grid so it can be switched off and measured against its own
control, and each has a numerical-equality test in the suite.

1. **Cross-entropy on the local row shard** — `--ces shard` (new default), `--ces scan` is the
   old code kept as the control. `_ce_local` sums NLL over the rows a chip already owns and
   one `psum` at the end replaces the `all-gather` inside the loop. Two changes ride along:
   logits stay in bf16, and the chosen logit is `sum(h · embed[target])` instead of a slice of
   the logit matrix, so the matmul output has one consumer and the fp32 copy is never
   materialised. Loss values shift in the fourth digit; `tests/test_ce_head.py` checks value
   *and* gradient against the old version to 1e-5, sharded and unsharded.
2. **bf16 attention scores** — `--score-dtypes bf16` (default), `f32` is the control. The
   scale now multiplies `q` before the einsum, which leaves the einsum output with a single
   consumer so XLA can fuse it into the softmax; the softmax still computes in fp32.
3. **`ep-gather`** — the capacity buffer is filled by reading instead of writing: an int32
   scatter of "which assignment sits in this slot" over `(E·cap,)`, then one row gather.
   Replaces a scatter into `(E·cap+1, d)`. Bit-identical to `ep`, tested.
4. **`ep:N`** — the buffer is cut into N pieces along the capacity axis and each piece does its
   own `all_to_all` → experts → `all_to_all`. With one piece there is nothing for the exchange
   to overlap with, and the profile charged 104.8 ms to exchanges that no arithmetic was
   waiting behind. Bit-identical to `ep`, tested.
5. **int8 expert weights** — `--quants experts`, below.

### int8 masters in JAX: the gradient has to be smuggled out

The obvious implementation does not exist. `jax.grad` will not return a cotangent for an int8
leaf — the tangent type of an integer array is `float0` — so storing weights as codes and
asking autodiff for `dL/dW` is a dead end, and `allow_int=True` only replaces the error with
zeros. Dequantising the whole stack outside the layer loop gets the gradient but also puts a
full bf16 copy next to the codes, which costs more than it saves.

What works: the dequantised weight is `code · scale + shadow`, where `shadow` is a zeros array
of the same shape that travels into the layer `scan` as one more `xs`. Adding a broadcast zero
is removed from the forward pass by the algebraic simplifier, and the backward pass leaves the
cotangent we need sitting on it. The bf16 copy exists for one layer at a time, not for the
stack.

The optimizer step then works one layer at a time (`jax.lax.map` over the leading axis):
dequantise to fp32, ordinary Adafactor step, recompute the per-output-channel scale as the new
`absmax/127`, pack back with **stochastic rounding**. Rounding has to be stochastic: the grid
step is about 3% of the weight σ and a typical Adafactor update is around `lr`, i.e. a few grid
steps for the large ones and a fraction of one for the rest — deterministic rounding would
throw the small ones away every step forever. Recomputing the scale from the updated weights is
what keeps clipping from ever happening as the weights grow.

Scales are per output channel (`absmax` over the input axis), fp32, shape `(L, E, 1, h)` —
1/2048 of the weight mass. At init the scale is analytic, `4σ/127`, so quantisation fuses into
the generator and the float tensor is never written to HBM: normal → scale → round → pack in
one pass.

Checked on two CPU devices before spending queue time (`d=512, L=4, E=32`, 0.11B): arguments
0.10 → 0.06 GB, temporaries 0.37 → 0.35. The saving is the whole expected half of the expert
mass and the shadow costs nothing — if it had been materialised, temporaries would have gone up
by exactly the amount the arguments went down.

**What int8 cannot save**: the gradient. It stays bf16 and it is the same size as the bf16
weights were, so at E=176 the floor is codes 3.3 + gradients 6.6 GB per chip rather than
6.6 + 6.6. That is why the estimate is ~35B total rather than ~54B.

### Optimizer cost, 1.12B dense `replica`, B=8, T=2048, chunked 256

| optimizer | ms | tok/s | args GB |
|---|---:|---:|---:|
| none (grads only) | 276.0 | 59 368 | 2.09 |
| **adamw** | **296.1** | **55 335** | 6.28 |
| adafactor | 395.5 | 41 426 | 2.10 |
| muon (+adafactor on the rest) | 762.8 | 21 477 | 6.03 |

**AdamW is cheaper in time than Adafactor here** — +20 ms versus +120 ms over the
gradient-only baseline. Adafactor's factored second moment costs two reductions and a
broadcast per tensor; AdamW is pure elementwise. Adafactor still buys memory: 2.10 GB of
arguments against 6.28. Muon as written costs +487 ms, i.e. 2.8× the whole step — five
Newton–Schulz iterations on every matrix. Levers if we want it: fewer iterations, bf16
inside the iteration.

---

## TPU v5e-8 — memory

Read the ceiling off `compiled.memory_analysis()`, never off the allocator.
`device.memory_stats()` only sees buffers JAX hands out (arguments and results); XLA:TPU
places everything internal inside the executable, which is why refusals arrive at compile
time before a byte is allocated. And `peak_bytes_in_use` is a **process** high-water mark
that never resets — it was reported per row until 2026-09-06 and was wrong from the third
row onward.

### The bf16 ceiling, measured 2026-09-06 — 27.1B fits at B=8, 29.5B does not

`ep` + `expert`, T=2048, chunked 256, remat 1, Adafactor, cf 1.0, B=8. Total size is
`0.48 + 0.151·E` billion (dense part plus 0.151B per expert).

| E | total | ms | tok/s | args GB | temp GB | sum | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 72 | 11.35B | — | — | 3.47 | 4.63 | 8.10 | fits |
| 144 | 22.22B | 393.4 | 41 645 | 6.07 | 7.30 | 13.37 | fits |
| 160 | 24.64B | 408.5 | 40 110 | 6.65 | 7.94 | 14.59 | fits |
| **176** | **27.06B** | **424.6** | **38 584** | **7.23** | **8.43** | **15.66** | **fits, 90 MB spare** |
| 192 | 29.47B | — | — | — | — | — | OOM, HLO temporaries |
| 216 / 256 | 33.1 / 39.1B | — | — | — | — | — | OOM |

**27B fits, and the fitted-slope method that predicted it was right to 1%.** (Scaling the
5.12 GB total through the origin had said 35B; fitting a slope plus a fixed part on two sizes
had said ~22B for the safe point and ~27B for the edge. Use the second method.)

Per 16 experts, i.e. per 2.42B parameters: **+16 ms and +1.1 GB per chip**, dead linear across
all three points. E=192 would need ~16.8 GB, which is why it refuses.

**But 27B only fits at B=8, and B=8 costs more than the parameters buy.** At 27.06B/B=8 it is
38 584 tok/s against 45 632 at 22.22B/B=16 — 15% fewer tokens per second for 22% more
parameters, and B=16 at E=176 needs about 0.65 GB it does not have. Halving expert storage
with int8 is exactly what would remove that trade.

⚠ The comparison assumes B is free to choose for quality, and it is not: B=8 doubles the
number of optimizer steps per token, and the one paired experiment we have says steps, not
tokens, bind the loss early. So this is a throughput argument, not a settled decision.

### Storage dtype and why a total cannot be extrapolated

`zero3`, E=72 = 11.35B total / 0.78B active, B=8, T=2048, chunked 256, remat 1, Adafactor.
Same shape and same script in all three rows:

| weights | jax | args GB | temp GB | sum |
|---|---|---:|---:|---:|
| fp32 | 0.10.2 | 5.29 | 9.48 | 14.77 |
| bf16 | 0.10.2 | 2.65 | 2.47 | 5.12 |
| bf16 | 0.11.1 | 2.65 | **4.23** | 6.88 |

fp32 → bf16 costs nothing in time to a tenth of a millisecond: arithmetic is bf16 either way
(`.astype(DT)` at every use), only storage changes. In fp32 the ceiling is also a
measurement: E=88 (13.76B) refuses at compile time, so ~12.2B — bf16 storage buys 1.8× of
headroom and no speed.

Temp is two things at once: a gradient-shaped part that scales with parameter count and an
activation + collective-buffer part that scales with B, T, d and cap and does not. Only a
two-point fit separates them, which is why the slope-plus-fixed-part method predicted the
27B edge and scaling a total through the origin missed it by 30%.

⚠ **Memory numbers do not survive the jax upgrade.** Identical config, temp 2.47 → 4.23
across jax 0.10.2 → 0.11.1 (4 Sep `moe-ceiling` against 6 Sep `tpu-am`).

Worth noticing: 2.47 GB of temporaries is *less* than a single bf16 copy of the gradients
(2.65 GB). With donated arguments and the layer loop as a `scan`, XLA folds the optimizer
update into the backward rather than materialising a whole gradient tree — which is why a
hand sum of weights + gradients + activations mispredicts the ceiling in both directions.

**Remat cannot be switched off.** Without it the 11.35B bf16 model's temporaries go
2.47 → 8.69 GB and execution fails even though the plan formally fits — `memory_analysis`
does not count buffers alive at argument donation. Its 1.33× is not a lever.

**A Python loop over layers does not compile at this scale.** An unrolled 24-layer MoE
graph killed the host process twice (692 s and 781 s, no output). Parameters are stacked
with the layer as leading axis and driven by `jax.lax.scan`: one-layer graph, 15–19 s to
compile. Consequence for sharding: split axis 1, because axis 0 is now the layer index.

---

## TPU v5e-8 — sharding

**Replication beats ZeRO-3 by 2.3–3.2× wherever the parameters fit on one chip.** Same
result on 2×T4 (1.83–1.96×) and for the same reason: weights are gathered once per layer,
so the collective is paid L times per step, and neither PCIe nor the v5e interconnect earns
that back. Decomposition on a 0.77B dense model: 749.4 ms with ZeRO-3 versus **253.1 ms**
replicated, everything else identical.

Rule for dense: **`replica` up to ~1.2B, `zero3` above.** For MoE the third option is now
live and is the plan of record: **`expert` + `ep`** below.

### The optimizer silently promoted bf16 parameters to fp32 (found and fixed 2026-09-06)

This is the single most expensive bug in this file, because it corrupted *time* and *memory*
measurements at once without ever raising an error.

`_adafactor_leaf` keeps the factored second-moment rows `r`/`c` in float32. Mixing them into
the update makes the returned parameter float32 even when the input was bfloat16. Same hole
in `adamw_update` and in muon's masked branch. Consequences, in order:

1. The `jit` signature changes between call 1 and call 2, so **the step compiles a second
   time**, on top of a live first executable. Reproduced on 8 fake CPU devices:
   `компиляций 1→2` in `replica`, `zero3` and `expert` alike.
2. Parameter memory doubles. Observed on TPU as live HBM 2.66 → 5.35 GB under zero3.
3. `memory_analysis()` was taken from the first (bf16) compilation while the timing loop ran
   the second (fp32) one — **the plan and the measurement describe different programs.**

Fix: `.astype(p.dtype)` on the returned parameter in all three optimizers. Guarded by
`test_update_preserves_parameter_dtype`, parametrized over adafactor/adamw/muon, two steps,
every leaf must stay bf16.

**This is what killed `expert` sharding**, which had been parked as "broken, mechanism
unknown": its plan was *smaller* than zero3's (3.47+2.32 = 5.79 GB against 6.88) yet it died
on the second call with `RESOURCE_EXHAUSTED ... HLO temporaries`. It was paying for two
executables. zero3 did the same and survived only because it started lower.

A second, independent cause was found in the same session and both fixes are needed:
without explicit `out_shardings`, the output sharding of some leaves differs from the input
(3 leaves under zero3, 9 under expert, e.g. `P(None,'x',None,None)` → `P(None,'x')`), which
is itself enough to force a recompile.

### `ep` — expert parallelism, 4.7× less collective traffic

`moe_ffn_ep` in `scripts/bench_moe_jax.py`. Each chip owns `E/chips` experts and `B/chips`
of the batch, routes its own tokens locally, then one `jax.lax.all_to_all` ships token rows
to the chip that owns the expert and a second one ships the results back. Requires
`--shardings expert` and `E % chips == 0 and B % chips == 0`. Needs `jax.shard_map` with
`check_vma=False`; the transpose of `all_to_all` is `all_to_all`, so gradients reach every
expert — pinned by `test_expert_parallel_gradient_flows_to_every_expert`.

Collective bytes per step, counted off the compiled HLO on 8 fake CPU devices, toy shape
d=512, EH=256, E=32, L=2, T=256, B=8:

| sharding + dispatch | collectives MiB | temp MiB |
|---|---:|---:|
| zero3 + cumsum (previous default) | 106.4 | 57.1 |
| zero3 + gather | 91.7 | 54.9 |
| expert + cumsum | 93.8 | 88.6 |
| expert + gather | 57.8 | 76.9 |
| **expert + ep** | **22.6** | 59.6 |

End to end on those 8 fake devices: ep 33 ms against cumsum 84 and gather 76, i.e. 2.5×.
**On TPU it delivered 2.34×** (389.2 ms against 911.6) — the static HLO byte count predicted
the real speedup to within 7%, on a laptop, before any queue time was spent. That is the
strongest argument for keeping `коллективы_ГБ` in every row.

⚠ The largest collectives in the zero3 plans are `f32[4096,512]` and `u32[4096,512]` — the
**token buffer (N·k, d)**, not the expert weights. So the "27% of the step is dispatch"
number is the cost of replicating activations across chips, not of computing indices, and
`ep` attacks the right thing.

⚠ `all_to_all` is emitted in HLO as a **tuple-shaped** collective, so a regex that reads one
shape per op silently under-counts it. Sum every shape to the left of the op name.

---

## What the Kaggle TPU image can and cannot do

The stock image ships **jax 0.10.2 with libtpu built 12 June 2025**. On it: every Pallas
kernel refuses to launch (`Pallas TPU requires a libtpu version that's at most a month
old`) and `ragged_dot` has a working forward but no gradient
(`UNIMPLEMENTED: Ragged dot fwd with rhs_contracting_dim != 1 - NYI`).

**`pip install -U jax[tpu]` inside the kernel, before importing jax, fixes all of it.**
`scripts/tpu_probe.py --pip 'jax[tpu],optax'` → **15 of 15 checks pass**; the run comes up
as jax 0.11.1 with 8 chips visible. Pinning `jax[tpu]==0.10.2` is a no-op: that extra
requires `libtpu==0.0.42.*`, exactly the June build already installed. The version triple
jax+jaxlib+libtpu has to move together. `scripts/bench_moe_jax.py --pip` does the same.

Unlocked by the upgrade: `ragged_dot` gradient, `megablox.gmm`, `splash_attention` forward
and backward, causal and windowed.

**`megablox` exposes only `gmm`** — no `tgmm`, no `custom_vjp`. The derivative with respect
to expert weights is a contraction along the ragged token axis, which upstream megablox
does with a separate `tgmm` kernel that JAX never got. So a hand-written vjp is not
available either.

And the flip side, which settles the design: **the capacity buffer exists for the sake of
the gradient.** At shape (E, cap, d) the weight derivative is a plain batched
`einsum('ecd,ech->edh')` and autodiff takes it. The buffer stays; what is worth optimising
is how it is filled, which is what `gather` does.

**Splash cannot be called under `jit` with a mesh.** `NotImplementedError: Mosaic kernels
cannot be automatically partitioned. Please wrap the call in a shard_map.` — all 16 splash
rows of the first `tpu-dense` run died on this, `replica` included, so it is not about a
split tensor: XLA:SPMD cannot partition a Mosaic kernel at all. Fixed by wrapping the call
in `jax.shard_map` with explicit specs (`attend_splash(..., mesh=)`); guarded by
`test_splash_runs_under_sharded_jit`, which needs two devices —
`tests/conftest.py` sets `--xla_force_host_platform_device_count=2`.

`jax.nn.dot_product_attention` on TPU is not a substitute: slower than our own chunked
attention (793 vs 749 ms) and asks three times the temporary memory without remat (15.01 vs
6.48 GB), i.e. it materialises T×T. The real replacement was always splash.

---

## Kaggle GPUs

Dense model, JAX 0.7.2 (the GPU image), fp16 compute, full step with Adafactor + remat +
chunked CE, T=2048, B=8. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` is required — XLA:GPU takes
only 75% of the card by default (11.92 of 16 GB observed on P100).

| params | 2×T4 zero3 tok/s | 2×T4 replica tok/s | 1×P100 tok/s | 1×T4 torch tok/s |
|---:|---:|---:|---:|---:|
| 0.30B | 2064 | **3846** | 1724 | **4872** |
| 0.45B | 1604 | **3029** | 1303 | 2945 |
| 0.64B | 1566 | **2864** | 1064 | 2523 |
| 1.12B | 1000 | **1958** | 711 | 1424 |

Ceiling with replication is one card: fp32 weights + gradient + Adafactor = 12 B/param,
13.4 GB / 12 ≈ **1.1–1.2B**, confirmed by the 1.12B row at 10.05 GB. ZeRO-3 raises it to
~2.5B at half the speed.

**The GPU track is not a pretraining machine.** 4872 tok/s × 30 h = 526M tokens/week.

**PyTorch is >2× faster than JAX per card** (0.3B on one card beats JAX on two) because
cutlass skips the upper triangle. That gap should close now that we know splash exists on
TPU and why chunked was slow. Two things block torch on Kaggle anyway:

* **DDP on 2×T4 hangs.** The first `all_reduce` (297M elements) and even a 2-element
  `all_gather` time out and SIGABRT. `NCCL_P2P_DISABLE=1` does not help. JAX uses both
  cards with no configuration at all. `gloo` untried.
* **`jax.nn.dot_product_attention` has no fused path on sm_75 either** and materialises T×T
  per layer: 134 MB per layer at T=2048, 16 heads, B=2/card, 2.7 GB over 20 layers before
  checkpointing. A 520M model OOMs on 2×T4 although its fp32 weights + gradients are only
  4.2 of 15 GB. **Two T4s are not 32 GB** under data parallelism — each card holds a full
  set of weights and gradients, and there is no NVLink between them.
* **`enable_gqa=True` disables the memory-efficient SDPA kernel on sm_75.** cutlass
  EFFICIENT_ATTENTION returns `No available kernel` and torch silently falls back to MATH
  with a full T×T. That cost 11.4 GiB of peak on a 300M model and OOM on everything larger.
  Fix: `repeat_interleave` the keys to the full head count. Peak 11418 → 3766 MiB,
  throughput 1382 → 4872 tok/s. FLASH needs sm_80+ and is unavailable.
* `torch.compile` OOMs on T4 for all our shapes; inductor prints `Not enough SMs to use
  max_autotune_gemm`.

---

## Attention

**Sliding window is free at T=2048 and worth +30% at T=8192** (1742.6 → 1339.8 ms at
window 2048). At T=2048 the window equals the context so the code path is identical to
0.1 ms — a useful check that the windowed path adds no overhead.

`attend_chunked` computes attention in query blocks with a static `window + block_len`
slice taken by `dynamic_slice`, so the graph shape does not depend on T. Verified against a
direct masked reference for windows {0,8,16,32,64,128} × block {8,16}, and splash verified
against it forward and backward in `interpret` mode. `tests/test_attn_window.py`.

**The 3:1 hybrid costs 2.2%** — first TPU measurement, `moe-prof`, 22.22B, `ep`, splash 512,
window 2048, B=8, T=8192, cf 1.0. All 24 layers windowed: 1651.1 ms / 39 693 tok/s. Six of 24
full (`--full-everys 4`, a `lax.cond` on the layer index inside the scan; both branches
compile, one executes): 1687.4 ms / 38 838. So the layout every frontier vendor uses is
essentially free at 8k, and the `lax.cond` adds nothing measurable.

**Long context is even cheaper than the last number said.** At cf 1.0 the same shape gives
T=2048 → 41 645 and T=8192 → 39 693 tok/s: **4.7% per token for 4× the length**, against 8%
measured at cf 1.25.

**KDA**, for the small model. Our `kda_chunkwise` follows Kimi Linear Eq. 6–9 and agrees
with the naive recurrence to 4.4e-16 in float64. `fla.ops.kda.chunk_kda` agrees with it to
1.2e-6 in fp32 and is **38× faster**; it is integrated (`model/kda_head.py`,
`backend="auto"`) and our two implementations stay as the reference pair.

⚠ **chunk = 16 is a hard ceiling for our formulation, not for KDA.** We materialise `K/Γ`
for the sake of the matmul, and `1/Γ ≤ e^{5C}`; at C=16 that is e^80 against an fp32 limit
of e^88.7, at C=32 it is already inf and inf·0 after masking is nan. `fla` uses 32–64
because inside the Triton kernel the tile sits in SRAM and `exp(G_r − G_i)` is computed
pairwise, where the difference is never positive. In pure PyTorch the pairwise form is a
(C,C,D) tensor: ~100 MiB per layer at C=64.

**Three structurally different kinds of decode state, and this is the whole point of linear
attention.** MLA's latent `c` grows with length; KDA's recurrent state `S` is `(B,H,d_k,d_v)`
and **does not grow at all**; ShortConv needs only the last K−1 inputs. Verified by transfer:
prefill 24 tokens with `chunk_kda` then 8 steps of `fused_recurrent_kda` against one run over
32 — output 1.6e-7, state 6.6e-7, and greedy generation with the cache matches full recompute
token for token (`tests/test_generate.py`). ⚠ The cache key must be the **execution** index,
not the layer index: in a looped model one layer's two passes sit at different stack depths.

`b_α` init is taken from `fla/layers/kda.py:180-185`: sample `dt` log-uniformly from
[0.001, 0.1] i.i.d. per channel, then inverse-softplus. The placeholder we had gave every
channel of a head the same decay, i.e. KDA degenerated to a per-head scalar gate at init.
`b_α` and `A` carry `._no_weight_decay`.

---

## MoE mechanics

These were established on the small model and carry to both.

**Fixed capacity makes the model non-causal in the token axis.** `cap` is computed from the
number of tokens in *this* forward and the overflow is dropped, so a token's output depends
on which other tokens ride along. Found while checking the generation cache: prefill of 5
against a full run of 9 differed by 0.38 in logits with identical weights; with capacity
removed, 5e-7. `set_full_capacity(True)` is used in generation and NLL only, never in
training.

**Dropping costs real loss.** cf 1.5 → 2.0 is worth **0.040 nats** for 5% of speed, on a
seed noise of 0.033. Mixing documents inside a micro-batch is worth another 0.041: B=2 with
T=672 beats B=1 with T=1344 on every axis at equal token count, and halving the document
length cost nothing measurable. Recommended: **B≥2, cf≈2.0**.

**Quantile Balancing has to be iterated.** K3 Algorithm 1 alternates α and β to
convergence; one application of Eq. 14 per optimizer step leaves peak/ideal load at 2.34
against 1.08 for four iterations. With the solver fixed, the dropped fraction holds at
0.005 instead of 0.2–0.6. The residual skew is not the solver's fault: QB equalises *mean*
load while capacity is checked per micro-batch, and experts specialise by domain. More
independent documents per forward is the cure, not more capacity headroom.

**The router must be fp32.** In bf16, 6% of tokens get a different top-k at init because
the scores are nearly equal and the discrete decision flips on rounding. DeepSeek and
Megatron do the same. Cost measured on the small model: ~9% of throughput.

QB bias updates and router-score collection must live **outside** `forward`. A
state-mutating forward breaks gradient checkpointing — the recompute re-ran routing after
the bias had moved and the per-expert counts no longer matched.

**Sorting does not beat a prefix sum** for slot assignment: `_slots_sort` is 2–3% slower
than `_slots_cumsum` and the feared growth of cumsum with E did not appear (E 24 → 72 cost
6%). Both are kept, bit-identical, `cumsum` is the default and `gather` is the fast filler.

**Total parameters are nearly free in time on TPU**: E 24 → 72 (4.09B → 11.35B) costs 6% of
speed at constant active mass. What costs is the dispatch: ~27% of the step, and independent
of E, which is the signature of moving activations rather than computing indices.

⚠ **But they were not free in loss on the small model.** A16 (64 experts, 225.7M/48.7M
active) against M_experts (80 experts, 273.0M/**48.8M active** — same active mass), 250
steps, same seed and a loader deterministic in `(seed, step)` so both saw identical
sequences in identical order: A16 won at every checkpoint (val CE 5.8858 against 5.9480 at
step 249) **and ran 33% faster** (3362 against 2250 tok/s). The speed part is understood —
expert FLOPs do not depend on `n_routed` (`n·cap = m·k·cf`) but the bmm splits into more and
smaller GEMMs, and Newton–Schulz in Muon grows linearly with the expert count. The loss part
is one seed per arm against a 0.033-nat seed noise, so 0.06 is suggestive, not proven. Do
not scale E on the TPU model on the strength of the time number alone.

**Batch on MoE depends on the sharding.** Under zero3 it bought nothing (B=8 → 32 was 4.61×
the time for 4× the tokens). Under `expert` + `ep` it does: B=8 → 16 is +9…+11%. So the older
conclusion "the MoE step is bound by work per token, not by collectives" held only while the
collectives were the zero3 ones.

---

## Looped transformers

`notes/looped.md` has the budget derivation; `scripts/loop_budget.py` reproduces the table
and it is a test.

**Looped won its first paired comparison.** L12_4x2 (12 layers, loop over 4–7 twice,
224.7M) against A16 (16 layers, 225.7M), same data, same seed, same steps: better val CE at
every checkpoint by 0.05–0.15 nats, reaching at step 203 what A16 needs 250 steps for. Seed
noise on this pair is 0.033, so the effect is real. Also slightly cheaper in memory —
activations count per execution, weights per distinct layer.

Two facts worth not rediscovering: parameters are restored with **experts, not width**
(one routed expert = `3 · moe_latent · expert_hidden`, so depth and width stay matched by
construction), and `is_mla = (layer + 1) % 4 == 0` only ends the backbone on global
attention when `n_layers % 4 == 0` — L=18 and L=10 silently break the K3 rule.

---

## Optimizer

**Muon's step scale was wrong by ~30×** and it is a subtle failure: `lr=0.02` with a
`sqrt(max(m,n))` multiplier mixed Jordan's recipe with Moonlight's. Correct (Moonlight,
arXiv 2502.16982, Lemma 1 + Eq. 3): the post-orthogonalisation RMS is `sqrt(1/max(m,n))`,
multiplied by **`0.2·sqrt(max(m,n))`**, after which Muon and AdamW share one lr. The
symptom was the MoE router collapsing in three steps and loss falling "too fast" into a
degenerate solution. `test_update_rms_matches_the_moonlight_target` pins RMS = 0.2.

Batching Newton–Schulz across parameters of the same shape is pointless — a (64,256,256)
stack already saturates a GPU (1.01×). The only remaining lever is the iteration count.

**Optimizer steps, not tokens, bound the loss at this stage — so maximising tok/s can lose.**
Same wall clock, same model: accum=4 / 550 steps / 3.4M tokens / ~35 min → val CE **5.778**;
accum=16 / 250 steps / 5.4M tokens / 30.8 min → **5.886**. More tokens, worse loss. Note the
throughput benchmark points the other way (accum 8 → 2950 tok/s, 16 → 3340, 32 → 3400),
because compilation made forward+backward cheaper while Muon's fixed per-step cost stayed.
Choose accumulation on validation loss, never on the tok/s column.

**Nexus (arXiv 2604.09258v2) — read 6 Sep, rejected for now, on memory.** Algorithm 3, p. 8:
keep `inner_model = model.clone()`, take a Normalized-SGD step on the clone at every
mini-batch, and at each gradient-accumulation boundary hand the outer optimizer the
displacement `ĝ = inner_model − model` instead of the summed gradient, then re-clone. The
claim (Table 1, p. 10) is equal pretraining loss with better downstream: 3B dense, loss
1.606 → 1.602, GSM8k 44.0 → 59.0, MATH 32.0 → 40.0, average accuracy 37.1 → 40.3. Gains grow
with scale, 130M → 2.3B.

Three reasons it is not ours yet, in order:
1. **A full extra parameter copy.** At 22.2B bf16 that is +5.55 GB per chip on top of
   6.07 args + 7.98 temp — 19.6 of 15.75. It does not fit, and the paper measures neither its
   memory nor its wall clock (only the assertion on p. 9 that both are near zero).
2. **It needs gradient accumulation to mean anything.** With `accum_steps = 1`, which is our
   configuration, `ĝ` is one NSGD step and Nexus degenerates to running the outer optimizer on
   a normalised gradient.
3. Dense only, AdamW/Muon/SGDM only, no MoE, no sharding, no factored second moment, no
   statement about bf16 anywhere in 49 pages.

Revisit if we end up with accumulation *and* HBM headroom — i.e. after int8, not before.

Our JAX Muon (`newton_schulz` + `muon_mask` + `muon_update` in `bench_moe_jax.py`) applies
to attention and dense-FFN matrices with Adafactor on everything else; `muon-all` adds the
experts. Singular values after 5 iterations land in [0.682, 1.135]. Two bugs the tests
caught: the mask added keys that were not in the parameter tree, and a single shared
`jnp.zeros(())` reused across leaves triggered `Attempt to donate the same buffer twice`.

---

## Data and tokenizer

Corpus encoded to flat `uint16` shards at `data/tokens/<group>/NNNN.bin` + `manifest.json`,
documents separated by `<|endoftext|>` (id 0), groups kept in separate directories so the
mix is chosen after the counts are known.

| group | tokens | bytes/token | note |
|---|---:|---:|---|
| math | 8.80B | 2.78 | synthetic **Question N / Answer N**; 73% of windows contain code |
| science | 6.63B | 3.01 | arxiv; densest formulas, almost no code (0.5%) |
| web | 4.41B | 3.83 | |
| code | 1.22B | 3.26 | |
| **total** | **21.06B** | | |

Verified independently of the encoder: ids in range, memmap round-trips, and decode →
re-encode reproduces ids exactly on 360 sampled documents.

Three traps in this pipeline, all of which recur when the corpus is redone:

* ⚠ Shards overshoot their nominal size (rollover is checked after a whole batch) — the
  loader must read actual file sizes.
* ⚠ **The default mix silently drops `science`.** `web=0.6, math=0.25, code=0.15` has no
  science weight, and science is 31.5% of the tokens. `--mix natural` weights by token count.
* ⚠ **An interrupted encode duplicates tokens.** Shards were opened in append mode, so an
  interrupt mid-parquet (2.4 GB shards) re-encodes the whole file on restart and the
  duplicate shows up only as strange training behaviour. Fixed by truncating to the last
  committed count before writing; port the guard, not just the intent.

`science` is a broad arXiv dump, not an ML corpus: a keyword pass over 1200 windows found
ML/AI 10.3%, physics 10.2%, pure maths 6.8%, econometrics 1.5%, bio/med 1.2%, and **71.6%
unidentified**. It buys breadth, not a domain.

**Document lengths, measured 2026-09-06** (`~/.claude/scratch/doclen.py`, gaps between id 0
over 4 shards per group). This decides what sequence length the corpus can actually support:

| group | docs | median | mean | >2k | >8k | >16k | share of **tokens** in docs >16k |
|---|---:|---:|---:|---:|---:|---:|---:|
| math | 361k | 961 | 1116 | 4.6% | 0.3% | 0.2% | 3.5% |
| **science** | 22k | **16 340** | 20 614 | 99.3% | 84.3% | 49.9% | **75.3%** |
| web | 187k | 817 | 2167 | 18.9% | 3.9% | 1.5% | 33.2% |
| code | 483k | 422 | 832 | 8.2% | 0.5% | 0.1% | 4.3% |

Weighted by group token counts: **55% of the corpus sits in documents longer than 2k, 42%
longer than 8k, 32% longer than 16k.** All of the long text is `science`, which currently has
weight **zero** in the default mix; `math` is useless for long context (3.5% of its tokens in
documents over 16k). Training at 16–32k is therefore first a decision about the mix and only
second one about attention.

Tokenizer: ByteLevel BPE, **V=16384**, no normalizer (round-trip losslessness matters with
a third of the corpus being code), digits split individually, `max_token_length=16`, 32
special slots. Pre-tokenizer is cl100k's `Split` regex with `\p{N}` instead of `\p{N}{1,3}`.
**Math does not compress better with a bigger vocabulary**: 12k → 32k is 2.7× the vocab for
+4.0% on math against +12.7% on web, because individual digit splitting plus dense LaTeX
punctuation leaves little for BPE to merge. 32k costs +12% compute and buys nothing where
it matters. Will be redone for the new plan.

⚠ `swallow-math` is an LLM rewrite of `finemath`, not an independent source (confirmed by
reading pairs and by the dataset card, arXiv:2505.02881) — finemath dropped. Jaccard and
rare-token containment both **failed** to detect the duplication.

Sampler: the stream is cut into non-overlapping slots of T+1 tokens and traversed by a
pseudorandom permutation (Feistel network with a cycle walk, O(1) memory). The group is
picked by a schedule with an exact proportion per 1000 sequences, not multinomially — at
250 steps the multinomial noise in the mix was comparable to the effects being measured.
`batch(step)` is a pure function of (seed, step).

---

## Logging discipline for benchmark runs

Every expensive bug in this file was invisible because the harness printed a *result* and not
the *evidence behind it*. A row that says `944.3 ms` says nothing about whether the program
timed is the program the plan described. So `scripts/bench_moe_jax.py` emits all of the
following on every row, and anything added later goes in the same place. The cost is one
`lower().compile()` we already pay.

| field | source | the failure it catches |
|---|---|---|
| `аргументы_ГБ`, `врем_ГБ`, `выход_ГБ` | `compiled.memory_analysis()` | the only honest memory figure — XLA:TPU keeps internals inside the executable, so the allocator cannot see them |
| `коллективы_ГБ` | every shape on every `all-gather / all-reduce / all-to-all / reduce-scatter / collective-permute` in `compiled.as_text()`, summed | tells sharding schemes apart *statically*, on a laptop, before spending queue time |
| `компиляций` | `step._cache_size()` after two calls | anything above 1 means the timing loop ran a different program from the one measured |
| `дрейф_параметров` | `(dtype, shape, sharding)` of every leaf before and after one step | the cause of the above, named directly |
| `живых_ГБ` | `device.memory_stats()["bytes_in_use"]` | buffers that were not freed — donation silently not happening |

Two rules that go with it:

* **An alarm fires inline, not only in the table.** `компиляций > 1` and any parameter drift
  print a `ВНИМАНИЕ` line, so a bad run is visible while reading the log rather than while
  reading a column afterwards.
* **Every alarm gets a negative control before it is trusted.** The drift alarm was checked
  by monkeypatching `_adafactor_leaf` back to its broken form: it printed both warnings and
  `дрейф: 14 | компиляций: 2`. An alarm that has never fired is not known to work.

Two things this replaces, both of which produced wrong entries in this ledger:
`peak_bytes_in_use` (a **process** high-water mark that never resets, so it is monotonic
across rows and meaningless from the third one on), and reading a `memory_analysis` from the
first compilation as if it described the run.

⚠ **`moe-fast` predates all five fields.** Its 26 rows carry only `аргументы_ГБ`, `врем_ГБ`
and `живых_ГБ`; the kernel was pushed before the columns were added, and Kaggle uploads a
snapshot of the file, so an edit made after the push does not reach a queued run. The dtype
fix *was* in that snapshot — `живых_ГБ` 3.54 against `аргументы_ГБ` 3.47 is the evidence —
but `компиляций` was not checked on any `moe-fast` row. First run with the full set:
`moe-prof`, 6 Sep.

`--profile DIR` writes one extra step per row under `jax.profiler.trace` (after the timing
loop, so it cannot pollute it) and `scripts/prof_ops.py` reads the xplane locally through
`jax.profiler.ProfileData`. That is the difference between knowing the step takes 718 ms and
knowing which op does. ⚠ There is **no `hlo_category` stat** in the xplane events — the only
stats are `device_offset_ps`, `device_duration_ps` and `Time Scale Multiplier` — so the tool
groups by the root of the instruction name, and that grouping cannot be read as "so much went
to matmuls": XLA names a fusion after its root op, so a matmul fused with an add becomes
`convolution_add_fusion` and one fused with anything else becomes `fusion.NNN`. Collectives
and scatter/gather are honest by name; arithmetic has to be recognised by operand shapes in
the per-instruction list.

Also useful and cheap: `+` splits `argv` into independent grids inside one kernel
(`... --out x.json + --shard --experts 144 ...`), so one 15–25 minute TPU queue slot covers
several unrelated sweeps. An exception in one row is caught, recorded in that row, and the
grid continues.

---

## Measurement traps that already cost a run

* **In-training `val_loss` is not comparable across runs.** `MixedLoader` seeds batches as
  `manual_seed(seed + offset + step)`, so different seeds see different validation batches:
  an apparent 0.27-nat seed effect was really 0.033. Cross-run conclusions go through
  `scripts/eval_ckpt.py` on one fixed grid. Nor is it comparable across different T.
* **Do not benchmark on a busy accelerator.** A first pass of the local ceiling numbers was
  taken while training ran on the same card; everything came out 2–3× low with a
  characteristic 1.9 ms floor.
* **Peak memory from an allocator lies** under CUDA graphs (torch does not see their pool)
  and on TPU (`peak_bytes_in_use` never resets). Use `mem_get_info` and
  `memory_analysis()`.
* **`nvidia-smi` and `cudaMemGetInfo` disagree by ~340 MiB** under WDDM; torch's number is
  the one you can actually allocate.
* The FLOP column is `6 · active · tokens`. It ignores attention entirely and counts the
  capacity-buffer padding as useful work — comparable between rows, not a utilisation
  figure.

---

## Open

* **≥2× throughput at fixed T, active mass and total mass** — the requirement of 6 Sep and
  the thing everything else waits on. Profiled; the work is now named and none of it is
  arithmetic:
  1. **cross-entropy head, 25.6% of the step** — kill the in-loop `all-gather` by computing
     the loss on the local row shard, and stop materialising fp32 logits. Worth ~140 ms of
     718 and changes no maths. Needs a loss-equality test and a control row.
  2. **`all_to_all` 14.6% + capacity buffer 12.1%** — the token→expert→token path. Overlap or
     a cheaper filler; this is a rewrite, not a flag.
  3. **attention's fp32 score tensor, 12.3%** — 67 MB per query block per layer.
  4. **layer-stack slicing 11.5%** — the price of `scan` over stacked parameters.
* **int8 is for memory, not for speed** — matmuls are under a fifth of the step, so halving
  them buys at most 1.09×. What it buys is the 5.55 GB of bf16 expert weights, and with them
  27B at B=16 or a weaker remat policy.
* 27.06B fits at B=8 with 90 MB spare and is 15% slower than 22.22B at B=16. Decide once the
  CE fix has moved the baseline, and decide on loss, not only on tok/s.
* **Intra-document masking and retriever packing are unimplemented loader work**, ruled in
  on 6 Sep as necessary: several unrelated documents in one window is cheap
  in FLOPs and harmful to the gradient.
* `jnp.mean(g*g)` inside `_adafactor_leaf` reduces a bf16 array, so the factored second moment
  accumulates over 1024–2048 elements in bf16. Noticed, not measured, not fixed.
* Two init-scale mismatches found in review and never fixed: `ShortConv` keeps kaiming
  std 0.29 against `fla`'s 0.02 (14.5× off), and the completed-block RMS of `AttnRes` is
  ~30× the embedding RMS at init. Both resurface at any larger scale.
* Muon on TPU costs 2.8× the step; unmeasured with fewer iterations or bf16 inside.
* remat=2 measured, remat=3 never ran.
* `scripts/bench_dense_jax.py` still holds a duplicated, un-windowed copy of
  `attend_chunked`.
* `scripts/kaggle_run.py` has no retry on push despite observed silent failures, and its
  docstring accelerator list does not mention the `T4x2` → P100 trap.
* DDP/NCCL on Kaggle 2×T4 unresolved; `gloo` untried.
