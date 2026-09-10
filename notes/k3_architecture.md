# Kimi K3 — implementation notes from the tech report

Source: `k3_tech_report.pdf` (47 pages), read page by page as images, 29 Jul 2026.
This file is the **working replacement for the PDF: look here when implementing, do not
re-read the PDF.** If you need a full page, render it: `pymupdf`, `Matrix(2.2, 2.2)`,
command at the end of this file.

---

## Page map (where to look if you need the original)

| Page | Contents | Relevance to us |
|---:|---|---|
| 1–2 | Title, overview, contribution list | low |
| **3** | **Figure 2 — overall architecture diagram** (block, KDA module, LatentMoE, AttnRes links) | 🔴 key |
| **4** | §2.1 Hybrid Attention, §2.1.1 KDA: recurrence, parameterization, chunkwise form (Eq. 1–4) | 🔴 key |
| **5** | Lower-bounded decay (Eq. 5), full-rank gate (Eq. 6); §2.1.2 Gated MLA + NoPE (Eq. 7) | 🔴 key |
| **6** | §2.2 **Attention Residuals** — Full (Eq. 8–9) and Block (Eq. 10) forms; §2.3 Stable LatentMoE | 🔴 key |
| **7** | Figure 4 (GLU/SwiGLU/SiTU-GLU), Eq. 11 (LatentMoE forward), §2.3.1 Normalized LatentMoE, §2.3.2 **SiTU-GLU (Eq. 12)** | 🔴 key |
| **8** | Figure 5 (QB illustration), §2.3.3 **Quantile Balancing** (Eq. 13) | 🔴 key |
| **9** | QB update (Eq. 14), histogram estimation; §2.4 Native Vision (MoonViT-V2) | 🟡 QB yes, vision no |
| **10** | §2.5 **Per-Head Muon**; §3.1 pretraining data; §3.2 scaling law (cosine > WSD) | 🔴 key |
| **11** | Figure 7 (scaling curves), **Table 1 — full K2 vs K3 configuration**; §3.3 training recipe | 🔴 key |
| **12** | §3.4 long context (NoPE, progressive extension 8k→64k→256k→1M); §4.1 SFT | 🟡 |
| **13** | RL: partial rollout, **Reasoning Effort RL (token budget)**, Agentic GRM | 🟡 directly on our reasoning track |
| **14** | MOPD (Eq. 15), MXFP4 QAT, **Draft Model Fine-Tuning / EAGLE-3 + LK loss (Eq. 16)** | 🟡 for MTP |
| 15–24 | RL environments, task synthesis, agentic environments | low |
| 25–36 | Infrastructure (§5), evals | low (not our scale) |
| 37–42 | Appendix A, start of appendices | low |
| **43** | **Appendix B — SiTU-GLU formally** (Eq. 18–19); Appendix C — QB derivation (Eq. 20–23) | 🔴 |
| **44** | **Algorithm 1 — alternating QB solver**; exact coordinate minimization (Eq. 24–27) | 🔴 |
| **45** | Appendix D — histogram quantile estimation; Appendix E — MoonEP | 🟡 / no |
| **46** | Figure 16 + Appendix F — **XTML chat template** | 🟡 for SFT |
| 47 | Continuation of F | 🟡 |

---

## 1. Overall block structure (pp. 3, 4)

The model mixes information along three axes:
- **along the sequence** — Hybrid Attention (KDA + MLA)
- **along depth** — Attention Residuals
- **along width** — Stable LatentMoE

**Block pattern (3:1):**
```
KDA  → Stable LatentMoE
KDA  → Stable LatentMoE
KDA  → Stable LatentMoE
MLA  → Stable LatentMoE        ← one global layer per block
```
Repeated through the whole backbone. **One extra Gated MLA is placed at the very end of
the backbone**, so the last layer always performs global attention.
Totals for K3: 69 KDA + 24 MLA = 93 layers.

Every sub-layer (both attention and MoE) connects to the rest not through an ordinary
residual but through AttnRes (see §4).

---

## 2. KDA — Kimi Delta Attention (pp. 4–5)

Linear attention on the delta rule with a **channel-wise forget gate**. This is what
replaces softmax attention in 3 layers out of 4 and gives cheap long context.

### 2.1 Recurrence (Eq. 1)

```
S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t k_t v_tᵀ
õ_t = S_tᵀ q_t
```

- `S_t ∈ R^{d_k × d_v}` — recurrent state (a matrix, not a vector!)
- `α_t ∈ (0,1)^{d_k}` — **per-channel** retention coefficient for one step
- `β_t ∈ (0,1)` — delta-rule write strength
- `(I − β k kᵀ)` is a Householder-like projection: "erase whatever is already stored
  along direction k", then `+ β k vᵀ` writes the new value. Hence "delta rule".

### 2.2 Head parameterization (Eq. 2)

```
q_t, k_t = L2Norm( Swish( ShortConv( W_{q/k} x_t ) ) )   ∈ R^{d_k}
v_t      =        Swish( ShortConv( W_v     x_t ) )      ∈ R^{d_v}
β_t      = Sigmoid( W_β x_t )                            ∈ (0,1)
z_t      = W_α↑ W_α↓ x_t + b_α                           ∈ R^{d_k}   (low-rank!)
```
- ShortConv — short causal convolution over time (depthwise, kernel usually 3–4)
- L2Norm on q and k is mandatory, otherwise the delta rule falls apart
- `z_t` — the decay logit, computed through a **low-rank** projection (W↓ then W↑) plus a
  head-specific bias `b_α`

### 2.3 Lower-bounded decay — the main departure from Kimi Linear (Eq. 5, p. 5)

Kimi Linear (and GDN, and Mamba-2) used an unbounded negative softplus:
`g = −e^A · Softplus(z) ∈ (−∞, 0)`.

K3 replaces it with a **lower-bounded scaled sigmoid**:
```
g_t = g_min · Sigmoid( e^{A_h} · z_t )   ∈ (g_min, 0)
α_t = exp(g_t)                            ∈ (e^{g_min}, 1)

g_min = −5 (fixed),  A_h — learnable per-head log-scale, initialized A_h = 0
```

**Why.** In the chunkwise form the keys are divided by the cumulative decay `1/Γ`. If decay
is unbounded, `1/Γ` grows without limit and overflows fp16/bf16. With `g_min = −5` every
retention factor is > e^{−5} ≈ 6.7·10⁻³, the cumulative log-decay over a 16-token tile lies
in (−80, 0), and the rescaling multiplier stays below e^{80} — inside the dynamic range.
**Practical payoff:** both diagonal and off-diagonal tiles are computed as dense matmuls on
Tensor Cores; the separate slow "position-pair diagonal" path from Kimi Linear disappears.

✅ **For us (corrected 29 Jul, see the bf16 ledger entry).** We train in **bf16**, whose
dynamic range is `log(max) = 88.72` — e^{80} fits, so **`g_min = −5` works as published, no
change needed**. On this card bf16 is *emulated*: stored in 16 bits, computed in fp32,
which is exactly what we want here.
*Superseded:* an earlier version of this note assumed fp16 (`log(max) = 11.09`) and planned
to either raise `g_min` or accumulate the rescale in fp32. fp16 measured 5× slower than
fp32 on the 1660 Ti and was dropped entirely — the problem is moot.

### 2.4 Chunkwise parallel form (Eq. 3–4)

Recurrent between chunks, parallel inside a chunk. Chunk size `C`.

```
γ^{i→j}_[t] = Π_{r=i..j} α^r_[t]          # per-channel cumulative decay
Γ^{1→C}_[t] ∈ R^{C × d_k}                 # γ¹…γ^C stacked as rows

Ṽ_[t] = U_[t] − W_[t] S_[t]               # pseudo-value, U and W from the UT transform

A_[t] = Tril[ (Q_[t] ⊙ Γ^{1→C}) (K_[t] / Γ^{1→C})ᵀ ]
O_[t] = (Γ^{1→C} ⊙ Q_[t]) S_[t]  +  A_[t] Ṽ_[t]
         └── inter-chunk ──┘        └─ intra-chunk ─┘
```
`Tril` zeroes the strictly upper triangle but **keeps the diagonal** — because every output
reads the state *after* the current token's update.

The UT transform and the full derivation are not in the K3 report; it defers to the Kimi
Linear paper [64]. **They are reproduced in the second half of this file (Eq. 6–7), so that
open question is closed** — either use that or take the kernel from `fla`
(flash-linear-attention).

### 2.5 Full-rank output gate (Eq. 6)

Kimi Linear used a low-rank output gate, K3 uses a full-rank one:
```
y_t = W_o [ Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t) ]
```
The RMSNorm is **head-wise** and is applied to the recurrent output before the gate.

---

## 3. Gated MLA + NoPE (pp. 5–6)

MLA = Multi-head Latent Attention from DeepSeek-V2: KV is compressed into a
low-dimensional latent `c_t = W_c x_t`, and it is `c_t` that gets cached; the full K and V
are reconstructed by learnable up-projections at attention time. This shrinks the KV cache.

What K3 adds:
1. **NoPE in all MLA layers** — no positional encoding at all. Positional information comes
   from the KDA layers (their recurrence and decay are inherently position-sensitive).
   Consequence: context extends to 1M **without** retuning the RoPE base and without YaRN.
2. **Full-rank channel-wise output gate** (Eq. 7), symmetric with KDA:
   ```
   y_t = W_o [ Sigmoid(W_g x_t) ⊙ õ_t ]
   ```
3. The attention output is kept in **FP32 during training** — to remove the biased rounding
   error of flash-attention. Costs a doubled on-chip footprint for the output tile; they
   redesigned the kernel to overlap it with KV staging buffers.

---

## 4. Attention Residuals (p. 6) — already built in an earlier project

**Idea.** An ordinary residual compresses all previous information into a single state
`h_l` — a bottleneck analogous to an RNN over time. The transformer replaced recurrence
over time with attention; AttnRes does **the same thing over depth**: each layer
selectively retrieves representations from all preceding layers instead of accumulating
them uniformly.

### 4.1 Full AttnRes (Eq. 8–9)

For layer `l`: a learnable **pseudo-query** `q_l = w_l ∈ R^d` (one per layer), with keys
and values:
```
k_i = v_i = { h_1            if i = 0      (token embedding)
            { f_i(h_i)       if 1 ≤ i ≤ l−1 (output of layer i)
```
The attention kernel is a softmax with an RMSNorm inside:
```
φ(q, k) = exp( qᵀ · RMSNorm(k) )

α_{i→l} = φ(q_l, k_i) / Σ_{j=0}^{l−1} φ(q_l, k_j)
h_l     = Σ_{i=0}^{l−1} α_{i→l} · v_i
```
**The RMSNorm inside the kernel is mandatory** — otherwise layers with large output norm
dominate the weights.

Cost: O(L²d) arithmetic (tolerable, L < 100), but O(Ld) memory to keep all layer outputs
alive.

### 4.2 Block AttnRes (Eq. 10) — what is actually used

The L layers are split into **N blocks** of S = L/N layers. Inside a block, layer outputs
are reduced by **summation** into a single representation:
```
b_n = Σ_{j ∈ B_n} f_j(h_j)          # full sum over block n
b_n^i                                # partial sum over the first i layers of the block
b_0 = h_1                            # the embedding is always available as a source
```
Between blocks — full attention over just the N block representations:
```
V = [b_0, b_1, …, b_{n−1}]ᵀ                 if i = 1 (first layer of block n)
V = [b_0, b_1, …, b_{n−1}, b_n^{i−1}]ᵀ      if i ≥ 2
```
Keys and weights follow Eq. 8–9. The final output layer aggregates all N block
representations.

Memory and communication drop from O(Ld) to O(Nd). The block structure additionally
**bounds the state at inference time**.

📌 **Empirical result: N ≈ 8 recovers almost the entire gain** at every scale.
K3: 93 layers → 8 blocks of 12 layers (last block incomplete), 9 blocks counting the
embedding.

---

## 5. Stable LatentMoE (pp. 6–8)

### 5.1 The basic LatentMoE idea

In an ordinary MoE every selected expert receives the full `d`-dimensional representation
of the token, so weight traffic and communication grow with the number of active experts.
LatentMoE **decouples model width from routed-expert width**:
- **shared experts** operate at full width `d` (common transformations)
- **routed experts** operate in a compact latent of width `ℓ`

Forward (Eq. 11):
```
u = Σ_{i ∈ T_k(x)}  p_i · E_i^routed( W↓ x )        # W↓: R^d → R^ℓ
y = Σ_{j=1..N_s}    E_j^shared(x)  +  W↑ RMSNorm(u) # W↑: R^ℓ → R^d
```
K3: `N_s = 2` shared experts per layer, 896 routed, 16 active → **sparsity 56**.
Latent `ℓ = 3584 = 0.5 · d`, where `d = 7168`.

### 5.2 Two fixes against instability

Extreme sparsity amplifies two failure modes:
1. The routed path chains `W↓`, a gated multi-branch FFN and `W↑` into what is nearly four
   sequential matmuls — a badly conditioned structure, activations blow up.
2. Balancing ~10³ experts leaves the regime where aux-loss-free bias updates behave well.

**Fix 1 — Normalized LatentMoE (§2.3.1).** An RMSNorm **between expert aggregation and the
up-projection** (visible in Eq. 11). The original LatentMoE applied `W↑` directly to `u`,
whose scale floats depending on which experts were selected and on the routing weights.
Besides stabilizing, this **consistently improves validation loss and benchmarks**.

**Fix 2 — SiTU-GLU (§2.3.2, Appendix B on p. 43).**

The problem with SwiGLU: both factors are unbounded, and coinciding large coordinates
produce activation outliers and overflow risk in low precision. Plain GLU has a bounded
sigmoid gate but loses Swish's approximately-linear positive regime.

The fix is a soft cap `softcap(x, β) = β·tanh(x/β)`, applied to the gate's linear factor
**and independently to the up branch**:
```
SiTU-GLU(x) = [ β₁ · tanh(W_g x / β₁) ⊙ Sigmoid(W_g x) ] ⊙ [ β₂ · tanh(W_u x / β₂) ]

β₁ = 4   (gate branch)
β₂ = 25  (up branch)
```
Properties (p. 43):
- near zero, `β·tanh(z/β) = z + O(z³/β²)` → **matches SwiGLU to first order**
- as β₁, β₂ → ∞ it **exactly recovers SwiGLU**
- the output is bounded: `‖SiTU-GLU(x)‖_∞ ≤ β₁β₂ = 100`
- unlike hard clamping of pre-activations, **a soft cap keeps gradients nonzero** far from
  the saturation boundary — that is what gives the better training behaviour

### 5.3 Quantile Balancing (§2.3.3, pp. 8–9; derivation on pp. 43–44)

Aux-loss-free routing: a bias `b_j` is added to the router score **for Top-k selection
only**, and **not** to the mixture weights:
```
s_i   = Sigmoid(W_r x_i)
T_i   = argtop_k( s_i + b )
p_{i,j} = s_{i,j} / Σ_{r ∈ T_i} s_{i,r}      # no b here!
```
Because `b` does not enter `p`, it controls dispatch only and never touches gradient
optimization of the router.

**What came before (DeepSeek):** a fixed step `b_j ← b_j + γ·sign(ℓ̄ − ℓ_j)`. γ trades slow
adaptation against load oscillation. With 896 experts this behaves badly.

**QB:** sets each expert's bias to the **router-score quantile corresponding to the target
load**, in a single forward pass.
- target load `q := mk/n` tokens per expert (m tokens, n experts, Top-k)
- instead of Top-k, take **Top-(k+1)** on the biased score `s_i + b`: the first k are the
  real routes, and the (k+1)-th entry gives the **threshold `α_i`** an expert must exceed
  to enter token i's Top-k. This removes the need for a separate token-side quantile.
- update (Eq. 14):
```
b̂_j^{(t+1)} ← − quantile_{1−k/n}( s_{:,j} − α^{(t)} )
b^{(t+1)}   ← b̂^{(t+1)} − mean( b̂^{(t+1)} ) · 1
```
The second line removes the common shift, which does not change Top-k. For causality the
update takes effect **only on the next step** — a batch is never routed by a bias derived
from itself. At inference the bias is **frozen**.

**Why it works (pp. 43–44).** Maximum balanced assignment is a linear program; its dual
objective
`L(α, β) = Σ max(0, s_{i,j} − α_i − β_j) + k·Σα_i + (mk/n)·Σβ_j`
is minimized coordinate-wise, and **each subproblem has a closed-form solution — the very
same (1−k/n) quantile** along the token and expert axes respectively. Hence the name.
DeepSeek's sign update is SignSGD on the same dual: it keeps only the direction, whereas QB
**jumps straight to the exact minimum**. That is why QB has no learning-rate-style
hyperparameter and equilibrates within a few steps even at ~10³ experts.

**Algorithm 1 (p. 44) — alternating solver:**
```
Input: score matrix s ∈ R^{m×n};  Output: assignment x ∈ {0,1}^{m×n}
1  β ← 0_{1×n}
2  for t = 1..T:
3      α ← desc_sort(s − β, axis=1)[:, k:k+1]
4      β ← desc_sort(s − α, axis=0)[mk/n : mk/n+1]
5  end
6  return x, where x_{i,j} = 1 if j ∈ argtop_k(s_i − β)
```

**Histogram quantile estimation (Appendix D, p. 45)** is needed only when sharding across
many ranks — **we do not need it**: on a single GPU the quantile is computed exactly. Noted
briefly in case it ever comes up: the "required bias" `r_{i,j} := α_i − s_{i,j}` is
histogrammed into B ≈ 1000 bins over the interval `[b_min − 1, b_max + 1]`; counters are
additive, one all-reduce per layer per step.

---

## 6. Per-Head Muon (§2.5, p. 10)

Muon applies to matrix parameters. For **attention projections** K3 refines it: instead of
Newton–Schulz orthogonalization over the full Q/K/V matrix, **split the momentum matrix
along the head dimension and orthogonalize each head's block separately**.

**Why.** Full-matrix orthogonalization treats all heads as one coupled block: heads with a
large gradient/momentum scale dominate the shared update direction, while small-scale heads
receive under-normalized updates. Per-head orthogonalization equalizes update scale across
heads.

As a side effect it is **cheaper**: Newton–Schulz on tall per-head blocks costs less than on
the full projection matrix.

📌 For us this is a cheap patch on top of the Muon already written in `my_model`.

---

## 7. Hyperparameters and training recipe

### Table 1 (p. 11) — K3 configuration

| Parameter | Kimi K2 | **Kimi K3** |
|---|---|---|
| Architecture | MoE | MoE |
| Layers | 61 | **93** |
| Total parameters | 1.04T | **2.78T** |
| Active parameters | 32.6B | **104.2B** |
| Hidden dimension | 7168 | **7168** |
| **Latent MoE dimension** | — | **3584 (0.5×)** |
| MoE hidden dim per expert | 2048 | **3072** |
| Routed experts | 384 | **896** |
| Active experts per token | 8 | **16** |
| Shared experts | 1 | **2** |
| Attention heads | 64 | **96** |
| Dense layers | 1 | **1** |
| Vocabulary size | 160K | **160K** |
| Training context | 128K | **1M** |
| Attention mechanism | MLA | **Hybrid KDA–MLA** |
| Activation | SwiGLU | **SiTU-GLU** |
| Attention layer composition | 61 MLA | **69 KDA + 24 MLA** |
| MTP layers | 1 | **1** |

Separately: the first layer is **dense** (not MoE). This is standard and it matters.

### Recipe (§3.3, pp. 11–12)

- Optimizer: **Per-Head Muon** + weight clipping from K2
- MoE balancing: **QB**
- LR schedule: **cosine decay with 1% linear warmup**
- **Weight decay = 0.1** throughout
- Context: start at **8k**, then extend to 64k in a separate phase

### Scaling law (§3.2, p. 10) — an important conclusion

They ran an **independent hyperparameter search for each schedule** and found that **cosine
decay consistently yields lower final loss than WSD**.

The methodological point worth remembering: cosine and WSD have **substantially different
optimal peak LR and batch size**, even at the same model size and token budget. Comparing
them on a shared hyperparameter set is unfair — whichever schedule those hyperparameters
happen to suit will win. Prior work where WSD won probably suffered from exactly this.

In total, the architecture + data + training improvements give a **≈2.5× gain in scaling
efficiency** over K2.

### Long context (§3.4, p. 12)

NoPE + KDA → extrapolation to 1M **without modifying the positional encoding**. A
four-stage program: 8K → 64K during pretraining, 256K → 1M during cooldown. The expensive
long sequences are concentrated in a small fraction of the budget.

Separately: **length by itself does not produce long-context ability**. They synthesize data
by shuffling and concatenating documents and subtasks so that the task is solvable **only**
by reaching for information scattered across the whole context. Otherwise attention
degenerates into local patterns.

---

## 8. Post-training — what relates to our reasoning plans

### Reasoning Effort RL (p. 13) — directly on topic for test-time compute

A per-task budget control mechanism:
- each task `x` is assigned an initial token budget `b_0(x)`, estimated by a cold-start
  model
- the task reward is **overwritten with −1** if the trajectory's total budget `T(y)` exceeds
  a threshold `τ · b_0(x)`
- for general tasks `T(y)` counts **thinking** tokens; for agentic tasks, all output tokens
  including tool-call arguments
- training proceeds **in stages over the multiplier τ**: first a *max-budget* variant with
  large τ (but capped, to suppress overthinking), then τ is annealed downward → producing
  *high-* and *low-effort* experts
- trajectories from all levels are pooled together for SFT and distillation

📌 This is exactly the mechanism that produces "caveman style": penalizing budget overrun
pushes the policy toward maximum information per token.

### MOPD — Multi-Teacher On-Policy Distillation (pp. 13–14)

9 experts (3 domains × 3 effort levels) are merged into one model. Per-token reward:
```
r_opd(y_t | e, x, y_<t) = clip( sg( log [ π_teacher(y_t|x,y_<t) / π_θ(y_t|e,x,y_<t) ] ), −R_max, R_max )
```
`sg` is stop-gradient. A dense reward signal, embedded in the same RL framework. They tried
finer top-k distillation objectives and **saw no advantage**.

### MTP / EAGLE-3 draft head (p. 14)

K3 pretrains with an **MTP layer that structurally mirrors a backbone block**. That layer is
then fine-tuned into a draft model in the EAGLE-3 style: the target model is frozen, only
the draft layer and its feature-fusion projection are updated.

- the draft is unrolled for **7 steps** during training; after the first step it consumes
  its own outputs
- **the draft's input fuses low/mid/high-level features of the target model** — the outputs
  of the 1st, 4th and last AttnRes blocks. They are concatenated and projected to hidden
  size by a matrix `W_E3` **without bias**, initialized as `[0 0 I]` — so that at the start
  the fused representation matches the high-level feature `h_h` that the MTP layer was
  pretrained on, with low/mid mixed in gradually.
- the loss is not KL but the **LK loss** (Eq. 16), which directly maximizes acceptance rate:
```
L_LK = − log Σ_{x ∈ V} min( p(x), q(x) )
```
`p` is the target model's distribution, `q` the draft's, both at temperature 1, **without**
an auxiliary ground-truth cross-entropy term. Motivation: minimizing KL does not guarantee
maximum acceptance rate for a capacity-limited draft.

### XTML chat template (Appendix F, pp. 46–47)

An XML-like markup where the angle brackets are replaced by **three reserved special
tokens**: `[open]`, `[sep]`, `[close]`, plus `[end_of_msg]` as a stop marker.
An element looks like: `[open]tag attr="value"[sep] ... [close]tag[sep]`.

Three goals: **extensibility** (new capabilities arrive as backward-compatible message
formats rather than template revisions), **low alignment tax** (the format is learned from
minimal supervised data), and **decoding friendliness** (simple encoders, streaming parsers,
grammar-constrained enforcers).

The assistant message body is split into three channels: **`think`, `response`, `tools`**.

Context layout:
- **global options** (`tool-declare`, `thinking-effort`) — **before** all input messages:
  they govern the whole session, change rarely, and changing them invalidates the KV cache
  anyway
- **input messages** — system, user, assistant, tool
- **one-shot options** (`tool_choice`, `response_format`) — **after** the input messages, so
  that per-request changes **do not touch the KV cache of the history**

📌 For our SFT: separating `think` / `response` with special tokens is what we need for
adaptive reasoning and for measuring "how many tokens went into thinking".

---

## 9. What we take and what we drop at our scale

| Component | Decision | Why |
|---|---|---|
| **AttnRes (Block, N ≈ 8)** | ✅ take | the full version already exists in an earlier project; the block form is a direct improvement |
| **Stable LatentMoE** | ✅ take | trying MoE is a goal of the project; the latent makes experts cheap |
| **RMSNorm before W↑** | ✅ take | one line, improves loss |
| **SiTU-GLU** | ✅ take | one formula; a bounded output is cheap insurance in 16-bit storage |
| **Quantile Balancing** | ✅ take | on a single GPU the exact quantile is trivial, no histogram needed |
| **KDA** | ⚠️ take, carefully | needs the UT transform (derived below) plus either our own kernel or `fla`; the fp16 range conflict is gone now that we train in bf16 |
| **Gated MLA + NoPE** | ✅ take | cheaper KV cache, NoPE removes all RoPE fiddling |
| **Hybrid 3:1 + final MLA** | ✅ take | this is the "mamba-like attention" we wanted |
| **Per-Head Muon** | ✅ take | cheap patch on top of the existing Muon |
| **cosine + 1% warmup, wd 0.1** | ✅ take | their scaling law says cosine > WSD outright |
| **Dense first layer** | ✅ take | standard, stabilizes the start |
| **MTP / EAGLE-3 draft** | 🟡 later | needed for test-time compute (more samples in the same time), but after the base model |
| **Reasoning Effort RL** | 🟡 later | directly on topic for caveman style, an RL-stage item |
| **XTML chat template** | 🟡 at SFT | a simplified version: `think`/`response` channels |
| **MoonViT-V2 / vision** | ❌ no | we are text-only |
| **MXFP4 QAT** | ❌ no | Turing has no FP8/FP4 |
| **Histogram quantile** | ❌ no | single GPU |
| **MoonEP, infrastructure §5** | ❌ no | not our scale |
| **MOPD** | ❌ no | needs 9 expert models |

### Open implementation questions

1. ~~**KDA in fp16.**~~ **Closed 29 Jul.** fp16 measured 5× slower than fp32 on the 1660 Ti
   and was dropped; bf16 has `log(max) = 88.72`, so the e^{80} rescale fits and `g_min = −5`
   needs no adjustment.
2. ~~**UT transform** not derived in the K3 report.~~ **Closed** — derived in the Kimi Linear
   notes below (Eq. 6–7); a ready kernel also exists in `fla/ops/kda`.
3. **Parameter split at ~200M total**: how much goes into routed experts, how much into
   shared, what latent `ℓ`. K3 uses `ℓ = 0.5·d` — start from that.
4. **Expert count at our size**: 896 is obviously wrong. K3's sparsity is 56; at ~200M total
   and 30–50M active we need sparsity ~4–6, i.e. roughly 32–64 routed at top-4…8 plus 1–2
   shared. To be computed exactly at config time. Note that both Quantile Balancing and the
   latent-expert scheme were motivated by *extreme* sparsity — it is not obvious they behave
   the same in our regime.
5. **Blocks vs. depth.** N ≈ 8 is K3's empirical optimum at **93 layers** (12 layers per
   block). At ~20 layers, 8 blocks means ~2.5 layers each — a different regime, and the block
   reduction has little left to reduce. N may need to drop with depth.

---

## How to re-read a page

```bash
uv run python -c "
import pymupdf
d = pymupdf.open('k3_tech_report.pdf')
d[N-1].get_pixmap(matrix=pymupdf.Matrix(2.2, 2.2)).save('/tmp/k3_pN.png')
"
```
where `N` is the page number from the map above.

---
---

# Kimi Linear — notes (the source of KDA)

Source: `kimi_linear.pdf` (arXiv 2510.26692v2, 28 pages), downloaded and read 29 Jul 2026.
K3 cites this paper for the KDA derivation and does not reproduce it. It is here.

Open source: **`fla` (flash-linear-attention), `fla/ops/kda`** — the authors released the
KDA kernel there. Repository: https://github.com/MoonshotAI/Kimi-Linear

## Page map

| Page | Contents | Relevance |
|---:|---|---|
| **3** | §2.2 Derivation chain: linear attention → DeltaNet → Gated DeltaNet | 🔴 understanding |
| **4** | §3 KDA (Eq. 1), §3.1 chunkwise: WY representation (Eq. 3–5), **UT transform (Eq. 6–7)**, state update (Eq. 8) | 🔴 implementation |
| **5** | Chunk output (Eq. 9), §3.2 efficiency vs DPLR; §4 neural parameterization of KDA | 🔴 implementation |
| **6** | Figure 3 — Kimi Linear block diagram; output gate (Eq. 10); 3:1 hybrid; NoPE | 🔴 |
| 10 | §5.4 training recipe (MuonClip, WSD, ctx 4096, lr 1.1e-3, batch 32M tokens) | 🟡 |
| 18 | §7.2 discussion of hybrids: intra-layer vs inter-layer | 🟡 |

## The derivation chain (p. 3) — what makes KDA make sense

The key frame: **linear attention is online learning**. The state `S` is not a "cache" but
fast weights that are trained during the pass over the sequence. Every step is a gradient
descent step on some loss. Different linear-attention variants = different losses.

**1. Naive linear attention**
```
S_t = S_{t−1} + k_t v_tᵀ ,     o_t = S_tᵀ q_t
```
This is gradient descent on `L_t(S) = −⟨Sᵀk_t, v_t⟩` — "just reinforce recent key-value
pairs". The problem: **there is no criterion for what to erase**. The state grows without
bound and old associations interfere with new ones over long context.

**2. DeltaNet — descent on reconstruction error**
```
L_t(S) = ½ ‖Sᵀ k_t − v_t‖²
S_t = S_{t−1} − β_t ∇_S L_t(S_{t−1}) = (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ
```
Now `β_t` is literally the **learning rate** of one online-learning step. The rule is "keep
correcting S toward mapping k_t ↦ v_t". The rank-1 structure of the update is equivalent to
a generalized Householder transformation → hardware-friendly chunkwise parallelization.

**3. Gated DeltaNet (GDN) — forgetting as weight decay**
DeltaNet is stable but **stores stale associations forever**. GDN adds a scalar forget gate
`α_t ∈ [0,1]`:
```
S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ
```
`α_t` acts as **weight decay on the fast weights** — data-dependent L2 regularization.

📌 Important observation by the authors: GDN can be read as a **multiplicative positional
encoding** in which the transition matrix is data-dependent and learnable — a relaxation of
the orthogonality constraint that RoPE imposes. This is where NoPE comes from: position is
already encoded in the decay.

**4. KDA — channel-wise forgetting**
GDN (and Mamba-2) use a **scalar** forget gate per head. KDA makes it **channel-wise** (as
in GLA): every feature dimension gets its own forgetting rate.
```
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ ,   α_t ∈ (0,1)^{d_k}
```
Motivation: the RNN state's finite memory is used more precisely — some channels can be kept
"long", others "short".

**Why this exact form rather than general DPLR.** The general Diagonal-Plus-Low-Rank form is
`S_t = (D − a_t b_tᵀ) S_{t−1} + k_t v_tᵀ` with independent `a` and `b`. Fine-grained decay in
that form creates numerical problems in the divisions (intra-chunk, Eq. 9), which is why GLA
computes in the log domain with secondary chunking in full precision — and that **destroys
the ability to use half-precision matmuls**. KDA **ties both `a` and `b` to `k`**, which
removes the bottleneck: second-level chunk matrix computations drop from 4 to 2, and three
further matmuls disappear. **Operator efficiency roughly doubles relative to general DPLR.**

## The chunkwise algorithm (pp. 4–5) — what the K3 report leaves out

Notation: chunk length `C`, `Γ^{1→C}_[t] ∈ R^{C×d_k}` is the stack of cumulative decays,
`Tril`/`StrictTril` are lower-triangular masks with and without the diagonal.

**WY representation (Eq. 3–5).** A series of rank-1 updates is packed into a compact form
through auxiliary vectors `w_t ∈ R^{d_k}` and `u_t ∈ R^{d_v}`:
```
P^r = Diag(γ^r) − Σ_{i=1..r}   Diag(γ^{i→r}) k^i w^iᵀ
H^r =            Σ_{i=1..t}   Diag(γ^{i→r}) k^i u^iᵀ

w^r = β^r ( Diag(γ^r) k^r − Σ_{i=1..r−1} w^i ( k^iᵀ Diag(γ^{i→r}) k^r ) )
u^r = β^r (            v^r − Σ_{i=1..r−1} u^i ( k^iᵀ Diag(γ^{i→r}) k^r ) )
```
The form of `P` is taken from Comba [40] to avoid an extra matrix inversion downstream.

**UT transform (Eq. 6–7) — here it is.** Needed to eliminate non-matmul FLOPs:
```
M_[t] = ( I + StrictTril[ Diag(β) (Γ^{1→C} ⊙ K) (K / Γ^{1→C})ᵀ ] )^{−1} · Diag(β)

W_[t] = M_[t] ( Γ^{1→C} ⊙ K_[t] )
U_[t] = M_[t] V_[t]
```
The inverse of the lower-triangular matrix is computed iteratively row by row by **forward
substitution** (Gaussian elimination), not through a general solver.

**State update between chunks (Eq. 8):**
```
S_[t+1] = Diag(γ^C) S_[t] + ( Γ^{i→C} ⊙ K_[t] )ᵀ ( U_[t] − W_[t] S_[t] )
```

**Chunk output (Eq. 9)** — the same as Eq. 4 in the K3 report, but here the origin of the
"pseudo-value" is visible:
```
O_[t] = (Γ^{1→C} ⊙ Q_[t]) S_[t]  +  Tril[ (Γ^{1→C} ⊙ Q_[t]) (K_[t] / Γ^{1→C})ᵀ ] (U_[t] − W_[t] S_[t])
         └──── inter-chunk ────┘         └──────── intra-chunk ────────┘         └── pseudo-value ──┘
```
Strategy: **inter-block recurrent, intra-block parallel** — to load the Tensor Cores with
matrix multiplications as much as possible.

## Parameterization (pp. 5–6)

```
q^h, k^h = L2Norm( Swish( ShortConv( W_{q/k} x_t ) ) )   ∈ R^{d_k}
v^h      =         Swish( ShortConv( W_v     x_t ) )     ∈ R^{d_v}
α^h      = f( W_α↑ W_α↓ x_t )                            ∈ [0,1]^{d_k}
β^h      = Sigmoid( W_β x_t )                            ∈ [0,1]
```
- **`d_k = d_v = 128` in all experiments** — the head dimension
- the rank of the low-rank projection `W_α↓`/`W_α↑` **equals the head dimension**
- L2Norm on q and k is "for eigenvalue stability"
- `f(·)` is a decay function "as in GDN and Mamba": in Kimi Linear it is `−e^A·Softplus(z)`;
  **K3 replaced it with the lower-bounded sigmoid** (see above)

**Output gate (Eq. 10)** — in Kimi Linear it is **low-rank**:
```
o_t = W_o ( Sigmoid( W_g↑ W_g↓ x_t ) ⊙ RMSNorm( KDA(q, k, v, α, β) ) )
```
The low rank was chosen "for a fair parameter-count comparison"; quality is comparable to
full rank. **K3 switched to full rank.** Noted separately: the gate **mitigates Attention
Sink**.

## Hybrid design and ablations

- **Layerwise, not headwise.** Mix whole layers rather than heads inside a layer — for
  infrastructure simplicity and training stability.
- **Exactly 3:1 (3 KDA : 1 full MLA)** — the best quality/throughput compromise. Ablations:
  - **more KDA (e.g. 7:1)** — comparable train loss, but **noticeably worse validation**
  - **less KDA (1:1)** — inference cost rises with no gain
  - **pure linear** — bad
- **NoPE on all MLA layers.** All responsibility for position is delegated to the KDA layers.
  KDA is declared the primary position-aware operator — a role analogous to, or stronger
  than, that of short convolutions or SWA.
- The first layer is **dense, no MoE**, "for stable training".
- For the linear component they deliberately do **not** use Mamba-2: KDA is better
  specifically at **retrieval and copying**.
- Worth recording: pure linear attention is bad at **exact retrieval from memory and exact
  copying** — which is the whole reason a hybrid is needed.

## Kimi Linear training recipe (p. 10)

| Parameter | Value |
|---|---|
| Context | 4096 |
| Optimizer | MuonClip |
| LR schedule | **WSD** (K3 later moved to cosine) |
| Peak LR | 1.1e-3 |
| Global batch | 32M tokens |
| Tokens | 1.4T (comparisons) / 5.7T (final checkpoint) |
| MoE | 256 experts, 8 active, 1 of them shared → sparsity 32 |
| Size | 48B total / 3B active |

## What this changes for us

1. **The UT transform is found** — open question №2 from the K3 notes is closed (Eq. 6–7).
2. **A ready kernel exists** — `fla/ops/kda` in flash-linear-attention. We probably will not
   have to write our own, but we must check whether it runs on sm_75 (Triton may require
   Ampere+).
   *Update 31 Jul:* a minimal Triton `tl.dot` kernel compiles and returns correct results on
   this card in fp32/fp16/bf16 (`rel_err ~1e-7`), so Triton itself is not a blocker. Speed
   is untested.
3. **d_k = d_v = 128** — a concrete starting point for head size.
4. **The 3:1 ratio is confirmed by ablation**, including the finding that 7:1 gives a
   deceptively good train loss with poor validation. Worth re-checking ourselves at our
   scale — it is a cheap ablation.
5. **The first layer is dense** — repeated in both K3 and Kimi Linear. We take it.
