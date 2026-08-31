"""KDA — Kimi Delta Attention, built up step by step.

Step 1: the bare recurrence on toy numbers (d_k = d_v = 2).

    S_t = (I - beta_t * k_t k_t^T) @ Diag(alpha_t) @ S_{t-1}  +  beta_t * k_t v_t^T
    o_t = S_t^T @ q_t

Run: uv run python model/kda.py
"""

import torch

# Printing only, no effect on the math: 3 decimals, never switch to
# scientific notation, so small decayed numbers stay readable.
torch.set_printoptions(precision=3, sci_mode=False)


def kda_step(S, k, v, alpha, beta):
    """One step of the KDA recurrence.

    S     : (d_k, d_v) — the state: an associative key -> value memory
    k     : (d_k,)     — key (L2-normalized in the real model)
    v     : (d_v,)     — value
    alpha : (d_k,)     — per-channel retention in (0,1): how much of S survives
    beta  : scalar     — write strength, the learning rate of this online step
    """
    # 1. forget. torch.diag turns the vector alpha (d_k,) into a diagonal
    #    matrix (d_k, d_k), and `@` is matrix multiplication. Multiplying from
    #    the left scales row i of S by alpha[i] — each key channel decays at
    #    its own rate.                                    (d_k,d_k) @ (d_k,d_v)
    S = torch.diag(alpha) @ S

    # 2. erase. `k @ S` is a vector-matrix product: (d_k,) @ (d_k,d_v) -> (d_v,),
    #    which is exactly what is currently stored under key k. torch.outer
    #    builds the (d_k, d_v) matrix k[:,None] * that[None,:], i.e. "the old
    #    content placed back along direction k". Subtracting it clears the slot.
    S = S - beta * torch.outer(k, k @ S)

    # 3. write. Same outer product, now with the new value v: put v in the slot.
    S = S + beta * torch.outer(k, v)
    return S


def naive_step(S, k, v, alpha, beta):
    """Naive linear attention — identical, but WITHOUT the erase step."""
    return torch.diag(alpha) @ S + beta * torch.outer(k, v)


def read(S, q):
    """Read from memory: o = S^T @ q.

    S.T transposes (d_k, d_v) -> (d_v, d_k), then `@ q` with q of shape (d_k,)
    gives (d_v,). With an orthonormal key, this returns the stored value exactly.
    """
    return S.T @ q


def fmt(t):
    """Format a 1-D tensor as fixed-point text.

    torch.set_printoptions does not apply to .tolist(), so numbers were coming
    out as 2.6999998092651367. Format each element explicitly instead.
    """
    return "[" + "  ".join(f"{x:7.3f}" for x in t.tolist()) + "]"


def show(tag, S, reads=()):
    print(f"  {tag}")
    print(f"    S = {fmt(S[0])}")
    print(f"        {fmt(S[1])}")
    for name, q in reads:
        print(f"    read({name}) -> {fmt(read(S, q))}")
    print()


# torch.tensor([...]) builds a tensor from a Python list. The trailing dots
# make these floats — with ints torch would create an integer tensor and the
# decay multiplications below would truncate.
e1 = torch.tensor([1.0, 0.0])
e2 = torch.tensor([0.0, 1.0])
keep = torch.tensor([1.0, 1.0])   # alpha = 1 -> forget nothing
full = torch.tensor(1.0)          # beta = 1 -> write at full strength


# ---------------------------------------------------------------------
print("SCENE 1. Two orthogonal keys. beta=1, alpha=1 (no forgetting).")
print("Expectation: S becomes an exact k -> v table, and reads return v exactly.\n")

S = torch.zeros(2, 2)   # torch.zeros(rows, cols) -> a (2,2) tensor of zeros
show("start", S)

S = kda_step(S, k=e1, v=torch.tensor([3.0, 1.0]), alpha=keep, beta=full)
show("after (k=e1, v=[3,1])", S, [("e1", e1)])

S = kda_step(S, k=e2, v=torch.tensor([-2.0, 4.0]), alpha=keep, beta=full)
show("after (k=e2, v=[-2,4])", S, [("e1", e1), ("e2", e2)])


# ---------------------------------------------------------------------
print("-" * 70)
print("SCENE 2. Overwrite: same key e1, new value [0,5].")
print("Delta rule on the left, naive linear attention on the right.\n")

S_delta = torch.zeros(2, 2)
S_naive = torch.zeros(2, 2)
for k, v in [(e1, torch.tensor([3.0, 1.0])), (e1, torch.tensor([0.0, 5.0]))]:
    S_delta = kda_step(S_delta, k, v, keep, full)
    S_naive = naive_step(S_naive, k, v, keep, full)

show("delta rule", S_delta, [("e1", e1)])
show("naive", S_naive, [("e1", e1)])


# ---------------------------------------------------------------------
print("-" * 70)
print("SCENE 3. Per-channel forgetting: alpha = [0.9, 0.1].")
print("Channel 1 holds its memory a long time, channel 2 forgets almost at once.\n")

alpha = torch.tensor([0.9, 0.1])
S = torch.zeros(2, 2)
S = kda_step(S, k=e1, v=torch.tensor([3.0, 1.0]), alpha=keep, beta=full)
S = kda_step(S, k=e2, v=torch.tensor([-2.0, 4.0]), alpha=keep, beta=full)
show("both written", S, [("e1", e1), ("e2", e2)])

# beta = 0 means "write nothing", so only the decay from step 1 acts.
for step in range(1, 4):
    S = kda_step(S, k=torch.zeros(2), v=torch.zeros(2), alpha=alpha, beta=torch.tensor(0.0))
    show(f"+{step} decay step (writing nothing)", S, [("e1", e1), ("e2", e2)])


# ---------------------------------------------------------------------
print("-" * 70)
print("SCENE 4. Non-orthogonal keys: k2 = [0.707, 0.707] overlaps with k1 = e1.")
print("Question: what survives of the k1 memory, and is v2 still read back exactly?\n")

# A unit-length key at 45 degrees to both axes: overlap cos(45) = 0.707 with e1.
k2 = torch.tensor([1.0, 1.0]) / torch.tensor(2.0).sqrt()
v1 = torch.tensor([3.0, 1.0])
v2 = torch.tensor([-2.0, 4.0])

S_delta = torch.zeros(2, 2)
S_naive = torch.zeros(2, 2)
for k, v in [(e1, v1), (k2, v2)]:
    S_delta = kda_step(S_delta, k, v, keep, full)
    S_naive = naive_step(S_naive, k, v, keep, full)

show("delta rule", S_delta, [("e1", e1), ("k2", k2)])
show("naive", S_naive, [("e1", e1), ("k2", k2)])
print(f"    v1 was {fmt(v1)},  v2 was {fmt(v2)}\n")


# ---------------------------------------------------------------------
print("-" * 70)
print("SCENE 5. Why L2Norm(k) is mandatory: the same key written 4 times,")
print("once with ||k|| = 1 and once with ||k|| = 2, beta = 1 in both cases.\n")

v = torch.tensor([1.0, 0.0])
for name, k in [("||k|| = 1  ->  k = [1, 0]", e1), ("||k|| = 2  ->  k = [2, 0]", 2 * e1)]:
    print(f"  {name}")
    S = torch.zeros(2, 2)
    for step in range(1, 5):
        S = kda_step(S, k=k, v=v, alpha=keep, beta=full)
        # S.norm() is the Frobenius norm: sqrt of the sum of all squared
        # entries. One number for "how big the whole state is".
        print(f"    step {step}: S[0] = {fmt(S[0])}   ||S|| = {S.norm():9.3f}")
    print()

# The erase step multiplies the component along k by (1 - beta * ||k||^2):
# that factor is 0 for a unit key (perfect erase), and -3 for ||k|| = 2
# (sign flip AND 3x amplification) -- which is the divergence above.
for kn in (0.5, 1.0, 1.5, 2.0):
    print(f"  ||k|| = {kn}  ->  erase factor 1 - ||k||^2 = {1 - kn**2:+.2f}")
