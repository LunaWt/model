"""Что влезает и что быстрее на одном чипе TPU v5e: dense против MoE, полное
внимание против блочно-локального.

    colab --auth=adc exec -s tpu -f scripts/bench_tpu_arch.py

Меряется полный шаг обучения, вместе с оптимизатором — иначе потолок по памяти
не имеет смысла: веса это меньше половины того, что лежит в HBM.

Три вопроса, три столбца в выводе.

1. **Сколько параметров влезает.** Adam держит на параметр: bf16-веса 2 Б,
   градиент 2 Б, master-копию fp32 4 Б и два момента fp32 по 4 Б — 16 Б всего.
   16 ГБ HBM делится на 16 и даёт ~0.9 млрд, и это до активаций. Поэтому
   меряются и `adamw`, и `adafactor` (второй момент факторизован по строкам и
   столбцам, вместо 4 Б/пар выходит ~0).

2. **MoE против dense при равной активной массе.** На одном чипе MoE не экономит
   память — веса всех экспертов лежат в HBM целиком, — зато экономит вычисление.
   Вопрос ровно в том, окупается ли это на TPU, где рой мелких матмулов и так
   был нашей главной бедой. Поэтому экспертов мало и они широкие.

   Раскладка токенов по экспертам сделана через scatter/gather с фиксированной
   ёмкостью, без одноразового тензора [N, k, E, cap]: он для E=16 и cap≈640
   занял бы сотни мегабайт на ровном месте.

3. **Полное внимание против блочно-локального при T=4096.** Локальное окно здесь
   не самописное ядро, а перестановка формы: последовательность режется на блоки
   по W, каждый блок смотрит на себя и на предыдущий. Получается полоса шириной
   до 2W, стоимость O(T·W) вместо O(T²), и всё это обычные матмулы, которые XLA
   сплавляет сам. Ровно то, что делает Gemma-style sliding window, но без
   зависимости от FlashAttention.
"""

from __future__ import annotations

import gc
import json
import os
import time

import jax
import jax.numpy as jnp
import optax
from jax import random

DT = jnp.bfloat16


def rms_norm(x, w):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 ** 2, axis=-1, keepdims=True) + 1e-6)
    return (y * w).astype(DT)


def attend_full(q, k, v, scale):
    T = q.shape[2]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) * scale
    logits = jnp.where(mask, logits, -jnp.inf)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, -1).astype(DT), v)


def attend_local(q, k, v, scale, window):
    """Полосовое внимание: блок смотрит на себя и на предыдущий блок."""
    B, H, T, D = q.shape
    W = window
    n = T // W
    qb = q.reshape(B, H, n, W, D)
    kb = k.reshape(B, H, n, W, D)
    vb = v.reshape(B, H, n, W, D)
    # к каждому блоку приклеиваем предыдущий: получается ключей 2W
    kprev = jnp.concatenate([jnp.zeros_like(kb[:, :, :1]), kb[:, :, :-1]], axis=2)
    vprev = jnp.concatenate([jnp.zeros_like(vb[:, :, :1]), vb[:, :, :-1]], axis=2)
    kk = jnp.concatenate([kprev, kb], axis=3)
    vv = jnp.concatenate([vprev, vb], axis=3)

    logits = jnp.einsum("bhnqd,bhnkd->bhnqk", qb, kk).astype(jnp.float32) * scale
    qpos = jnp.arange(W)[:, None] + W          # позиция запроса в склейке 2W
    kpos = jnp.arange(2 * W)[None, :]
    band = kpos <= qpos
    first = jnp.arange(n)[:, None, None] > 0   # у нулевого блока нет предыдущего
    band = jnp.where(first, band, band & (kpos[None] >= W))
    logits = jnp.where(band[None, None], logits, -jnp.inf)
    o = jnp.einsum("bhnqk,bhnkd->bhnqd", jax.nn.softmax(logits, -1).astype(DT), vv)
    return o.reshape(B, H, T, D)


def moe_ffn(p, x, top_k, cap_factor):
    B, T, d = x.shape
    N = B * T
    E = p["w_in"].shape[0]
    xf = x.reshape(N, d)
    gate = jax.nn.softmax(xf.astype(jnp.float32) @ p["gate"], axis=-1)
    vals, idx = jax.lax.top_k(gate, top_k)
    cap = max(1, int(N * top_k / E * cap_factor))

    onehot = jax.nn.one_hot(idx.reshape(-1), E, dtype=jnp.int32)
    pos = (jnp.cumsum(onehot, axis=0) - onehot)
    pos = (pos * onehot).sum(-1).reshape(N, top_k)
    keep = pos < cap
    slot = jnp.where(keep, idx * cap + pos, E * cap)

    src = jnp.repeat(xf, top_k, axis=0)
    buf = jnp.zeros((E * cap + 1, d), x.dtype).at[slot.reshape(-1)].set(src)
    buf = buf[: E * cap].reshape(E, cap, d)

    g = jnp.einsum("ecd,edh->ech", buf, p["w_gate"].astype(DT))
    u = jnp.einsum("ecd,edh->ech", buf, p["w_in"].astype(DT))
    h = jnp.einsum("ech,ehd->ecd", jax.nn.silu(g) * u, p["w_out"].astype(DT))

    flat = jnp.concatenate([h.reshape(E * cap, d), jnp.zeros((1, d), h.dtype)])
    w = (vals * keep).astype(DT)
    return (flat[slot] * w[..., None]).sum(1).reshape(B, T, d)


def block(p, x, cfg):
    B, T, d = x.shape
    heads, kv = cfg["heads"], cfg["kv"]
    d_head = d // heads
    h = rms_norm(x, p["norm1"])
    q = (h @ p["q"].astype(DT)).reshape(B, T, heads, d_head).transpose(0, 2, 1, 3)
    k = (h @ p["k"].astype(DT)).reshape(B, T, kv, d_head).transpose(0, 2, 1, 3)
    v = (h @ p["v"].astype(DT)).reshape(B, T, kv, d_head).transpose(0, 2, 1, 3)
    rep = heads // kv
    k, v = jnp.repeat(k, rep, 1), jnp.repeat(v, rep, 1)
    scale = d_head ** -0.5
    o = (attend_full(q, k, v, scale) if cfg["window"] == 0
         else attend_local(q, k, v, scale, cfg["window"]))
    x = x + o.transpose(0, 2, 1, 3).reshape(B, T, d) @ p["o"].astype(DT)

    hh = rms_norm(x, p["norm2"])
    if "w_in" in p:
        return x + moe_ffn(p, hh, cfg["top_k"], cfg["cap_factor"])
    g, u = jnp.split(hh @ p["up"].astype(DT), 2, axis=-1)
    return x + (jax.nn.silu(g) * u) @ p["down"].astype(DT)


def init_params(key, cfg):
    d, hidden, E = cfg["d"], cfg["hidden"], cfg["experts"]
    heads, kv = cfg["heads"], cfg["kv"]
    d_head = d // heads
    ks = random.split(key, cfg["layers"] * 8 + 4)
    i = [0]

    def take(shape, scale):
        p = random.normal(ks[i[0]], shape, dtype=jnp.float32) * scale
        i[0] += 1
        return p

    params = {"embed": take((cfg["vocab"], d), d ** -0.5),
              "final_norm": jnp.ones((d,), jnp.float32), "blocks": []}
    for _ in range(cfg["layers"]):
        b = {"norm1": jnp.ones((d,), jnp.float32),
             "q": take((d, d), d ** -0.5),
             "k": take((d, kv * d_head), d ** -0.5),
             "v": take((d, kv * d_head), d ** -0.5),
             "o": take((d, d), d ** -0.5),
             "norm2": jnp.ones((d,), jnp.float32)}
        if E:
            b["gate"] = take((d, E), d ** -0.5)
            b["w_gate"] = take((E, d, hidden), d ** -0.5)
            b["w_in"] = take((E, d, hidden), d ** -0.5)
            b["w_out"] = take((E, hidden, d), hidden ** -0.5)
        else:
            b["up"] = take((d, 2 * hidden), d ** -0.5)
            b["down"] = take((hidden, d), hidden ** -0.5)
        params["blocks"].append(b)
    return params


def loss_fn(params, tokens, targets, cfg):
    x = params["embed"].astype(DT)[tokens]
    for p in params["blocks"]:
        x = block(p, x, cfg)
    logits = (rms_norm(x, params["final_norm"]) @ params["embed"].astype(DT).T).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, -1)
    return -jnp.mean(jnp.take_along_axis(logp, targets[..., None], -1))


def active_params(params, cfg) -> int:
    total = sum(x.size for x in jax.tree.leaves(params))
    if not cfg["experts"]:
        return total
    per_layer = sum(x.size for x in jax.tree.leaves(params["blocks"][0])
                    if x.ndim == 3)
    unused = per_layer * (1 - cfg["top_k"] / cfg["experts"])
    return int(total - unused * cfg["layers"])


def run(name, cfg, opt_name):
    dev = jax.devices()[0]
    key = random.PRNGKey(0)
    B, T = cfg["batch"], cfg["seq"]
    tokens = random.randint(key, (B, T), 0, cfg["vocab"])
    targets = random.randint(random.fold_in(key, 1), (B, T), 0, cfg["vocab"])
    params = init_params(random.fold_in(key, 2), cfg)
    total = sum(x.size for x in jax.tree.leaves(params))
    act = active_params(params, cfg)

    tx = optax.adamw(1e-3) if opt_name == "adamw" else optax.adafactor(1e-3)
    state = tx.init(params)

    @jax.jit
    def step(params, state, tokens, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets, cfg)
        upd, state = tx.update(grads, state, params)
        return optax.apply_updates(params, upd), state, loss

    t0 = time.perf_counter()
    params, state, loss = step(params, state, tokens, targets)
    jax.block_until_ready(loss)
    compile_s = time.perf_counter() - t0

    best = float("inf")
    for _ in range(4):
        t1 = time.perf_counter()
        params, state, loss = step(params, state, tokens, targets)
        jax.block_until_ready(loss)
        best = min(best, time.perf_counter() - t1)

    peak = dev.memory_stats()["peak_bytes_in_use"] / 2**30
    row = {"имя": name, "опт": opt_name, "всего_М": round(total / 1e6, 1),
           "активных_М": round(act / 1e6, 1), "B": B, "T": T,
           "окно": cfg["window"], "экспертов": cfg["experts"],
           "компиляция_с": round(compile_s, 1), "мс": round(best * 1000, 1),
           "ток_с": round(B * T / best), "ТFLOPс": round(6 * act * B * T / 1e9 / (best * 1000), 1),
           "пик_ГБ": round(peak, 2)}
    del params, state, tx
    gc.collect()
    return row


def main() -> None:
    dev = jax.devices()[0]
    lim = dev.memory_stats()["bytes_limit"] / 2**30
    print(f"{dev.device_kind} | jax {jax.__version__} | HBM {lim:.2f} ГБ", flush=True)

    base = dict(vocab=32768, heads=16, kv=4, top_k=2, cap_factor=1.25,
                batch=4, seq=1024, window=0, experts=0)
    plans = []

    # 1. потолок по параметрам: dense, растущий d, два оптимизатора
    for d, layers, hidden in [(1536, 16, 4096), (2048, 20, 5461), (2048, 28, 5461),
                              (2560, 28, 6826), (3072, 28, 8192)]:
        for opt in ("adamw", "adafactor"):
            plans.append((f"dense {d}x{layers}", {**base, "d": d, "layers": layers,
                                                  "hidden": hidden}, opt))

    # 2. MoE: мало широких экспертов против dense той же активной массы
    for E, k, hid in [(8, 2, 4096), (16, 2, 4096), (16, 4, 2048)]:
        plans.append((f"moe {E}x{hid} top{k}",
                      {**base, "d": 2048, "layers": 20, "hidden": hid,
                       "experts": E, "top_k": k}, "adafactor"))

    # 3. длина и окно
    for T, W in [(2048, 0), (2048, 512), (4096, 0), (4096, 512), (4096, 1024)]:
        plans.append((f"T={T} окно={W or 'полное'}",
                      {**base, "d": 2048, "layers": 20, "hidden": 5461,
                       "batch": 2, "seq": T, "window": W}, "adafactor"))

    rows = []
    for name, cfg, opt in plans:
        try:
            row = run(name, cfg, opt)
        except Exception as exc:  # noqa: BLE001
            row = {"имя": name, "опт": opt, "ошибка": f"{type(exc).__name__}: {str(exc)[:70]}"}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with open("/content/arch.json", "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    print("ГОТОВО", flush=True)


if __name__ == "__main__":
    main()
