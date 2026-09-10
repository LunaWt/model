"""Тот же плотный трансформер на JAX — чтобы померить TPU в тех же токенах в секунду.

    colab --auth=adc exec -s tpu -f scripts/bench_dense_jax.py

Слепок `scripts/bench_dense_ref.py`: те же формы, та же длина, тот же батч,
RMSNorm + GQA + SwiGLU и связанные эмбеддинги. Отличий два, и оба неустранимы:

  * на TPU нет «неоткомпилированного» режима — XLA компилирует граф целиком, так
    что это JAX+XLA против PyTorch eager. Сравнение честное практически (так их и
    запускают), но не по-архитектурно чистое;
  * счёт в bf16, потому что у TPU это родной тип, а fp16 там смысла не имеет.

Считаем FLOP так же: 6 · параметры · токены, чтобы столбец ТFLOP/с лежал в одной
шкале с замерами на T4 и на GTX 1660 Ti.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
from jax import random

DT = jnp.bfloat16


def set_dtype(name: str) -> None:
    global DT
    DT = getattr(jnp, name)


def init_params(key, vocab, d, layers, heads, kv, hidden):
    keys = random.split(key, layers * 6 + 2)
    i = 0

    def take(shape, scale):
        nonlocal i
        p = random.normal(keys[i], shape, dtype=jnp.float32) * scale
        i += 1
        return p

    d_head = d // heads
    params = {"embed": take((vocab, d), d ** -0.5), "final_norm": jnp.ones((d,), jnp.float32),
              "blocks": []}
    for _ in range(layers):
        params["blocks"].append({
            "norm1": jnp.ones((d,), jnp.float32),
            "q": take((d, d), d ** -0.5),
            "k": take((d, kv * d_head), d ** -0.5),
            "v": take((d, kv * d_head), d ** -0.5),
            "o": take((d, d), d ** -0.5),
            "norm2": jnp.ones((d,), jnp.float32),
            "up": take((d, 2 * hidden), d ** -0.5),
            "down": take((hidden, d), hidden ** -0.5),
        })
    return params


def rms_norm(x, w):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 ** 2, axis=-1, keepdims=True) + 1e-6)
    return (y * w).astype(DT)


ATTN_BLOCK = 0


def attend_chunked(q, k, v, scale, block_len, window=0, f32_scores=False):
    """Внимание блоками запросов; `window` > 0 — скользящее окно вместо полного.

    Окно снимает квадратичность и по арифметике, и по памяти: блоку запросов длиной
    `block_len` нужны только ключи из [i·block_len − window, (i+1)·block_len), то есть
    ровно `window + block_len` штук. Длина среза статическая, начало — `dynamic_slice`,
    так что форма графа от T не зависит и матрица логитов не растёт вместе с длиной.

    Очки остаются в типе счёта, а во float32 переводятся уже внутри softmax. Разница
    не в арифметике — она в обоих случаях fp32, — а в том, что выезжает в HBM: профиль
    `moe-prof` нашёл здесь `f32[2,4,4,256,2048]`, то есть 67 МБ на блок запросов на
    слой и 12.9 ГБ за проход, 12.3% шага. Масштаб теперь применяется к запросам до
    einsum, чтобы у выхода einsum был единственный потребитель и XLA мог слить его с
    softmax. `--attn-f32-scores` возвращает старое поведение для сравнения.
    """
    B, T, H, D = q.shape
    G = k.shape[2]
    rep = H // G
    nb = T // block_len
    span = T if not window or window + block_len >= T else window + block_len
    q = (q * scale).astype(q.dtype)
    qb = q.reshape(B, nb, block_len, G, rep, D).transpose(1, 0, 2, 3, 4, 5)

    def stepf(_, xs):
        qi, i = xs
        qpos = i * block_len + jnp.arange(block_len)
        if span == T:
            ki, vi, start = k, v, 0
        else:
            start = jnp.clip((i + 1) * block_len - span, 0, T - span)
            ki = jax.lax.dynamic_slice_in_dim(k, start, span, 1)
            vi = jax.lax.dynamic_slice_in_dim(v, start, span, 1)
        kpos = start + jnp.arange(span)
        logits = jnp.einsum("bqgrd,bkgd->bgrqk", qi, ki)
        if f32_scores:
            logits = logits.astype(jnp.float32)
        keep = qpos[:, None] >= kpos[None, :]
        if window:
            keep &= qpos[:, None] - kpos[None, :] < window
        masked = jnp.where(keep, logits.astype(jnp.float32), -jnp.inf)
        a = jax.nn.softmax(masked, axis=-1).astype(q.dtype)
        return None, jnp.einsum("bgrqk,bkgd->bqgrd", a, vi)

    _, out = jax.lax.scan(jax.checkpoint(stepf), None, (qb, jnp.arange(nb)))
    return out.transpose(1, 0, 2, 3, 4, 5).reshape(B, T, H, D)


def block(p, x, heads, kv):
    B, T, d = x.shape
    d_head = d // heads
    h = rms_norm(x, p["norm1"])
    q = (h @ p["q"].astype(DT)).reshape(B, T, heads, d_head)
    k = (h @ p["k"].astype(DT)).reshape(B, T, kv, d_head)
    v = (h @ p["v"].astype(DT)).reshape(B, T, kv, d_head)
    o = (attend_chunked(q, k, v, d_head ** -0.5, ATTN_BLOCK) if ATTN_BLOCK else
         jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=d_head ** -0.5))
    x = x + o.reshape(B, T, d) @ p["o"].astype(DT)

    g, u = jnp.split(rms_norm(x, p["norm2"]) @ p["up"].astype(DT), 2, axis=-1)
    return x + (jax.nn.silu(g) * u) @ p["down"].astype(DT)


def loss_fn(params, tokens, targets, heads, kv, vocab):
    x = params["embed"].astype(DT)[tokens]
    for p in params["blocks"]:
        x = block(p, x, heads, kv)
    h = rms_norm(x, params["final_norm"])
    logits = (h @ params["embed"].astype(DT).T).astype(jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)
    return -jnp.mean(picked)


def sharded_loss(mesh, heads, kv, vocab):
    """Батч по чипам, параметры продублированы, потери усредняются через pmean.

    Через `NamedSharding` на входах это не собирается: в JAX 0.10 выборка строк
    эмбеддинга продублированной матрицей по разложенным индексам требует явного
    `out_sharding` (ShardingTypeError). Внутри `shard_map` каждый чип видит свой
    локальный кусок как обычный массив, и вопрос не возникает.
    """
    from jax.sharding import PartitionSpec as P

    smap = getattr(jax, "shard_map", None)
    if smap is None:
        from jax.experimental.shard_map import shard_map as smap

    def local(params, tokens, targets):
        return jax.lax.pmean(loss_fn(params, tokens, targets, heads, kv, vocab), "dp")

    kw = dict(mesh=mesh, in_specs=(P(), P("dp", None), P("dp", None)), out_specs=P())
    try:
        return smap(local, **kw, check_rep=False)
    except TypeError:
        return smap(local, **kw)


def _drop_static(fn):
    return lambda params, tokens, targets, *_: fn(params, tokens, targets)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=672)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-block", type=int, default=0,
                   help="0 — внимание одним куском, иначе длина блока запросов")
    p.add_argument("--shard", action="store_true",
                   help="разложить батч по всем чипам, параметры продублировать")
    p.add_argument("--shapes", nargs="*",
                   default=["512:16:8:2:1365", "768:10:12:4:2048", "1024:8:16:4:2730",
                            "1536:20:16:4:4096", "2048:24:16:4:5461"])
    return p.parse_args()


def main() -> None:
    global ATTN_BLOCK
    a = parse_args()
    set_dtype(a.dtype)
    ATTN_BLOCK = a.attn_block
    if ATTN_BLOCK and a.seq_len % ATTN_BLOCK:
        raise SystemExit(f"T={a.seq_len} не делится на блок {ATTN_BLOCK}")
    dev = jax.devices()[0]
    print(f"{dev.device_kind} | jax {jax.__version__} | устройств {jax.device_count()} | "
          f"B={a.batch_size} T={a.seq_len} {a.dtype} | блок внимания {ATTN_BLOCK or 'нет'}")
    print(f"\n{'d:слоёв:голов':<18} {'всего М':>8} {'компиляция с':>13} {'мс':>8} "
          f"{'ток/с':>9} {'ТFLOP/с':>8}")

    key = random.PRNGKey(0)
    tokens = random.randint(key, (a.batch_size, a.seq_len), 0, a.vocab)
    targets = random.randint(random.fold_in(key, 1), (a.batch_size, a.seq_len), 0, a.vocab)

    replicated = batched = mesh = None
    if a.shard:
        from jax.sharding import NamedSharding, PartitionSpec as P

        n_dev = jax.device_count()
        if a.batch_size % n_dev:
            raise SystemExit(f"батч {a.batch_size} не делится на {n_dev} чипов")
        mesh = jax.make_mesh((n_dev,), ("dp",))
        replicated = NamedSharding(mesh, P())
        batched = NamedSharding(mesh, P("dp", None))
        tokens = jax.device_put(tokens, batched)
        targets = jax.device_put(targets, batched)

    for shape in a.shapes:
        d, layers, heads, kv, hidden = (int(v) for v in shape.split(":"))
        try:
            params = init_params(random.fold_in(key, 2), a.vocab, d, layers, heads, kv, hidden)
            if a.shard:
                params = jax.device_put(params, replicated)
            n = sum(x.size for x in jax.tree.leaves(params))
            grad_fn = (jax.jit(jax.value_and_grad(sharded_loss(mesh, heads, kv, a.vocab)))
                       if a.shard else
                       jax.jit(jax.value_and_grad(loss_fn), static_argnums=(3, 4, 5)))
            if a.shard:
                grad_fn = _drop_static(grad_fn)
            t0 = time.perf_counter()
            out = grad_fn(params, tokens, targets, heads, kv, a.vocab)
            jax.block_until_ready(out)
            compile_s = time.perf_counter() - t0

            best = float("inf")
            for _ in range(a.reps):
                t0 = time.perf_counter()
                jax.block_until_ready(grad_fn(params, tokens, targets, heads, kv, a.vocab))
                best = min(best, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            print(f"{shape:<18} {'':>8} {type(exc).__name__}: {str(exc)[:60]}")
            continue

        toks = a.batch_size * a.seq_len
        gflop = 6 * n * toks / 1e9
        print(f"{shape:<18} {n / 1e6:>8.1f} {compile_s:>13.1f} {best * 1000:>8.1f} "
              f"{toks / best:>9.0f} {gflop / (best * 1000):>8.2f}")
        del params
    print("\nпримечание: JAX+XLA всегда компилирует; замер на T4 сделан в eager PyTorch")


if __name__ == "__main__":
    main()
