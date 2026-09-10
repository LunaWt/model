"""Скользящее окно: маска обязана совпасть с прямым вычислением, а не примерно.

Проверять надо именно на границах блоков — там `dynamic_slice` может съехать, и
ошибка будет выглядеть как «чуть хуже сходится», а не как падение.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from scripts.bench_moe_jax import attend_chunked

B, T, H, KV, D = 2, 64, 4, 2, 8


def qkv(seed: int = 0):
    k = random.split(random.PRNGKey(seed), 3)
    return (random.normal(k[0], (B, T, H, D), jnp.float32),
            random.normal(k[1], (B, T, KV, D), jnp.float32),
            random.normal(k[2], (B, T, KV, D), jnp.float32))


def reference(q, k, v, scale, window):
    k = jnp.repeat(k, H // KV, axis=2)
    v = jnp.repeat(v, H // KV, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale
    pos = jnp.arange(T)
    keep = pos[:, None] >= pos[None, :]
    if window:
        keep &= pos[:, None] - pos[None, :] < window
    a = jax.nn.softmax(jnp.where(keep, logits, -jnp.inf), axis=-1)
    return jnp.einsum("bhqk,bkhd->bqhd", a, v)


@pytest.mark.parametrize("window", [0, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("block_len", [8, 16])
def test_window_matches_direct_mask(window, block_len):
    q, k, v = qkv()
    got = attend_chunked(q, k, v, D ** -0.5, block_len, window)
    ref = reference(q, k, v, D ** -0.5, window)
    assert jnp.abs(got - ref).max() < 1e-4


def test_window_actually_cuts_something():
    """Страховка от теста вхолостую: окно должно менять результат."""
    q, k, v = qkv(1)
    full = attend_chunked(q, k, v, D ** -0.5, 8, 0)
    win = attend_chunked(q, k, v, D ** -0.5, 8, 8)
    assert jnp.abs(full - win).max() > 1e-2


def test_window_wider_than_context_equals_full():
    q, k, v = qkv(2)
    assert jnp.allclose(attend_chunked(q, k, v, D ** -0.5, 8, 0),
                        attend_chunked(q, k, v, D ** -0.5, 8, 4 * T), atol=1e-5)


def test_gradient_flows_through_the_window():
    """Окно режется `dynamic_slice`; без правильного транспонирования градиент по k нулевой."""
    q, k, v = qkv(3)
    g = jax.grad(lambda kk: attend_chunked(q, kk, v, D ** -0.5, 8, 16).sum())(k)
    assert jnp.isfinite(g).all() and jnp.abs(g).max() > 0


SPLASH_T, SPLASH_H, SPLASH_D = 256, 2, 128


@pytest.mark.parametrize("window", [0, 128])
def test_splash_kernel_matches_the_same_mask(window):
    """Pallas-ядро гоняется в режиме `interpret` на CPU: проверяется не скорость, а
    что маска, масштаб и раскладка голов заведены правильно."""
    from scripts.bench_moe_jax import attend_splash

    t, h, d = SPLASH_T, SPLASH_H, SPLASH_D
    k = random.split(random.PRNGKey(7), 3)
    q, kk, vv = (random.normal(k[i], (1, t, h, d), jnp.float32) for i in range(3))
    got = attend_splash(q, kk, vv, d ** -0.5, 128, window, interpret=True)
    ref = attend_chunked(q, kk, vv, d ** -0.5, 128, window)
    assert jnp.abs(got - ref).max() < 1e-4


@pytest.mark.parametrize("window", [0, 128])
def test_splash_gradient_matches_too(window):
    """Прямой проход ничего не говорит про обучение: у ядра свой vjp, и он тоже под remat."""
    from scripts.bench_moe_jax import attend_splash

    t, h, d = SPLASH_T, SPLASH_H, SPLASH_D
    k = random.split(random.PRNGKey(9), 3)
    q, kk, vv = (random.normal(k[i], (1, t, h, d), jnp.float32) for i in range(3))
    arg = (q, kk, vv)
    got = jax.grad(jax.checkpoint(
        lambda a, b, c: attend_splash(a, b, c, d ** -0.5, 128, window, True).sum()),
        argnums=(0, 1, 2))(*arg)
    ref = jax.grad(lambda a, b, c: attend_chunked(a, b, c, d ** -0.5, 128, window).sum(),
                   argnums=(0, 1, 2))(*arg)
    assert all(jnp.abs(x - y).max() < 1e-4 for x, y in zip(got, ref))


def test_splash_runs_under_sharded_jit():
    """XLA:SPMD не умеет резать ядро Mosaic сам — вызов обязан быть в `shard_map`.

    Без него `jit` с мешем падает «Mosaic kernels cannot be automatically partitioned»,
    причём даже когда резать нечего. Тест ловит именно это: батч разложен по чипам.
    """
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from scripts.bench_moe_jax import attend_splash

    n = jax.device_count()
    t, h, d = SPLASH_T, SPLASH_H, SPLASH_D
    mesh = Mesh(np.array(jax.devices()).reshape(n), ("x",))
    k = random.split(random.PRNGKey(11), 3)
    arg = [random.normal(k[i], (n, t, h, d), jnp.float32) for i in range(3)]
    arg = [jax.device_put(x, NamedSharding(mesh, P("x", None, None, None))) for x in arg]

    got = jax.jit(lambda *a: attend_splash(*a, d ** -0.5, 128, 128, True, mesh))(*arg)
    ref = attend_chunked(*arg, d ** -0.5, 128, 128)
    assert jnp.abs(got - ref).max() < 1e-4
