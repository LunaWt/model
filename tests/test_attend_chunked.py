"""Блочное внимание должно совпадать с одноимённым сплошным."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from scripts.bench_dense_jax import attend_chunked
from scripts.bench_moe_jax import attend_chunked as attend_moe

B, T, H, KV, D = 2, 64, 4, 2, 16
SCALE = D ** -0.5


def qkv(seed: int = 0):
    k1, k2, k3 = random.split(random.PRNGKey(seed), 3)
    return (random.normal(k1, (B, T, H, D)),
            random.normal(k2, (B, T, KV, D)),
            random.normal(k3, (B, T, KV, D)))


@pytest.mark.parametrize("block", [8, 16, 32, 64])
def test_matches_full_attention(block):
    q, k, v = qkv()
    ref = jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=SCALE)
    got = attend_chunked(q, k, v, SCALE, block)
    assert jnp.abs(ref - got).max() < 2e-5


def test_is_causal():
    """Правка последнего токена не должна менять выход в первой половине."""
    q, k, v = qkv()
    got = attend_chunked(q, k, v, SCALE, 16)
    k2 = k.at[:, -1].add(100.0)
    v2 = v.at[:, -1].add(100.0)
    moved = attend_chunked(q, k2, v2, SCALE, 16)
    assert jnp.abs(got[:, : T - 1] - moved[:, : T - 1]).max() < 1e-6
    assert jnp.abs(got[:, -1] - moved[:, -1]).max() > 1.0


def test_gradient_matches_full_attention():
    q, k, v = qkv(1)

    def loss(fn):
        return lambda q, k, v: jnp.sum(fn(q, k, v) ** 2)

    full = jax.grad(loss(lambda q, k, v: jax.nn.dot_product_attention(
        q, k, v, is_causal=True, scale=SCALE)), argnums=(0, 1, 2))(q, k, v)
    chunk = jax.grad(loss(lambda q, k, v: attend_chunked(q, k, v, SCALE, 16)),
                     argnums=(0, 1, 2))(q, k, v)
    for a, b in zip(full, chunk):
        assert jnp.abs(a - b).max() < 1e-4


@pytest.mark.parametrize("window", [0, 32])
def test_bf16_scores_match_fp32_scores(window):
    """Очки перестали выезжать в HBM во float32 (12.3% шага в профиле `moe-prof`).
    Softmax по-прежнему считается в fp32, поэтому расхождение обязано быть на уровне
    округления bf16 у выхода, а не на уровне другой арифметики."""
    q, k, v = (x.astype(jnp.bfloat16) for x in qkv(2))
    hi = attend_moe(q, k, v, SCALE, 16, window, f32_scores=True).astype(jnp.float32)
    lo = attend_moe(q, k, v, SCALE, 16, window, f32_scores=False).astype(jnp.float32)
    assert jnp.abs(hi - lo).max() < 2e-2


def test_moe_copy_matches_full_attention():
    """Масштаб теперь применяется к запросам до einsum: одно лишнее округление bf16,
    но в fp32 это обязано совпасть с эталоном как раньше."""
    q, k, v = qkv(3)
    ref = jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=SCALE)
    assert jnp.abs(ref - attend_moe(q, k, v, SCALE, 16)).max() < 2e-5


def test_the_two_copies_have_not_drifted():
    """Kaggle грузит ровно один файл, соседние в кернел не приедут (`kaggle_run.build`),
    поэтому дублирование здесь вынужденное. Что не вынужденное — расхождение: плотный
    прогон служит контролем для MoE, и внимание в них обязано быть одним и тем же."""
    import inspect

    from scripts import bench_dense_jax, bench_moe_jax
    assert (inspect.getsource(bench_dense_jax.attend_chunked)
            == inspect.getsource(bench_moe_jax.attend_chunked))
