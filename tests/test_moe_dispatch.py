"""Раскладка токенов по экспертам: сортировка обязана совпадать с префиксной суммой."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from scripts.bench_moe_jax import _slots_cumsum, _slots_sort, moe_ffn

N, TOP_K = 256, 2


def routing(E: int, seed: int = 0):
    gate = random.normal(random.PRNGKey(seed), (N, E))
    return jax.lax.top_k(gate, TOP_K)[1]


@pytest.mark.parametrize("E", [4, 8, 24])
def test_sort_gives_the_same_positions_as_cumsum(E):
    idx = routing(E)
    assert jnp.array_equal(_slots_cumsum(idx, N, TOP_K, E), _slots_sort(idx, N, TOP_K, E))


@pytest.mark.parametrize("E", [4, 8, 24])
def test_positions_are_a_dense_ranking_inside_each_expert(E):
    """У эксперта с n назначениями позиции обязаны быть ровно 0..n-1, без дыр."""
    idx = routing(E, 1)
    pos = _slots_sort(idx, N, TOP_K, E)
    for e in range(E):
        got = jnp.sort(pos.reshape(-1)[idx.reshape(-1) == e])
        assert jnp.array_equal(got, jnp.arange(got.size))


def params(E: int, d: int, h: int, seed: int = 2):
    k = random.split(random.PRNGKey(seed), 4)
    return {"gate": random.normal(k[0], (d, E)) * d ** -0.5,
            "w_gate": random.normal(k[1], (E, d, h)) * d ** -0.5,
            "w_in": random.normal(k[2], (E, d, h)) * d ** -0.5,
            "w_out": random.normal(k[3], (E, h, d)) * h ** -0.5}


@pytest.mark.parametrize("cap_factor", [1.0, 1.25, 4.0])
def test_both_dispatches_give_the_same_output(cap_factor):
    """Включая cap_factor=1.0, где переполнение есть и токены выбрасываются."""
    E, d, h = 8, 32, 16
    x = random.normal(random.PRNGKey(3), (2, N // 2, d), jnp.bfloat16)
    p = params(E, d, h)
    a = moe_ffn(p, x, TOP_K, cap_factor, "cumsum")
    b = moe_ffn(p, x, TOP_K, cap_factor, "sort")
    assert jnp.array_equal(a, b)


def test_ragged_matches_capacity_dispatch_when_nothing_is_dropped():
    """При запасе по ёмкости обе раскладки считают одно и то же, с точностью до порядка сложения."""
    E, d, h = 8, 32, 16
    x = random.normal(random.PRNGKey(3), (2, N // 2, d), jnp.bfloat16)
    p = params(E, d, h)
    ref = moe_ffn(p, x, TOP_K, 8.0, "cumsum").astype(jnp.float32)
    got = moe_ffn(p, x, TOP_K, 8.0, "ragged").astype(jnp.float32)
    assert jnp.abs(ref - got).max() < 2e-2


def test_ragged_drops_nothing_at_capacity_one():
    """Смысл ragged: буфера ёмкости нет, поэтому cap_factor на результат не влияет."""
    E, d, h = 8, 32, 16
    x = random.normal(random.PRNGKey(3), (2, N // 2, d), jnp.bfloat16)
    p = params(E, d, h)
    assert jnp.array_equal(moe_ffn(p, x, TOP_K, 1.0, "ragged"),
                           moe_ffn(p, x, TOP_K, 8.0, "ragged"))


def test_low_capacity_drops_tokens():
    """Проверка, что тест выше не проходит вхолостую: при cap_factor=1.0 выбросы есть."""
    E, d, h = 8, 32, 16
    x = random.normal(random.PRNGKey(3), (2, N // 2, d), jnp.bfloat16)
    p = params(E, d, h)
    tight = moe_ffn(p, x, TOP_K, 1.0, "sort")
    loose = moe_ffn(p, x, TOP_K, 4.0, "sort")
    assert jnp.abs(tight.astype(jnp.float32) - loose.astype(jnp.float32)).max() > 0


@pytest.mark.parametrize("cap_factor", [1.0, 1.25, 4.0])
def test_gather_dispatch_equals_scatter_dispatch(cap_factor):
    """`gather` меняет только способ набить буфер, результат обязан совпасть до бита."""
    E, d, h = 8, 32, 16
    x = random.normal(random.PRNGKey(3), (2, N // 2, d), jnp.bfloat16)
    p = params(E, d, h)
    assert jnp.array_equal(moe_ffn(p, x, TOP_K, cap_factor, "cumsum"),
                           moe_ffn(p, x, TOP_K, cap_factor, "gather"))


def mesh2():
    import numpy as np
    from jax.sharding import Mesh
    n = jax.device_count()
    return Mesh(np.array(jax.devices()).reshape(n), ("x",)), n


def test_expert_parallel_matches_the_global_dispatch_at_ample_capacity():
    """`ep` считает то же самое — но ёмкость у него локальная, поэтому сверять можно
    только с запасом, при котором никто ничего не выбрасывает.

    С маленьким `cap_factor` расхождение законно: корзина на чип в P раз мельче
    глобальной, разброс заполнения относительно больше, и выбрасывается больше.
    """
    mesh, n = mesh2()
    E, d, h = 8 * n, 32, 16
    B = 2 * n
    x = random.normal(random.PRNGKey(3), (B, 16, d), jnp.bfloat16)
    p = params(E, d, h)
    ref = moe_ffn(p, x, TOP_K, 64.0, "cumsum").astype(jnp.float32)
    got = jax.jit(lambda pp, xx: moe_ffn(pp, xx, TOP_K, 64.0, "ep", mesh))(
        p, x).astype(jnp.float32)
    assert jnp.abs(ref - got).max() < 2e-2


def test_expert_parallel_gradient_flows_to_every_expert():
    """`all_to_all` транспонируется в `all_to_all`; без этого градиент до чужих
    экспертов не доходит и обучается только своя доля."""
    mesh, n = mesh2()
    E, d, h = 8 * n, 32, 16
    B = 2 * n
    x = random.normal(random.PRNGKey(4), (B, 16, d), jnp.bfloat16)
    p = params(E, d, h)
    g = jax.jit(jax.grad(lambda pp: moe_ffn(pp, x, TOP_K, 64.0, "ep", mesh)
                         .astype(jnp.float32).sum()))(p)
    per_expert = jnp.abs(g["w_in"].astype(jnp.float32)).sum(axis=(1, 2))
    assert jnp.isfinite(per_expert).all()
    assert (per_expert > 0).all(), per_expert


@pytest.mark.parametrize("cap_factor", [1.0, 4.0])
def test_expert_parallel_gather_equals_scatter(cap_factor):
    """`ep-gather` меняет только способ набить буфер до `all_to_all`; и раскладка, и
    сборка обратно обязаны дать тот же результат до бита."""
    mesh, n = mesh2()
    E, d, h = 8 * n, 32, 16
    B = 2 * n
    x = random.normal(random.PRNGKey(5), (B, 16, d), jnp.bfloat16)
    p = params(E, d, h)
    fn = lambda disp: jax.jit(lambda pp, xx: moe_ffn(pp, xx, TOP_K, cap_factor, disp, mesh))(p, x)
    assert jnp.array_equal(fn("ep"), fn("ep-gather"))


@pytest.mark.parametrize("spec", ["ep:2", "ep:4", "ep-gather:2"])
def test_chunked_all_to_all_equals_one_exchange(spec):
    """Резать буфер по ёмкости на куски — это только про планировщик: маршрутизация,
    ёмкость и порядок сложения те же, значит и результат обязан быть тем же."""
    mesh, n = mesh2()
    E, d, h = 8 * n, 32, 16
    B = 2 * n
    x = random.normal(random.PRNGKey(6), (B, 16, d), jnp.bfloat16)
    p = params(E, d, h)
    one = "ep-gather" if spec.startswith("ep-gather") else "ep"
    fn = lambda disp: jax.jit(lambda pp, xx: moe_ffn(pp, xx, TOP_K, 4.0, disp, mesh))(p, x)
    assert jnp.array_equal(fn(one), fn(spec))
