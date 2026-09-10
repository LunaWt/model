"""Muon: Ньютон–Шульц обязан ортогонализовать, а маска — не течь.

Коэффициенты (3.4445, −4.7750, 2.0315) сознательно не сходятся к единице: они
загоняют все сингулярные значения в примерно [0.68, 1.13] за пять шагов и там
держат. Тест проверяет именно это, а не близость к точной ортогонализации.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from scripts.bench_moe_jax import (adafactor_init, adafactor_update, adamw_init,
                                   adamw_update, muon_init, muon_mask, muon_update,
                                   newton_schulz)


@pytest.mark.parametrize("shape", [(64, 64), (64, 256), (256, 64)])
def test_singular_values_land_in_the_expected_band(shape):
    x = random.normal(random.PRNGKey(0), shape)
    s = np.linalg.svd(np.asarray(newton_schulz(x, 5)), compute_uv=False)
    assert 0.6 < s.min() and s.max() < 1.2


def test_leading_axes_are_independent():
    """Слои и эксперты идут одним вызовом; каждая матрица обязана считаться сама по себе."""
    x = random.normal(random.PRNGKey(1), (3, 2, 32, 48))
    got = newton_schulz(x, 5)
    for i in range(3):
        for j in range(2):
            assert jnp.allclose(got[i, j], newton_schulz(x[i, j], 5), atol=1e-5)


def tiny(seed: int = 0):
    k = random.split(random.PRNGKey(seed), 5)
    return {"embed": random.normal(k[0], (16, 8)),
            "final_norm": jnp.ones(8),
            "blocks": {"q": random.normal(k[1], (2, 8, 8)),
                       "up": random.normal(k[2], (2, 8, 16)),
                       "down": random.normal(k[3], (2, 16, 8)),
                       "norm1": jnp.ones((2, 8)),
                       "w_in": random.normal(k[4], (2, 4, 8, 6))}}


def test_mask_picks_matrices_and_leaves_the_rest():
    m = muon_mask(tiny(), experts=False)
    assert m["blocks"]["q"] and m["blocks"]["up"] and m["blocks"]["down"]
    assert not m["blocks"]["norm1"] and not m["embed"] and not m["blocks"]["w_in"]
    assert muon_mask(tiny(), experts=True)["blocks"]["w_in"]


def test_unmasked_leaves_match_plain_adafactor():
    """Проверка, что вторая ветка — действительно Adafactor, а не что-то похожее."""
    p, g = tiny(), jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, tiny())
    mask = muon_mask(p, experts=False)
    new_m, _ = muon_update(p, g, muon_init(p, mask), mask)
    new_a, _ = adafactor_update(p, g, adafactor_init(p))
    for path in (("embed",), ("blocks", "norm1"), ("blocks", "w_in")):
        a, b = new_m, new_a
        for key in path:
            a, b = a[key], b[key]
        assert jnp.allclose(a, b, atol=1e-6)


def test_masked_leaves_do_not_match_adafactor():
    """Страховка от теста вхолостую: на матрицах ветки обязаны разойтись."""
    p, g = tiny(1), jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, tiny(1))
    mask = muon_mask(p, experts=False)
    new_m, _ = muon_update(p, g, muon_init(p, mask), mask)
    new_a, _ = adafactor_update(p, g, adafactor_init(p))
    assert jnp.abs(new_m["blocks"]["q"] - new_a["blocks"]["q"]).max() > 1e-3


def test_state_keeps_shape_across_two_steps():
    p, g = tiny(2), jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, tiny(2))
    mask = muon_mask(p, experts=True)
    st = muon_init(p, mask)
    for _ in range(2):
        p, st = muon_update(p, g, st, mask)
    assert st["мом"]["blocks"]["w_in"].shape == (2, 4, 8, 6)
    assert st["мом"]["embed"].shape == ()
    assert all(jnp.isfinite(x).all() for x in jax.tree.leaves(p))


@pytest.mark.parametrize("opt", ["adafactor", "adamw", "muon"])
def test_update_preserves_parameter_dtype(opt):
    """bf16-параметр обязан остаться bf16 после шага.

    Состояние Adafactor живёт в float32, и без явного приведения шаг возвращает
    fp32-параметр. Дальше рушится всё сразу: у `jit` меняется сигнатура и он
    компилирует шаг второй раз поверх живого первого, память параметров удваивается,
    а `memory_analysis`, снятый по первой компиляции, меряет уже не ту программу.
    """
    p = jax.tree.map(lambda x: x.astype(jnp.bfloat16), tiny(3))
    g = jax.tree.map(lambda x: jnp.full(x.shape, 0.1, jnp.bfloat16), p)
    if opt == "muon":
        mask = muon_mask(p, experts=True)
        step = lambda pp, st: muon_update(pp, g, st, mask)  # noqa: E731
        st = muon_init(p, mask)
    elif opt == "adamw":
        step = lambda pp, st: adamw_update(pp, g, st)  # noqa: E731
        st = adamw_init(p)
    else:
        step = lambda pp, st: adafactor_update(pp, g, st)  # noqa: E731
        st = adafactor_init(p)

    for _ in range(2):
        p, st = step(p, st)
    bad = [x.dtype for x in jax.tree.leaves(p) if x.dtype != jnp.bfloat16]
    assert not bad, bad
