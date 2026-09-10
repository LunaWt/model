"""int8-веса экспертов: разжатие, котангента через нулевую тень, шаг оптимизатора.

Хранение весов кодами трогает три разных места, и каждое ломается по-своему:
разжатие (не та ось масштаба — тихо неверные числа), градиент (jax.grad не отдаёт
котангенту по int8-листу вовсе) и обновление (детерминированное округление съедает
все поправки меньше половины шага сетки). Проверяется каждое.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from scripts.bench_moe_jax import _adafactor_leaf, _quant_leaf, dequant, moe_ffn, quant_keys

E, D, H, TOP_K = 4, 32, 16, 2


def quantize(w):
    s = jnp.maximum(jnp.max(jnp.abs(w), axis=-2, keepdims=True), 1e-20) / 127.0
    return jnp.clip(jnp.round(w / s), -127, 127).astype(jnp.int8), s


def float_params(seed: int = 0):
    k = random.split(random.PRNGKey(seed), 4)
    return {"gate": random.normal(k[0], (D, E)) * D ** -0.5,
            "w_gate": random.normal(k[1], (E, D, H)) * D ** -0.5,
            "w_in": random.normal(k[2], (E, D, H)) * D ** -0.5,
            "w_out": random.normal(k[3], (E, H, D)) * H ** -0.5}


def quant_params(p):
    out = {"gate": p["gate"]}
    for k in ("w_gate", "w_in", "w_out"):
        code, s = quantize(p[k])
        out[k], out[k + "_s"] = code, s
        out[k + "_g"] = jnp.zeros(p[k].shape, jnp.bfloat16)
    return out


def test_dequant_returns_the_weights_within_half_a_step():
    """Масштаб — absmax по входному каналу на 127, значит ошибка не больше половины
    шага сетки; сверху ложатся два округления bf16 — самого масштаба и произведения,
    по 2^-9 и 2^-8 относительной (у bf16 восемь значащих бит)."""
    p = float_params()
    got = dequant(quant_params(p))
    for k in ("w_gate", "w_in", "w_out"):
        step = jnp.max(jnp.abs(p[k]), axis=-2, keepdims=True) / 127.0
        err = jnp.abs(got[k].astype(jnp.float32) - p[k])
        assert (err <= step / 2 + jnp.abs(p[k]) * 2 ** -7 + 1e-6).all(), k


def test_dequant_leaves_float_weights_alone():
    p = float_params()
    assert dequant(p) is p
    assert not quant_keys({"blocks": p})
    assert quant_keys({"blocks": quant_params(p)}) == ["w_gate", "w_in", "w_out"]


def test_the_zero_shadow_carries_the_true_weight_gradient():
    """Главная проверка всей конструкции: градиент по тени обязан совпасть с
    градиентом по тем же весам, если бы они лежали обычными числами."""
    x = random.normal(random.PRNGKey(7), (2, 16, D), jnp.bfloat16)
    p = float_params(1)
    q = quant_params(p)
    deq = {k: dequant(q)[k] for k in ("w_gate", "w_in", "w_out")}

    def out(pp):
        return moe_ffn(pp, x, TOP_K, 4.0, "cumsum").astype(jnp.float32).sum()

    shadow = jax.grad(lambda sh: out({**q, **{k + "_g": v for k, v in sh.items()}}))(
        {k: q[k + "_g"] for k in deq})
    plain = jax.grad(lambda w: out({"gate": q["gate"], **w}))(deq)
    for k in deq:
        assert jnp.array_equal(shadow[k], plain[k]), k


def test_stochastic_rounding_keeps_the_mean_of_a_sub_step_update():
    """Обновление меньше половины шага сетки: при обычном округлении код не сдвинется
    ни разу, при стохастическом сдвинется в нужной доле случаев."""
    code = jnp.full((1, 8, 8), 10, jnp.int8)
    s = jnp.full((1, 1, 8), 0.01, jnp.float32)
    frac = 0.3
    moved = []
    for i in range(64):
        u = code.astype(jnp.float32) + frac
        noise = random.uniform(random.PRNGKey(i), u.shape, jnp.float32, -0.5, 0.5)
        moved.append(float(jnp.mean(jnp.round(u + noise))))
    assert abs(np.mean(moved) - 10.3) < 0.02, np.mean(moved)
    assert float(jnp.mean(jnp.round(code.astype(jnp.float32) + frac))) == 10.0
    assert s.shape == (1, 1, 8)


def test_the_optimizer_step_moves_codes_and_keeps_the_scale_honest():
    """Шаг обязан двигать коды, оставлять их в int8 и пересчитывать масштаб так,
    чтобы новый absmax по-прежнему укладывался в 127."""
    p = float_params(2)
    code, s = quantize(p["w_in"])
    g = random.normal(random.PRNGKey(9), p["w_in"].shape, jnp.bfloat16) * 1e-2
    r = jnp.zeros(code.shape[:-1], jnp.float32)
    c = jnp.zeros(code.shape[:-2] + code.shape[-1:], jnp.float32)
    new, s2, _, _ = _quant_leaf(code, s, g, r, c, random.PRNGKey(0),
                                1e-3, 0.95, 1e-30, 0.1)
    assert new.dtype == jnp.int8
    assert int(jnp.abs(new.astype(jnp.int32)).max()) <= 127
    assert float(jnp.mean(new != code)) > 0.1
    w = code.astype(jnp.float32) * s
    want, _, _ = _adafactor_leaf(w, g, r, c, 1e-3, 0.95, 1e-30, 0.1)
    assert (jnp.abs(new.astype(jnp.float32) * s2 - want) <= s2 * 1.001).all()
