"""Голова кросс-энтропии: три реализации обязаны давать один лосс и один градиент.

`scan` — прежняя версия (fp32-логиты, all-gather внутри цикла), `local` — новая
арифметика без шардинга, `shard` — она же по локальным строкам с одним psum. Правка
меняет и значения, и то, что выезжает в HBM, поэтому сверяется и то и другое: сам
лосс, градиент по скрытому состоянию и градиент по эмбеддингу.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.sharding import Mesh

from scripts.bench_moe_jax import _chunked_ce_legacy, chunked_ce

N, D, V, CHUNK = 64, 16, 48, 16


def data(seed: int = 0, dt=jnp.float32):
    k1, k2, k3 = random.split(random.PRNGKey(seed), 3)
    return (random.normal(k1, (N, D), dt) * 0.5,
            random.normal(k2, (V, D), dt) * D ** -0.5,
            random.randint(k3, (N,), 0, V))


def mesh1():
    n = jax.device_count()
    return Mesh(np.array(jax.devices()).reshape(n), ("x",)), n


def reference(h, embed, targets):
    logits = (h.astype(jnp.float32) @ embed.astype(jnp.float32).T)
    lse = jax.nn.logsumexp(logits, axis=-1)
    return jnp.mean(lse - jnp.take_along_axis(logits, targets[:, None], axis=-1)[:, 0])


def test_local_matches_the_plain_formula():
    h, embed, tgt = data()
    got = chunked_ce(h, embed, tgt, CHUNK, shard=False)
    assert abs(float(got) - float(reference(h, embed, tgt))) < 1e-5


def test_local_matches_the_legacy_scan():
    h, embed, tgt = data(1)
    old = _chunked_ce_legacy(h, embed, tgt, CHUNK)
    new = chunked_ce(h, embed, tgt, CHUNK, shard=False)
    assert abs(float(old) - float(new)) < 1e-5


def test_sharded_matches_the_legacy_scan():
    """Каждый чип считает свои строки, psum складывает; делитель — все строки."""
    mesh, _ = mesh1()
    h, embed, tgt = data(2)
    old = _chunked_ce_legacy(h, embed, tgt, CHUNK)
    new = jax.jit(lambda a, b, t: chunked_ce(a, b, t, CHUNK, mesh=mesh))(h, embed, tgt)
    assert abs(float(old) - float(new)) < 1e-5


@pytest.mark.parametrize("shard", [False, True])
def test_gradients_match_the_legacy_scan(shard):
    mesh, _ = mesh1()
    h, embed, tgt = data(3)
    gold = jax.grad(lambda a, b: _chunked_ce_legacy(a, b, tgt, CHUNK), argnums=(0, 1))
    new = jax.jit(jax.grad(
        lambda a, b: chunked_ce(a, b, tgt, CHUNK, mesh=mesh, shard=shard), argnums=(0, 1)))
    for a, b in zip(gold(h, embed), new(h, embed)):
        assert jnp.abs(a - b).max() < 1e-5, jnp.abs(a - b).max()


def test_bf16_logits_stay_close_to_fp32_logits():
    """Логиты в bf16 вдвое дешевле по HBM; вопрос ровно один — сколько это стоит по
    числу. Разница обязана держаться в пределах шага bf16 у логита порядка единицы."""
    h, embed, tgt = data(4, jnp.bfloat16)
    lo = chunked_ce(h, embed, tgt, CHUNK, shard=False, f32_logits=False)
    hi = chunked_ce(h, embed, tgt, CHUNK, shard=False, f32_logits=True)
    assert abs(float(lo) - float(hi)) < 5e-3


def test_the_loss_actually_depends_on_the_target():
    """Проверка, что тесты выше не проходят вхолостую: на самом вероятном токене
    лосс обязан быть заметно ниже, чем на случайном."""
    h, embed, tgt = data(5)
    best = jnp.argmax(h @ embed.T, axis=-1)
    assert float(chunked_ce(h, embed, best, CHUNK, shard=False)) + 0.5 < float(
        chunked_ce(h, embed, tgt, CHUNK, shard=False))
