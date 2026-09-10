"""Quantile Balancing действительно выравнивает нагрузку экспертов.

Метрика — доля слотов, выброшенных по переполнению ёмкости. Именно она, а не
«коэффициент вариации», потому что платим мы ровно за неё: переполненный слот
не считается вообще, вес токена по этому эксперту зануляется.
"""

from __future__ import annotations

import torch

from model.model import K3Config, LatentMoE


def cfg(**kw) -> K3Config:
    base = dict(vocab_size=64, d_model=64, n_layers=4, n_heads=2, d_head=16,
                moe_latent=32, n_routed=32, top_k=4, n_shared=1,
                expert_hidden=32, shared_hidden=32, n_blocks=2)
    return K3Config(**{**base, **kw})


def skewed_scores(m: int, n: int, strength: float, seed: int = 0) -> torch.Tensor:
    """Скоры с заранее перекошенной популярностью экспертов.

    Роутер в обучении именно так и ломается: часть экспертов систематически
    получает более высокий скор на любом токене, и без биаса top-k уходит к ним.
    """
    g = torch.Generator().manual_seed(seed)
    bias = torch.linspace(-strength, strength, n)
    return torch.sigmoid(torch.randn(m, n, generator=g) + bias)


def expert_counts(moe: LatentMoE, s: torch.Tensor) -> torch.Tensor:
    idx = torch.topk(s + moe.qb_bias, moe.cfg.top_k, dim=-1).indices
    return torch.bincount(idx.reshape(-1), minlength=s.shape[1])


def load_ratio(moe: LatentMoE, s: torch.Tensor) -> float:
    """Нагрузка самого популярного эксперта, делённая на идеально ровную.

    Мера самого выравнивания, не зависящая от capacity_factor: 1.0 — идеал.
    """
    counts = expert_counts(moe, s)
    return float(counts.max()) / (s.shape[0] * moe.cfg.top_k / s.shape[1])


def overflow_fraction(moe: LatentMoE, s: torch.Tensor) -> float:
    """Какая доля из m*k слотов не влезла бы в полки ёмкости cap."""
    m, n = s.shape
    counts = expert_counts(moe, s)
    cap = max(1, int(m * moe.cfg.top_k / n * moe.cfg.capacity_factor))
    return float((counts - cap).clamp(min=0).sum()) / (m * moe.cfg.top_k)


def run_qb(iters: int, strength: float, metric=load_ratio) -> float:
    c = cfg(qb_iters=iters)
    moe = LatentMoE(c)
    s = skewed_scores(512, c.n_routed, strength)
    moe._scores = [s]
    moe.update_router_bias()
    return metric(moe, s)


def test_without_balancing_a_skewed_router_overflows():
    c = cfg()
    moe = LatentMoE(c)
    s = skewed_scores(512, c.n_routed, strength=2.0)
    assert load_ratio(moe, s) > 2.0
    assert overflow_fraction(moe, s) > 0.15


def test_the_alternating_solver_beats_a_single_iteration():
    """Одна итерация Eq. 14 — это НЕ решение Algorithm 1, и разница видна.

    Смотрим на перекос нагрузки, а не на переполнение: при `capacity_factor`
    с запасом обе версии дают ноль выброшенных слотов, и тест бы ничего не
    проверял.
    """
    one = run_qb(iters=1, strength=2.0)
    many = run_qb(iters=8, strength=2.0)
    assert many < one, (one, many)
    assert many < 1.15, many


def test_default_iteration_count_balances_a_hard_skew():
    assert run_qb(iters=K3Config().qb_iters, strength=3.0) < 1.2


def test_bias_stays_centred_so_it_cannot_drift():
    c = cfg()
    moe = LatentMoE(c)
    s = skewed_scores(512, c.n_routed, strength=2.0)
    for _ in range(20):
        moe._scores = [s]
        moe.update_router_bias()
    assert abs(float(moe.qb_bias.mean())) < 1e-5
    assert float(moe.qb_bias.abs().max()) < 10


def test_bias_is_not_applied_to_the_mixture_weights():
    """b управляет только выбором top-k; p считается из чистых скоров (§2.3.3)."""
    c = cfg()
    moe = LatentMoE(c)
    moe.eval()
    x = torch.randn(1, 8, c.d_model)
    y0 = moe(x)
    moe.qb_bias.copy_(torch.zeros_like(moe.qb_bias) + 5.0)   # общий сдвиг
    assert torch.allclose(y0, moe(x), atol=1e-5), \
        "равномерный сдвиг биаса не меняет ни top-k, ни веса смеси"
