"""Muon: ортогонализация, поголовное применение, расписание lr."""

from __future__ import annotations

import math

import torch

from model.configs import CONFIGS
from model.model import K3Model
from model.optim import (
    Muon,
    build_optimizers,
    lr_multiplier,
    orthogonalize,
    tag_head_params,
)


def singular_values(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(x.double())


def test_orthogonalize_flattens_the_spectrum():
    torch.manual_seed(0)
    # намеренно плохо обусловленная матрица: спектр от 1 до 100
    u, _ = torch.linalg.qr(torch.randn(64, 64))
    v, _ = torch.linalg.qr(torch.randn(32, 32))
    s = torch.logspace(0, 2, 32)
    G = u[:, :32] @ torch.diag(s) @ v.T

    before = singular_values(G)
    after = singular_values(orthogonalize(G, steps=5))
    assert before.max() / before.min() > 50
    # квинтика Джордана намеренно не сходится к ровно 1: коэффициенты
    # (3.4445, −4.7750, 2.0315) подобраны под скорость первых шагов, и спектр
    # оседает полосой примерно [0.7, 1.3]. Важно, что обусловленность падает
    # с сотни до единиц, а не что все числа равны единице.
    assert after.max() / after.min() < 2.0
    assert after.min() > 0.6 and after.max() < 1.4


def test_orthogonalize_is_batched_over_leading_dims():
    torch.manual_seed(0)
    G = torch.randn(4, 16, 32)
    out = orthogonalize(G)
    assert out.shape == G.shape
    for i in range(4):
        one = orthogonalize(G[i])
        assert torch.allclose(out[i], one, atol=1e-5)


def test_orthogonalize_is_scale_invariant():
    torch.manual_seed(0)
    G = torch.randn(16, 32)
    assert torch.allclose(orthogonalize(G), orthogonalize(G * 1000), atol=1e-4)


def test_muon_state_is_one_tensor_per_param_in_bf16():
    p = torch.nn.Parameter(torch.randn(16, 32))
    opt = Muon([p], lr=0.01)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    st = opt.state[p]
    assert list(st) == ["buf"] and st["buf"].dtype == torch.bfloat16
    assert not torch.allclose(before, p.detach())


def test_update_rms_matches_the_moonlight_target():
    """Moonlight (2502.16982), Лемма 1: RMS шага Muon = sqrt(1/max(m,n)).

    Множитель γ = 0.2·sqrt(max(m,n)) обязан привести его к 0.2 НЕЗАВИСИМО от
    формы — иначе Muon и AdamW нельзя держать на одном lr, а промах в этом
    множителе стоит порядка величины по скорости обучения.
    """
    torch.manual_seed(0)
    for shape in ((512, 512), (64, 512), (512, 64), (128, 2048)):
        p = torch.nn.Parameter(torch.zeros(shape))
        p.grad = torch.randn(shape)
        opt = Muon([p], lr=1.0, momentum=0.0, nesterov=False)
        opt.step()
        rms = p.detach().square().mean().sqrt().item()
        assert 0.15 < rms < 0.25, (shape, rms)


def test_muon_head_split_differs_from_whole_matrix():
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    grad = torch.randn(8, 32)

    a = torch.nn.Parameter(w.clone())
    b = torch.nn.Parameter(w.clone())
    b._muon_head_dim = 4

    for p, opt in ((a, Muon([a], lr=0.1)), (b, Muon([b], lr=0.1))):
        p.grad = grad.clone()
        opt.step()
    assert not torch.allclose(a.detach(), b.detach())


def test_attention_projections_are_tagged_per_head():
    m = K3Model(CONFIGS["tiny"])
    n = tag_head_params(m)
    assert n > 0
    kda = m.attn[0]
    assert kda.W_q.weight._muon_head_dim == kda.n_heads
    assert not hasattr(m.ffn[0].W_g.weight, "_muon_head_dim")


def test_optimizer_split_covers_every_parameter_exactly_once():
    m = K3Model(CONFIGS["tiny"])
    muon, adamw = build_optimizers(m)
    seen = [p for g in muon.param_groups for p in g["params"]]
    seen += [p for g in adamw.param_groups for p in g["params"]]
    ids = {id(p) for p in seen}
    assert len(ids) == len(seen), "параметр попал в два оптимизатора"
    assert ids == {id(p) for p in m.parameters() if p.requires_grad}

    embed = m.embed.weight
    assert any(p is embed for g in adamw.param_groups for p in g["params"]), \
        "эмбеддинг обновляется AdamW: строки разрежены, ортогонализация не про них"
    assert all(p.ndim >= 2 for g in muon.param_groups for p in g["params"])


def test_no_weight_decay_on_kda_gate_parameters():
    m = K3Model(CONFIGS["tiny"])
    _, adamw = build_optimizers(m, weight_decay=0.1)
    plain = {id(p) for g in adamw.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    assert id(m.attn[0].b_alpha) in plain
    assert id(m.attn[0].A) in plain


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    total = 1000
    assert lr_multiplier(0, total) == pytest_approx(1 / 10)
    assert lr_multiplier(9, total) == pytest_approx(1.0)
    assert lr_multiplier(total - 1, total) == pytest_approx(0.1, tol=1e-3)
    mid = lr_multiplier(total // 2, total)
    assert 0.4 < mid < 0.7
    xs = [lr_multiplier(s, total) for s in range(10, total)]
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:])), "после прогрева lr не растёт"


def pytest_approx(v: float, tol: float = 1e-9):
    class _A:
        def __eq__(self, other):
            return math.isclose(other, v, rel_tol=tol, abs_tol=tol)
    return _A()


def test_resume_keeps_the_muon_moment_in_bf16():
    """Базовый load_state_dict приводит состояние к dtype параметра.

    Параметры fp32, момент bf16 — без нашего переопределения возобновление молча
    удваивает состояние Muon (на A16 это +451 МиБ при потолке 6 ГБ).
    """
    p = torch.nn.Parameter(torch.randn(16, 32))
    opt = Muon([p], lr=0.01)
    p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()

    q = torch.nn.Parameter(torch.randn(16, 32))
    opt2 = Muon([q], lr=0.01)
    opt2.load_state_dict(sd)
    assert opt2.state[q]["buf"].dtype == torch.bfloat16
    assert torch.equal(opt2.state[q]["buf"], opt.state[p]["buf"])
