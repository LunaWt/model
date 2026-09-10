"""Chunkwise-KDA должна давать ровно то же, что наивная рекуррентность.

Это единственное место, где корректность KDA вообще проверяема: наивный цикл
переписан прямо с Eq. 1 и очевиден глазами, chunkwise-форма — нет. Все тесты
идут в float64, чтобы расхождение алгоритмов не пряталось за шумом bf16.
"""

import pytest
import torch
import torch.nn.functional as F

from model.kda_head import KDA, ShortConv, kda_chunkwise, kda_recurrent

B, H, T, D = 2, 3, 48, 16
DT = torch.float64


def _inputs(seed=0, t=T, g_min=-5.0):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, H, t, D, dtype=DT), dim=-1)
    k = F.normalize(torch.randn(B, H, t, D, dtype=DT), dim=-1)
    v = torch.randn(B, H, t, D, dtype=DT)
    # g = log α, ровно в том диапазоне, который даёт параметризация K3
    g = g_min * torch.sigmoid(torch.randn(B, H, t, D, dtype=DT))
    beta = torch.sigmoid(torch.randn(B, H, t, dtype=DT))
    return q, k, v, g, beta


@pytest.mark.parametrize("chunk", [1, 2, 4, 8, 16])
def test_chunkwise_matches_recurrent(chunk):
    q, k, v, g, beta = _inputs()
    ref = kda_recurrent(q, k, v, g, beta)
    got = kda_chunkwise(q, k, v, g, beta, chunk=chunk)
    assert got.shape == ref.shape
    assert (got - ref).abs().max() < 1e-13, (got - ref).abs().max()


@pytest.mark.parametrize("t", [1, 15, 17, 31, 33, 47])
def test_sequence_length_not_multiple_of_chunk(t):
    """Хвост добивается паддингом; он не должен ни течь в состояние, ни в выход."""
    q, k, v, g, beta = _inputs(seed=1, t=t)
    ref = kda_recurrent(q, k, v, g, beta)
    got = kda_chunkwise(q, k, v, g, beta, chunk=16)
    assert got.shape == ref.shape
    assert (got - ref).abs().max() < 1e-13, (got - ref).abs().max()


def test_causality():
    """Выход в позиции t не должен зависеть ни от чего после t."""
    q, k, v, g, beta = _inputs(seed=2)
    base = kda_chunkwise(q, k, v, g, beta, chunk=16)

    cut = 20                       # ломаем всё начиная с этой позиции
    v2 = v.clone()
    v2[:, :, cut:] += 10.0
    k2 = k.clone()
    k2[:, :, cut:] = F.normalize(torch.randn_like(k2[:, :, cut:]), dim=-1)
    other = kda_chunkwise(q, k2, v2, g, beta, chunk=16)

    assert (other[:, :, :cut] - base[:, :, :cut]).abs().max() < 1e-13
    assert (other[:, :, cut:] - base[:, :, cut:]).abs().max() > 1e-3   # иначе тест пустой


def test_gradients_match_recurrent():
    """Backward тоже должен совпадать — иначе обучение поедет молча."""
    q, k, v, g, beta = _inputs(seed=3)
    grads = []
    for fn in (kda_recurrent, lambda *a: kda_chunkwise(*a, chunk=16)):
        ins = [t.clone().requires_grad_() for t in (q, k, v, g, beta)]
        out = fn(*ins)
        (out * torch.arange(1, D + 1, dtype=DT)).sum().backward()
        grads.append([t.grad for t in ins])
    for name, a, b in zip(["q", "k", "v", "g", "beta"], *grads):
        assert (a - b).abs().max() < 1e-11, f"grad {name}: {(a - b).abs().max()}"


def test_zero_decay_is_pure_delta_rule():
    """g = 0 (α = 1) — ничего не забывается; проверяем, что exp/log нигде не потерян."""
    q, k, v, _, beta = _inputs(seed=4)
    g = torch.zeros_like(v)
    ref = kda_recurrent(q, k, v, g, beta)
    got = kda_chunkwise(q, k, v, g, beta, chunk=16)
    assert (got - ref).abs().max() < 1e-13


def test_chunk_size_guard_rejects_overflowing_configs():
    """chunk·|g_min| > 80 переполняет 1/Γ — слой обязан отказаться сразу."""
    KDA(d_model=32, n_heads=2, d_head=16, chunk_size=16, g_min=-5.0)   # 80, ровно предел
    with pytest.raises(ValueError, match="chunk_size"):
        KDA(d_model=32, n_heads=2, d_head=16, chunk_size=32, g_min=-5.0)


def test_layer_chunkwise_equals_layer_recurrent():
    """Тот же тест, но через сам слой: параметризация + рекуррентность + выход."""
    torch.manual_seed(5)
    layer = KDA(d_model=32, n_heads=2, d_head=16, chunk_size=16).to(DT)
    x = torch.randn(2, 40, 32, dtype=DT)

    y_chunk = layer(x)
    layer.chunk_size = 0
    y_loop = layer(x)
    assert (y_chunk - y_loop).abs().max() < 1e-12, (y_chunk - y_loop).abs().max()


def test_shortconv_is_causal():
    """ShortConv не должна пропускать будущее: меняем хвост — начало не шевелится."""
    torch.manual_seed(6)
    conv = ShortConv(channels=8, kernel_size=4).to(DT)
    x = torch.randn(1, 20, 8, dtype=DT)
    y = conv(x)
    x2 = x.clone()
    x2[:, 12:] += 5.0
    y2 = conv(x2)
    assert (y2[:, :12] - y[:, :12]).abs().max() < 1e-14
    assert (y2[:, 12:] - y[:, 12:]).abs().max() > 1e-3
