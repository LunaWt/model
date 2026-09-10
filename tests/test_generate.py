"""Кэш генерации: он обязан давать ровно то же, что полный пересчёт."""

from __future__ import annotations

import torch

from model.configs import CONFIGS
from model.generate import generate
from model.kda_head import KDA, ShortConv, kda_recurrent
from model.model import K3Config, K3Model


def tiny_model(**over) -> K3Model:
    cfg = K3Config(**{**vars(CONFIGS["tiny"]).copy(), **over}) if over else CONFIGS["tiny"]
    torch.manual_seed(0)
    m = K3Model(cfg)
    m.eval()
    return m


def test_short_conv_cache_matches_the_full_pass():
    torch.manual_seed(0)
    conv = ShortConv(8, kernel_size=4)
    x = torch.randn(2, 12, 8)
    full = conv(x)

    y0, st = conv.forward_cached(x[:, :5], None)
    outs = [y0]
    for t in range(5, 12):
        y, st = conv.forward_cached(x[:, t:t + 1], st)
        outs.append(y)
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-6)


def test_short_conv_state_holds_exactly_kernel_minus_one_inputs():
    conv = ShortConv(4, kernel_size=4)
    _, st = conv.forward_cached(torch.randn(1, 7, 4), None)
    assert st.shape == (1, 4, 3)


def test_kda_recurrent_carries_state_between_calls():
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 9, 8
    q = torch.nn.functional.normalize(torch.randn(B, H, T, D), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D), dim=-1)
    v = torch.randn(B, H, T, D)
    g = -5.0 * torch.sigmoid(torch.randn(B, H, T, D))
    beta = torch.sigmoid(torch.randn(B, H, T))

    full = kda_recurrent(q, k, v, g, beta)
    o1, S = kda_recurrent(q[:, :, :4], k[:, :, :4], v[:, :, :4], g[:, :, :4], beta[:, :, :4],
                          output_final_state=True)
    o2, _ = kda_recurrent(q[:, :, 4:], k[:, :, 4:], v[:, :, 4:], g[:, :, 4:], beta[:, :, 4:],
                          initial_state=S, output_final_state=True)
    assert torch.allclose(torch.cat([o1, o2], dim=2), full, atol=1e-6)


def test_kda_layer_cache_matches_the_full_pass():
    torch.manual_seed(0)
    layer = KDA(32, n_heads=2, d_head=16, backend="recurrent")
    layer.eval()
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        full = layer(x)
        cache: dict = {}
        outs = [layer(x[:, :6], cache=cache)]
        for t in range(6, 10):
            outs.append(layer(x[:, t:t + 1], cache=cache))
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=1e-5)


def test_greedy_generation_with_cache_equals_full_recompute():
    m = tiny_model()
    prompt = torch.randint(0, m.cfg.vocab_size, (1, 7))
    kw = dict(max_new_tokens=12, temperature=0.0)
    a = generate(m, prompt, use_cache=True, **kw)
    b = generate(m, prompt, use_cache=False, **kw)
    assert torch.equal(a, b), (a.tolist(), b.tolist())


def test_greedy_generation_with_cache_equals_full_recompute_when_looped():
    """Ключ кэша — номер ИСПОЛНЕНИЯ: у зациклённого слоя два прохода, две памяти.

    При ключе по номеру слоя второй проход затирал бы состояние первого, и
    расхождение вылезло бы только на зациклённых конфигах.
    """
    m = tiny_model(loop_span=(1, 2), loop_r=2, n_blocks=2)
    assert len(m.execution_order) > m.cfg.n_layers
    prompt = torch.randint(0, m.cfg.vocab_size, (1, 5))
    kw = dict(max_new_tokens=10, temperature=0.0)
    assert torch.equal(generate(m, prompt, use_cache=True, **kw),
                       generate(m, prompt, use_cache=False, **kw))


def test_cache_grows_only_in_the_mla_layers():
    """12 из 16 слоёв гибрида несут кэш ПОСТОЯННОГО размера — это и есть довод
    за KDA на длинном контексте, и он должен быть виден в самом кэше."""
    m = tiny_model()
    cache: dict = {}
    m(torch.randint(0, m.cfg.vocab_size, (1, 6)), cache=cache)
    sizes = {i: {k: tuple(v.shape) for k, v in c.items()} for i, c in cache.items()}
    m(torch.randint(0, m.cfg.vocab_size, (1, 1)), cache=cache)

    grown = 0
    for i, c in cache.items():
        for key, v in c.items():
            if tuple(v.shape) != sizes[i][key]:
                assert key == "c", f"вырос не-MLA кэш: слой {i}, {key}"
                grown += 1
    assert grown == sum(1 for c in cache.values() if "c" in c) > 0


def test_fixed_expert_capacity_makes_the_model_non_causal():
    """Почему генерация вообще требует `set_full_capacity`.

    При фиксированной полке выход токена зависит от того, какие ещё токены
    попали в этот forward. Тест фиксирует и сам эффект, и то, что снятие
    ёмкости его убирает.
    """
    m = tiny_model()
    x = torch.randint(0, m.cfg.vocab_size, (1, 12))
    with torch.no_grad():
        short, long = m(x[:, :5]), m(x)
        drifted = (short[0, :5] - long[0, :5]).abs().max().item()
        m.set_full_capacity(True)
        short, long = m(x[:, :5]), m(x)
        clean = (short[0, :5] - long[0, :5]).abs().max().item()
        m.set_full_capacity(False)
    assert drifted > 1e-2, drifted
    assert clean < 1e-4, clean
