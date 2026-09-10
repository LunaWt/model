"""Зациклённый стек: порядок исполнения, блоки, градиенты общих весов."""

from __future__ import annotations

import pytest
import torch

from model.configs import CONFIGS
from model.model import GatedMLA, K3Config, K3Model, LatentMoE


def tiny(**kw) -> K3Config:
    base = dict(vocab_size=64, d_model=32, n_layers=4, n_heads=2, d_head=16,
                max_seq_len=32, mla_kv_latent=16, mla_q_latent=16, moe_latent=16,
                n_routed=4, n_shared=1, top_k=2, expert_hidden=16, shared_hidden=16,
                dense_hidden=32, n_blocks=2, kda_backend="chunkwise")
    return K3Config(**{**base, **kw})


def test_execution_order_repeats_only_the_span():
    cfg = tiny(n_layers=12, loop_span=(4, 7), loop_r=2, n_blocks=4)
    assert cfg.execution_order() == [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11]


def test_execution_order_is_identity_without_a_loop():
    assert tiny(n_layers=4).execution_order() == [0, 1, 2, 3]
    assert tiny(n_layers=4, loop_span=(1, 2), loop_r=1).execution_order() == [0, 1, 2, 3]


def test_blocks_are_cut_over_executions_not_layers():
    # 4 слоя, span (1,2) x2 -> 6 исполнений; на 2 блока не делится 4, делится 6
    tiny(n_layers=4, loop_span=(1, 2), loop_r=2, n_blocks=2)
    with pytest.raises(AssertionError, match="n_blocks"):
        tiny(n_layers=4, loop_span=(1, 2), loop_r=2, n_blocks=4)


def test_layer_type_follows_layer_index_not_execution_position():
    cfg = tiny(n_layers=4, attn_pattern=4, loop_span=(1, 2), loop_r=2, n_blocks=2)
    m = K3Model(cfg)
    mla = [i for i, a in enumerate(m.attn) if isinstance(a, GatedMLA)]
    assert mla == [3], "MLA должен остаться последним слоем, а не съехать на повтор"
    assert m.execution_order == [0, 1, 2, 1, 2, 3]


def test_looped_forward_shape_and_shared_gradients():
    cfg = tiny(n_layers=4, loop_span=(1, 2), loop_r=2, n_blocks=2)
    m = K3Model(cfg)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(tok)
    assert out.shape == (2, 16, cfg.vocab_size)
    out.sum().backward()

    missing = [n for n, p in m.named_parameters()
               if p.grad is None and n != "res_attn.0.q"]
    assert not missing, missing


def test_looped_layer_gradient_is_the_sum_over_visits():
    """У общих весов градиент обязан накапливаться по обоим визитам.

    Проверяем на одном подслое: строим модель, где визит 2 отключён (r=1) и где
    он есть (r=2) при одинаковых весах и входе, — градиент зациклённого слоя
    должен отличаться, а незациклённого нулевого слоя быть тем же порядком.
    """
    torch.manual_seed(0)
    cfg1 = tiny(n_layers=4, loop_span=(1, 2), loop_r=1, n_blocks=2)
    m1 = K3Model(cfg1)
    cfg2 = tiny(n_layers=4, loop_span=(1, 2), loop_r=2, n_blocks=2)
    m2 = K3Model(cfg2)
    m2.load_state_dict(m1.state_dict())

    tok = torch.randint(0, cfg1.vocab_size, (1, 16))
    for m in (m1, m2):
        m(tok).square().sum().backward()

    g1 = m1.attn[1].W_q.weight.grad
    g2 = m2.attn[1].W_q.weight.grad
    assert not torch.allclose(g1, g2)
    assert g2.abs().sum() > 0


def test_loop_residual_scale_off_changes_output():
    torch.manual_seed(0)
    a = K3Model(tiny(n_layers=4, loop_span=(1, 2), loop_r=2, n_blocks=2, loop_res_scale=True))
    b = K3Model(tiny(n_layers=4, loop_span=(1, 2), loop_r=2, n_blocks=2, loop_res_scale=False))
    b.load_state_dict(a.state_dict())
    tok = torch.randint(0, 64, (1, 16))
    assert not torch.allclose(a(tok), b(tok))


def test_executed_params_exceed_active_only_when_looped():
    plain = K3Model(tiny(n_layers=4, n_blocks=2)).count_params()
    assert plain["executed"] == plain["active"]
    looped = K3Model(tiny(n_layers=4, loop_span=(1, 2), loop_r=2, n_blocks=2)).count_params()
    assert looped["executed"] > looped["active"]


def test_visit_scores_are_keyed_so_recompute_cannot_double_count():
    cfg = tiny(n_layers=4, loop_span=(1, 2), loop_r=2, n_blocks=2)
    m = K3Model(cfg)
    m.train()
    tok = torch.randint(0, cfg.vocab_size, (1, 16))
    m(tok)
    moe = next(f for f in m.ffn if isinstance(f, LatentMoE))
    assert set(moe._visit_scores) == {0, 1}, "зациклённый MoE должен видеть два визита"

    m(tok)   # повторный forward без harvest — перезапись, а не накопление
    assert set(moe._visit_scores) == {0, 1}

    m.harvest_router_scores()
    assert len(moe._scores) == 1 and moe._scores[0].shape[0] == 2 * 16
    assert not moe._visit_scores


def test_named_configs_reproduce_the_looped_budget_table():
    """notes/looped.md §4: каждый зациклённый конфиг сматчен со своим контролем."""
    got = {}
    for name in ("A16", "L12_4x2", "A24", "L16_8x2", "L12_6x3"):
        m = K3Model(CONFIGS[name])
        n = m.count_params()
        got[name] = (len(m.execution_order), n["total"], n["executed"],
                     sum(1 for i in m.execution_order if isinstance(m.attn[i], GatedMLA)))
        assert isinstance(m.attn[CONFIGS[name].n_layers - 1], GatedMLA), \
            f"{name}: стек должен заканчиваться MLA"

    for looped, control, kv in (("L12_4x2", "A16", 4), ("L16_8x2", "A24", 6)):
        e_l, tot_l, exe_l, kv_l = got[looped]
        e_c, tot_c, exe_c, kv_c = got[control]
        assert e_l == e_c and kv_l == kv_c == kv
        assert abs(tot_l / tot_c - 1) < 0.01
        assert abs(exe_l / exe_c - 1) < 0.02
