"""chunked_cross_entropy должна совпадать с обычной CE — по значению и по градиентам."""

import pytest
import torch

from model.losses import IGNORE_INDEX, chunked_cross_entropy, plain_cross_entropy

V, D, N = 512, 64, 130          # N нарочно не кратно chunk


def _inputs(seed=0, dtype=torch.float64, ignore=0):
    torch.manual_seed(seed)
    h = torch.randn(N, D, dtype=dtype, requires_grad=True)
    w = torch.randn(V, D, dtype=dtype, requires_grad=True) * 0.05
    w = w.detach().requires_grad_()
    t = torch.randint(0, V, (N,))
    if ignore:
        t[torch.randperm(N)[:ignore]] = IGNORE_INDEX
    return h, w, t


def _grads(loss, *tensors):
    loss.backward()
    out = [x.grad.clone() for x in tensors]
    for x in tensors:
        x.grad = None
    return out


def _close(got, ref, atol):
    """allclose с rtol=0.

    По умолчанию torch.allclose держит rtol=1e-5, и на градиентах порядка 1e-2
    это допуск 1e-7 — который в float64-тесте пропускал ошибку в семь порядков.
    """
    return torch.allclose(got, ref, rtol=0.0, atol=atol)


@pytest.mark.parametrize("chunk", [1, 7, 64, 512])
@pytest.mark.parametrize("n_ignore", [0, 17])
def test_matches_plain_cross_entropy(chunk, n_ignore):
    """float64, CPU: расхождение должно быть на уровне машинной точности."""
    h, w, t = _inputs(ignore=n_ignore)

    ref = plain_cross_entropy(h, w, t)
    ref_gh, ref_gw = _grads(ref, h, w)

    got = chunked_cross_entropy(h, w, t, chunk=chunk, matmul_dtype=torch.float64)
    got_gh, got_gw = _grads(got, h, w)

    assert _close(got, ref, 1e-12), f"loss: {got.item()} vs {ref.item()}"
    assert _close(got_gh, ref_gh, 1e-12), f"grad_h: {(got_gh - ref_gh).abs().max()}"
    assert _close(got_gw, ref_gw, 1e-12), f"grad_W: {(got_gw - ref_gw).abs().max()}"


def test_chunk_size_does_not_change_result():
    """Разбиение на куски — деталь реализации, ответ от неё зависеть не должен."""
    h, w, t = _inputs(seed=1, ignore=11)
    outs = []
    for chunk in (3, 16, 129, 1000):
        loss = chunked_cross_entropy(h, w, t, chunk=chunk, matmul_dtype=torch.float64)
        outs.append((loss.detach().clone(), *_grads(loss, h, w)))
    for loss, gh, gw in outs[1:]:
        assert _close(loss, outs[0][0], 1e-12)
        assert _close(gh, outs[0][1], 1e-12)
        assert _close(gw, outs[0][2], 1e-12)


def test_all_tokens_ignored_gives_zero_not_nan():
    """Вырожденный случай: делить на число валидных токенов нельзя, если их 0."""
    h, w, t = _inputs(seed=2)
    t[:] = IGNORE_INDEX
    loss = chunked_cross_entropy(h, w, t, chunk=16, matmul_dtype=torch.float64)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().max() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна CUDA")
def test_matches_autocast_path_on_gpu():
    """bf16 на GPU: сверяем с тем, что даёт обычный путь под autocast."""
    torch.manual_seed(3)
    h = torch.randn(N, D, device="cuda", requires_grad=True)
    w = (torch.randn(V, D, device="cuda") * 0.05).requires_grad_()
    t = torch.randint(0, V, (N,), device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = plain_cross_entropy(h, w, t)
    ref_gh, ref_gw = _grads(ref, h, w)

    got = chunked_cross_entropy(h, w, t, chunk=32)
    got_gh, got_gw = _grads(got, h, w)

    # bf16-матмулы: сравниваем по относительной величине, а не побитово
    assert (got - ref).abs() < 2e-2 * ref.abs()
    assert (got_gh - ref_gh).abs().max() < 2e-2 * ref_gh.abs().max()
    assert (got_gw - ref_gw).abs().max() < 2e-2 * ref_gw.abs().max()
