"""Cross-entropy, не материализующая логиты на всю последовательность.

# В чём проблема

Обычный путь такой:

    logits = lm_head(h)                     # (N, V)  — весь батч сразу
    loss   = F.cross_entropy(logits, tgt)

Тензор (N, V) живёт до конца backward, потому что производная softmax считается
из него. При T=1536 и V=16384 это 48 МиБ в bf16, плюс копия в fp32, плюс
сохранённый log_softmax — около 350 МиБ на ровном месте.

# Что делает этот файл

Считает то же самое кусками по `chunk` токенов и **не сохраняет ничего**
размера (N, V): в backward каждый кусок логитов пересчитывается заново,
используется и выбрасывается. Живёт одновременно ровно один кусок.

Градиент выписан руками, а не получен автоградом, — для CE он элементарный:

    dL/dlogits = (softmax(logits) − onehot(target)) / N_valid
    dL/dh      = dL/dlogits @ W
    dL/dW      = dL/dlogitsᵀ @ h

Пересчёт логитов в backward стоит одного лишнего матмула (N, d) @ (d, V).
Это тот же приём, что в cut-cross-entropy и Liger, только без Triton — их
ядра на sm_75 не поедут, а выигрыш у нас всё равно упирается в V=16384.

# Замечание про torch.compile

Внутри `torch.compile` собственная autograd.Function рвёт граф. Поэтому
loss считается СНАРУЖИ скомпилированной модели:

    logits_h = compiled_model.body(tokens)     # компилируем модель
    loss = chunked_cross_entropy(h, W, tgt)    # а loss — обычным питоном
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


class _ChunkedCE(torch.autograd.Function):
    """Реализация. Пользоваться через `chunked_cross_entropy` ниже."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk, ignore_index, matmul_dtype):
        # hidden  (N, d) — скрытые состояния
        # weight  (V, d) — веса lm_head (у нас связаны с эмбеддингом)
        # targets (N,)   — правильные токены
        N = hidden.shape[0]
        valid = targets != ignore_index
        n_valid = valid.sum().clamp(min=1)

        # Тип накопления: матмулы можно и в bf16, но logsumexp по 16k элементам
        # и суммирование по всем токенам — нет. promote_types поднимает bf16/fp16
        # до fp32 и НЕ опускает fp64 (важно: на fp64 стоят тесты точности).
        acc = torch.promote_types(hidden.dtype, torch.float32)

        loss_sum = hidden.new_zeros((), dtype=acc)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            logits = torch.matmul(hidden[s:e].to(matmul_dtype),
                                  weight.to(matmul_dtype).t()).to(acc)
            lse = torch.logsumexp(logits, dim=-1)                       # (c,)
            tgt = targets[s:e].clamp(min=0).unsqueeze(1)
            # −log p(target) = logsumexp(logits) − logits[target]
            nll = lse - logits.gather(1, tgt).squeeze(1)
            loss_sum = loss_sum + torch.where(valid[s:e], nll, 0.0).sum()

        ctx.save_for_backward(hidden, weight, targets, n_valid)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.matmul_dtype = matmul_dtype
        ctx.acc = acc
        return loss_sum / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, targets, n_valid = ctx.saved_tensors
        chunk, ignore_index = ctx.chunk, ctx.ignore_index
        mdt, acc = ctx.matmul_dtype, ctx.acc
        N = hidden.shape[0]

        # grad_out — скаляр (обычно 1.0), делённый на число валидных токенов,
        # потому что loss это среднее
        scale = (grad_out / n_valid).to(acc)

        grad_h = torch.zeros_like(hidden)
        # grad_W копится по кускам, поэтому тип накопления берём повышенный:
        # в bf16 сумма 3000 слагаемых потеряла бы больше, чем сам матмул
        grad_W = torch.zeros_like(weight, dtype=acc)

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            h_c = hidden[s:e].to(mdt)
            # пересчитываем те же логиты — вместо того чтобы хранить их
            logits = torch.matmul(h_c, weight.to(mdt).t()).to(acc)
            p = torch.softmax(logits, dim=-1)                            # (c, V)

            t_c = targets[s:e]
            keep = t_c != ignore_index
            # dL/dlogits = softmax − onehot
            p.scatter_add_(1, t_c.clamp(min=0).unsqueeze(1),
                           torch.full((e - s, 1), -1.0, dtype=p.dtype, device=p.device))
            p = p * (scale * keep.to(acc)).unsqueeze(1)                  # игнорируемые строки -> 0

            p_m = p.to(mdt)
            grad_h[s:e] = torch.matmul(p_m, weight.to(mdt)).to(grad_h.dtype)
            grad_W += torch.matmul(p_m.t(), h_c).to(acc)
            del logits, p, p_m

        return grad_h, grad_W.to(weight.dtype), None, None, None, None


def chunked_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk: int = 512,
    ignore_index: int = IGNORE_INDEX,
    matmul_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Средняя CE по токенам, без материализации всех логитов.

    Аргументы:
        hidden       (..., d) — выход модели ДО lm_head
        weight       (V, d)   — `lm_head.weight`
        targets      (...)    — целевые токены той же формы, что hidden без d
        chunk                 — сколько токенов считать за раз; меньше = меньше
                                памяти и чуть больше запусков ядер
        ignore_index          — метка «этот токен не считать»
        matmul_dtype          — в чём делать матмулы; по умолчанию bf16 на CUDA
                                (как сделал бы autocast) и dtype входа иначе

    Возвращает скаляр. Градиент течёт в `hidden` и в `weight`.
    """
    d = hidden.shape[-1]
    h = hidden.reshape(-1, d)
    t = targets.reshape(-1)
    if matmul_dtype is None:
        matmul_dtype = torch.bfloat16 if h.is_cuda else h.dtype
    return _ChunkedCE.apply(h, weight, t, chunk, ignore_index, matmul_dtype)


def plain_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Обычный путь — эталон для тестов и запасной вариант.

    Логиты поднимаются до fp32 (или остаются fp64) перед softmax по той же
    причине, что и в chunked-версии: log_softmax по 16k элементам в bf16 —
    это заметная потеря разрядов.
    """
    logits = F.linear(hidden, weight)
    acc = torch.promote_types(logits.dtype, torch.float32)
    return F.cross_entropy(
        logits.reshape(-1, weight.shape[0]).to(acc),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
