"""Muon для матриц + AdamW для всего остального, и косинусное расписание.

# Зачем Muon здесь, а не «просто AdamW»

Две причины, и вторая на этой карте решающая.

1. Метод. AdamW делит градиент на покоординатную оценку масштаба — у него нет
   понятия «матрица». Muon берёт градиент матрицы, добавляет момент и
   ОРТОГОНАЛИЗУЕТ результат: заменяет `G = U S Vᵀ` на `U Vᵀ`, то есть выкидывает
   сингулярные числа, оставляя только направления. Шаг получается одинаково
   сильным по всем направлениям — не даёт нескольким крупным сингулярным
   направлениям съесть весь шаг.

2. Память. AdamW держит ДВА fp32-состояния на параметр, Muon — ОДНО (момент).
   При 225.7M параметров это 903 МиБ разницы, и ровно из-за неё Config F с
   fp32-AdamW не влезал в 6 ГБ (замер 2 сен: 6151 МиБ, скорость падала с 445 до
   168 ток/с на подкачке в хост-память). Момент дополнительно держим в bf16 —
   ещё −450 МиБ; после ортогонализации абсолютный масштаб момента всё равно
   выбрасывается, так что терять там особо нечего.

Ортогонализацию считаем итерацией Ньютона–Шульца (5 шагов, коэффициенты Келлера
Джордана) — только матричные умножения, без SVD, и работает батчево.

# Per-head (K3, §3.1)

Матрица внимания формы (H·D, d_model) — это H независимых голов, поставленных
друг на друга. Ортогонализовать её целиком значит смешивать головы; K3 применяет
Muon поголовно. Параметр, помеченный `_muon_head_dim = H`, разворачивается в
(H, D, d_model), и Ньютон–Шульц идёт батчем по головам. Веса экспертов
(n_experts, ℓ, hidden) уже трёхмерные и обрабатываются так же — по эксперту.

# Что уходит в AdamW

Всё одномерное (RMSNorm, `b_α`, `A`, псевдо-запросы AttnRes) и эмбеддинг. Для
эмбеддинга ортогонализация бессмысленна: строки обновляются разреженно, по тем
токенам, что встретились в батче, и «направления матрицы» тут не та величина.
"""

from __future__ import annotations

import math

import torch
from torch import nn

NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Ньютон–Шульц: приближает U Vᵀ из SVD G = U S Vᵀ. Батчево по ведущим осям.

    Итерация квинтическая и сходится к матрице с сингулярными числами ≈ 1 из
    любого стартового масштаба, поэтому вход нормируется по спектральной норме
    сверху (норма Фробениуса — верхняя оценка, её и берём).
    """
    a, b, c = NS_COEFFS
    X = G.float()
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


RMS_TARGET = 0.2


class Muon(torch.optim.Optimizer):
    """Momentum + ортогонализация. Одно состояние на параметр.

    Масштаб шага — по Moonlight (arXiv 2502.16982, Лемма 1 и Eq. 3), а не по
    исходному посту Джордана. Лемма: у полноранговой матрицы (m, n) шаг после
    ортогонализации имеет RMS ровно `sqrt(1/max(m, n))` — то есть зависит от
    ФОРМЫ параметра, а не от градиента. У Adam RMS шага ≈ 1. Поэтому вклад
    домножается на

        γ = 0.2 · sqrt(max(m, n))

    и после этого Muon и AdamW живут на ОДНОМ lr и одном расписании. 0.2, а не
    1.0, потому что реальный RMS Adam из-за β₁ < β₂ и ε заметно меньше единицы.

    ⚠️ Это ровно та строка, где легко промахнуться на порядок: без множителя
    0.2 и на «родном» lr Джордана (0.02) шаг получается примерно в 30 раз
    больше нужного. Замерено 2 сен: роутер MoE схлопывался за 3 шага, до 70%
    слотов улетало в переполнение.
    """

    def __init__(self, params, lr: float = 6e-4, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0,
                 state_dtype: torch.dtype = torch.bfloat16):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay,
                                      state_dtype=state_dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mom, nesterov = group["momentum"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p, dtype=group["state_dtype"])
                buf = st["buf"]
                buf.mul_(mom).add_(g.to(buf.dtype), alpha=1 - mom)
                upd = g.add(buf.to(g.dtype), alpha=mom) if nesterov else buf.to(g.dtype)

                heads = getattr(p, "_muon_head_dim", None)
                shaped = upd.view(heads, -1, upd.shape[-1]) if heads else upd
                o = orthogonalize(shaped, steps=group["ns_steps"]).view_as(upd)

                m, n = shaped.shape[-2], shaped.shape[-1]
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(o, alpha=-group["lr"] * RMS_TARGET * math.sqrt(max(m, n)))
        return loss

    def load_state_dict(self, state_dict) -> None:
        """То же, что у базового класса, но момент возвращается в `state_dtype`.

        `Optimizer.load_state_dict` приводит любое вещественное состояние к dtype
        САМОГО ПАРАМЕТРА (optimizer.py, `_process_value_according_to_param_policy`).
        Параметры у нас fp32, момент — bf16, так что после возобновления он молча
        становился бы fp32: +451 МиБ на A16, ровно тот запас, которого на 6 ГБ нет.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            dt = group["state_dtype"]
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None and "buf" in st:
                    st["buf"] = st["buf"].to(dt)


def tag_head_params(model: nn.Module) -> int:
    """Пометить матрицы внимания как поголовные для Muon. Возвращает их число."""
    from model.kda_head import KDA
    from model.model import GatedMLA

    tagged = 0
    for mod in model.modules():
        if not isinstance(mod, (KDA, GatedMLA)):
            continue
        h = mod.n_heads
        for name in ("W_q", "W_k", "W_v", "W_g", "W_uq", "W_uk", "W_uv"):
            lin = getattr(mod, name, None)
            if lin is not None and lin.weight.shape[0] % h == 0:
                lin.weight._muon_head_dim = h
                tagged += 1
    return tagged


def build_optimizers(model: nn.Module, lr_muon: float = 6e-4, lr_adam: float = 6e-4,
                     weight_decay: float = 0.1, betas: tuple[float, float] = (0.9, 0.95),
                     state_dtype: torch.dtype = torch.bfloat16) -> tuple[Muon, torch.optim.AdamW]:
    """Раскладка параметров на два оптимизатора.

    Muon: всё с ndim >= 2, кроме эмбеддинга (он же lm_head — веса связаны).
    AdamW: эмбеддинг и всё одномерное; weight decay только на эмбеддинг, потому
    что decay на `b_α` тянет α к e^(−2.5) и стирает состояние KDA каждый шаг.

    lr у обоих по умолчанию один и тот же — в этом весь смысл RMS-выравнивания
    в Muon (см. его docstring).
    """
    tag_head_params(model)
    embed = model.embed.weight

    muon_params, adam_decay, adam_plain = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p is embed:
            adam_decay.append(p)
        elif p.ndim >= 2:
            muon_params.append(p)
        else:
            adam_plain.append(p)

    muon = Muon(muon_params, lr=lr_muon, weight_decay=weight_decay, state_dtype=state_dtype)
    adamw = torch.optim.AdamW(
        [{"params": adam_decay, "weight_decay": weight_decay},
         {"params": adam_plain, "weight_decay": 0.0}],
        lr=lr_adam, betas=betas, eps=1e-8,
    )
    return muon, adamw


def lr_multiplier(step: int, total_steps: int, warmup_frac: float = 0.01,
                  min_frac: float = 0.1) -> float:
    """Косинус с линейным прогревом. Возвращает множитель к базовому lr."""
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total_steps - warmup)
    t = min(1.0, t)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))
