"""KDA — Kimi Delta Attention, полный слой.

Две реализации одной и той же рекуррентности:

  * `kda_recurrent` — наивный цикл по токенам. Медленный (T последовательных
    шагов, GPU простаивает), зато читается напрямую по формуле. Это ЭТАЛОН:
    его единственная задача — служить образцом правды в тестах.
  * `kda_chunkwise` — chunkwise-параллельная форма (Kimi Linear Eq. 6–9).
    Рабочая. Сверена с эталоном до 4.4e-16 в float64.

Слой `KDA` берёт вторую; `chunk_size=0` переключает на первую.

Формулы (K3 tech report, Eq. 1, 2, 5, 6):

    S_t = (I − β_t k_t k_t^T) · Diag(α_t) · S_{t−1} + β_t k_t v_t^T
    õ_t = S_t^T q_t
    y_t = W_o [ Sigmoid(W_g x_t) ⊙ RMSNorm(õ_t) ]

    q_t, k_t = L2Norm( Swish( ShortConv( W_{q/k} x_t ) ) )
    v_t      =         Swish( ShortConv( W_v     x_t ) )
    β_t      = Sigmoid( W_β x_t )
    z_t      = W_α↑ W_α↓ x_t + b_α
    g_t      = g_min · Sigmoid( e^{A_h} · z_t ),   α_t = exp(g_t)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConv(nn.Module):
    """Короткая причинная depthwise-свёртка по оси времени.

    Каждый канал свёртывается сам с собой (groups=channels), окно смотрит
    только назад: [t−K+1 … t]. Даёт токену локальный n-граммный контекст до
    того, как он превратится в ключ/значение — у KDA нет ни softmax, ни
    позиционных кодировок, взять этот контекст больше неоткуда.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            groups=channels,   # depthwise: (C, 1, K) весов вместо (C, C, K)
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, C)
        x = x.transpose(1, 2)                        # (B, C, T): Conv1d хочет канал вторым
        # K-1 нулей СЛЕВА и ни одного справа -> окно смотрит только назад.
        # Длина после этого T+K-1, свёртка окном K даёт ровно T обратно.
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)                     # обратно в (B, T, C)


def kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Наивная рекуррентность — ЭТАЛОН. Один шаг на токен.

        S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t k_t v_tᵀ
        õ_t = S_tᵀ q_t

    Формы: q, k, v, g — (B, H, T, D); beta — (B, H, T). Возврат (B, H, T, D).
    `g` это ЛОГАРИФМ затухания (g = log α), так удобнее: дальше всё считается
    через кумулятивные суммы, а не произведения.

    Медленно: T последовательных шагов, на каждом матрица 128×128 — GPU
    простаивает. Существует ради тестов: chunkwise-версия сверяется с ней.
    """
    B, H, T, D = q.shape
    alpha = g.exp()
    S = q.new_zeros(B, H, D, D)
    outs = []
    for t in range(T):
        S = alpha[:, :, t].unsqueeze(-1) * S
        read_k = torch.einsum("bhk,bhkv->bhv", k[:, :, t], S)
        delta = v[:, :, t] - read_k
        S = S + beta[:, :, t][..., None, None] * torch.einsum(
            "bhk,bhv->bhkv", k[:, :, t], delta
        )
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, dim=2)


def kda_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk: int = 16,
) -> torch.Tensor:
    """Chunkwise-параллельная форма. То же, что kda_recurrent, но быстро.

    Kimi Linear, Eq. 6–9 (в K3-отчёте это Eq. 3–4; вывод — в
    `notes/k3_architecture.md`). Идея: **между блоками рекуррентно, внутри
    блока параллельно.** Вместо T последовательных шагов остаётся T/C шагов,
    а вся работа внутри блока — плотные матричные умножения.

    ## Как это выходит

    Перепишем шаг, сгруппировав по k_t:

        S_t = Diag(α_t) S_{t−1} + k_t u_tᵀ,   u_t = β_t (v_t − S_{t−1}ᵀ Diag(α_t) k_t)

    `u_t` — «сколько ещё дописать под ключ k_t». Разворачивая внутри блока и
    обозначая Γ_r = α_1 ⊙ … ⊙ α_r (кумулятивное затухание от начала блока):

        S_r = Diag(Γ_r) S_0 + Σ_{i≤r} Diag(Γ_r/Γ_i) k_i u_iᵀ

    Подставив это обратно в определение u_t, получаем систему на все u блока
    сразу — треугольную, потому что u_t зависит только от предыдущих:

        (I + A) U = Diag(β) (V − (Γ⊙K) S_0),   A[r,i] = β_r ⟨k_i/Γ_i, Γ_r⊙k_r⟩, i<r

    Это и есть **UT-преобразование**: одна треугольная система вместо C
    последовательных шагов. Дальше всё — матмулы.

    ## Почему chunk = 16, а не 64

    В матрицу A входит `K / Γ`, то есть ДЕЛЕНИЕ на кумулятивное затухание.
    Γ ≥ e^{g_min·C}, значит 1/Γ ≤ e^{5C}. При C=16 это e^80 ≈ 5.5·10³⁴ —
    влезает (предел fp32 и bf16 одинаков, 3.4·10³⁸). При C=32 было бы e^160,
    то есть inf, а inf·0 после маскирования даст nan. Ровно ради этого в K3 и
    введена нижняя граница g_min = −5: без неё чанковая форма не считается
    вообще. Считаем в fp32 независимо от autocast — тензоры тут мелкие,
    экономить нечего, а запас по порядку нужен.
    """
    B, H, T, D = q.shape
    C = chunk
    # bf16 поднимаем до fp32 (нужен запас по порядку под 1/Γ), fp64 не трогаем —
    # на нём стоят тесты точности
    dt = torch.promote_types(q.dtype, torch.float32)

    pad = (-T) % C
    if pad:
        # добиваем до кратности C: k=v=0 (ничего не пишется), β=0 (нулевая
        # сила записи), g=0 (α=1, ничего не забывается) -> состояние не
        # меняется, а выходы этих позиций мы всё равно отрежем
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        g = F.pad(g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))

    n = (T + pad) // C
    shp = (B, H, n, C, D)
    q, k, v, g = (t.to(dt).reshape(shp) for t in (q, k, v, g))
    beta = beta.to(dt).reshape(B, H, n, C, 1)

    # Γ_r = произведение α по всем позициям блока до r включительно.
    # В логарифме это обычная кумулятивная сумма — устойчивее произведения.
    # ВНИМАНИЕ: Gc — это ЛОГАРИФМ Γ. Само затухание получается только через
    # .exp(), и забыть его здесь очень легко.
    Gc = g.cumsum(dim=-2)                       # (B,H,n,C,D), значения ≤ 0
    Gamma = Gc.exp()                            # Γ ∈ (0, 1]
    Kg = k * Gamma                              # Γ ⊙ K
    Kd = k * (-Gc).exp()                        # K / Γ   <- вот тут и растёт
    Qg = q * Gamma                              # Γ ⊙ Q

    # A = StrictTril[ Diag(β) (Γ⊙K)(K/Γ)ᵀ ]; строго ниже диагонали, потому что
    # u_r зависит от предыдущих u, но не от себя
    A = (Kg @ Kd.transpose(-1, -2)).tril(-1) * beta
    A = A + torch.eye(C, dtype=dt, device=q.device)

    # (I+A) — нижнетреугольная с единицами на диагонали, решаем прямой
    # подстановкой. U = M V, W = M (Γ⊙K), где M = (I+A)^{-1} Diag(β)
    U = torch.linalg.solve_triangular(A, beta * v, upper=False, unitriangular=True)
    W = torch.linalg.solve_triangular(A, beta * Kg, upper=False, unitriangular=True)

    gamma_C = Gamma[..., -1, :]                 # (B,H,n,D) — затухание за весь блок
    Kc = Kd * gamma_C.unsqueeze(-2)             # Γ^{i→C} ⊙ K, для переноса в S

    # для выхода маска ВКЛЮЧАЕТ диагональ: токен читает состояние уже ПОСЛЕ
    # собственной записи
    Aq = (Qg @ Kd.transpose(-1, -2)).tril(0)

    S = q.new_zeros(B, H, D, D)
    outs = []
    for i in range(n):
        Vt = U[:, :, i] - W[:, :, i] @ S        # псевдо-значение, Eq. 8
        outs.append(Qg[:, :, i] @ S + Aq[:, :, i] @ Vt)   # межблочное + внутриблочное
        S = gamma_C[:, :, i].unsqueeze(-1) * S + Kc[:, :, i].transpose(-1, -2) @ Vt

    o = torch.stack(outs, dim=2).reshape(B, H, n * C, D)
    return o[:, :, :T]


class KDA(nn.Module):
    """Многоголовый слой KDA целиком: x -> y, обе размерности d_model.

    Аргументы:
        d_model     ширина модели
        n_heads     число голов
        d_head      d_k = d_v, размер головы (в статье везде 128)
        conv_kernel окно ShortConv
        g_min       нижняя граница лог-затухания; α ∈ (e^g_min, 1)
        alpha_rank  ранг low-rank проекции для логита затухания
                    (в статье = d_head); None -> d_head
        chunk_size  размер блока chunkwise-формы; 0 -> наивный цикл (эталон).
                    Потолок задан g_min: нужно chunk·|g_min| <= 80, иначе
                    1/Γ переполняется (см. kda_chunkwise)
        dt_range    диапазон log-равномерной выборки dt для инициализации b_α;
                    (0.001, 0.1) — как в fla и в Mamba-2/GDN, откуда K3 её и
                    берёт («b_α initialized following [64, 24, 139]», стр. 5).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int = 128,
        conv_kernel: int = 4,
        g_min: float = -5.0,
        alpha_rank: int | None = None,
        dt_range: tuple[float, float] = (0.001, 0.1),
        chunk_size: int = 16,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.g_min = g_min
        if chunk_size and chunk_size * abs(g_min) > 80.0:
            raise ValueError(
                f"chunk_size={chunk_size} при g_min={g_min}: 1/Γ доходит до "
                f"e^{chunk_size * abs(g_min):.0f}, а предел fp32/bf16 — e^88. "
                "Уменьши chunk_size или подними g_min."
            )
        self.chunk_size = chunk_size
        inner = n_heads * d_head          # ширина всех голов вместе

        # --- проекции входа -------------------------------------------------
        self.W_q = nn.Linear(d_model, inner, bias=False)
        self.W_k = nn.Linear(d_model, inner, bias=False)
        self.W_v = nn.Linear(d_model, inner, bias=False)

        # своя свёртка на каждый из трёх путей
        self.conv_q = ShortConv(inner, conv_kernel)
        self.conv_k = ShortConv(inner, conv_kernel)
        self.conv_v = ShortConv(inner, conv_kernel)

        # --- сила записи β: одно число на голову ----------------------------
        self.W_beta = nn.Linear(d_model, n_heads, bias=False)

        # --- логит затухания z: low-rank d -> r -> inner --------------------
        r = alpha_rank if alpha_rank is not None else d_head
        self.W_a_down = nn.Linear(d_model, r, bias=False)
        self.W_a_up = nn.Linear(r, inner, bias=False)

        # b_α — отдельный параметр, а не bias внутри Linear: так его не затрёт
        # общая инициализация весов модели.
        #
        # Схема из fla/layers/kda.py (и Mamba-2/GDN до неё): берём шаг dt
        # ЛОГ-РАВНОМЕРНО из [0.001, 0.1] независимо для каждого из H*d_k
        # каналов, и прогоняем через обратный softplus. Смысл: инициализация
        # задаёт разброс ВРЕМЁН ЖИЗНИ памяти, равномерный в логарифме, а не
        # разброс самих коэффициентов. Итог: α ∈ [0.62, 0.995], полураспад от
        # ~2 до ~140 токенов, и каналы ВНУТРИ одной головы разные — без этого
        # KDA на старте вырождается в скалярный гейт на голову, то есть в GDN.
        dt = torch.exp(
            torch.rand(inner) * (math.log(dt_range[1]) - math.log(dt_range[0]))
            + math.log(dt_range[0])
        ).clamp(min=1e-4)
        self.b_alpha = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # обучаемый лог-масштаб крутизны, свой на голову; e^A, старт A=0 -> 1
        self.A = nn.Parameter(torch.zeros(n_heads))

        # Ни b_α, ни A не должны получать weight decay: рецепт K3 — wd=0.1 на
        # всё подряд, а b_α -> 0 даёт α = e^(-2.5) = 0.082, полураспад 0.28
        # токена — состояние стирается каждый шаг и KDA перестаёт что-либо
        # переносить. fla помечает оба параметра ровно так же.
        self.b_alpha._no_weight_decay = True
        self.A._no_weight_decay = True

        # --- выходной путь --------------------------------------------------
        # RMSNorm по последней оси = по d_v одной головы -> нормировка headwise
        self.o_norm = nn.RMSNorm(d_head)
        self.W_g = nn.Linear(d_model, inner, bias=False)   # full-rank гейт (K3)
        self.W_o = nn.Linear(inner, d_model, bias=False)

    # ------------------------------------------------------------------
    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, H*D) -> (B, H, T, D). view режет ось, transpose меняет местами."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        # --- q, k, v --------------------------------------------------------
        # ВАЖЕН ПОРЯДОК: сначала режем на головы, потом L2Norm. F.normalize
        # работает по последней оси, и на склеенном (B,T,H*D) он нормировал бы
        # все головы как один вектор — тогда ‖k‖=1 внутри головы не гарантирован
        # и дельта-правило теряет устойчивость.
        q = self._heads(F.silu(self.conv_q(self.W_q(x))))          # (B, H, T, D)
        k = self._heads(F.silu(self.conv_k(self.W_k(x))))
        v = self._heads(F.silu(self.conv_v(self.W_v(x))))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # v НЕ нормируется: это содержимое памяти, его величина несёт смысл

        # --- β --------------------------------------------------------------
        beta = torch.sigmoid(self.W_beta(x)).transpose(1, 2)       # (B, H, T)

        # --- α ---------------------------------------------------------------
        z = self.W_a_up(self.W_a_down(x)) + self.b_alpha           # (B, T, H*D)
        z = self._heads(z)                                         # (B, H, T, D)
        g = self.g_min * torch.sigmoid(self.A.exp().view(1, H, 1, 1) * z)
        # дальше везде работаем с ЛОГАРИФМОМ затухания: chunkwise-форма
        # считает кумулятивные суммы, а не произведения

        # --- рекуррентность ----------------------------------------------------
        # S: (B, H, d_k, d_v). Оси B и H — независимые экземпляры памяти.
        if self.chunk_size:
            o = kda_chunkwise(q, k, v, g, beta, chunk=self.chunk_size)
        else:
            o = kda_recurrent(q, k, v, g, beta)                    # эталон, медленно
        o = o.to(x.dtype)                                          # (B, H, T, d_v)

        # --- выход: RMSNorm (headwise) -> гейт -> W_o -------------------------
        o = self.o_norm(o)
        o = o.transpose(1, 2).reshape(B, T, H * D)                 # склеили головы
        return self.W_o(torch.sigmoid(self.W_g(x)) * o)
