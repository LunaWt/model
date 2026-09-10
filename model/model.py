"""Полная архитектура K3 в нашем масштабе (~200M total / ~40M active).

Что взято из отчёта (notes/k3_architecture.md, раздел 9 «что берём»):

  * гибрид 3 KDA : 1 Gated MLA, NoPE во всех MLA-слоях
  * Block Attention Residuals вместо обычного residual
  * Stable LatentMoE: routed-эксперты в латенте ℓ, RMSNorm перед W↑,
    SiTU-GLU вместо SwiGLU, Quantile Balancing вместо aux-loss
  * первый слой dense (без MoE)

⚠️ КОНФИГ — НЕ ИЗ СТАТЬИ. Числа в K3Config это наши решения по открытым
вопросам 3/4/5 из notes/k3_architecture.md, они помечены ниже. Их надо
пересматривать осознанно, а не считать «как в статье».

Запуск проверки форм:  uv run python model/model.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from model.kda_head import KDA

# =====================================================================
# Конфиг
# =====================================================================

@dataclass
class K3Config:
    """Конфиг F (принят 31 июл 2026): 225.7M total / 48.7M active.

    Как он получился. Всегда-активный «пол» при d_model=512 — это эмбеддинг
    (8.4M) + внимание + dense-слой. У конфига A внимание съедало 21.1M из-за
    4 голов × 128, и активных выходило 82.4M — MoE физически не мог быть
    разреженным. F режет внимание вдвое (2 головы), а освободившийся бюджет
    отдаёт числу экспертов: 64 вместо 32 при top-4 вместо top-8, то есть
    sparsity 16 вместо 4 — ближе к режиму K3 (56), где QB вообще имеет смысл.
    """

    # --- базовое ---
    vocab_size: int = 16384          # наш токенизатор
    d_model: int = 512
    n_layers: int = 16               # 12 KDA + 4 MLA при паттерне 3:1
    max_seq_len: int = 1536          # упирается в 6 ГБ: ~1.5 МиБ активаций на
                                     # токен контекста, 2048 не влезает

    # --- внимание ---
    n_heads: int = 2
    d_head: int = 128                # d_k = d_v; в статье 128 во всех экспериментах
    kda_conv_kernel: int = 4
    kda_g_min: float = -5.0
    attn_pattern: int = 4            # каждый attn_pattern-й слой — MLA, остальные KDA

    # --- MLA: размеры латентов (DeepSeek-V2, без RoPE-части благодаря NoPE) ---
    mla_kv_latent: int = 64          # масштабируется вместе с n_heads·d_head
    mla_q_latent: int = 96

    # --- MoE ---
    # sparsity = 64/4 = 16. У K3 — 896/16 = 56; наш режим реже, чем у A
    # (было 4), но плотнее статьи. Ни QB, ни латентные эксперты на такой
    # плотности никем не проверялись — это ровно то, что мы хотим измерить.
    moe_latent: int = 256            # ℓ = 0.5 · d_model, как в K3
    n_routed: int = 64
    n_shared: int = 2                # K3 фиксирует N_s = 2
    top_k: int = 4
    expert_hidden: int = 256         # скрытая ширина routed-эксперта (внутри ℓ)
    # Запас «полки» эксперта — компромисс, а не оптимум. Замер 2 сен на
    # чекпоинте A16/550 (scripts/router_load.py, 8 батчей):
    #     cf     1.25   1.50   2.00   2.50   3.00
    #     drop   .113   .068   .030   .017   .010    <- доля выброшенных слотов
    #     ток/с  1172    971    722     —      —     <- цена по скорости
    # Ёмкость входит в размер буфера экспертов линейно, поэтому bmm дорожает
    # ровно во столько же раз: 1.25 -> 2.0 это −38% скорости за +8% доставленных
    # слотов. На этой карте так себе сделка, отсюда 1.5.
    # Откуда перекос, если QB его выравнивает: QB держит СРЕДНЮЮ нагрузку
    # (пик/идеал по всем батчам сразу 1.08), а ёмкость проверяется на КАЖДОМ
    # микро-батче, где пик/идеал 2.05. Разница — цена того, что при B=1 микро-батч
    # это один кусок одного документа одной группы, а эксперты специализируются
    # по домену. Настоящее лекарство — больше независимых документов в forward,
    # а не запас ёмкости.
    capacity_factor: float = 1.5
    qb_iters: int = 4                # итераций чередующегося решателя QB на шаг
                                     # (Algorithm 1, p. 44)
    shared_hidden: int = 256         # скрытая ширина shared-эксперта (при d_model)
    dense_hidden: int = 1024         # FFN первого (dense) слоя

    # --- SiTU-GLU (Eq. 12, Appendix B) ---
    situ_beta_gate: float = 4.0
    situ_beta_up: float = 25.0

    # --- AttnRes ---
    # ⚠️ открытый вопрос 5: N≈8 — оптимум K3 при 93 слоях (12 слоёв на блок).
    # У нас 16 слоёв, 8 блоков дали бы по 2 слоя — сокращать почти нечего.
    n_blocks: int = 4

    # --- looped (notes/looped.md) ---
    # loop_span=(a, b) — слои a..b включительно исполняются loop_r раз подряд
    # теми же весами. None -> обычный стек.
    loop_span: tuple[int, int] | None = None
    loop_r: int = 1
    # SMELT шаг 5: вклад зациклённых подслоёв в частичную сумму блока делится
    # на r, иначе одни и те же веса пишут в неё r раз в согласованную сторону.
    loop_res_scale: bool = True

    kda_backend: str = "auto"

    def execution_order(self) -> list[int]:
        if self.loop_span is None or self.loop_r == 1:
            return list(range(self.n_layers))
        a, b = self.loop_span
        return (list(range(a)) + list(range(a, b + 1)) * self.loop_r
                + list(range(b + 1, self.n_layers)))

    def __post_init__(self):
        if self.loop_span is not None:
            a, b = self.loop_span
            assert 0 <= a <= b < self.n_layers, f"loop_span={self.loop_span} вне [0, {self.n_layers})"
            assert self.loop_r >= 1
        n_exec = len(self.execution_order())
        assert n_exec % self.n_blocks == 0, (
            f"исполнений {n_exec} не делится на n_blocks={self.n_blocks}; "
            "блоки AttnRes режутся по исполнениям, а не по слоям"
        )


# =====================================================================
# Кирпичи
# =====================================================================

class SiTUGLU(nn.Module):
    """SiTU-GLU (K3 Eq. 12) — SwiGLU с мягким ограничением сверху.

        softcap(x, β) = β · tanh(x / β)

        SiTU-GLU(x) = [ β₁·tanh(W_g x / β₁) ⊙ Sigmoid(W_g x) ]
                      ⊙ [ β₂·tanh(W_u x / β₂) ]

    У SwiGLU обе ветки не ограничены: совпавшие большие координаты дают
    выбросы активаций. Мягкая крышка ограничивает выход (‖·‖∞ ≤ β₁β₂ = 100),
    но, в отличие от жёсткого clamp, НЕ обнуляет градиент за границей.
    Возле нуля β·tanh(z/β) = z + O(z³/β²), то есть в первом порядке
    совпадает со SwiGLU; при β→∞ переходит в него точно.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 beta_gate: float = 4.0, beta_up: float = 25.0):
        super().__init__()
        self.b1, self.b2 = beta_gate, beta_up
        self.W_g = nn.Linear(d_in, d_hidden, bias=False)
        self.W_u = nn.Linear(d_in, d_hidden, bias=False)
        self.W_d = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.W_g(x)
        gate = self.b1 * torch.tanh(g / self.b1) * torch.sigmoid(g)
        up = self.b2 * torch.tanh(self.W_u(x) / self.b2)
        return self.W_d(gate * up)


class GroupedSiTUGLU(nn.Module):
    """n экспертов одинаковой формы, посчитанных ТРЕМЯ вызовами bmm.

    Наивно эксперты — это список модулей и цикл по ним. Арифметика там
    копеечная (матрицы вида 96×256 @ 256×256), но каждый вызов это запуск
    ядра, а на WDDM-карте запуск стоит на порядок дороже обычного. Замер:
    64 отдельных matmul — 10.35 мс, один bmm на те же данные — 0.98 мс.

    torch.bmm перемножает СТОПКИ матриц: (n, a, b) @ (n, b, c) -> (n, a, c),
    n независимых умножений одним ядром. Поэтому веса всех экспертов лежат
    одним параметром с ведущей осью n, а не n отдельными Linear.
    """

    def __init__(self, n_experts: int, d_in: int, d_hidden: int, d_out: int,
                 beta_gate: float = 4.0, beta_up: float = 25.0):
        super().__init__()
        self.n_experts = n_experts
        self.b1, self.b2 = beta_gate, beta_up
        # Инициализируем здесь: _init_weights модели ходит по nn.Linear /
        # nn.Embedding и трёхмерные Parameter не увидит.
        mk = lambda a, b: nn.Parameter(torch.randn(n_experts, a, b) * 0.02)
        self.W_g = mk(d_in, d_hidden)
        self.W_u = mk(d_in, d_hidden)
        self.W_d = mk(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, cap, d_in) -> (n, cap, d_out)
        g = torch.bmm(x, self.W_g)
        gate = self.b1 * torch.tanh(g / self.b1) * torch.sigmoid(g)
        up = self.b2 * torch.tanh(torch.bmm(x, self.W_u) / self.b2)
        return torch.bmm(gate * up, self.W_d)


class AttnRes(nn.Module):
    """Один узел Attention Residuals (K3 Eq. 8–9).

    Обычный residual сжимает всё предыдущее в одно состояние — узкое горло,
    аналогичное RNN, только по глубине. AttnRes заменяет его на выбор:
    подслой сам решает, из каких предыдущих представлений собрать свой вход.

        φ(q, k)  = exp( q^T · RMSNorm(k) )
        α_{i→l}  = φ(q_l, k_i) / Σ_j φ(q_l, k_j)
        h_l      = Σ_i α_{i→l} · v_i          (k_i = v_i)

    q_l — обучаемый псевдо-запрос, один вектор на подслой (не зависит от
    токена). А вот ключи зависят, поэтому веса всё равно свои для каждой
    позиции. RMSNorm внутри ядра обязателен: без него слои с большой нормой
    выхода перетягивают внимание на себя.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # старт нулями -> все логиты равны -> softmax равномерный ->
        # h_l = среднее источников. Мягкий, ни на что не претендующий старт.
        self.q = nn.Parameter(torch.zeros(d_model))
        self.k_norm = nn.RMSNorm(d_model, elementwise_affine=False)

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        # sources: список (B, T, d) -> (B, T, M, d), M = число источников
        src = torch.stack(sources, dim=2)
        logits = torch.einsum("btmd,d->btm", self.k_norm(src), self.q)
        w = logits.softmax(dim=-1)
        return torch.einsum("btm,btmd->btd", w, src)


class GatedMLA(nn.Module):
    """Multi-head Latent Attention + NoPE + полноранговый выходной гейт.

    MLA (DeepSeek-V2): K и V не хранятся напрямую, а сжимаются в латент
    c_t = W_dkv x_t; кэшируется именно c_t, а полные K и V восстанавливаются
    up-проекциями в момент внимания. KV-кэш ужимается в d_model/d_c раз.

    NoPE: позиционных кодировок нет вообще — за позицию отвечают KDA-слои,
    их затухание само по себе позиционно. Это заодно выкидывает всю возню с
    RoPE-базой при удлинении контекста, и убирает «decoupled RoPE»-часть
    головы, которая в оригинальном MLA нужна была только ради RoPE.

        y_t = W_o [ Sigmoid(W_g x_t) ⊙ õ_t ]      (Eq. 7; RMSNorm тут НЕТ,
                                                   в отличие от KDA)
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 kv_latent: int, q_latent: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        inner = n_heads * d_head

        self.W_dq = nn.Linear(d_model, q_latent, bias=False)
        self.q_norm = nn.RMSNorm(q_latent)
        self.W_uq = nn.Linear(q_latent, inner, bias=False)

        self.W_dkv = nn.Linear(d_model, kv_latent, bias=False)
        self.kv_norm = nn.RMSNorm(kv_latent)
        self.W_uk = nn.Linear(kv_latent, inner, bias=False)
        self.W_uv = nn.Linear(kv_latent, inner, bias=False)

        self.W_g = nn.Linear(d_model, inner, bias=False)
        self.W_o = nn.Linear(inner, d_model, bias=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self._heads(self.W_uq(self.q_norm(self.W_dq(x))))
        c = self.kv_norm(self.W_dkv(x))          # (B, T, kv_latent) — это и кэшируется
        if cache is not None:
            past = cache.get("c")
            c = c if past is None else torch.cat([past, c], dim=1)
            cache["c"] = c
        k = self._heads(self.W_uk(c))
        v = self._heads(self.W_uv(c))
        # причинная маска SDPA выравнивается по левому верхнему углу, поэтому при
        # разной длине q и k она означала бы не то. Два поддерживаемых режима:
        # префилл (длины равны, маска нужна) и декодирование по одному токену
        # (q длины 1 смотрит на весь кэш, маска не нужна).
        is_causal = c.shape[1] == T
        assert is_causal or T == 1, f"частичный префилл не поддержан: T={T}, кэш={c.shape[1]}"

        # SDPA сам применит масштаб 1/sqrt(d_head) и причинную маску.
        #
        # Считаем в fp32, а не в bf16 — вопреки тому, что делаем везде ещё.
        # Причина: на sm_75 у SDPA в bf16 ЕДИНСТВЕННЫЙ доступный бэкенд — MATH,
        # то есть наивное внимание, материализующее матрицу T×T целиком.
        # Cutlass-ядро memory-efficient (та же идея, что у FlashAttention: не
        # держать T×T в памяти) на Turing собрано, но только под fp16/fp32 —
        # bf16 в нём требует sm_80. Замерено на B=1 H=2 T=1536 d_head=128,
        # fwd+bwd:
        #     MATH    bf16   4.12 мс   пик 95.0 MiB
        #     mem_eff fp32   2.15 мс   пик 27.5 MiB   <- 1.9x и 3.5x
        #     mem_eff fp16   5.03 мс             (fp16 на этой карте медленный)
        # То есть fp32 здесь не «дороже и точнее», а прямо дешевле обоих.
        with torch.autocast(x.device.type, enabled=False):
            o = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=is_causal
            )
        o = o.to(x.dtype).transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.W_o(torch.sigmoid(self.W_g(x)) * o)


class LatentMoE(nn.Module):
    """Stable LatentMoE (K3 Eq. 11 + §2.3.1 + Quantile Balancing).

        u = Σ_{i ∈ TopK}  p_i · E_i^routed( W↓ x )        W↓: d -> ℓ
        y = Σ_j E_j^shared(x)  +  W↑ RMSNorm(u)           W↑: ℓ -> d

    Идея латента: ширина модели и ширина routed-эксперта развязаны. Эксперты
    работают в узком ℓ, поэтому их можно держать много и дёшево, а общая
    ширина d при этом не страдает.

    RMSNorm между агрегацией и W↑ (Fix 1) — масштаб u зависит от того, какие
    эксперты выбрались и с какими весами, и без нормировки плывёт от токена
    к токену. Кроме стабильности это просто улучшает loss.
    """

    def __init__(self, cfg: K3Config):
        super().__init__()
        self.cfg = cfg
        d, ell = cfg.d_model, cfg.moe_latent

        self.W_down = nn.Linear(d, ell, bias=False)
        self.experts = GroupedSiTUGLU(cfg.n_routed, ell, cfg.expert_hidden, ell,
                                      cfg.situ_beta_gate, cfg.situ_beta_up)
        self.u_norm = nn.RMSNorm(ell)
        self.W_up = nn.Linear(ell, d, bias=False)

        self.shared = nn.ModuleList([
            SiTUGLU(d, cfg.shared_hidden, d, cfg.situ_beta_gate, cfg.situ_beta_up)
            for _ in range(cfg.n_shared)
        ])

        self.W_router = nn.Linear(d, cfg.n_routed, bias=False)
        # bias для балансировки: НЕ параметр (градиента не получает),
        # а буфер — считается правилом QB и едет в чекпоинте
        self.register_buffer("qb_bias", torch.zeros(cfg.n_routed))
        # скоры роутера, накопленные за микро-батчи текущего шага оптимизатора
        self._scores: list[torch.Tensor] = []
        # скоры текущего forward, по одному входу на визит: ключ — номер визита.
        # Именно СЛОВАРЬ, а не список: при gradient checkpointing forward
        # прогоняется второй раз в backward, и список бы удвоил выборку, а
        # перезапись по ключу идемпотентна.
        self._visit_scores: dict[int, torch.Tensor] = {}
        # ёмкость по факту, а не по формуле — только для генерации, см.
        # K3Model.set_full_capacity. По умолчанию ВЫКЛЮЧЕНО: путь обучения и
        # замера val должен остаться ровно тем же.
        self.full_capacity = False

    # ------------------------------------------------------------------
    def harvest_scores(self) -> None:
        """Перенести скоры текущего forward в накопитель шага оптимизатора.

        Зовётся из цикла обучения ПОСЛЕ backward каждого микро-батча (см.
        `K3Model.harvest_router_scores`). Разделение forward и накопления —
        чтобы пересчёт под gradient checkpointing не удваивал выборку.
        """
        if self._visit_scores:
            self._scores.append(torch.cat(list(self._visit_scores.values()), dim=0))
            self._visit_scores.clear()

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """Quantile Balancing, Eq. 14. Вызывается ИЗ ЦИКЛА ОБУЧЕНИЯ, раз на шаг
        оптимизатора — не из forward.

        Смысл: выставить каждому эксперту такой bias, при котором ровно доля
        k/n токенов проходит его порог. DeepSeek двигал bias фиксированным шагом
        по знаку ошибки (SignSGD по двойственной задаче); QB прыгает сразу в
        точный минимум — у него нет шага обучения и он выравнивается за
        считанные шаги даже при тысяче экспертов.

        Почему НЕ внутри forward, хотя так короче:
          * `m` в Eq. 14 — это батч шага оптимизатора. При градиентной
            аккумуляции обновление из forward сработало бы accum_steps раз,
            каждый по более шумному квантилю;
          * forward, меняющий состояние, ломает любой пересчёт — от
            gradient checkpointing до простого повторного прогона: маршрутизация
            во втором проходе получится другая.
        Причинность сохраняется: bias, посчитанный по этому батчу, действует
        только со следующего шага. На инференсе заморожен (буфер не трогаем).

        Это ЧЕРЕДУЮЩИЙСЯ решатель (Algorithm 1, p. 44), а не одна формула: порог
        α зависит от b, а b — от α, и Eq. 14 это один шаг покоординатной
        минимизации двойственной задачи. Замер 2 сен на статичной матрице скоров
        (512 токенов, 32 эксперта, top-4, перекос 2.0): без биаса переполнение
        0.454, после 1 итерации 0.0063, после 2 — ровно 0. Так что итерации
        стоят почти ничего и доводят решение до точного, но ⚠️ **не они лечат
        перекос, который виден в обучении** — см. `capacity_factor`.
        """
        if not self._scores:
            return
        s = torch.cat(self._scores, dim=0)
        self._scores.clear()

        k, n = self.cfg.top_k, self.cfg.n_routed
        b = self.qb_bias
        for _ in range(self.cfg.qb_iters):
            # порог α_i: (k+1)-й по величине БИАСОВАННЫЙ скор токена i.
            # Эксперт входит в Top-k токена i ровно если s_ij + b_j > α_i.
            alpha = torch.topk(s + b, k + 1, dim=-1).values[:, -1:]          # (m, 1)
            b = -torch.quantile(s - alpha, 1.0 - k / n, dim=0)               # (n,)
            # общий сдвиг Top-k не меняет -> убираем, чтобы bias не уплывал
            b = b - b.mean()
        self.qb_bias.copy_(b)

    def forward(self, x: torch.Tensor, visit: int = 0) -> torch.Tensor:
        B, T, d = x.shape
        m, k, n = B * T, self.cfg.top_k, self.cfg.n_routed
        flat = x.reshape(m, d)

        # --- маршрутизация (Eq. 13) ---
        # Роутер считается в fp32 ПРИНУДИТЕЛЬНО, даже под autocast. Замерено:
        # в bf16 у 95 из 1536 токенов (6%) получается другой top-k, чем в fp32.
        # Причина не в матмуле, а в том, что решение здесь ДИСКРЕТНОЕ: при
        # инициализации все сигмоидные скоры лежат около 0.5 и различаются в
        # третьем-четвёртом знаке, а у bf16 знаков всего ~3. Округление
        # переворачивает порядок — и токен уходит к другим экспертам целиком.
        # Матрица тут крошечная (d × n_routed), так что fp32 бесплатен.
        # DeepSeek и Megatron делают ровно это.
        with torch.autocast(flat.device.type, enabled=False):
            s = torch.sigmoid(F.linear(flat.float(), self.W_router.weight.float()))
        idx = torch.topk(s + self.qb_bias, k, dim=-1).indices  # (m, k)
        gathered = s.gather(-1, idx)                          # bias сюда НЕ входит
        p = gathered / gathered.sum(-1, keepdim=True)

        # --- диспетчеризация: раскладка по «полкам» фиксированной ёмкости ---
        # Каждый токен идёт к k экспертам, значит всего m*k «слотов». Сортируем
        # слоты по номеру эксперта и укладываем в буфер (n, cap, ℓ) — по полке
        # на эксперта. Дальше все эксперты считаются одним bmm.
        #
        # Размеры буфера фиксированы заранее, а не выводятся из данных. Цена —
        # cap выбирается с запасом, и если эксперт переполнился, лишние токены к
        # нему НЕ ПОПАДАЮТ (их вес зануляется). Это стандартный token dropping
        # из MoE-практики; QB как раз и держит нагрузку ровной, так что при
        # сошедшемся балансе переполнений почти нет.
        slot_expert = idx.reshape(-1)                          # (m*k,)
        slot_weight = p.reshape(-1)
        order = torch.argsort(slot_expert)                     # группируем по эксперту
        sorted_expert = slot_expert[order]
        slot_token = (order // k)[:]                           # слот j -> токен j // k
        src_tok = slot_token

        # Счётчик слотов на эксперта. Это scatter_add_, а НЕ torch.bincount:
        # у bincount размер выхода в принципе зависит от данных, поэтому он
        # (а) синхронизирует GPU с CPU, (б) заставляет Dynamo рвать граф —
        # замерено, один bincount на слой давал 8 разрывов на модель и убивал
        # torch.compile. Здесь длина известна (n), и всё остаётся на устройстве.
        counts = torch.zeros(n, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, slot_expert, torch.ones_like(slot_expert))
        starts = torch.cumsum(counts, 0) - counts              # начало полки каждого эксперта
        pos = torch.arange(m * k, device=x.device) - starts[sorted_expert]   # место внутри полки
        if self.full_capacity:
            cap = max(1, int(counts.max().item()))
        else:
            cap = max(1, int(m * k / n * self.cfg.capacity_factor))
        overflow = pos >= cap
        # переполненные слоты сваливаем в служебную строку cap — её выход
        # никуда не пойдёт, потому что вес обнулён
        pos = torch.where(overflow, torch.full_like(pos, cap), pos)

        h = self.W_down(flat)                                  # (m, ℓ)
        buf = h.new_zeros(n, cap + 1, h.shape[-1])
        buf[sorted_expert, pos] = h[src_tok]
        out = self.experts(buf)                                # (n, cap+1, ℓ), один bmm-путь
        vals = out[sorted_expert, pos]

        w = slot_weight[order] * (~overflow)                   # вес выброшенных слотов = 0
        # Под autocast выход экспертов bf16, а веса маршрутизации остаются fp32
        # (sigmoid и деление autocast не трогает), поэтому произведение — fp32.
        # Копим сумму в этом же типе: смесь из top_k слагаемых с весами ~0.25
        # — ровно то место, где не хочется терять разряды.
        vals = vals * w.unsqueeze(-1)
        u = torch.zeros(m, h.shape[-1], dtype=vals.dtype, device=h.device)
        u.index_add_(0, src_tok, vals)                         # раскладываем обратно по токенам
        self.dropped = overflow.sum()                          # для мониторинга, без синхронизации

        y = self.W_up(self.u_norm(u))
        for exp in self.shared:
            y = y + exp(flat)

        if self.training:
            self._visit_scores[visit] = s.detach().float()

        return y.view(B, T, d)


# =====================================================================
# Модель
# =====================================================================

class K3Model(nn.Module):
    """Стек подслоёв, связанных Block Attention Residuals.

    Обычный трансформер:  h ← h + f(h)  на каждом подслое.
    Здесь вместо этого:

      * L слоёв делятся на N блоков по S = L/N слоёв;
      * ВНУТРИ блока выходы подслоёв просто СУММИРУЮТСЯ в b_n
        (b_n^i — частичная сумма по первым i подслоям блока);
      * МЕЖДУ блоками — полноценное внимание по N представлениям блоков;
      * b_0 — эмбеддинг, он доступен как источник всегда.

    Память падает с O(L·d) до O(N·d): живыми держим N представлений блоков,
    а не выход каждого слоя.

    Источники для подслоя i блока n (Eq. 10):
        i = 1 : [b_0, …, b_{n−1}]
        i ≥ 2 : [b_0, …, b_{n−1}, b_n^{i−1}]
    """

    def __init__(self, cfg: K3Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.attn = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.attn_norm = nn.ModuleList()
        self.ffn_norm = nn.ModuleList()
        self.res_attn = nn.ModuleList()
        self.res_ffn = nn.ModuleList()

        for layer in range(cfg.n_layers):
            # 3 KDA : 1 MLA. Последний слой стека при n_layers % pattern == 0
            # оказывается MLA — как в K3, где backbone заканчивается глобальным
            # вниманием.
            is_mla = (layer + 1) % cfg.attn_pattern == 0
            if is_mla:
                self.attn.append(GatedMLA(cfg.d_model, cfg.n_heads, cfg.d_head,
                                          cfg.mla_kv_latent, cfg.mla_q_latent))
            else:
                self.attn.append(KDA(cfg.d_model, cfg.n_heads, cfg.d_head,
                                     cfg.kda_conv_kernel, cfg.kda_g_min,
                                     backend=cfg.kda_backend))

            # первый слой dense — «for stable training», повторено и в K3,
            # и в Kimi Linear
            if layer == 0:
                self.ffn.append(SiTUGLU(cfg.d_model, cfg.dense_hidden, cfg.d_model,
                                        cfg.situ_beta_gate, cfg.situ_beta_up))
            else:
                self.ffn.append(LatentMoE(cfg))

            # pre-norm на входе каждого подслоя
            self.attn_norm.append(nn.RMSNorm(cfg.d_model))
            self.ffn_norm.append(nn.RMSNorm(cfg.d_model))
            self.res_attn.append(AttnRes(cfg.d_model))
            self.res_ffn.append(AttnRes(cfg.d_model))

        # финальный узел агрегирует все N представлений блоков + эмбеддинг
        self.res_out = AttnRes(cfg.d_model)
        self.out_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight        # tied embeddings

        self.execution_order = cfg.execution_order()
        span = cfg.loop_span
        self.looped_layers = frozenset(range(span[0], span[1] + 1)) if span else frozenset()
        self.loop_scale = 1.0 / cfg.loop_r if (span and cfg.loop_res_scale) else 1.0

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Инициализация GPT-2-стиля: N(0, 0.02) на все матрицы.

        По умолчанию nn.Embedding даёт N(0, 1). При связанных эмбеддингах
        (lm_head.weight — та же матрица) и RMSNorm перед головой это даёт
        логиты порядка ±‖h‖·‖e‖ ≈ ±100 и стартовый loss в районе 85 при
        ln(V)=5.5 — то есть модель стартует хуже равномерного угадывания.
        Трогаем только веса Linear/Embedding: b_α, A_h и псевдо-запросы
        AttnRes инициализируются осмысленно у себя и здесь не участвуют.
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def body(self, tokens: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        """Стек без lm_head: (B, T) -> (B, T, d_model) после out_norm.

        Отделено от `forward`, потому что chunked_cross_entropy сама умножает на
        lm_head и логиты (B, T, V) материализовать не надо.

        `cache` — состояние генерации, словарь по НОМЕРУ ИСПОЛНЕНИЯ (позиции в
        `execution_order`), а не по номеру слоя: у зациклённого слоя два прохода
        стоят в разных местах стека и памяти внимания у них разные.
        """
        h0 = self.embed(tokens)
        blocks: list[torch.Tensor] = [h0]      # b_0, потом b_1 … b_N
        partial: torch.Tensor | None = None    # b_n^i, частичная сумма блока
        order = self.execution_order
        per_block = len(order) // self.cfg.n_blocks
        visits: dict[int, int] = {}

        for step, layer in enumerate(order):
            if step % per_block == 0:
                partial = None                 # начался новый блок

            visit = visits.get(layer, 0)
            visits[layer] = visit + 1
            scale = self.loop_scale if layer in self.looped_layers else 1.0

            for is_attn, res, norm, sub in (
                (True, self.res_attn[layer], self.attn_norm[layer], self.attn[layer]),
                (False, self.res_ffn[layer], self.ffn_norm[layer], self.ffn[layer]),
            ):
                sources = blocks if partial is None else blocks + [partial]
                h = res(sources)
                if isinstance(sub, LatentMoE):
                    out = sub(norm(h), visit=visit)
                elif is_attn and cache is not None:
                    out = sub(norm(h), cache=cache.setdefault(step, {}))
                else:
                    out = sub(norm(h))
                out = out * scale if scale != 1.0 else out
                partial = out if partial is None else partial + out

            if (step + 1) % per_block == 0:
                blocks.append(partial)         # блок закрыт
                partial = None

        return self.out_norm(self.res_out(blocks))

    def forward(self, tokens: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        # tokens: (B, T) int64 -> логиты (B, T, vocab)
        return self.lm_head(self.body(tokens, cache))

    def set_full_capacity(self, flag: bool) -> None:
        """Снять/вернуть ограничение ёмкости экспертов. Для генерации — снять.

        ⚠️ При фиксированной ёмкости модель НЕ причинна по токенам. Полка
        эксперта размера `cap = m·k/n · capacity_factor` считается от числа
        токенов В ЭТОМ forward, и лишние слоты выбрасываются — значит выход
        токена зависит от того, какие ещё токены прогоняются рядом. При обучении
        это стандартная практика MoE и цена за фиксированные формы, но при
        генерации это означает, что префилл длины T и пошаговое декодирование
        считают РАЗНОЕ. Замер 2 сен на `tiny`: расхождение логитов 0.38 при
        одинаковых весах и входе; со снятой ёмкостью — 5e-7, то есть уровень
        порядка суммирования.

        Снятая ёмкость стоит одной синхронизации с хостом на слой (`counts.max()`),
        поэтому в обучении её включать нельзя.
        """
        for ffn in self.ffn:
            if isinstance(ffn, LatentMoE):
                ffn.full_capacity = flag

    def harvest_router_scores(self) -> None:
        """Собрать скоры роутера после backward микро-батча (см. LatentMoE.harvest_scores)."""
        for ffn in self.ffn:
            if isinstance(ffn, LatentMoE):
                ffn.harvest_scores()

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """Обновить QB-bias во всех MoE-слоях. Звать из цикла обучения ПОСЛЕ
        optimizer.step(), один раз на шаг:

            loss.backward(); model.harvest_router_scores()
            opt.step(); opt.zero_grad()
            model.update_router_bias()
        """
        for ffn in self.ffn:
            if isinstance(ffn, LatentMoE):
                ffn.update_router_bias()

    # ------------------------------------------------------------------
    def count_params(self) -> dict[str, int]:
        """Всего / активных / исполняемых на токен.

        `active` — различные параметры, участвующие в одном токене: routed-эксперты
        считаются top_k из n_routed. У зациклённой модели этого мало: одни и те же
        веса работают r раз, и прокси FLOPs — именно `executed`, сумма по порядку
        исполнения, где повторный слой считается заново.
        """
        total = sum(p.numel() for p in self.parameters())

        def layer_active(i: int) -> int:
            n = sum(p.numel() for p in self.attn[i].parameters())
            ffn = self.ffn[i]
            n += sum(p.numel() for p in ffn.parameters())
            if isinstance(ffn, LatentMoE):
                per_expert = sum(p.numel() for p in ffn.experts.parameters()) // self.cfg.n_routed
                n -= per_expert * (self.cfg.n_routed - self.cfg.top_k)
            for mod in (self.attn_norm[i], self.ffn_norm[i], self.res_attn[i], self.res_ffn[i]):
                n += sum(p.numel() for p in mod.parameters())
            return n

        per_layer = [layer_active(i) for i in range(self.cfg.n_layers)]
        floor = self.embed.weight.numel() + sum(p.numel() for p in self.res_out.parameters()) \
            + sum(p.numel() for p in self.out_norm.parameters())
        return {
            "total": total,
            "active": floor + sum(per_layer),
            "executed": floor + sum(per_layer[i] for i in self.execution_order),
        }


def param_groups(model: nn.Module, weight_decay: float = 0.1) -> list[dict]:
    """Две группы для будущего оптимизатора: с weight decay и без.

    Рецепт K3 — wd = 0.1 на всё, но decay применяют только к матрицам. Без
    decay идут:
      * всё одномерное — веса RMSNorm, смещения, псевдо-запросы AttnRes;
      * всё, помеченное `._no_weight_decay` — `b_α` и `A` внутри KDA.

    Почему это не косметика: decay тянет `b_α` к нулю, а при `b_α = 0`
    получается α = e^(−2.5) = 0.082 — полураспад 0.28 токена. Рекуррентное
    состояние стиралось бы каждый шаг, молча и без единой ошибки.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if getattr(p, "_no_weight_decay", False) or p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = K3Config()
    m = K3Model(cfg)
    n = m.count_params()
    print(f"total    {n['total'] / 1e6:7.1f}M")
    print(f"active   {n['active'] / 1e6:7.1f}M")
    print(f"executed {n['executed'] / 1e6:7.1f}M")

    tok = torch.randint(0, cfg.vocab_size, (2, 64))
    out = m(tok)
    print("вход ", tuple(tok.shape), "-> выход", tuple(out.shape))
