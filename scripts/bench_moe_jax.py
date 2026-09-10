"""Сетка замеров MoE на TPU v5e-8: что влезает и что из этого быстро.

    uv run python -m scripts.kaggle_run scripts/bench_moe_jax.py \
        --name moe-tpu --accelerator tpuV5e8 --shard

Меряется полный шаг обучения вместе с оптимизатором. Без оптимизатора потолок по
памяти бессмысленен: веса — меньше половины того, что лежит в HBM. На параметр
уходит 16 байт с Adam (веса fp32 4, градиент 4, два момента по 4) и 8 байт с
Adafactor, у которого второй момент факторизован в две строки вместо матрицы.

Шардинг — ZeRO-3 по восьми чипам: каждый параметр режется по нулевой оси, то есть
эксперты (E, d, h) расходятся по чипам целиком, а плотные матрицы — построчно.
Состояние оптимизатора наследует ту же раскладку. На чип приходится 16N/8 = 2N
байт с Adam и 1N с Adafactor, и именно это делает 8 млрд параметров обсуждаемыми
на 8 x 16 ГБ. Эмбеддинг оставлен продублированным: он маленький, а шардить его по
словарю — значит собирать строки коллективом на каждой выборке токена.

Две правки, без которых замер меряет не модель:

  * внимание блоками запросов (`attend_chunked`) — иначе матрица T x T живёт
    целиком на каждый слой и упирается память, а не арифметика;
  * кросс-энтропия кусками (`chunked_ce`) — иначе логиты (B, T, 32768) в fp32
    просят гигабайты одним куском. Ровно на этом упал прогон 520M на T4.

Флопы считаем по активным параметрам: 6 · активные · токены. Столбец сравним с
замерами на T4 и на GTX 1660 Ti. Он занижен там, где часть работы уходит впустую
(набивка буфера ёмкости) и не учитывает внимание вовсе — сравнивать строки между
собой можно, называть это утилизацией железа нельзя.

Оси перебора, каждая списком: число экспертов, раскладка токенов, шардинг, длина
контекста, реализация внимания, длина окна, оптимизатор. Каждая строка сетки —
отдельная компиляция и замер, ошибка одной строки не роняет прогон.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from functools import partial

if "--pip" in sys.argv:
    import subprocess
    _specs = sys.argv[sys.argv.index("--pip") + 1].split(",")
    _r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *_specs],
                        capture_output=True, text=True)
    print(f"pip {_specs} код {_r.returncode}: {(_r.stdout + _r.stderr).strip()[-400:]}",
          flush=True)
"""Образ Kaggle TPU приезжает с jax 0.10.2 и libtpu, собранным 12 июня 2025. Любое
ядро Pallas на нём отказывается запускаться («requires a libtpu version that's at
most a month old»), а у `ragged_dot` не работает обратный проход. `pip install -U
jax[tpu]` тянет согласованную свежую тройку jax+jaxlib+libtpu, и обе дыры
закрываются — проверено `scripts/tpu_probe.py`, 15 проверок из 15."""

_tpu = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--libtpu=")]
if _tpu:
    os.environ["LIBTPU_INIT_ARGS"] = _tpu[0]
_xla = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--xla=")]
if _xla:
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " " + " ".join(_xla)
"""`--xla=...` и `--libtpu=...` разбираются до импорта jax: обе переменные читаются
один раз при инициализации бэкенда, и внутри сегмента их уже не поменять. Поэтому
флаги компилятора перебираются отдельными запусками ядра, а не осью сетки."""
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")
"""XLA:GPU по умолчанию забирает 75% карты и на этом останавливается: на P100
замер видел 11.92 ГБ из 16. На TPU переменная не действует."""

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import random  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

DT = jnp.bfloat16
"""Тип счёта. На TPU и на Ampere+ это bfloat16; у T4 (sm_75) тензорных ядер под
bfloat16 нет вовсе, там нужен float16, иначе XLA считает эмуляцией."""

SPLASH_INTERPRET = False
"""`--splash-interpret` гоняет ядро Mosaic интерпретатором: на CPU проверяется вся
обвязка шага целиком (scan, remat, shard_map, cond), не занимая слот TPU. Медленно."""


def rms_norm(x, w):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 ** 2, axis=-1, keepdims=True) + 1e-6)
    return (y * w).astype(DT)


def attend_chunked(q, k, v, scale, block_len, window=0, f32_scores=False, seg=None):
    """Внимание блоками запросов; `window` > 0 — скользящее окно вместо полного.

    Окно снимает квадратичность и по арифметике, и по памяти: блоку запросов длиной
    `block_len` нужны только ключи из [i·block_len − window, (i+1)·block_len), то есть
    ровно `window + block_len` штук. Длина среза статическая, начало — `dynamic_slice`,
    так что форма графа от T не зависит и матрица логитов не растёт вместе с длиной.

    Очки остаются в типе счёта, а во float32 переводятся уже внутри softmax. Разница
    не в арифметике — она в обоих случаях fp32, — а в том, что выезжает в HBM: профиль
    `moe-prof` нашёл здесь `f32[2,4,4,256,2048]`, то есть 67 МБ на блок запросов на
    слой и 12.9 ГБ за проход, 12.3% шага. Масштаб теперь применяется к запросам до
    einsum, чтобы у выхода einsum был единственный потребитель и XLA мог слить его с
    softmax. `--attn-f32-scores` возвращает старое поведение для сравнения.

    `seg` — (B, T) номера документов. Токен видит только свой: без этого соседний
    несвязанный документ в окне работает как шум, и модель учится не дальней связи, а
    тому, что всё до последней границы можно игнорировать (arXiv 2402.13991).
    """
    B, T, H, D = q.shape
    G = k.shape[2]
    rep = H // G
    nb = T // block_len
    span = T if not window or window + block_len >= T else window + block_len
    q = (q * scale).astype(q.dtype)
    qb = q.reshape(B, nb, block_len, G, rep, D).transpose(1, 0, 2, 3, 4, 5)

    def stepf(_, xs):
        qi, i = xs
        qpos = i * block_len + jnp.arange(block_len)
        if span == T:
            ki, vi, start = k, v, 0
        else:
            start = jnp.clip((i + 1) * block_len - span, 0, T - span)
            ki = jax.lax.dynamic_slice_in_dim(k, start, span, 1)
            vi = jax.lax.dynamic_slice_in_dim(v, start, span, 1)
        kpos = start + jnp.arange(span)
        logits = jnp.einsum("bqgrd,bkgd->bgrqk", qi, ki)
        if f32_scores:
            logits = logits.astype(jnp.float32)
        keep = qpos[:, None] >= kpos[None, :]
        if window:
            keep &= qpos[:, None] - kpos[None, :] < window
        if seg is not None:
            sq = jax.lax.dynamic_slice_in_dim(seg, i * block_len, block_len, 1)
            sk = jax.lax.dynamic_slice_in_dim(seg, start, span, 1)
            keep = keep & (sq[:, :, None] == sk[:, None, :])[:, None, None]
        masked = jnp.where(keep, logits.astype(jnp.float32), -jnp.inf)
        a = jax.nn.softmax(masked, axis=-1).astype(q.dtype)
        return None, jnp.einsum("bgrqk,bkgd->bqgrd", a, vi)

    _, out = jax.lax.scan(jax.checkpoint(stepf), None, (qb, jnp.arange(nb)))
    return out.transpose(1, 0, 2, 3, 4, 5).reshape(B, T, H, D)


def attend_splash(q, k, v, scale, block_len, window=0, interpret=False, mesh=None):
    """Готовое Pallas-ядро TPU (`splash_attention`): маска живёт в описании, не в HBM.

    Отличие от `attend_chunked` не в маске, а в том, что softmax считается внутри
    ядра поблочно и матрица логитов вообще не выезжает в память. Масштаб ядру не
    передать, поэтому запросы масштабируются заранее. GQA разворачивается повтором
    ключей: у `make_splash_mha` число голов должно совпадать.

    Вызов обязан быть внутри `shard_map`. XLA:SPMD не умеет разрезать ядро Mosaic и
    отвечает «Mosaic kernels cannot be automatically partitioned» — на этом упали все
    splash-строки прогона `tpu-dense`, включая `replica`, где резать было нечего.
    Внутри `shard_map` каждому чипу достаётся свой кусок батча целиком, ядро видит
    обычные (B/чипов, T, H, D), а голов и позиций у него по-прежнему по одному шарду.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (  # noqa: PLC0415
        splash_attention_kernel as sk, splash_attention_mask as sm)

    B, T, H, D = q.shape
    one = sm.LocalMask((T, T), (window - 1, 0), 0) if window else sm.CausalMask((T, T))
    blk = min(block_len, T)
    kern = sk.make_splash_mha(
        sm.MultiHeadMask([one] * H), head_shards=1, q_seq_shards=1,
        interpret=interpret or SPLASH_INTERPRET,
        block_sizes=sk.BlockSizes(block_q=blk, block_kv=blk, block_kv_compute=blk,
                                  block_q_dkv=blk, block_kv_dkv=blk,
                                  block_kv_dkv_compute=blk,
                                  block_q_dq=blk, block_kv_dq=blk))
    hm = lambda a: a.transpose(0, 2, 1, 3)  # noqa: E731

    def call(q, k, v):
        rep = H // k.shape[2]
        return hm(jax.vmap(kern)(hm(q), hm(jnp.repeat(k, rep, axis=2)),
                                 hm(jnp.repeat(v, rep, axis=2))))

    q = (q * scale).astype(q.dtype)
    if mesh is None:
        return call(q, k, v)
    spec = P("x", None, None, None)
    return jax.shard_map(call, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec,
                         check_vma=False)(q, k, v)


def _ce_local(h, embed, targets, chunk, f32_logits):
    """Сумма NLL по строкам, кусками по `chunk`, без логитов на все строки сразу.

    Логиты остаются в типе счёта, а во float32 переводятся внутри редукции: разница
    не в точности (складывается всё равно fp32), а в том, что выезжает в HBM —
    (chunk, V) в bf16 вдвое меньше, чем в fp32.

    Взятый логит считается не выборкой из матрицы, а скалярным произведением строки
    на `embed[target]`. Математически то же самое, но у выхода матмула остаётся один
    потребитель, поэтому fp32-копия логитов не нужна вовсе. Профиль `moe-prof` мерил
    здесь два полных тензора (f32 и bf16) на кусок.
    """
    n, d = h.shape
    nc = max(1, n // chunk)
    c = n // nc

    def stepf(acc, xs):
        hi, ti = xs
        logits = hi @ embed.T
        if f32_logits:
            logits = logits.astype(jnp.float32)
        m = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
        z = jnp.sum(jnp.exp(logits.astype(jnp.float32) - m.astype(jnp.float32)), axis=-1)
        lse = m[:, 0].astype(jnp.float32) + jnp.log(z)
        picked = jnp.sum(hi.astype(jnp.float32) * embed[ti].astype(jnp.float32), axis=-1)
        return acc + jnp.sum(lse - picked), None

    total, _ = jax.lax.scan(jax.checkpoint(stepf),
                            jnp.float32(0),
                            (h[: nc * c].reshape(nc, c, d),
                             targets[: nc * c].reshape(nc, c)))
    return total, nc * c


def chunked_ce(h, embed, targets, chunk, mesh=None, shard=True, f32_logits=False):
    """Кросс-энтропия на локальных строках: каждый чип считает свои и складывает psum.

    Иначе `h` разрезано по строкам, `reshape(nc, chunk, d)` кладёт кусок целиком на
    один чип, а `lax.scan` требует его на всех — и XLA ставит `all-gather` всего
    скрытого состояния **внутрь тела цикла**. В профиле `moe-prof` это 105 из 718 мс:
    134 МБ пересобирались 32 раза за проход и четыре прохода за шаг. Лосс — сумма по
    строкам, и строки у каждого чипа свои, так что общаться нужно один раз в конце.
    """
    if mesh is None or not shard:
        total, cnt = _ce_local(h, embed, targets, chunk, f32_logits)
        return total / cnt

    chips = mesh.shape["x"]
    n_loc = h.shape[0] // chips
    nc = max(1, n_loc // chunk)
    fn = jax.shard_map(
        lambda a, b, t: jax.lax.psum(_ce_local(a, b, t, chunk, f32_logits)[0], "x"),
        mesh=mesh, in_specs=(P("x", None), P(), P("x")), out_specs=P(),
        check_vma=False)
    return fn(h, embed, targets) / (chips * nc * (n_loc // nc))


def _chunked_ce_legacy(h, embed, targets, chunk):
    """Прежняя версия, оставлена как контроль в сетке (`--ces scan`)."""
    n, d = h.shape
    nc = n // chunk
    hc = h[: nc * chunk].reshape(nc, chunk, d)
    tc = targets[: nc * chunk].reshape(nc, chunk)

    def stepf(acc, xs):
        hi, ti = xs
        logits = (hi @ embed.T).astype(jnp.float32)
        lse = jax.nn.logsumexp(logits, axis=-1)
        picked = jnp.take_along_axis(logits, ti[:, None], axis=-1)[:, 0]
        return acc + jnp.sum(lse - picked), None

    total, _ = jax.lax.scan(jax.checkpoint(stepf), jnp.float32(0), (hc, tc))
    return total / (nc * chunk)


def _slots_cumsum(idx, N, top_k, E):
    """Позиция токена внутри буфера эксперта через префиксную сумму по one-hot.

    В лоб: матрица (N·k, E) и `cumsum` по ней. Работы пропорционально N·k·E, то
    есть при 88 экспертах вчетверо больше, чем при 24.
    """
    onehot = jax.nn.one_hot(idx.reshape(-1), E, dtype=jnp.int32)
    pos = (jnp.cumsum(onehot, axis=0) - onehot)
    return (pos * onehot).sum(-1).reshape(N, top_k)


def _slots_sort(idx, N, top_k, E):
    """То же самое через сортировку: после неё позиция — это `arange` минус начало блока.

    Сортируется вектор длиной N·k, поэтому работа от E не зависит вовсе.
    `jnp.argsort` устойчива по умолчанию, значит порядок внутри эксперта — исходный,
    и позиции совпадают с `_slots_cumsum` до бита (это проверяет тест).
    """
    flat = idx.reshape(-1)
    order = jnp.argsort(flat)
    ordered = flat[order]
    starts = jnp.searchsorted(ordered, jnp.arange(E), side="left")
    rank = jnp.arange(flat.size) - starts[ordered]
    return jnp.zeros(flat.size, jnp.int32).at[order].set(rank).reshape(N, top_k)


QUANT_KEYS = ("w_gate", "w_in", "w_out")
"""Что можно хранить в int8. Только эксперты: они и есть вся масса модели, а
плотная часть, эмбеддинг и роутер вместе меньше их одной двадцатой."""


def dequant(p):
    """Разжать веса экспертов, если они лежат кодами.

    `+ p[k + "_g"]` — нулевой тензор, единственный смысл которого в том, что он
    дифференцируем. jax.grad не отдаёт градиент по int8-листу (касательный тип у
    целых — float0), а нам нужен ровно dL/dW полной формы, чтобы обновить коды.
    Прибавление нуля XLA выкидывает из прямого прохода, а в обратном на этом месте
    остаётся нужная котангента. Ноль едет внутрь `scan` вместе с весами слоя,
    поэтому распакованная копия живёт один слой, а не всю стопку.
    """
    if p["w_in"].dtype != jnp.int8:
        return p
    scale = lambda k: jax.lax.stop_gradient(p[k + "_s"]).astype(DT)  # noqa: E731
    return {**p, **{k: p[k].astype(DT) * scale(k) + p[k + "_g"] for k in QUANT_KEYS}}


def quant_keys(params):
    b = params.get("blocks", {})
    return [k for k in QUANT_KEYS if k in b and b[k].dtype == jnp.int8]


def _experts(p, buf):
    g = jnp.einsum("ecd,edh->ech", buf, p["w_gate"].astype(DT))
    u = jnp.einsum("ecd,edh->ech", buf, p["w_in"].astype(DT))
    return jnp.einsum("ech,ehd->ecd", jax.nn.silu(g) * u, p["w_out"].astype(DT))


def moe_ffn_ragged(p, x, top_k):
    """Без буфера ёмкости вообще: токены сортируются по эксперту и идут в `ragged_dot`.

    `jax.lax.ragged_dot(lhs, rhs, group_sizes)` умножает подряд идущие куски строк на
    свою матрицу каждый; на TPU это опускается в Pallas-ядро `megablox.gmm`. Отсюда три
    отличия от раскладки по ёмкости: не считается 25% арифметики на набивке, ни один
    токен не выбрасывается по переполнению, и вместо scatter в (E·cap+1, d) с последующим
    gather остаётся одна перестановка туда и одна обратно.
    """
    B, T, d = x.shape
    N = B * T
    E = p["w_in"].shape[0]
    xf = x.reshape(N, d)
    gate = jax.nn.softmax(xf.astype(jnp.float32) @ p["gate"], axis=-1)
    vals, idx = jax.lax.top_k(gate, top_k)

    flat = idx.reshape(-1)
    order = jnp.argsort(flat)
    sizes = jnp.bincount(flat, length=E)
    src = xf[order // top_k]

    rd = lambda a, w: jax.lax.ragged_dot(a, w.astype(DT), sizes)  # noqa: E731
    h = rd(jax.nn.silu(rd(src, p["w_gate"])) * rd(src, p["w_in"]), p["w_out"])

    out = jnp.zeros((flat.size, d), h.dtype).at[order].set(h)
    return (out.reshape(N, top_k, d) * vals[..., None].astype(DT)).sum(1).reshape(B, T, d)


def moe_ffn_ep(p, x, top_k, cap_factor, mesh, filler="scatter", chunks=1):
    """Экспертный параллелизм своими руками: эксперты стоят, ездят токены.

    Зачем. В автоматическом режиме XLA раскладывает буфер ёмкости сам, и по HLO видно
    что он выбирает: `all-gather` буфера (N·k, d) на каждый чип плюс `all-reduce`
    обратно, на каждый слой. Это ровно те 27% шага, которые вычитание `--dispatches none`
    показывало как «цену раскладки»: платится не арифметика индексов, а P-кратный
    перенос активаций.

    Здесь тот же обмен делается одним `all_to_all` в каждую сторону. Каждый чип
    маршрутизирует свои токены по ВСЕМ E экспертам, режет буфер на P кусков по
    назначению, обменивается — и получает по P·cap слотов на каждого своего эксперта.
    Переносится ровно (P−1)/P своего куска вместо P копий целого.

    Ёмкость считается от локального числа токенов, поэтому корзины мельче и разброс
    заполнения относительно больше: при том же `cap_factor` выбрасывается больше
    токенов, чем в глобальной раскладке. Это цена, а не ошибка.

    `chunks` > 1 режет буфер по оси ёмкости и делает обмен и арифметику по кускам.
    Одним куском перекрывать нечего: пока едет буфер, ядрам нечего считать, и профиль
    `moe-prof` мерил на этих двух обменах 104.8 мс из 718. Кусками между ними
    появляется независимость — обмен куска i+1 не ждёт матмулов куска i, — и планировщик
    XLA может их совместить. Результат обязан совпадать до бита: режется буфер, а не
    маршрутизация (это проверяет тест).
    """
    B, T, d = x.shape
    E = p["w_in"].shape[0]
    chips = mesh.shape["x"]
    if E % chips or B % chips:
        raise ValueError(f"экспертный параллелизм требует E % чипов == 0 и B % чипов == 0, "
                         f"а тут E={E}, B={B}, чипов={chips}")
    e_loc = E // chips
    n_loc = B * T // chips
    cap = max(1, int(n_loc * top_k / E * cap_factor))
    nch = chunks if cap % chunks == 0 else 1

    def local(xl, gate_w, wg, wi, wo):
        n = xl.shape[0] * xl.shape[1]
        xf = xl.reshape(n, d)
        gate = jax.nn.softmax(xf.astype(jnp.float32) @ gate_w, axis=-1)
        vals, idx = jax.lax.top_k(gate, top_k)
        pos = _slots_cumsum(idx, n, top_k, E)
        keep = pos < cap
        slot = jnp.where(keep, idx * cap + pos, E * cap)
        if filler == "gather":
            miss = n * top_k
            inv = jnp.full((E * cap,), miss, jnp.int32).at[slot.reshape(-1)].set(
                jnp.arange(miss), mode="drop")
            ok = inv < miss
            buf = jnp.where(ok[:, None], xf[jnp.where(ok, inv, 0) // top_k], 0)
        else:
            src = jnp.repeat(xf, top_k, axis=0)
            buf = jnp.zeros((E * cap + 1, d), xf.dtype).at[slot.reshape(-1)].set(
                src)[: E * cap]

        cs = cap // nch
        parts = []
        for c in range(nch):
            part = jax.lax.dynamic_slice_in_dim(
                buf.reshape(E, cap, d), c * cs, cs, 1).reshape(chips, e_loc * cs, d)
            recv = jax.lax.all_to_all(part, "x", 0, 0, tiled=True)
            mine = recv.reshape(chips, e_loc, cs, d).transpose(1, 0, 2, 3)
            mine = mine.reshape(e_loc, chips * cs, d)
            g = jnp.einsum("ecd,edh->ech", mine, wg.astype(DT))
            u = jnp.einsum("ecd,edh->ech", mine, wi.astype(DT))
            out = jnp.einsum("ech,ehd->ecd", jax.nn.silu(g) * u, wo.astype(DT))
            back = out.reshape(e_loc, chips, cs, d).transpose(1, 0, 2, 3)
            back = back.reshape(chips, e_loc * cs, d)
            parts.append(jax.lax.all_to_all(back, "x", 0, 0, tiled=True)
                         .reshape(chips, e_loc, cs, d))
        got = jnp.stack(parts, axis=2).reshape(E * cap, d)
        w = (vals * keep).astype(DT)
        if filler == "gather":
            picked = got[jnp.where(keep, slot, 0)]
        else:
            picked = jnp.concatenate([got, jnp.zeros((1, d), got.dtype)])[slot]
        return (picked * w[..., None]).sum(1).reshape(xl.shape)

    bat = P("x", None, None)
    exp = P("x", None, None)
    return jax.shard_map(local, mesh=mesh, out_specs=bat, check_vma=False,
                         in_specs=(bat, P(), exp, exp, exp))(
        x, p["gate"], p["w_gate"], p["w_in"], p["w_out"])


def moe_ffn(p, x, top_k, cap_factor, dispatch="cumsum", mesh=None):
    p = dequant(p)
    if dispatch == "ragged":
        return moe_ffn_ragged(p, x, top_k)
    if dispatch.startswith("ep"):
        how, _, nch = dispatch.partition(":")
        return moe_ffn_ep(p, x, top_k, cap_factor, mesh,
                          "gather" if how == "ep-gather" else "scatter", int(nch or 1))
    B, T, d = x.shape
    N = B * T
    E = p["w_in"].shape[0]
    xf = x.reshape(N, d)
    gate = jax.nn.softmax(xf.astype(jnp.float32) @ p["gate"], axis=-1)
    vals, idx = jax.lax.top_k(gate, top_k)
    cap = max(1, int(N * top_k / E * cap_factor))
    src = jnp.repeat(xf, top_k, axis=0)

    if dispatch == "none":
        buf = jnp.concatenate(
            [src, jnp.zeros((E * cap - src.shape[0], d), src.dtype)]).reshape(E, cap, d)
        h = _experts(p, buf).reshape(E * cap, d)[: N * top_k]
        return (h.reshape(N, top_k, d) * vals[..., None].astype(DT)).sum(1).reshape(B, T, d)

    pos = (_slots_sort if dispatch == "sort" else _slots_cumsum)(idx, N, top_k, E)
    keep = pos < cap
    slot = jnp.where(keep, idx * cap + pos, E * cap)

    if dispatch == "gather":
        # Тот же буфер, но собранный чтением, а не записью: сначалаint32-scatter
        # «какое назначение лежит в этой ячейке», потом одна выборка строк по
        # готовым индексам. Широкий scatter на (E·cap+1, d) заменяется на узкий.
        miss = N * top_k
        inv = jnp.full((E * cap,), miss, jnp.int32).at[slot.reshape(-1)].set(
            jnp.arange(miss), mode="drop")
        tok = jnp.where(inv < miss, inv, 0) // top_k
        buf = jnp.where((inv < miss)[:, None], xf[tok], 0)
    else:
        buf = jnp.zeros((E * cap + 1, d), x.dtype).at[slot.reshape(-1)].set(src)[: E * cap]
    h = _experts(p, buf.reshape(E, cap, d))

    flat = jnp.concatenate([h.reshape(E * cap, d), jnp.zeros((1, d), h.dtype)])
    w = (vals * keep).astype(DT)
    return (flat[slot] * w[..., None]).sum(1).reshape(B, T, d)


def dense_ffn(p, x):
    g, u = jnp.split(x @ p["up"].astype(DT), 2, axis=-1)
    return (jax.nn.silu(g) * u) @ p["down"].astype(DT)


def attention(q, k, v, scale, cfg, window):
    if cfg["attn"] == "sdpa":
        return jax.nn.dot_product_attention(q, k, v, is_causal=True, scale=scale)
    if cfg["attn"] == "splash":
        return attend_splash(q, k, v, scale, cfg["attn_block"], window, mesh=cfg["mesh"])
    return attend_chunked(q, k, v, scale, cfg["attn_block"], window,
                          f32_scores=cfg.get("attn_f32_scores", False))


def block(p, x, cfg, is_full=None):
    B, T, d = x.shape
    heads, kv = cfg["heads"], cfg["kv"]
    d_head = d // heads
    h = rms_norm(x, p["norm1"])
    q = (h @ p["q"].astype(DT)).reshape(B, T, heads, d_head)
    k = (h @ p["k"].astype(DT)).reshape(B, T, kv, d_head)
    v = (h @ p["v"].astype(DT)).reshape(B, T, kv, d_head)
    scale = d_head ** -0.5
    w = cfg["window"]
    if not w:
        o = attention(q, k, v, scale, cfg, 0)
    elif is_full is None:
        o = attention(q, k, v, scale, cfg, w)
    else:
        o = jax.lax.cond(is_full,
                         lambda: attention(q, k, v, scale, cfg, 0),
                         lambda: attention(q, k, v, scale, cfg, w))
    x = x + o.reshape(B, T, d) @ p["o"].astype(DT)

    hh = rms_norm(x, p["norm2"])
    out = dense_ffn(p, hh)
    if cfg["experts"]:
        out = out + moe_ffn(p, hh, cfg["top_k"], cfg["cap_factor"], cfg["dispatch"],
                            cfg["mesh"])
    return x + out


def init_params(take, cfg):
    """Слои сложены в стопку: у каждого тензора ведущая ось — номер слоя.

    Это не косметика. Питоновский цикл по 24 слоям разворачивает граф в 24 копии,
    и XLA:TPU умирает на компиляции: прогон 4 млрд параметров держался 692 с и был
    убит по памяти хоста, не напечатав ни строки. Со стопкой граф глубиной в один
    слой, а `jax.lax.scan` повторяет его L раз.
    """
    d, E, L = cfg["d"], cfg["experts"], cfg["layers"]
    heads, kv = cfg["heads"], cfg["kv"]
    d_head = d // heads
    h = cfg["expert_hidden"]
    sh = cfg["shared_hidden"]
    blocks = {"norm1": take((L, d), 0.0) + 1.0,
              "q": take((L, d, d), d ** -0.5),
              "k": take((L, d, kv * d_head), d ** -0.5),
              "v": take((L, d, kv * d_head), d ** -0.5),
              "o": take((L, d, d), d ** -0.5),
              "norm2": take((L, d), 0.0) + 1.0,
              "up": take((L, d, 2 * sh), d ** -0.5),
              "down": take((L, sh, d), sh ** -0.5)}
    if E:
        blocks["gate"] = take((L, d, E), d ** -0.5)
        q = cfg.get("quant") == "experts"
        for k, shape, sc in (("w_gate", (L, E, d, h), d ** -0.5),
                             ("w_in", (L, E, d, h), d ** -0.5),
                             ("w_out", (L, E, h, d), h ** -0.5)):
            got = take(shape, sc, quant=q)
            blocks[k], blocks[k + "_s"] = got if q else (got, None)
            if not q:
                del blocks[k + "_s"]
    return {"embed": take((cfg["vocab"], d), d ** -0.5),
            "final_norm": take((d,), 0.0) + 1.0,
            "blocks": blocks}


REMAT = {
    0: None,
    1: None,
    2: jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    3: jax.checkpoint_policies.dots_saveable,
}


def loss_fn(params, shadow, tokens, targets, cfg):
    """Слои идут одним `scan`; чередование «окно / полное внимание» — через `lax.cond`.

    Ветка выбирается по номеру слоя, поэтому обе компилируются, а исполняется одна.
    Разложить стопку на две (окна отдельно, полные отдельно) нельзя: порядок слоёв
    в чередовании и есть смысл конструкции.

    `shadow` — нули формы весов экспертов, которые едут через тот же `scan`, чтобы
    из обратного прохода вышла котангента по int8-весам (см. `dequant`). Пусто, если
    веса не квантованы.
    """
    x = params["embed"].astype(DT)[tokens]
    fe = cfg["full_every"]
    xs = {**params["blocks"], **{k + "_g": v for k, v in shadow.items()}}
    if cfg["window"] and fe:
        full = (jnp.arange(cfg["layers"]) % fe) == fe - 1
        xs = (xs, full)

    def body(x, p):
        p, is_full = p if isinstance(p, tuple) else (p, None)
        return block(p, x, cfg, is_full), None

    step = jax.checkpoint(body, policy=REMAT[cfg["remat"]]) if cfg["remat"] else body
    x, _ = jax.lax.scan(step, x, xs)
    h = rms_norm(x, params["final_norm"]).reshape(-1, x.shape[-1])
    emb, tgt = params["embed"].astype(DT), targets.reshape(-1)
    if cfg.get("ce", "shard") == "scan":
        return _chunked_ce_legacy(h, emb, tgt, cfg["ce_chunk"])
    return chunked_ce(h, emb, tgt, cfg["ce_chunk"], mesh=cfg["mesh"],
                      shard=cfg.get("ce", "shard") == "shard",
                      f32_logits=cfg.get("ce_f32_logits", False))


def adamw_init(params):
    zero = lambda x: jnp.zeros_like(x)  # noqa: E731
    return {"m": jax.tree.map(zero, params), "v": jax.tree.map(zero, params)}


def adamw_update(params, grads, state, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    new = jax.tree.map(
        lambda p, m, v: (p - lr * (m / (jnp.sqrt(v) + eps) + wd * p)).astype(p.dtype),
        params, m, v)
    return new, {"m": m, "v": v}


def _strip(tree, drop):
    return {**tree, "blocks": {k: v for k, v in tree["blocks"].items() if k not in drop}}


def adafactor_init(params):
    """Второй момент факторизован: вместо матрицы — строка и столбец.

    Это и есть вся разница по памяти с Adam: 8 байт на параметр против 16.
    """
    r = jax.tree.map(lambda x: jnp.zeros(x.shape[:-1] if x.ndim >= 2 else x.shape,
                                         jnp.float32), params)
    c = jax.tree.map(lambda x: jnp.zeros(x.shape[:-2] + x.shape[-1:] if x.ndim >= 2 else (),
                                         jnp.float32), params)
    qk = quant_keys(params)
    if not qk:
        return {"r": r, "c": c}
    drop = {k + "_s" for k in qk}
    return {"r": _strip(r, drop), "c": _strip(c, drop), "ключ": random.PRNGKey(7)}


def _unzip(tree, n):
    pick = lambda i: jax.tree.map(lambda t: t[i], tree,  # noqa: E731
                                  is_leaf=lambda t: isinstance(t, tuple))
    return [pick(i) for i in range(n)]


def _adafactor_leaf(p, g, r, c, lr, b2, eps, wd):
    """Строки `r` и `c` живут в float32, поэтому шаг обязан приводиться обратно к типу
    параметра. Без `.astype` bf16-параметр после первого шага становится fp32: меняется
    сигнатура `jit`, шаг компилируется второй раз поверх живого первого, а память
    параметров удваивается — ровно это роняло `expert`-шардинг и портило все замеры
    памяти, снятые по первой компиляции."""
    if p.ndim < 2:
        r = b2 * r + (1 - b2) * (g * g + eps).astype(jnp.float32)
        return (p - lr * (g / (jnp.sqrt(r) + eps) + wd * p)).astype(p.dtype), r, c
    """`dtype=` у среднего копит сумму во float32, не создавая float32-копии
    градиента: свёртка идёт по 1024–2048 элементам, и в bf16 (8 бит мантиссы) хвост
    слагаемых в такой сумме теряется целиком."""
    r = b2 * r + (1 - b2) * (jnp.mean(g * g, axis=-1, dtype=jnp.float32) + eps)
    c = b2 * c + (1 - b2) * (jnp.mean(g * g, axis=-2, dtype=jnp.float32) + eps)
    den = jnp.sqrt(r[..., None] * c[..., None, :] / jnp.mean(r, axis=-1)[..., None, None])
    return (p - lr * (g / (den + eps) + wd * p)).astype(p.dtype), r, c


def _quant_leaf(code, s, g, r, c, key, lr, b2, eps, wd):
    """Шаг по квантованному весу: разжать, обычный шаг в float32, пересчитать масштаб
    по новому absmax, упаковать обратно со стохастическим округлением.

    Округление обязано быть стохастическим. Шаг сетки int8 — это absmax/127, у нас
    примерно 3% от σ весов, а типичное обновление Adafactor — около lr, то есть
    единицы шагов сетки. Детерминированное округление отбрасывало бы всё, что меньше
    половины шага, и мелкие поправки не накапливались бы никогда; стохастическое
    сохраняет матожидание.
    """
    w = code.astype(jnp.float32) * s
    w, r, c = _adafactor_leaf(w, g, r, c, lr, b2, eps, wd)
    s = jnp.maximum(jnp.max(jnp.abs(w), axis=-2, keepdims=True), 1e-20) / 127.0
    u = w / s + random.uniform(key, w.shape, jnp.float32, -0.5, 0.5)
    return jnp.clip(jnp.round(u), -127, 127).astype(jnp.int8), s, r, c


def _quant_stack(code, s, g, r, c, key, lr, b2, eps, wd):
    """Тот же шаг по одному слою за раз: распакованная стопка целиком — это 4 байта
    на параметр, вчетверо больше самих кодов, и весь выигрыш по памяти съедается."""
    keys = random.split(key, code.shape[0])
    return jax.lax.map(lambda t: _quant_leaf(*t, lr, b2, eps, wd),
                       (code, s, g, r, c, keys))


def adafactor_update(params, grads, state, lr=1e-3, b2=0.95, eps=1e-30, wd=0.1):
    qk = quant_keys(params)
    if qk:
        drop = set(qk) | {k + "_s" for k in qk}
        key = state["ключ"]
        nb, nr, nc = {}, {}, {}
        for i, k in enumerate(qk):
            nb[k], nb[k + "_s"], nr[k], nc[k] = _quant_stack(
                params["blocks"][k], params["blocks"][k + "_s"], grads["blocks"][k],
                state["r"]["blocks"][k], state["c"]["blocks"][k],
                random.fold_in(key, i), lr, b2, eps, wd)
        rest, st = adafactor_update(
            _strip(params, drop), _strip(grads, drop),
            {"r": _strip(state["r"], drop), "c": _strip(state["c"], drop)},
            lr, b2, eps, wd)
        return ({**rest, "blocks": {**rest["blocks"], **nb}},
                {"r": {**st["r"], "blocks": {**st["r"]["blocks"], **nr}},
                 "c": {**st["c"], "blocks": {**st["c"]["blocks"], **nc}},
                 "ключ": random.fold_in(key, 101)})

    out = jax.tree.map(lambda p, g, r, c: _adafactor_leaf(p, g, r, c, lr, b2, eps, wd),
                       params, grads, state["r"], state["c"])
    new, r, c = _unzip(out, 3)
    return new, {"r": r, "c": c}


def newton_schulz(x, steps=5):
    """Приблизить ортогонализацию матрицы пятистепенной итерацией Ньютона–Шульца.

    Считает не сам ортогональный множитель, а его грубое приближение: коэффициенты
    подобраны так, чтобы сингулярные значения быстро сгонялись к единице, не сходясь
    точно. Ведущие оси свободные, поэтому одна и та же функция обрабатывает слой
    (L, d, h) и всех экспертов (L, E, d, h) сразу.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    t = x.astype(jnp.float32)
    flip = t.shape[-2] > t.shape[-1]
    if flip:
        t = jnp.swapaxes(t, -1, -2)
    t = t / (jnp.linalg.norm(t, axis=(-2, -1), keepdims=True) + 1e-7)
    for _ in range(steps):
        s = t @ jnp.swapaxes(t, -1, -2)
        t = a * t + (b * s + c * (s @ s)) @ t
    return jnp.swapaxes(t, -1, -2) if flip else t


def muon_mask(params, experts):
    """Кому достаётся Muon. Эмбеддинг, нормировки и роутер — не матрицы преобразования,
    их ортогонализировать нечего, они идут в Adafactor."""
    mask = jax.tree.map(lambda _: False, params)
    names = ("q", "k", "v", "o", "up", "down")
    if experts:
        names += ("w_gate", "w_in", "w_out")
    for k in names:
        if k in mask["blocks"]:
            mask["blocks"][k] = True
    return mask


def muon_init(params, mask):
    af = adafactor_init(params)
    nil = lambda: jnp.zeros((), jnp.float32)  # noqa: E731
    drop = lambda t: jax.tree.map(lambda x, m: nil() if m else x, t, mask)  # noqa: E731
    return {"мом": jax.tree.map(lambda x, m: jnp.zeros_like(x, jnp.float32) if m else nil(),
                                params, mask),
            "r": drop(af["r"]), "c": drop(af["c"])}


def muon_update(params, grads, state, mask, lr=1e-3, mu=0.95, b2=0.95, eps=1e-30,
                wd=0.1, steps=5):
    """Muon там, где маска, Adafactor на остальном — одним проходом по дереву.

    Флаг в маске питоновский, поэтому ветка выбирается при трассировке и вторая
    половина в граф не попадает вовсе.
    """
    def one(p, g, m, r, c, mk):
        if not mk:
            p, r, c = _adafactor_leaf(p, g, r, c, lr, b2, eps, wd)
            return p, m, r, c
        m = mu * m + g.astype(jnp.float32)
        fan = max(p.shape[-2] / p.shape[-1], 1.0) ** 0.5
        upd = p - lr * (fan * newton_schulz(m, steps).astype(p.dtype) + wd * p)
        return upd.astype(p.dtype), m, r, c

    out = jax.tree.map(one, params, grads, state["мом"], state["r"], state["c"], mask)
    new, m, r, c = _unzip(out, 4)
    return new, {"мом": m, "r": r, "c": c}


def spec_of_shape(shape, n, mode="zero3"):
    """Три раскладки параметров по чипам.

    `zero3` — режем всё по первой оси после слоёв. Нулевую ось трогать нельзя: у
    сложенных в стопку параметров это номер слоя, по которому идёт `scan`.

    `expert` — по чипам режутся только эксперты (единственные четырёхмерные
    тензоры, (слои, E, d, h)), остальное продублировано. Эксперты тогда никуда не
    ездят, ездят токены; плотная часть работает как обычный data-parallel, и
    all-gather весов на каждом слое исчезает. Замер `moe-decomp` показал, что
    именно эти сборы и стоят две трети шага.

    `replica` — полная копия на каждом чипе, только как вычитание.
    """
    if mode == "replica" or (mode == "expert" and len(shape) != 4):
        return P()
    for ax in (1, 0):
        if len(shape) > ax and shape[ax] % n == 0:
            return P(*([None] * ax), "x", *([None] * (len(shape) - ax - 1)))
    return P()


def spec_of(x, n, mode="zero3"):
    return spec_of_shape(x.shape, n, mode)


def make_alloc(mesh, n, key, pdt=jnp.float32, mode="zero3"):
    """Аллокатор по одному тензору за раз, сразу на нужные чипы.

    Целиком дерево параметров в один `jax.jit` заводить нельзя: на 4 млрд
    параметров XLA строит граф из тысяч крупных узлов, и процесс убивает
    OOM-killer хоста ещё до компиляции (проверено, «Killed» на 781 с). По одному
    тензору граф крошечный, а компиляции кешируются по форме — их выходит
    девять штук на всю модель.
    """
    cache: dict = {}
    ctr = [0]

    def make(shape, scale, quant):
        """Квантование живёт внутри `jit`, поэтому float-версия тензора не выезжает в
        HBM вовсе: генерация, деление, округление и упаковка сливаются в один проход.
        Масштаб берётся не по absmax выборки, а аналитически: у нормального веса с
        отклонением σ хвост за 4σ — 0.006% элементов, и обрезать их при инициализации
        дешевле, чем гнать по тензору отдельный проход редукции."""
        if not quant:
            return lambda kk: (random.normal(kk, shape, jnp.float32) * scale).astype(pdt)
        step = 4.0 * scale / 127.0
        s_shape = shape[:-2] + (1, shape[-1])

        def gen(kk):
            w = random.normal(kk, shape, jnp.float32) * scale
            return (jnp.clip(jnp.round(w / step), -127, 127).astype(jnp.int8),
                    jnp.full(s_shape, step, jnp.float32))

        return gen

    def alloc(shape, scale, quant=False):
        ctr[0] += 1
        k = random.fold_in(key, ctr[0])
        if mesh is None:
            return jax.jit(make(shape, scale, quant))(k)
        fn = cache.get((shape, scale, quant))
        if fn is None:
            out = NamedSharding(mesh, spec_of_shape(shape, n, mode))
            if quant:
                s_shape = shape[:-2] + (1, shape[-1])
                out = (out, NamedSharding(mesh, spec_of_shape(s_shape, n, mode)))
            fn = jax.jit(make(shape, scale, quant), out_shardings=out)
            cache[(shape, scale, quant)] = fn
        return fn(k)

    return alloc


def shard_state(tree, mesh, n, mode="zero3"):
    if mesh is None:
        return tree
    return jax.tree.map(
        lambda x: jax.device_put(x, NamedSharding(mesh, spec_of(x, n, mode))), tree)


def count(params, cfg) -> tuple[int, int]:
    total = sum(int(x.size) for x in jax.tree.leaves(params))
    if not cfg["experts"]:
        return total, total
    routed = sum(int(x.size) for k, x in params["blocks"].items() if k.startswith("w_"))
    idle = routed * (1 - cfg["top_k"] / cfg["experts"])
    return total, int(total - idle)


_is3 = lambda x: isinstance(x, tuple) and len(x) == 3  # noqa: E731

_SZ = {"bf16": 2, "f32": 4, "f16": 2, "s32": 4, "u32": 4, "s8": 1, "u8": 1, "pred": 1}
_COLL = "all-gather|all-reduce|all-to-all|reduce-scatter|collective-permute"


def collective_bytes(compiled) -> int:
    """Сколько байт гоняют коллективы за шаг, по скомпилированному HLO.

    Статическая цифра, не замер: сумма размеров результатов коллективных операций.
    Именно она показала, что MoE платит не за арифметику раскладки, а за перенос
    буфера токенов — а сравнить два варианта раскладки по ней можно, не занимая TPU.
    Тип результата у коллектива бывает кортежем, поэтому размеры собираются со всех
    форм слева от имени операции.
    """
    import re  # noqa: PLC0415

    total = 0
    for ln in compiled.as_text().splitlines():
        m = re.search(r"^\s*%?\S+ = (.*?) (" + _COLL + r")\(", ln)
        if not m:
            continue
        for sh in re.findall(r"([a-z0-9]+)\[([0-9,]*)\]", m.group(1)):
            n = _SZ.get(sh[0], 4)
            for d in (int(x) for x in sh[1].split(",") if x):
                n *= d
            total += n
    return total


def run(name, cfg, opt_name, mesh, n_dev):
    cfg = {**cfg, "mesh": mesh}
    key = random.PRNGKey(0)
    B, T = cfg["batch"], cfg["seq"]
    dev = jax.local_devices()[0]
    start = time.perf_counter()

    def log(what):
        used = (dev.memory_stats() or {}).get("bytes_in_use", 0) / 2**30
        rss = 0.0
        try:
            with open("/proc/self/statm") as f:
                rss = int(f.read().split()[1]) * 4096 / 2**30
        except OSError:
            pass
        print(f"  [{time.perf_counter() - start:6.1f}с] {name} {opt_name}: {what} "
              f"| HBM {used:.2f} ГБ | RSS хоста {rss:.1f} ГБ", flush=True)

    log("создаю параметры")
    pdt = getattr(jnp, cfg["param_dtype"])
    shd = cfg["sharding"]
    params = init_params(make_alloc(mesh, n_dev, random.fold_in(key, 2), pdt, shd), cfg)
    jax.block_until_ready(jax.tree.leaves(params)[0])
    total, active = count(params, cfg)
    log(f"параметры готовы: {total / 1e9:.2f} млрд всего, {active / 1e9:.2f} активных")

    if opt_name.startswith("muon"):
        mask = muon_mask(params, opt_name == "muon-all")
        init = partial(muon_init, mask=mask)
        upd = partial(muon_update, mask=mask)
    else:
        init, upd = {"adamw": (adamw_init, adamw_update),
                     "adafactor": (adafactor_init, adafactor_update),
                     "none": (lambda p: {"_": jnp.zeros((), jnp.float32)},
                              lambda p, g, s: (p, {"_": s["_"] + sum(
                                  jnp.sum(x).astype(jnp.float32)
                                  for x in jax.tree.leaves(g))}))}[opt_name]
    state = shard_state(init(params), mesh, n_dev, shd)
    jax.block_until_ready(jax.tree.leaves(state)[0])
    log("состояние оптимизатора готово")

    tokens = random.randint(key, (B, T), 0, cfg["vocab"])
    targets = random.randint(random.fold_in(key, 1), (B, T), 0, cfg["vocab"])
    if mesh is not None:
        ds = NamedSharding(mesh, P("x", None))
        tokens, targets = jax.device_put(tokens, ds), jax.device_put(targets, ds)

    qk = quant_keys(params)

    def _step(params, state, tokens, targets):
        shadow = {k: jnp.zeros(params["blocks"][k].shape, DT) for k in qk}
        loss, (grads, gshadow) = jax.value_and_grad(loss_fn, argnums=(0, 1),
                                                    allow_int=True)(
            params, shadow, tokens, targets, cfg)
        if qk:
            grads = {**grads, "blocks": {**grads["blocks"], **gshadow}}
        params, state = upd(params, grads, state)
        return params, state, loss

    if mesh is None:
        step = jax.jit(_step, donate_argnums=(0, 1))
    else:
        outs = jax.tree.map(lambda x: x.sharding, (params, state))
        step = jax.jit(_step, donate_argnums=(0, 1), out_shardings=(*outs, None))

    log("компилирую шаг")
    t0 = time.perf_counter()
    mem = {}
    try:
        comp = step.lower(params, state, tokens, targets).compile()
        ma = comp.memory_analysis()
        mem = {"аргументы_ГБ": round(ma.argument_size_in_bytes / 2**30, 2),
               "врем_ГБ": round(ma.temp_size_in_bytes / 2**30, 2),
               "выход_ГБ": round(ma.output_size_in_bytes / 2**30, 2),
               "коллективы_ГБ": round(collective_bytes(comp) / 2**30, 2)}
        log(f"план XLA: аргументы {mem['аргументы_ГБ']} + временные {mem['врем_ГБ']} + "
            f"выход {mem['выход_ГБ']} ГБ, коллективов {mem['коллективы_ГБ']} ГБ")
    except Exception as exc:  # noqa: BLE001
        log(f"memory_analysis недоступен: {type(exc).__name__}")
    before = jax.tree.map(lambda x: (x.dtype, x.shape, x.sharding), params)
    params, state, loss = step(params, state, tokens, targets)
    jax.block_until_ready(loss)
    compile_s = time.perf_counter() - t0
    log(f"шаг скомпилирован и прошёл за {compile_s:.1f} с")

    after = jax.tree.map(lambda x: (x.dtype, x.shape, x.sharding), params)
    drift = sum(a != b for a, b in zip(jax.tree.leaves(before, is_leaf=_is3),
                                       jax.tree.leaves(after, is_leaf=_is3)))
    if drift:
        log(f"ВНИМАНИЕ: шаг поменял тип/форму/шардинг у {drift} параметров — "
            f"следующий вызов пойдёт на вторую компиляцию поверх первой")

    loss0 = float(loss)
    best = float("inf")
    for _ in range(3):
        t1 = time.perf_counter()
        params, state, loss = step(params, state, tokens, targets)
        jax.block_until_ready(loss)
        best = min(best, time.perf_counter() - t1)
    builds = getattr(step, "_cache_size", lambda: -1)()
    log(f"замер закончен, компиляций {builds}")

    fit = {}
    if cfg.get("fit_steps"):
        """Токены случайные и одни и те же на всех шагах, так что это не обучение, а
        проверка обучаемости: лосс обязан ползти вниз от ln(V). Ровно этим сравниваются
        bf16-веса с int8: если код не двигается или стохастическое округление ломает
        накопление мелких поправок, кривая встанет, а не пойдёт медленнее."""
        for i in range(cfg["fit_steps"]):
            params, state, loss = step(params, state, tokens, targets)
            if i % 25 == 0 or i == cfg["fit_steps"] - 1:
                log(f"шаг {i + 4}: лосс {float(loss):.4f}")
        fit = {"лосс0": round(loss0, 4), "лоссN": round(float(loss), 4),
               "шагов": cfg["fit_steps"] + 4}

    if cfg.get("profile"):
        """Один лишний шаг под трассировщиком, уже после замера: xplane пишется на
        хост и стоит десятки миллисекунд, попадать в измеряемое время ему нельзя.
        Разбирается локально `scripts/prof_ops.py` — там видно, куда ушёл шаг, по
        категориям HLO, а не по догадкам."""
        sub = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")[:80]
        try:
            with jax.profiler.trace(os.path.join(cfg["profile"], sub)):
                params, state, loss = step(params, state, tokens, targets)
                jax.block_until_ready(loss)
            log(f"профиль записан в {sub}")
        except Exception as exc:  # noqa: BLE001
            log(f"профиль не снялся: {type(exc).__name__}: {exc}")
    if builds > 1:
        log("ВНИМАНИЕ: компиляций больше одной — время замерено не на той программе, "
            "которую описал план XLA")

    ms = dev.memory_stats() or {"bytes_in_use": 0, "bytes_limit": 0}
    row = {"имя": name, "опт": opt_name,
           "всего_млрд": round(total / 1e9, 2), "активных_млрд": round(active / 1e9, 2),
           "экспертов": cfg["experts"], "B": B, "T": T,
           "вним": cfg["attn"], "окно": cfg["window"], "полн_кажд": cfg["full_every"],
           "компиляция_с": round(compile_s, 1), "мс": round(best * 1000, 1),
           "ток_с": round(B * T / best),
           "ТFLOPс": round(6 * active * B * T / 1e9 / (best * 1000), 1),
           "живых_ГБ": round(ms["bytes_in_use"] / 2**30, 2),
           "лимит_ГБ": round(ms["bytes_limit"] / 2**30, 2),
           "компиляций": builds, "дрейф_параметров": drift, **mem, **fit}
    """`peak_bytes_in_use` здесь не годится: это высшая точка за жизнь процесса, она не
    сбрасывается между строками сетки и с третьей строки показывает чужой максимум.
    Потолок читается по плану XLA (аргументы + временные + выход), он посчитан для этой
    компиляции; `живых_ГБ` — что рантайм держит после замера."""
    del params, state
    return row


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2048)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv", type=int, default=4)
    p.add_argument("--shapes", nargs="*", default=[],
                   help="ось перебора форм, `d:слоёв:голов:kv:hidden`; перекрывает "
                        "--d/--layers/--heads/--kv/--shared-hidden")
    p.add_argument("--dtype", default="bfloat16",
                   help="тип счёта: bfloat16 на TPU и Ampere+, float16 на T4")
    p.add_argument("--expert-hidden", type=int, default=1024)
    p.add_argument("--shared-hidden", type=int, default=1024)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--cap-factor", type=float, default=1.25)
    p.add_argument("--cap-factors", nargs="*", type=float, default=[],
                   help="ось перебора запаса ёмкости; арифметика экспертов растёт линейно, "
                        "а у `ep` корзина считается на чип и потому мельче — выброс при "
                        "том же множителе выше, чем в глобальной раскладке")
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--seqs", nargs="*", type=int, default=[2048])
    p.add_argument("--windows", nargs="*", type=int, default=[0],
                   help="0 — полное причинное внимание; иначе длина скользящего окна. "
                        "Окно кратно длине блока запросов, иначе срез не выравнивается")
    p.add_argument("--full-everys", nargs="*", type=int, default=[0],
                   help="при окне: каждый N-й слой всё равно смотрит на весь контекст "
                        "(MiMo держит 10 полных слоёв из 70, DeepSeek V4 — тип слоя в конфиге); "
                        "0 — окно на всех слоях")
    p.add_argument("--attns", nargs="*", default=["chunked"],
                   help="chunked | splash | sdpa; splash — готовое Pallas-ядро TPU")
    p.add_argument("--batches", nargs="*", type=int, default=[8],
                   help="батч на шаг, режется по чипам; больше батч — реже платим за "
                        "коллективы ZeRO-3, они зависят от размера модели, а не от числа токенов")
    p.add_argument("--param-dtypes", nargs="*", default=["float32"],
                   help="bfloat16 вдвое опускает пол по памяти (веса + градиент 4 байта на "
                        "параметр вместо 8), но без мастер-копии fp32 мелкие обновления "
                        "теряются на округлении — для замера памяти годится, для обучения "
                        "нужно стохастическое округление")
    p.add_argument("--attn-blocks", nargs="*", type=int, default=[256],
                   help="длина блока запросов для chunked и splash")
    p.add_argument("--ce-chunk", type=int, default=1024)
    p.add_argument("--ces", nargs="*", default=["shard"],
                   help="как считать кросс-энтропию: shard — каждый чип на своих строках "
                        "(psum в конце), local — та же арифметика без shard_map, "
                        "scan — прежняя версия с fp32-логитами и all-gather в цикле")
    p.add_argument("--score-dtypes", nargs="*", default=["bf16"],
                   help="в чём `attend_chunked` держит очки в HBM; softmax в обоих "
                        "случаях считается в fp32")
    p.add_argument("--quants", nargs="*", default=["none"],
                   help="none | experts. experts хранит w_gate/w_in/w_out кодами int8 с "
                        "масштабом на выходной канал: байт на параметр вместо двух. "
                        "Скорости это не даёт (матмулы — меньше пятой части шага), даёт "
                        "место: градиент остаётся bf16, экономится ровно масса весов")
    p.add_argument("--fit-steps", type=int, default=0,
                   help="после замера прогнать N шагов на тех же случайных токенах и "
                        "напечатать лосс: проверка, что шаг вообще учит (для int8 — что "
                        "стохастическое округление не съедает обновление)")
    p.add_argument("--remats", nargs="*", type=int, default=[1],
                   help="чекпоинтинг активаций между слоями: 0 — нет (без него шаг падает "
                        "по памяти даже в bf16), 1 — пересчитывать слой целиком, "
                        "2 — сохранять выходы матмулов без батч-осей, 3 — все матмулы. "
                        "2 и 3 меняют арифметику на память: пересчёт остаётся только на "
                        "дешёвых поэлементных операциях")
    p.add_argument("--experts", nargs="*", type=int, default=[24, 40, 56, 72, 88])
    p.add_argument("--top-ks", nargs="*", type=int, default=[2])
    p.add_argument("--dispatches", nargs="*", default=["cumsum"],
                   help="cumsum | sort | gather | ep | ragged | none. gather набивает буфер "
                        "чтением вместо записи: широкий scatter заменён на узкий int32 "
                        "плюс одна выборка строк. ep — экспертный параллелизм с явным "
                        "all_to_all вместо автоматической раскладки XLA, нужен "
                        "`--shardings expert`; `ep:4` режет буфер на 4 куска, чтобы обмен "
                        "и арифметика перекрывались. none — слепой reshape без "
                        "маршрутизации, арифметика та же, цена раскладки видна как разница")
    p.add_argument("--opts", nargs="*", default=["adafactor"],
                   help="adafactor | adamw | muon | muon-all | none. muon — Ньютон–Шульц "
                        "на матрицах внимания и плотного FFN, Adafactor на остальном; "
                        "muon-all добавляет к нему экспертов. none считает градиенты и "
                        "складывает их в скаляр — вычитание цены шага оптимизатора")
    p.add_argument("--shardings", nargs="*", default=["zero3"],
                   help="zero3 | expert | replica; expert режет по чипам только экспертов, "
                        "плотную часть дублирует — коллективы на каждом слое исчезают")
    p.add_argument("--pip", default="",
                   help="список пакетов через запятую, ставится до импорта jax; "
                        "`jax[tpu]` разблокирует Pallas и градиент ragged_dot")
    p.add_argument("--libtpu", default="",
                   help="строка в LIBTPU_INIT_ARGS, ставится до импорта jax и действует на "
                        "весь прогон. Интересное: xla_tpu_enable_windowed_einsum_for_all_gather "
                        "(перекрыть сбор весов с матмулом, ровно наша беда с ZeRO-3) и "
                        "xla_tpu_scoped_vmem_limit_kib")
    p.add_argument("--splash-interpret", action="store_true",
                   help="гонять splash интерпретатором Mosaic: обвязка проверяется на CPU")
    p.add_argument("--shard", action="store_true")
    p.add_argument("--profile", default="",
                   help="каталог для xplane: каждая строка сетки пишет туда лишний шаг "
                        "под трассировщиком; разбирать `scripts/prof_ops.py`")
    p.add_argument("--out", default="/kaggle/working/moe.json")
    return p.parse_args(argv)


def run_grid(a, rows, mesh, n_dev) -> None:
    global DT, SPLASH_INTERPRET
    DT = getattr(jnp, a.dtype)
    SPLASH_INTERPRET = a.splash_interpret
    base = {k: getattr(a, k) for k in
            ("d", "layers", "heads", "kv", "expert_hidden", "shared_hidden",
             "top_k", "cap_factor", "vocab", "ce_chunk", "profile", "fit_steps")}
    shapes = a.shapes or [f"{a.d}:{a.layers}:{a.heads}:{a.kv}:{a.shared_hidden}"]
    caps = a.cap_factors or [a.cap_factor]
    grid = [(sh, E, k, disp, rm, pdt, rep, ab, B, opt, T, at, w, fe, cf, ce, sd, qz)
            for sh in shapes for E in a.experts for k in a.top_ks
            for disp in a.dispatches for rm in a.remats for pdt in a.param_dtypes
            for rep in a.shardings for ab in a.attn_blocks for B in a.batches
            for opt in a.opts for T in a.seqs for at in a.attns for w in a.windows
            for fe in a.full_everys for cf in caps for ce in a.ces
            for sd in a.score_dtypes for qz in a.quants]
    for sh, E, k, disp, rm, pdt, rep, ab, B, opt, T, at, w, fe, cf, ce, sd, qz in grid:
        d, layers, heads, kv, hidden = (int(v) for v in sh.split(":"))
        cfg = {**base, "d": d, "layers": layers, "heads": heads, "kv": kv,
               "shared_hidden": hidden, "cap_factor": cf,
               "experts": E, "top_k": k, "dispatch": disp, "remat": rm,
               "param_dtype": pdt, "sharding": rep, "attn_block": ab, "batch": B,
               "seq": T, "attn": at, "window": w, "full_every": fe, "ce": ce,
               "attn_f32_scores": sd == "f32", "ce_f32_logits": sd == "f32",
               "quant": qz}
        if qz != "none" and not opt.startswith(("adafactor", "none")):
            rows.append({"имя": f"{sh} E={E} {qz}", "опт": opt,
                         "ошибка": "int8 сделан только для adafactor и none"})
            continue
        win = f"окно{w}" + (f"/{fe}" if fe else "") if w else "полное"
        name = (f"{sh} E={E} top{k} {disp} remat={rm} {pdt} {at}={ab} {win} B={B} T={T}"
                + (f" {rep}" if rep != "zero3" else "")
                + (f" cf={cf}" if cf != 1.25 else "")
                + (f" ce={ce}" if ce != "shard" else "")
                + (f" очки={sd}" if sd != "bf16" else "")
                + (" int8" if qz != "none" else "")
                + (f" опт={opt}" if opt != "adafactor" else ""))
        try:
            row = run(name, cfg, opt, mesh, n_dev)
        except Exception as exc:  # noqa: BLE001
            row = {"имя": name, "опт": opt, "ошибка": f"{type(exc).__name__}: {str(exc)[:90]}"}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        try:
            with open(a.out, "w") as f:
                json.dump(rows, f, ensure_ascii=False, indent=1)
        except OSError:
            pass


def main() -> None:
    """Аргументы делятся знаком `+` на независимые сетки внутри одного прогона.

    Слот TPU выдаётся по одному и ждётся десятками минут, а перебор — декартово
    произведение осей: плотные формы и MoE в одну сетку не сводятся, там половина
    сочетаний бессмысленна. Сегменты разбирают argv по отдельности и пишут строки
    в общий файл.
    """
    segs, cur = [], []
    for tok in sys.argv[1:]:
        if tok == "+":
            segs.append(cur)
            cur = []
        else:
            cur.append(tok)
    segs.append(cur)
    args = [parse_args(s) for s in segs]

    dev = jax.local_devices()[0]
    n_dev = jax.device_count()
    shard = any(x.shard for x in args)
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev), ("x",)) if shard else None
    lim = (dev.memory_stats() or {"bytes_limit": 0})["bytes_limit"] / 2**30
    print(f"{dev.device_kind} | jax {jax.__version__} | чипов {n_dev} | "
          f"HBM {lim:.2f} ГБ на чип | шардинг {'да' if mesh else 'нет'} | "
          f"сеток {len(args)}", flush=True)

    rows = []
    for i, a in enumerate(args):
        print(f"=== сетка {i + 1} из {len(args)} ===", flush=True)
        run_grid(a, rows, mesh, n_dev)
    print("ГОТОВО", flush=True)


if __name__ == "__main__":
    main()
