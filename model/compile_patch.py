"""torch.compile на sm_75 (GTX 1660 Ti): снять запрет на bf16 и проверить графы.

# Запрет на bf16

Inductor отказывается компилировать граф, у которого хоть один вход или выход
в bfloat16, если карта не поддерживает bf16 аппаратно:

    torch/_inductor/compile_fx.py::_check_triton_bf16_support
      -> device_interface.is_bf16_supported(including_emulation=False)
      -> False на sm_75
      -> raise SkipFrame("BF16 is not supported")   # весь граф уходит в eager

У Turing (TU116) нет ни тензорных ядер, ни аппаратного bf16 — PyTorch хранит
такие тензоры в 16 битах, а считает в fp32. Проверка написана в расчёте на то,
что Triton не сумеет сгенерировать корректный код, но Triton поднимает bf16 в
fp32 внутри ядра и прекрасно справляется. Замер 2 сен на ядре `silu(x@w)`,
bf16 512×512: собралось, расхождение с eager 0.00098 — уровень округления bf16.
Под autocast(bf16) без этого патча компиляция «проходит», а ускорения нет.

# Что на этой карте всё равно не работает

`max_autotune_gemm` inductor отключает сам («Not enough SMs»: у карты 24 SM,
перебор конфигураций GEMM он считает бессмысленным) и отдаёт матмулы в cuBLAS.
То есть `mode="max-autotune"` здесь = обычный inductor + CUDA-графы.

`fla.ops.kda.chunk_kda` обёрнут в `torch.compiler.disable`, поэтому разрыв графа
на каждом KDA-слое неустраним со стороны нашего кода.

# CUDA-графы: замер 2 сен, A16, B=1 T=1344, forward+backward

    eager                       447 мс
    max-autotune, установившийся  314 мс   (1.42x)
    записано графов: 22 функции, 178 узлов, cudagraph_skips = 0

Записываются они на ВТОРОМ вызове скомпилированной функции: первый идёт
прогревочным. Последовательность вызовов: 7857 мс (компиляция), 4043 мс
(запись), дальше 314 мс. Поэтому прогрев обязателен, иначе первый шаг обучения
дороже остальных в 12 раз.

⚠️ Две вещи, каждая из которых стоила отладки:
  * при включённых графах `torch.cuda.max_memory_allocated()` не видит пул
    графов и занижает память вдвое — мерить через `torch.cuda.mem_get_info()`;
  * градиенты должны жить в СТАБИЛЬНЫХ буферах. `zero_grad(set_to_none=True)`
    освобождает `.grad`, следующий backward аллоцирует его уже внутри захвата, и
    накопление читает перезаписанную память: RuntimeError «accessing gradient
    tensor output of CUDAGraphs that has been overwritten». Лечится одним
    eager-прогоном до компиляции и `set_to_none=False` дальше.

Использование:

    from model.compile_patch import compile_body, compile_report, cudagraph_stats
    body = compile_body(model)          # компилируется метод, не модуль
    ...
    print(compile_report(), cudagraph_stats())
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

# Кеш inductor по умолчанию лежит в /tmp/torchinductor_$USER и пропадает при
# рестарте WSL. Уводим в постоянный каталог: правка любой строки архитектуры
# инвалидирует кеш целиком, терять его ещё и на рестартах не хочется.
CACHE_DIR = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR",
                                Path.home() / ".cache" / "torchinductor"))
TRITON_CACHE_DIR = Path(os.environ.get("TRITON_CACHE_DIR",
                                       Path.home() / ".cache" / "triton"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE_DIR))
os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))

_patched = False


def enable_bf16_compile() -> bool:
    """Снять запрет inductor на bf16-подграфы. Идемпотентна.

    False означает, что карта поддерживает bf16 нативно и патч не нужен
    (Ampere и новее).
    """
    global _patched
    if _patched:
        return True
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
        return False

    from torch._inductor import compile_fx

    if not hasattr(compile_fx, "_check_triton_bf16_support"):
        # молча ничего не делать нельзя: мы будем думать, что компиляция
        # включена, а она уходит в eager
        raise RuntimeError(
            "torch._inductor.compile_fx._check_triton_bf16_support не найдена — "
            f"проверь, что изменилось в torch {torch.__version__}"
        )
    compile_fx._check_triton_bf16_support = lambda graph: None
    _patched = True
    return True


def enable_gemm_autotune() -> None:
    """Разрешить inductor подбирать Triton-шаблоны для матмулов на маленькой карте.

    `torch._inductor.utils.is_big_gpu` требует >= 68 SM (порог поставлен по 3080).
    У нас 24, поэтому `_use_template_for_gpu` возвращает False, и `tuned_bmm` /
    `tuned_mm` даже не рассматривают Triton — всё уходит в ATen. Именно отсюда
    сообщение «Not enough SMs to use max_autotune_gemm mode».

    Порог — эвристика «на мелкой карте Triton всё равно проиграет cuBLAS, не
    трать время на автотюн», а не запрет корректности. Сняв его, мы не заставляем
    inductor брать Triton: автотюн ЗАМЕРЯЕТ шаблоны против ATen и берёт быстрейший.
    Цена — время компиляции.

    Отдельно важно для нас: под этот же гейт попадает `bmm`, а наши эксперты — это
    ровно три bmm, и в bf16 они уходят в MAGMA отдельными вызовами.

    Не включается по умолчанию: цену и выигрыш надо мерить (scripts/bench_bmm.py,
    scripts/compile_check.py --gemm-autotune).
    """
    from torch._inductor import config, utils

    utils.is_big_gpu = lambda *a, **k: True          # noqa: ARG005
    config.max_autotune_gemm = True


def compile_body(model, mode: str = "max-autotune", gemm_autotune: bool = False):
    """Скомпилировать `model.body`, оставив сам модуль несжатым.

    Компилируется связанный метод, а не модуль: `state_dict` модели остаётся без
    префикса `_orig_mod.`, поэтому чекпоинт совместим с запуском без компиляции.
    """
    enable_bf16_compile()
    if gemm_autotune:
        enable_gemm_autotune()
    torch._inductor.config.triton.cudagraphs = True
    return torch.compile(model.body, mode=mode)


def compile_report() -> dict:
    """Счётчики Dynamo/Inductor после прогона: разрывы графа и отказы от графов."""
    from torch._dynamo.utils import counters

    return {
        "graph_breaks": sum(counters["graph_break"].values()),
        "graph_break_reasons": {k.split("\n")[0]: v for k, v in counters["graph_break"].items()},
        "cudagraph_skips": counters["inductor"].get("cudagraph_skips", 0),
    }


def cudagraph_stats() -> dict:
    """Сколько CUDA-графов реально записано менеджером inductor'а.

    Единственная прямая проверка: `cudagraph_skips == 0` означает лишь, что
    inductor не отказывался вслух, а записанных графов при этом может не быть
    вообще — например, если все подграфы откатились в eager по bf16.
    """
    try:
        from torch._inductor.cudagraph_trees import get_manager
    except Exception:  # noqa: BLE001
        return {"managers": 0, "captured_funcs": 0, "captured_nodes": 0}

    funcs = nodes = managers = 0
    for dev in range(torch.cuda.device_count()):
        mgr = get_manager(dev, create_if_none_exists=False)
        if mgr is None:
            continue
        managers += 1
        funcs += len(mgr.ids_to_funcs)
        stack = [n for roots in (mgr.roots or {}).values() for n in roots]
        while stack:
            node = stack.pop()
            nodes += 1
            stack.extend(c for cs in (node.children or {}).values() for c in cs)
    return {"managers": managers, "captured_funcs": funcs, "captured_nodes": nodes}
