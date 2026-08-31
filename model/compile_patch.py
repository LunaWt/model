"""Разрешить torch.compile генерировать bf16-ядра на sm_75 (GTX 1660 Ti).

# Зачем

Inductor отказывается компилировать граф, у которого хоть один вход или выход
в bfloat16, если карта не поддерживает bf16 аппаратно:

    torch/_inductor/compile_fx.py::_check_triton_bf16_support
      -> device_interface.is_bf16_supported(including_emulation=False)
      -> False на sm_75
      -> raise SkipFrame("BF16 is not supported")   # весь граф уходит в eager

У Turing (TU116) нет ни тензорных ядер, ни аппаратного bf16 — PyTorch хранит
такие тензоры в 16 битах, а считает в fp32. Проверка написана в расчёте на то,
что Triton не сумеет сгенерировать корректный код.

# Почему её можно снять

Матричные умножения inductor у нас и так не трогает: он сам печатает
`Not enough SMs to use max_autotune_gemm mode` и отдаёт их в cuBLAS. Всё, что
он генерирует, — поэлементные ядра и редукции, а там bf16 это 16-битная
загрузка с расширением до fp32, обычная арифметика fp32 и обратное усечение.
Никакого bf16-железа для этого не нужно, и Triton это умеет.

# Что замерено (31 июл 2026, конфиг F, B=1 T=1536, KDA заменена на MLA)

    eager                          466.5 мс   3293 ток/с   пик 3917 MiB
    compile default                267.0 мс   5752 ток/с
    compile max-autotune-no-cudagraphs  256.7 мс  5983 ток/с  пик 2818 MiB

То есть **ускорение x1.4–1.8 и примерно −1.1 ГБ памяти**. Разброс от прогона к
прогону большой: под WDDM видеокарта делится с рабочим столом Windows.

Корректность: сравнение с эталоном в чистом fp32 показало, что
скомпилированная версия БЛИЖЕ к нему, чем eager (среднее |Δ| 9.2e-3 против
2.3e-2). Расхождение eager и compile между собой (до 1.7 по модулю) — это не
ошибка ядер, а перевороты маршрутизации MoE: сигмоидные скоры при
инициализации почти совпадают, и округление bf16 меняет top-k у ~6% токенов.

# Про CUDA-графы

`mode="max-autotune"` — это то же самое плюс CUDA-графы. С первого раза он
падает с `cudaErrorStreamCaptureInvalidated`, и причина понятна: автотюн
**бенчмаркает** ядра, бенчмарк требует синхронизации, а синхронизироваться
внутри захвата графа нельзя. Лечится прогревом — сначала скомпилировать
`max-autotune-no-cudagraphs` и прогнать пару шагов, тогда результаты автотюна
лягут в кеш, и во второй компиляции мерить будет уже нечего
(`compile_model(..., cudagraphs=True)` делает это само).

Но включать их незачем. Замерено тремя прогонами в отдельных процессах:

    max-autotune-no-cudagraphs   266.2 / 263.0 / 266.9 мс   занято на карте 3988 MiB
    max-autotune (графы)         270.9 / 249.4 мс           занято на карте 4156 MiB

Разброс перекрывает разницу, а свой пул графов стоит настоящих +168 МиБ.
Память у нас — связывающее ограничение, так что обмен плохой. Графы экономят
запуски ядер, а после слияния inductor'ом запусков осталось мало.

⚠️ При включённых графах `torch.cuda.max_memory_allocated()` врёт (показывает
1133 МиБ) — пул графов он не учитывает. Мерить только через
`torch.cuda.mem_get_info()`.

Использование:

    from model.compile_patch import enable_bf16_compile, compile_model
    enable_bf16_compile()
    model = compile_model(model)
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

COMPILE_MODE = "max-autotune-no-cudagraphs"

# Компиляция полной модели идёт ~45 минут, а кеш по умолчанию лежит в
# /tmp/torchinductor_$USER и пропадает при рестарте WSL. Уводим в постоянный
# каталог. Переменные читаются при импорте torch._inductor, поэтому ставим их
# здесь — до первого torch.compile, но после импорта torch это уже поздно
# менять через os.environ у самого inductor, так что дублируем и в конфиг.
CACHE_DIR = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR",
                                Path.home() / ".cache" / "torchinductor"))
TRITON_CACHE_DIR = Path(os.environ.get("TRITON_CACHE_DIR",
                                       Path.home() / ".cache" / "triton"))


def _setup_cache() -> None:
    """Постоянный каталог кеша. Один граф на всю модель — значит правка любой
    строки архитектуры инвалидирует кеш целиком и стоит полной перекомпиляции.
    Тем более не хочется терять его ещё и на каждом рестарте."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE_DIR))
    os.environ.setdefault("TRITON_CACHE_DIR", str(TRITON_CACHE_DIR))


_setup_cache()

_patched = False


def enable_bf16_compile() -> bool:
    """Снять запрет на bf16-компиляцию. Идемпотентна.

    Возвращает True, если запрет был снят, и False, если карта поддерживает
    bf16 нативно и патч не нужен (Ampere и новее — там всё работает само).
    """
    global _patched
    if _patched:
        return True

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
        return False    # нормальная карта, вмешиваться незачем

    import torch._inductor.compile_fx as compile_fx

    if not hasattr(compile_fx, "_check_triton_bf16_support"):
        # PyTorch переименовал или убрал проверку — молча ничего не делать
        # нельзя, иначе мы будем думать, что компиляция включена, а она нет.
        raise RuntimeError(
            "torch._inductor.compile_fx._check_triton_bf16_support не найдена — "
            f"проверь, что изменилось в torch {torch.__version__}"
        )

    compile_fx._check_triton_bf16_support = lambda graph: None
    _patched = True
    return True


def compile_model(
    model: nn.Module,
    mode: str = COMPILE_MODE,
    cudagraphs: bool = False,
    warmup: Callable[[nn.Module], None] | None = None,
    **kwargs,
) -> nn.Module:
    """torch.compile с нашими настройками, с включённым bf16 на этой карте.

    dynamic=False — формы у нас фиксированные (B, T заданы конфигом, ёмкость
    полки эксперта тоже), а динамические формы стоят перекомпиляций.

    cudagraphs=True переключает на `max-autotune` (с графами) и обходит
    падение при захвате: сначала компилирует без графов и вызывает `warmup`,
    чтобы автотюн отработал и лёг в кеш. `warmup(compiled_model)` должен
    прогнать хотя бы один полный шаг — forward и backward, иначе кеш backward
    останется пустым и захват снова упадёт.

    По умолчанию выключено: замеры не показали выигрыша, а память графы едят
    (см. шапку файла).
    """
    enable_bf16_compile()
    with warnings.catch_warnings():
        # «does not support bfloat16 compilation natively» больше не всплывёт,
        # но max_autotune_gemm предупредит про число SM — это ожидаемо
        warnings.filterwarnings("ignore", message=".*Not enough SMs.*")

        if not cudagraphs:
            return torch.compile(model, mode=mode, dynamic=False, **kwargs)

        if warmup is None:
            raise ValueError(
                "cudagraphs=True требует warmup: без прогретого кеша автотюна "
                "захват графа падает с cudaErrorStreamCaptureInvalidated"
            )
        primed = torch.compile(model, mode="max-autotune-no-cudagraphs",
                               dynamic=False, **kwargs)
        warmup(primed)
        torch._dynamo.reset()
        return torch.compile(model, mode="max-autotune", dynamic=False, **kwargs)
