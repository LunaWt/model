"""Запустить локальный скрипт на бесплатном ускорителе Kaggle и забрать вывод.

    uv run python -m scripts.kaggle_run scripts/bench_dense_jax.py \
        --name jax-tpu --accelerator tpuV5e8 -- --shard --batch-size 16

Квота на 4 сен 2026, из `kaggle quota` на аккаунте: 30 ч GPU и 20 ч TPU в неделю,
счётчики раздельные, сброс в 00:00 UTC в субботу. Один сеанс — 12 ч на GPU и 9 ч
на TPU (kaggle.com/docs/notebooks, kaggle.com/docs/tpu).

Как это устроено. Интерактивной сессии из терминала нет — только пакетный цикл
push → poll → fetch. `kernels push` возвращается сразу, поэтому ждать приходится
самим. Файл собирается в один самодостаточный `script.py`: Kaggle загружает
только `code_file`, соседние файлы не приедут, поэтому импортов из репозитория в
скрипте быть не должно. Аргументы подставляются через `sys.argv` в шапке.

Всё, что скрипт кладёт в /kaggle/working (20 ГБ), скачивается потом
`kaggle kernels output`. Диск сеанса стирается, /kaggle/working — нет.

Две ловушки, обе стоили целого прогона:

  * `machine_shape` уходит на сервер как есть, без проверки. Неизвестное значение
    молча превращается в один P100 — так `NvidiaTeslaT4x2` даёт не две T4, а одну
    P100, и понять это можно только по `device_kind` в выводе. Рабочие строки:
    `none`, `NvidiaTeslaT4` (×2), `NvidiaTeslaP100`, `tpuV5e8`;
  * загружается снимок файла на момент push. Правки, сделанные после, в уже стоящий
    в очереди прогон не попадут никогда.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")
WORK = Path.home() / ".claude/scratch/kg"


def build(script: Path, folder: Path, slug: str, user: str, acc: str, extra: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    body = [ln for ln in script.read_text().splitlines()
            if ln.strip() != "from __future__ import annotations"]
    argv = json.dumps([script.name, *extra])
    (folder / "script.py").write_text(
        f"import sys\nsys.argv = {argv}\n" + "\n".join(body) + "\n")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": str(acc.startswith("Nvidia")).lower(),
        "enable_tpu": str(acc.startswith("tpu") or acc.startswith("Tpu")).lower(),
        "enable_internet": "true",
        "machine_shape": acc,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))


def run(args: list[str]) -> str:
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def push(args: list[str], tries: int = 3, wait: int = 20) -> bool:
    """`kernels push` умеет вернуться молча: кода ошибки нет, версии тоже нет, и опрос
    статуса потом показывает прошлый прогон — то есть замер идёт по старому коду и это
    не видно ниоткуда. Единственный надёжный признак успеха — строка про push в ответе."""
    for i in range(tries):
        out = run(args)
        print(out, flush=True)
        if "successfully pushed" in out.lower():
            return True
        if i + 1 < tries:
            print(f"push не подтверждён, попытка {i + 2} из {tries} через {wait} с",
                  flush=True)
            time.sleep(wait)
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("script", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--user", default="luna87")
    p.add_argument("--accelerator", default="none",
                   help="none | NvidiaTeslaT4 | NvidiaTeslaP100 | tpuV5e8")
    p.add_argument("--poll", type=int, default=30, help="секунд между опросами")
    p.add_argument("--no-wait", action="store_true")
    a, extra = p.parse_known_args()
    extra = [v for v in extra if v != "--"]

    folder = WORK / a.name
    build(a.script, folder, a.name, a.user, a.accelerator, extra)
    cmd = ["kernels", "push", "-p", str(folder)]
    if a.accelerator != "none":
        cmd += ["--accelerator", a.accelerator]
    if not push(cmd):
        sys.exit("push не подтверждён три раза подряд — прогон не запущен")
    ref = f"{a.user}/{a.name}"
    if a.no_wait:
        print(f"опрос: kaggle kernels status {ref}")
        return

    t0 = time.time()
    while True:
        time.sleep(a.poll)
        st = run(["kernels", "status", ref])
        print(f"{time.time() - t0:6.0f}s {st}", flush=True)
        if "COMPLETE" in st or "ERROR" in st or "CANCEL" in st:
            break
    out = folder / "out"
    print(run(["kernels", "output", ref, "-p", str(out), "-o"]))
    log = next(out.glob("*.log"), None)
    if log:
        for ev in json.loads(log.read_text()):
            sys.stdout.write(ev["data"])


if __name__ == "__main__":
    main()
