"""Разбор xplane не должен падать на настоящем файле.

Профиль снимается на TPU в очереди по 15–25 минут; узнать, что парсер не читает
формат, надо здесь, а не после прогона. На CPU планов устройства нет, поэтому
проверяется ровно то, что проверяемо локально: файл открывается, диагностика
собирается, отчёт печатается.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from scripts.prof_ops import collect, main, report


def _trace(path):
    @jax.jit
    def f(x, w):
        return jnp.tanh(x @ w).sum()

    x, w = jnp.ones((64, 64)), jnp.full((64, 64), 0.01)
    f(x, w).block_until_ready()
    with jax.profiler.trace(str(path)):
        f(x, w).block_until_ready()
    return sorted(path.rglob("*.xplane.pb"))


def test_trace_is_written_and_parsed(tmp_path):
    files = _trace(tmp_path)
    assert files, "jax.profiler.trace не оставил ни одного xplane"
    planes, nest, async_ms, seen = collect(files[0])
    assert isinstance(planes, dict)
    assert isinstance(nest, dict)
    assert isinstance(async_ms, dict)
    assert isinstance(seen, list)


def test_report_survives_a_profile_without_device_planes(tmp_path, capsys):
    files = _trace(tmp_path)
    report(files[0], top=5)
    assert capsys.readouterr().out


def test_main_walks_a_directory(tmp_path, capsys):
    _trace(tmp_path)
    main([str(tmp_path), "--top", "3"])
    assert capsys.readouterr().out
