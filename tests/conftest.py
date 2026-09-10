"""Два поддельных устройства на CPU, чтобы шардинг проверялся тестами, а не только TPU.

Флаг читается XLA один раз при первом импорте jax, поэтому ставится здесь, до любого
теста. На однокарточные тесты это не влияет: без явного шардинга всё живёт на нулевом.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2").strip()
