"""Sanity checks for the training environment.

Cheap guard: if the CUDA build ever breaks (driver update, torch upgrade that
drops sm_75), this fails loudly instead of surfacing as a weird runtime error
in the middle of a training run.
"""

import torch


def test_cuda_available():
    assert torch.cuda.is_available(), "no CUDA device visible from WSL"


def test_gpu_arch_supported():
    """GTX 1660 Ti is Turing (sm_75). CUDA releases periodically drop old archs."""
    major, minor = torch.cuda.get_device_capability(0)
    assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()


def test_gpu_compute_actually_runs():
    """Arch in the list isn't proof — run a real kernel and a real backward pass."""
    x = torch.randn(512, 512, device="cuda", requires_grad=True)
    (x @ x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
