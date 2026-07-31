"""Numerical contract for the Ch16 causal attention benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "ch16_finetuning"
    / "gpu"
    / "04_flash_attention.py"
)
SPEC = importlib.util.spec_from_file_location("ch16_flash_attention", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_naive_and_sdpa_use_the_same_causal_semantics():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    naive = MODULE.naive_attention(q, k, v)
    sdpa = MODULE.sdpa_attention(q, k, v)

    torch.testing.assert_close(naive, sdpa, atol=1e-5, rtol=1e-5)
