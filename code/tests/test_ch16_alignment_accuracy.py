"""Ch16 alignment claims and teaching implementations must stay evidence-bounded."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent
SFT_CHAPTER = REPO_ROOT / "30_SFT_LoRA与QLoRA.md"
ALIGNMENT_CHAPTER = REPO_ROOT / "31_偏好对齐与强化学习.md"
GPU_DIR = CODE_ROOT / "ch30_lora_qlora" / "gpu"


def load_example(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, GPU_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chapter_removes_fixed_alignment_resource_and_quality_claims():
    text = SFT_CHAPTER.read_text(encoding="utf-8") + ALIGNMENT_CHAPTER.read_text(encoding="utf-8")
    forbidden = (
        "| **模型数量**",
        "单模型训练，省一半显存",
        "SGLang 是首选 rollout",
        "RL 后端首选",
        "奖励噪声几乎为零",
        "CoT 推理能力几乎**完全可迁移**",
        "OpenAI o1/o3、DeepSeek-R1、Claude 4",
        "rewards.groupby",
        "ref_chosen, ref_rejected",
        "GPT-5.5：五档",
    )
    for claim in forbidden:
        assert claim not in text


def test_chapter_uses_published_deepseek_distill_scope_and_values():
    text = ALIGNMENT_CHAPTER.read_text(encoding="utf-8")
    assert "Qwen2.5-Math-1.5B" in text
    assert "Qwen2.5-Math-7B" in text
    assert "DeepSeek-R1-Distill-Qwen-14B" in text
    assert "70.0 / 94.5" in text
    assert "AIME 2024 pass@1 / MATH-500 pass@1" in text
    assert "temperature" in text and "top-p 0.95" in text


def test_chapter_section_numbers_are_monotonic_at_the_end():
    text = ALIGNMENT_CHAPTER.read_text(encoding="utf-8")
    assert text.index("## 31.2 RL Post-Training") < text.index("## 🧭 本章小结")
    assert "### 31.2.7 章节速记卡" in text


def test_grpo_loss_is_grouped_differentiable_and_uses_k3_direction():
    torch = pytest.importorskip("torch")
    module = load_example("12_grpo_loss.py", "ch16_grpo")
    source = inspect.getsource(module.grpo_loss)
    assert "ref_log_probs - log_probs" in source

    log_probs = torch.full((4, 3), -1.0, requires_grad=True)
    old_log_probs = torch.full((4, 3), -1.0)
    ref_log_probs = torch.full((4, 3), -1.1)
    rewards = torch.tensor([0.0, 1.0, 10.0, 12.0])
    loss = module.grpo_loss(
        log_probs,
        old_log_probs,
        ref_log_probs,
        rewards,
        group_size=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert log_probs.grad is not None
    with pytest.raises(ValueError):
        module.grpo_loss(log_probs, old_log_probs, ref_log_probs, rewards, group_size=3)


def test_orpo_loss_is_reference_free_and_backpropagates_both_responses():
    torch = pytest.importorskip("torch")
    module = load_example("15_orpo_loss.py", "ch16_orpo")
    parameters = inspect.signature(module.orpo_loss).parameters
    assert not any("ref" in name for name in parameters)

    chosen = torch.full((2, 4), -1.0, requires_grad=True)
    rejected = torch.full((2, 4), -1.5, requires_grad=True)
    labels = torch.ones((2, 4), dtype=torch.long)
    labels[:, :1] = -100
    loss = module.orpo_loss(chosen, rejected, labels, labels.clone())
    loss.backward()
    assert torch.isfinite(loss)
    assert chosen.grad is not None
    assert rejected.grad is not None


def test_math_verifier_rejects_code_and_resource_abuse():
    module = load_example("14_math_verifier.py", "ch16_math_verifier")
    assert module.verify("1 + 2", "3")
    assert not module.verify("1", "__import__('os').system('whoami')")
    assert not module.verify("1", "9 ** 999999")


def test_sglang_client_is_loopback_only():
    module = load_example("13_sglang_rollout.py", "ch16_sglang")
    assert module.validate_loopback_base_url("http://127.0.0.1:30000/") == (
        "http://127.0.0.1:30000"
    )
    with pytest.raises(ValueError, match="只允许连接本机"):
        module.validate_loopback_base_url("https://example.com")


def test_adaptive_router_is_offline_and_product_neutral():
    script = GPU_DIR / "11_adaptive_inference.py"
    source = script.read_text(encoding="utf-8").lower()
    assert "deepseek-v4-pro" not in source
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=CODE_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("OK")


def test_ch16_source_ledger_contains_primary_papers_and_current_docs():
    ledger = (REPO_ROOT / "docs" / "AUTHORITATIVE_SOURCES.md").read_text(encoding="utf-8")
    row = next(line for line in ledger.splitlines() if line.startswith("| Ch31 |"))
    for marker in (
        "2402.03300",
        "2501.12948",
        "2403.07691",
        "trl/grpo_trainer",
        "docs.sglang.ai",
        "deepspeed.readthedocs.io",
        "developers.openai.com/api/docs/guides/latest-model",
    ):
        assert marker in row
