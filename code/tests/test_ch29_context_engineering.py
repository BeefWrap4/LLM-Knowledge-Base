"""第 29 章离线示例的准确性与边界测试。"""

from __future__ import annotations

import math
import sys
from importlib import util
from pathlib import Path
from types import ModuleType

CHAPTER_DIR = Path(__file__).parents[1] / "ch29_context_engineering" / "llm"


def load_example(filename: str) -> ModuleType:
    module_name = f"test_ch29_{Path(filename).stem}"
    spec = util.spec_from_file_location(module_name, CHAPTER_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TOKEN_ECONOMICS = load_example("02_token_economics.py")
CONTEXT_ROT = load_example("03_context_rot_demo.py")
LAYERED_MEMORY = load_example("08_pydantic_ai_memory.py")
PROMPT_CACHING = load_example("10_prompt_caching.py")


def test_synthetic_position_curve_is_symmetric_and_u_shaped():
    left = CONTEXT_ROT.synthetic_position_score(0.05)
    middle = CONTEXT_ROT.synthetic_position_score(0.50)
    right = CONTEXT_ROT.synthetic_position_score(0.95)

    assert math.isclose(left, right)
    assert left > middle


def test_context_rot_demo_discloses_that_it_is_not_an_experiment(capsys):
    CONTEXT_ROT.run_demo()
    output = capsys.readouterr().out

    assert "合成教学曲线" in output
    assert "不是模型实验" in output
    assert "必须在目标模型和任务上实测" in output


def test_token_cost_uses_injected_rate_card_and_explicit_components():
    rates = TOKEN_ECONOMICS.RateCard(
        label="test",
        currency="unit",
        checked_on="2026-07-31",
        source_url="https://example.test/pricing",
        uncached_input=1.0,
        cache_read=0.25,
        cache_write=1.5,
        output=4.0,
        storage_per_million_token_hour=0.5,
    )
    usage = TOKEN_ECONOMICS.TokenUsage(
        uncached_input_tokens=1_000_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=1_000_000,
        output_tokens=500_000,
        stored_token_hours=2_000_000,
    )

    cost = TOKEN_ECONOMICS.estimate_cost(rates, usage)

    assert cost.uncached_input == 1.0
    assert cost.cache_read == 0.5
    assert cost.cache_write == 1.5
    assert cost.output == 2.0
    assert cost.storage == 1.0
    assert cost.total == 6.0


def test_local_layered_memory_is_framework_neutral_and_respects_capacity():
    short_term = LAYERED_MEMORY.ShortTermMemory(capacity=2)
    short_term.add("user", "first")
    short_term.add("assistant", "second")
    short_term.add("user", "third")
    assert [message["content"] for message in short_term.to_messages()] == ["second", "third"]

    source = (CHAPTER_DIR / "08_pydantic_ai_memory.py").read_text(encoding="utf-8")
    assert "MemoryTool" not in source
    assert "PydanticAIStyleAgent" not in source
    assert "from pydantic_ai" not in source


def test_prompt_cache_example_separates_hit_discount_from_total_cost():
    first_write = PROMPT_CACHING.anthropic_normalized_input_cost(
        prefix_tokens=8_000,
        dynamic_tokens_per_turn=500,
        turns=1,
        cache_hits=0,
    )
    repeated_hits = PROMPT_CACHING.anthropic_normalized_input_cost(
        prefix_tokens=8_000,
        dynamic_tokens_per_turn=500,
        turns=10,
        cache_hits=9,
    )

    assert first_write.change_vs_no_cache > 0
    assert repeated_hits.change_vs_no_cache < 0
    gemini = next(
        policy for policy in PROMPT_CACHING.POLICY_SNAPSHOTS if policy.provider_scope.startswith("Gemini")
    )
    assert "不是免费命中" in gemini.billing_boundary
