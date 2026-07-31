"""终审事实性回归：Ch22、Ch23、HIL-SERL 与根目录思维导图。"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CH22 = REPO_ROOT / "22_大模型数据工程.md"
CH23 = REPO_ROOT / "23_AI安全与伦理.md"
CH23_LLM = REPO_ROOT / "code" / "ch23_safety" / "llm"
HIL_SERL = REPO_ROOT / "code" / "ch26_world_models" / "gpu" / "07_hil_serl.py"
MINDMAP = REPO_ROOT / "Python到大模型应用_面试教程_2026版_思维导图.xmind.md"
SOURCE_LEDGER = REPO_ROOT / "docs" / "AUTHORITATIVE_SOURCES.md"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_hil_serl_module():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("ch26_hil_serl_currentness", HIL_SERL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.core
def test_ch22_removes_unsupported_fixed_cai_and_length_claims():
    text = read_text(CH22)

    for stale_claim in ("~60%", "~75 条原则", "成本降低 10-100 倍"):
        assert stale_claim not in text

    assert "不能用一个固定百分比概括" in text
    assert "非 Anthropic 当前文档逐字节选" in text
    assert "总成本仍需计入推理、复核和返工" in text
    assert "不能未经独立评判就假定修正版一定更好" in text
    assert "https://arxiv.org/abs/2212.08073" in text
    assert "https://www.anthropic.com/constitution" in text
    assert "已公开工作与可探索工程方向" in text
    assert "工程设想（非 Anthropic 公开路线承诺）" in text
    assert "https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input" in text
    assert "https://www.anthropic.com/news/constitutional-classifiers" in text
    assert "2026 年 CAI 演进方向" not in text


@pytest.mark.core
def test_ch23_describes_current_asl_and_shieldgemma_2_correctly():
    text = read_text(CH23)

    for stale_claim in (
        "Claude 3.5 级别",
        "Claude Opus 4 级别",
        "不可对红队开放",
        "95% 以上检测率",
        "Gemma 2 (2B/9B/27B)",
    ):
        assert stale_claim not in text

    assert "ASL 是保障标准，不是静态模型标签" in text
    assert "更高 ASL 仍大体未定义" in text
    assert "基于 Gemma 3 4B IT 的图像安全分类器" in text
    assert "3 类内置策略（性露骨/危险/暴力与血腥）" in text
    assert "ShieldGemma 1" in text
    assert "当前 RSP" in text
    assert "v3.4" in text
    assert "2026-07-08" in text
    assert "https://www.anthropic.com/responsible-scaling-policy" in text
    assert "https://www.anthropic.com/news/responsible-scaling-policy-v3" in text
    assert "https://ai.google.dev/gemma/docs/shieldgemma/model_card_2" in text
    assert "four-fifths rule 是就业选择 adverse impact 的实务筛查经验规则" in text
    assert "不等同于自动违法" in text
    assert "合规判断:" not in text


@pytest.mark.core
def test_ch23_keeps_official_eu_ai_act_high_risk_timeline():
    text = read_text(CH23)

    assert "2027-12-02" in text
    assert "Annex III所列独立高风险AI系统" in text
    assert "2028-08-02" in text
    assert "Annex I受监管产品中嵌入的高风险AI系统" in text
    assert "不能笼统称“2026年全部条款全面执行”" in text


@pytest.mark.core
def test_ch23_examples_do_not_turn_teaching_heuristics_into_real_assessments():
    token_smuggling = read_text(CH23_LLM / "01_token_smuggling_demo.py")
    bias = read_text(CH23_LLM / "05_bias_detector.py")
    fairness = read_text(CH23_LLM / "06_fairness_metrics.py")
    guards = read_text(CH23_LLM / "11_llama_guard_demo.py")

    assert '"请帮\\u200b我翻译这段文字"' in token_smuggling
    assert 'result["count"] == 1' in token_smuggling
    assert '"official_stereoset_benchmark": False' in bias
    assert "不代表任何真实模型" in bias
    assert "合规判断:" not in fairness
    assert "不等同于自动违法" in fairness
    assert "ShieldGemma 1" in guards
    assert "ShieldGemma 2" in guards


@pytest.mark.core
def test_hil_serl_source_marks_critic_boundary_and_real_intervention_mechanism():
    text = read_text(HIL_SERL)

    for stale_claim in ("10x+", "1-2 小时", "reward += +1", "bonus reward"):
        assert stale_claim not in text

    assert "教学 critic" in text
    assert "人类干预会覆盖策略动作" in text
    assert "二元任务成功分类器" in text
    assert "50/50 RLPD 采样" in text
    assert "干预 transition 还会额外复制到 intervention/demo buffer" in text
    assert "不能外推固定倍数或时长" in text


@pytest.mark.core
def test_mindmap_avoids_implementation_and_benchmark_numbers_as_constants():
    text = read_text(MINDMAP)
    folded = text.casefold()

    for stale_claim in (
        "小整数缓存: cpython 缓存 -5~256",
        "小整数缓存: -5~256",
        "默认 (700, 10, 10)",
        "节省 50%+ 内存",
        "写入 ×1.25",
        "节省 90%",
        "显存 60%→95%",
        "吞吐 10-20×",
        "延迟 30s-5min",
        "成本 ×3-10",
        "100-500 tokens",
        "openai o3",
        "o3/r1",
        "q2_k (2.7gb)",
        "iphone 15 pro (3b q4)",
        "rtx 5090 d",
        "qlora 7b 只需 ~6gb",
        "3-5 个示例效果最佳",
        "temperature: 0 确定性",
        '"budget_tokens": 5000',
        "deepseek-v3 (236b 总参, 21b 激活)",
        "l(n, d) ∝ n^0.34 d^0.34",
    ):
        assert stale_claim not in folded

    assert text.count("小整数复用: CPython 实现细节") == 2
    assert "不背固定默认值" in text
    assert "收益受属性、继承和版本影响，需实测" in text
    assert "usage 缓存字段与实际账单" in text
    assert "吞吐/尾延迟收益依工作负载而变" in text
    assert "按当前模型目录与 API 指南核对推理控制字段" in text
    assert "文件大小取决于模型参数与量化元数据" in text
    assert "按模型权重、KV cache、激活、batch 与并发核算显存" in text


@pytest.mark.core
def test_authoritative_source_ledger_covers_the_corrected_claims():
    text = read_text(SOURCE_LEDGER)

    for source in (
        "https://docs.python.org/3/c-api/long.html",
        "https://docs.python.org/3.14/library/gc.html",
        "https://arxiv.org/abs/2212.08073",
        "https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input",
        "https://www.anthropic.com/responsible-scaling-policy",
        "https://www.anthropic.com/news/responsible-scaling-policy-v3",
        "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
        "https://ai.google.dev/gemma/docs/shieldgemma/model_card_2",
        "https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines",
        "https://arxiv.org/abs/2410.21845",
        "https://github.com/rail-berkeley/hil-serl",
    ):
        assert source in text


@pytest.mark.gpu
def test_hil_serl_intervention_overrides_action_without_reward_bonus():
    torch = pytest.importorskip("torch")
    module = load_hil_serl_module()

    state = torch.zeros(14)
    next_state = torch.ones(14)
    policy_action = torch.zeros(7)
    expert_action = torch.ones(7)
    transition = module.record_online_transition(
        state=state,
        policy_action=policy_action,
        expert_action=expert_action,
        classifier_reward=0.0,
        next_state=next_state,
        done=False,
        intervened=True,
    )

    assert torch.equal(transition.action, expert_action)
    assert transition.reward == 0.0
    assert transition.intervened is True
    assert transition.source == "online"

    online_buffer = []
    demo_or_intervention_buffer = []
    module.store_online_transition(
        transition,
        online_buffer=online_buffer,
        demo_or_intervention_buffer=demo_or_intervention_buffer,
    )
    assert len(online_buffer) == 1 and online_buffer[0] is transition
    assert len(demo_or_intervention_buffer) == 1
    assert demo_or_intervention_buffer[0] is transition

    demo = module.Transition(
        state=state,
        action=expert_action,
        reward=1.0,
        next_state=next_state,
        done=True,
        intervened=False,
        source="demo",
    )
    mixed = module.balanced_replay_batch([demo], [transition], per_buffer=1, seed=7)
    assert [item.source for item in mixed].count("demo") == 1
    assert [item.source for item in mixed].count("online") == 1


@pytest.mark.gpu
def test_hil_serl_mock_path_never_checks_hardware(monkeypatch, capsys):
    module = load_hil_serl_module()

    monkeypatch.setattr(sys, "argv", [str(HIL_SERL), "--mock"])

    def fail_if_called():
        pytest.fail("mock path must return before hardware probing")

    monkeypatch.setattr(module, "check_hardware", fail_if_called)
    module.main()
    output = capsys.readouterr().out

    assert "[SKIP]" in output
    assert output.rstrip().endswith("OK")
