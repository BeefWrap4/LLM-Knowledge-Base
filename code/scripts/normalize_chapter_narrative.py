#!/usr/bin/env python3
# ---
# code/scripts/normalize_chapter_narrative.py
# 统一 40 章的学习导航、正文编号和固定结尾栏目。
# Usage: python code/scripts/normalize_chapter_narrative.py [--check]
# ---
"""Normalize chapter-level narrative structure without deleting technical content."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
REPO = CODE.parent
UPDATED_AT = "2026-08-04T00:00:00.000Z"


@dataclass(frozen=True)
class ChapterMeta:
    position: str
    prerequisites: tuple[int, ...]
    objectives: tuple[str, str, str]
    related: tuple[int, ...]


CHAPTER_META: dict[int, ChapterMeta] = {
    1: ChapterMeta(
        "全书起点，建立后续 Web、数据与大模型代码共同依赖的 Python 基础。",
        (),
        (
            "解释 Python 执行模型、核心语法和常用数据结构",
            "编写包含函数、模块和异常处理的可运行程序",
            "诊断参数传递、作用域和数据结构相关的常见陷阱",
        ),
        (2, 5),
    ),
    2: ChapterMeta(
        "从语法进入对象语义，解决赋值、共享引用和复制行为容易混淆的问题。",
        (1,),
        (
            "区分对象身份、相等性、可变性与哈希性",
            "预测赋值、浅拷贝和深拷贝后的引用关系",
            "根据嵌套结构和性能约束选择复制策略",
        ),
        (3, 6),
    ),
    3: ChapterMeta(
        "建立 Python 面向对象模型，为框架源码、依赖注入和设计模式打基础。",
        (1, 2),
        (
            "解释类、对象、继承、描述符和元类的协作关系",
            "实现职责清晰且可测试的类层次",
            "诊断 MRO、属性查找和对象构造相关问题",
        ),
        (4, 9),
    ),
    4: ChapterMeta(
        "连接 Python 语言机制与工程抽象，支撑装饰器、中间件和资源管理。",
        (1, 3),
        (
            "解释闭包、装饰器、生成器和上下文管理协议",
            "实现可组合的函数增强与惰性数据处理",
            "根据状态、资源和可读性约束选择抽象方式",
        ),
        (5, 9),
    ),
    5: ChapterMeta(
        "建立并发模型，为异步 API、任务队列和高吞吐服务提供基础。",
        (1, 4),
        (
            "比较线程、进程和协程的调度与隔离边界",
            "实现能够正确取消、超时和回收资源的并发程序",
            "根据 CPU、I/O 和故障隔离需求选择并发方案",
        ),
        (9, 24),
    ),
    6: ChapterMeta(
        "解释 Python 内存与生命周期，为性能诊断和长时间服务稳定性打基础。",
        (1, 2),
        (
            "解释引用计数、循环垃圾回收和对象分配机制",
            "使用可复现方法定位内存增长与对象滞留",
            "根据证据选择 profiling、弱引用或生命周期修复方案",
        ),
        (11, 20),
    ),
    7: ChapterMeta(
        "建立复杂度与数据结构直觉，为检索、调度和算法面试提供工具。",
        (1,),
        (
            "根据访问模式选择合适的数据结构并分析复杂度",
            "实现常见排序、图、树和动态规划算法",
            "解释时间、空间、可维护性之间的工程取舍",
        ),
        (8, 10),
    ),
    8: ChapterMeta(
        "把 Python 基础扩展到可复现的数据处理与特征分析流程。",
        (1, 7),
        (
            "使用 NumPy、Pandas 和可视化工具处理结构化数据",
            "构建从读取、清洗到分析输出的可复现流程",
            "诊断副本、缺失值、泄漏和统计口径问题",
        ),
        (10, 22),
    ),
    9: ChapterMeta(
        "将 Python 能力组织成可部署 API，为 RAG、Agent 和模型服务提供入口。",
        (1, 5),
        (
            "解释 FastAPI 请求生命周期、类型校验和依赖注入",
            "实现包含异步、鉴权和错误处理的 API",
            "设计具备观测、限流和安全边界的服务接口",
        ),
        (14, 24),
    ),
    10: ChapterMeta(
        "建立传统机器学习问题求解闭环，为深度学习和模型评估提供基线。",
        (7, 8),
        (
            "解释监督学习、无监督学习和评估的基本假设",
            "实现包含预处理、训练和交叉验证的基线流程",
            "诊断数据泄漏、类别不平衡和过拟合问题",
        ),
        (11, 17),
    ),
    11: ChapterMeta(
        "从传统机器学习进入张量计算与神经网络训练，为 Transformer 奠基。",
        (1, 8, 10),
        (
            "解释张量、自动微分和神经网络训练循环",
            "实现可复现的 PyTorch 训练与评估流程",
            "诊断梯度、设备、精度和数据管道问题",
        ),
        (12, 19),
    ),
    12: ChapterMeta(
        "建立大模型共同原理，连接注意力、Transformer 架构和自回归生成。",
        (10, 11),
        (
            "推导注意力并跟踪 Transformer 中的张量形状",
            "解释训练、解码和 KV Cache 的数据流",
            "比较编码器、解码器及现代架构变体的适用边界",
        ),
        (13, 16),
    ),
    13: ChapterMeta(
        "从模型原理进入可控交互，建立 Prompt、上下文和生成参数的设计方法。",
        (12,),
        (
            "设计目标明确、约束可检验的 Prompt",
            "评估采样、缓存、推理模式和注入防御的效果",
            "根据任务风险与成本选择提示和生成策略",
        ),
        (14, 15),
    ),
    14: ChapterMeta(
        "构建外部知识增强闭环，是企业知识问答和 Agent 检索能力的核心。",
        (12, 13),
        (
            "解释 RAG 从摄取、索引、检索到生成的完整链路",
            "实现并评估一个可追踪来源的最小 RAG 系统",
            "根据坏例区分检索、上下文和生成问题",
        ),
        (15, 17),
    ),
    15: ChapterMeta(
        "把模型、工具和状态组织成可恢复任务系统，是应用工程主线的核心章节。",
        (13, 14),
        (
            "解释 ReAct、工具调用、MCP 和多 Agent 的执行边界",
            "设计具备状态、幂等和恢复能力的 Agent 工作流",
            "用评测与可观测性定位工具、副作用和编排故障",
        ),
        (20, 35),
    ),
    16: ChapterMeta(
        "连接训练后对齐、参数高效微调与推理部署，形成模型能力改造闭环。",
        (11, 12),
        (
            "比较 SFT、偏好优化和参数高效微调方法",
            "估算训练、权重和 KV Cache 的资源需求",
            "根据质量、成本和硬件选择优化与部署方案",
        ),
        (19, 25),
    ),
    17: ChapterMeta(
        "建立可测量的质量闭环，为 RAG、Agent、模型选择和上线门禁提供证据。",
        (10, 12),
        (
            "设计覆盖质量、安全和工程指标的评估体系",
            "构建数据集并正确使用自动指标与 LLM-as-Judge",
            "解释置信度、偏差、坏例和回归结果",
        ),
        (14, 20),
    ),
    18: ChapterMeta(
        "把 RAG、Agent 和微调能力映射到主流框架，重点训练抽象与选型判断。",
        (13, 14, 15),
        (
            "比较主流 LLM 框架的核心抽象和版本边界",
            "运行一个最小框架示例并识别状态与数据流",
            "根据可控性、生态和运维成本做出选型",
        ),
        (20, 24),
    ),
    19: ChapterMeta(
        "从单卡训练扩展到多设备系统，建立并行、通信和显存之间的统一模型。",
        (11, 12, 16),
        (
            "解释数据、张量、流水线并行及 ZeRO 的通信边界",
            "估算不同并行策略的显存与通信成本",
            "使用利用率和通信证据诊断扩展效率",
        ),
        (25, 36),
    ),
    20: ChapterMeta(
        "把实验、调用、成本和质量证据串成生产运行闭环。",
        (9, 15, 17),
        (
            "设计覆盖 trace、metric、log、eval 和 cost 的观测模型",
            "实现默认离线且可回归的 LLMOps 流程",
            "根据链路证据定位质量、延迟和成本异常",
        ),
        (24, 40),
    ),
    21: ChapterMeta(
        "从文本模型扩展到视觉、语音和生成模型，建立跨模态表示与对齐直觉。",
        (11, 12),
        (
            "解释多模态编码、对齐、融合与生成的主要架构",
            "比较视觉语言模型、扩散模型和统一模型的能力边界",
            "评估多模态数据、指标、安全和部署约束",
        ),
        (26, 39),
    ),
    22: ChapterMeta(
        "把模型效果追溯到数据生命周期，覆盖采集、治理、标注和持续迭代。",
        (8, 14, 16),
        (
            "设计从采集、清洗、去重到版本管理的数据流水线",
            "构建可审计的 SFT、偏好和 Agent 轨迹数据",
            "评估质量、偏差、许可和隐私风险",
        ),
        (17, 23),
    ),
    23: ChapterMeta(
        "为 Prompt、RAG、Agent 和数据链路建立威胁模型与治理边界。",
        (13, 15, 22),
        (
            "识别注入、越狱、隐私和 Agent 副作用威胁",
            "设计分层防御、权限隔离和审计机制",
            "依据测试证据与法规边界评估剩余风险",
        ),
        (20, 40),
    ),
    24: ChapterMeta(
        "把模型应用封装为可交付服务，连接容器、编排、网关和持续交付。",
        (9, 19, 20),
        (
            "容器化模型服务并解释运行时资源边界",
            "设计具备扩缩容、流量治理和故障恢复的部署架构",
            "验证 CI/CD、可观测性和供应链安全",
        ),
        (25, 37),
    ),
    25: ChapterMeta(
        "深入推理数据面与服务指标，是推理工程和容量规划的核心章节。",
        (12, 16, 19),
        (
            "解释 Prefill、Decode、KV Cache 和批处理的性能瓶颈",
            "用 TTFT、TPOT、吞吐和显存指标设计基准",
            "根据模型、硬件和流量选择推理引擎",
        ),
        (28, 37),
    ),
    26: ChapterMeta(
        "连接多模态模型与物理环境，建立世界模型、VLA 和机器人学习全景。",
        (11, 21),
        (
            "解释世界模型、VLA、模仿学习和机器人策略的关系",
            "设计数据采集、训练、仿真和控制闭环",
            "评估 sim-to-real、安全和硬件约束",
        ),
        (39,),
    ),
    27: ChapterMeta(
        "分析推理模型如何在推理时分配计算，并把能力提升转化为可评估预算。",
        (12, 16),
        (
            "解释 Test-Time Compute、过程奖励和搜索式推理",
            "设计同时约束正确率、延迟与 token 预算的评估",
            "判断增加推理计算何时产生净收益",
        ),
        (33,),
    ),
    28: ChapterMeta(
        "把模型部署约束下沉到个人设备、浏览器和边缘节点。",
        (12, 16, 25),
        (
            "比较 MLX、llama.cpp、WebGPU 和移动端推理栈",
            "估算端侧模型的内存、功耗和延迟需求",
            "根据隐私、体验和维护成本选择端云方案",
        ),
        (29,),
    ),
    29: ChapterMeta(
        "把上下文视为有限系统资源，统一 Prompt、检索、记忆和缓存设计。",
        (13, 14, 15),
        (
            "解释上下文组成、注意力退化和窗口经济学",
            "设计压缩、裁剪、记忆和缓存策略",
            "根据坏例诊断上下文缺失、污染和顺序问题",
        ),
        (35,),
    ),
    30: ChapterMeta(
        "在 Transformer 之外建立状态空间与线性序列模型的原理和选型框架。",
        (11, 12),
        (
            "解释 SSM、选择性扫描与 Mamba 的递推机制",
            "比较注意力、SSM 和混合架构的复杂度与状态",
            "根据任务证据判断替代或混合 Transformer 的价值",
        ),
        (32,),
    ),
    31: ChapterMeta(
        "区分参数知识、外部知识和可删除记忆，建立知识修改的工程边界。",
        (12, 14),
        (
            "区分知识编辑、持续学习、机器遗忘和 RAG",
            "解释 ROME、MEMIT 等方法的目标与评估",
            "根据可追溯性、更新频率和风险选择方案",
        ),
        (35,),
    ),
    32: ChapterMeta(
        "拆解现代高效大模型中的 MLA、MoE、MTP 和低精度训练机制。",
        (12, 19, 30),
        (
            "解释 MLA、细粒度 MoE、MTP 和 FP8 的协作方式",
            "计算关键结构对显存、通信和吞吐的影响",
            "区分公开证据、可复现实现和未经证实推断",
        ),
        (33, 37),
    ),
    33: ChapterMeta(
        "把训练异常转化为可观测、可归因和可恢复的诊断流程。",
        (11, 19),
        (
            "根据 loss、梯度、激活和数据证据定位训练异常",
            "设计 checkpoint、回放和恢复验收流程",
            "根据任务与稳定性证据选择优化器和裁剪策略",
        ),
        (36,),
    ),
    34: ChapterMeta(
        "解释文本如何进入模型，并把词表设计连接到成本、多语言和领域效果。",
        (1, 12, 22),
        (
            "解释 BPE、Unigram 和 SentencePiece 的训练与解码",
            "测量压缩率、未知词和多语言 token 效率",
            "根据领域数据与兼容性设计词表适配",
        ),
        (13,),
    ),
    35: ChapterMeta(
        "把 Agent 记忆从单一聊天历史升级为可治理的生产级状态系统。",
        (15, 29, 31),
        (
            "设计工作、情景、语义和程序四层记忆",
            "比较 Mem0、Zep、Graphiti 与 Letta 的数据模型",
            "处理检索、冲突、一致性、隐私和遗忘问题",
        ),
        (40,),
    ),
    36: ChapterMeta(
        "补齐 JAX、XLA 与 TPU 训练栈，建立函数变换和显式分片心智模型。",
        (11, 19, 33),
        (
            "解释 JAX 函数变换、XLA 编译和数组分片",
            "迁移一个 PyTorch 训练思路并识别语义差异",
            "根据硬件、生态和规模选择 JAX/TPU 路线",
        ),
        (37,),
    ),
    37: ChapterMeta(
        "深入 Prefill/Decode 分离与 KV 池化，分析跨节点推理的数据路径。",
        (19, 25),
        (
            "解释 PD 分离、KV 传输和资源池化的系统动机",
            "建立计算、网络、排队与命中率的延迟模型",
            "根据流量和基础设施判断分离部署是否值得",
        ),
        (24,),
    ),
    38: ChapterMeta(
        "建立无需完整再训练的权重组合方法，并强调评测与许可边界。",
        (16, 17),
        (
            "解释 Linear、SLERP、Task Arithmetic、TIES 和 DARE",
            "设计可复现的 MergeKit 合并与回滚计划",
            "评估能力干扰、安全、许可和部署收益",
        ),
        (40,),
    ),
    39: ChapterMeta(
        "把 Agent 扩展到真实 GUI 环境，连接数据、策略学习、沙箱和安全评测。",
        (15, 21, 23),
        (
            "解释 Computer-Use 任务、环境和评测协议",
            "设计从轨迹数据、SFT 到 RL 的训练闭环",
            "评估权限隔离、动作副作用和基准有效性",
        ),
        (40,),
    ),
    40: ChapterMeta(
        "将前 39 章知识压缩成可验证的项目证据、系统设计和面试表达。",
        (14, 15, 17, 20, 25),
        (
            "从公开 JD 提取岗位能力与证据要求",
            "用项目卡、指标卡和坏例卡组织真实经历",
            "完成分层回答、追问演练和 14 天训练计划",
        ),
        (),
    ),
}


CANONICAL_HEADINGS = {
    "summary": "## 🧭 本章小结",
    "selftest": "## ✅ 自测与练习",
    "code": "## 🧪 配套代码与验收",
    "interview": "## 🎯 面试题精讲",
    "quick": "## 📋 本章速查表",
    "knowledge": "## 🗺️ 知识地图",
    "related": "## 🔗 相关章节",
    "sources": "## 📖 一手参考资料",
}
APPENDIX_ORDER = ("summary", "selftest", "code", "interview", "quick", "knowledge", "related", "sources")


def chapter_files(repo: Path = REPO) -> list[Path]:
    return sorted(path for path in repo.glob("[0-9][0-9]_*.md") if 1 <= int(path.name[:2]) <= 40)


def _fence_aware_h2_indexes(lines: list[str]) -> list[int]:
    indexes: list[int] = []
    fence_char: str | None = None
    fence_length = 0
    for index, line in enumerate(lines):
        if fence_char is not None:
            if re.match(rf"^\s{{0,3}}{re.escape(fence_char)}{{{fence_length},}}\s*$", line):
                fence_char = None
                fence_length = 0
            continue
        fence_match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)
            fence_char = marker[0]
            fence_length = len(marker)
            continue
        if re.match(r"^##\s+", line):
            indexes.append(index)
    return indexes


def _classify_heading(title: str) -> str | None:
    plain = re.sub(r"^\d+(?:\.\d+)*(?:\.x)?\s+", "", title).strip()
    if re.search(r"本章小结|章节小结", plain):
        return "summary"
    if re.search(r"自测与练习|自测题|章节练习", plain):
        return "selftest"
    if re.search(r"配套代码|代码运行|运行说明|运行与验收|代码与验收", plain):
        return "code"
    if re.search(r"面试题|面试真题|真题精选|面试高频", plain):
        return "interview"
    if re.search(r"速查表|知识速查|检查清单", plain):
        return "quick"
    if re.search(
        r"思维导图|知识地图|内存布局图解|作用域查找流程图|GPU 调度与模型服务架构|推理网关流量管理", plain
    ):
        return "knowledge"
    if "相关章节" in plain:
        return "related"
    if re.search(
        r"一手参考|参考资料|参考文献|延伸阅读|资料来源|权威资料|官方资料|核心论文索引",
        plain,
    ):
        return "sources"
    return None


def _clean_body(lines: list[str]) -> list[str]:
    body = list(lines)
    while body and not body[0].strip():
        body.pop(0)
    while body and (not body[-1].strip() or body[-1].strip() == "---"):
        body.pop()
    return body


def _chapter_map(repo: Path) -> dict[int, Path]:
    return {int(path.name[:2]): path for path in chapter_files(repo)}


def _wikilink(chapter: int, paths: dict[int, Path]) -> str:
    return f"[[{paths[chapter].stem}]]"


def _code_directory(chapter: int, repo: Path) -> tuple[str | None, str | None]:
    matches = sorted((repo / "code").glob(f"ch{chapter:02d}_*"))
    if not matches:
        return None, None
    directory = matches[0]
    for tier in ("core", "llm", "gpu"):
        if (directory / tier).is_dir():
            return directory.relative_to(repo).as_posix(), tier
    return directory.relative_to(repo).as_posix(), None


def _navigation_block(
    chapter: int,
    meta: ChapterMeta,
    paths: dict[int, Path],
    code_dir: str | None,
    main_titles: list[str],
) -> list[str]:
    prereqs = "、".join(_wikilink(value, paths) for value in meta.prerequisites) or "无硬性先修"
    clean_titles = [
        re.sub(r"\s*[⭐★]+\s*$", "", re.sub(r"^\d+(?:\.\d+)*\s+", "", title)).strip() for title in main_titles
    ]
    if len(clean_titles) > 4:
        route = " → ".join([*clean_titles[:3], "…", clean_titles[-1]])
    else:
        route = " → ".join(clean_titles)
    route = route or "问题与直觉 → 原理与示例 → 工程边界 → 复习与验收"
    code_text = f"`{code_dir}/`" if code_dir else "本章暂无独立代码目录，使用正文推导、自测题和决策表验收"
    return [
        "> [!abstract] 本章导航",
        f"> **定位**：{meta.position}",
        ">",
        f"> **先修**：{prereqs}。",
        ">",
        "> **学习目标**：",
        *[f"> - {objective}。" for objective in meta.objectives],
        ">",
        f"> **建议路径**：{route}。先完成主线，再按需要阅读进阶内容。",
        ">",
        f"> **配套代码**：{code_text}。",
    ]


def _remove_existing_navigation(lines: list[str]) -> list[str]:
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == "> [!abstract] 本章导航")
    except StopIteration:
        return lines
    end = start
    while end < len(lines) and lines[end].startswith(">"):
        end += 1
    while end < len(lines) and not lines[end].strip():
        end += 1
    return [*lines[:start], *lines[end:]]


def _normalize_opening_context(lines: list[str]) -> list[str]:
    """Turn the legacy chapter-opening quote into one neutral reading note."""

    try:
        h1_index = next(index for index, line in enumerate(lines) if re.match(r"^#\s+", line))
    except StopIteration:
        return lines

    start = h1_index + 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or not lines[start].startswith(">"):
        return lines

    end = start
    while end < len(lines) and lines[end].startswith(">"):
        end += 1

    body: list[str] = []
    for line in lines[start:end]:
        stripped = line.strip()
        if stripped == "> [!info] 阅读提示":
            continue
        if "**面试频率**" in line or "**面试重要度**" in line:
            continue
        line = line.replace("🆕 **2026年更新**", "**版本与范围**")
        line = line.replace("🆕 **2026 年更新**", "**版本与范围**")
        line = line.replace("🆕 **2026年新内容**", "**版本与范围**")
        line = line.replace("首选架构", "常用架构")
        line = line.replace(
            "有效解决了模型知识过时、幻觉、无法访问私有数据三大痛点",
            "用于缓解模型知识过时、幻觉和无法访问私有数据等问题",
        )
        line = line.replace("完整覆盖 RAG 技术栈的每一个关键环节", "串联 RAG 技术栈的关键环节")
        line = line.replace("等最新趋势", "等截至审校日的技术方向")
        line = line.replace("爆炸式增长与洗牌", "持续演进")
        line = line.replace(
            '让 AI 拥有了"眼睛和耳朵"，能同时理解',
            "让 AI 能够联合处理",
        )
        line = line.replace("扩散模型的生成革命", "扩散模型的生成机制")
        line = line.replace("核心原理与2026年最新进展", "核心原理与截至审校日的进展")
        line = line.replace(
            "是通往通用人工智能（AGI）的必经之路",
            "是扩展模型感知与生成能力的重要方向",
        )
        body.append(line)

    while body and body[0].strip() == ">":
        body.pop(0)
    while body and body[-1].strip() == ">":
        body.pop()

    collapsed: list[str] = []
    for line in body:
        if line.strip() == ">" and collapsed and collapsed[-1].strip() == ">":
            continue
        collapsed.append(line)

    replacement = ["> [!info] 阅读提示", *collapsed] if collapsed else []
    before = lines[: h1_index + 1]
    after = lines[end:]
    while after and not after[0].strip():
        after.pop(0)
    return [*before, "", *replacement, "", *after] if replacement else [*before, "", *after]


def _neutralize_opening_prose(lines: list[str]) -> list[str]:
    """Remove unsupported frequency claims and promotional wording from the prelude."""

    replacements = {
        "并发编程是 Python 后端/高并发岗位面试的绝对核心，约 **95%** 的中高级岗位会深入考察。": (
            "并发编程是 Python 后端与高并发岗位的重要基础，中高级岗位常从原理、选型和故障处理继续追问。"
        ),
        "数据结构与算法是技术面试的核心环节，几乎每场面试都包含手写代码环节。": (
            "数据结构与算法是常见的技术面试环节，重点考察问题建模、复杂度分析和代码正确性。"
        ),
        "大模型应用开发的最后一公里是将模型能力封装为可调用的 API 服务。": (
            "大模型能力需要通过可调用的 API 服务进入可集成、可运维的应用系统。"
        ),
        "面试中，经典算法原理、模型评估指标、偏差-方差权衡等知识点的考察频率与大模型原理不相上下。": (
            "经典算法原理、模型评估指标和偏差—方差权衡，仍是理解大模型训练与评估的重要基础。"
        ),
        "面试中，PyTorch 核心操作、反向传播原理、CNN/RNN 架构细节、训练优化技巧等知识点的考察贯穿始终。": (
            "PyTorch 核心操作、反向传播、常见网络结构和训练诊断，是算法与模型工程岗位的常见考查内容。"
        ),
        "Transformer 架构统治了 NLP、计算机视觉、多模态等几乎所有深度学习领域。": (
            "Transformer 架构已广泛应用于 NLP、计算机视觉和多模态等深度学习任务。"
        ),
        "本章是全教程最重要的一章，每个知识点都可能直接决定面试成败。": (
            "本章是后续 Prompt、RAG、Agent、训练与推理章节的共同基础，学习时应同时关注公式、张量形状和工程边界。"
        ),
        "大模型推理是 2026 年大模型应用落地最关键的工程环节。": (
            "大模型推理是模型能力转化为稳定在线服务的关键工程环节。"
        ),
        "将大模型部署到边缘设备（手机、PC、嵌入式）是 2026 年最热的方向之一。": (
            "将大模型部署到手机、PC 和嵌入式设备，是持续发展的端侧工程方向。"
        ),
    }
    result: list[str] = []
    for line in lines:
        for old, new in replacements.items():
            line = line.replace(old, new)
        result.append(line)
    return result


def _ensure_opening_intro(lines: list[str], h1_index: int, meta: ChapterMeta) -> list[str]:
    content = [line for line in lines[h1_index + 1 :] if line.strip() and line.strip() != "---"]
    if content:
        return lines
    intro = (
        f"本章围绕“{meta.position.rstrip('。')}”展开。"
        "阅读时先建立问题边界和最小心智模型，再通过示例、失败条件与自测完成验证。"
    )
    return [*lines[: h1_index + 1], "", intro]


def _add_updated_frontmatter(lines: list[str]) -> list[str]:
    if not lines or lines[0].strip() != "---":
        return lines
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration:
        return lines
    updated_index = next((index for index in range(1, end) if lines[index].startswith("updated:")), None)
    if updated_index is None:
        created_index = next(
            (index for index in range(1, end) if lines[index].startswith("created:")), end - 1
        )
        lines.insert(created_index + 1, f"updated: {UPDATED_AT}")
    else:
        lines[updated_index] = f"updated: {UPDATED_AT}"
    return lines


def _normalize_h1(lines: list[str], chapter: int) -> list[str]:
    for index, line in enumerate(lines):
        if re.match(r"^#\s+", line):
            title = re.sub(
                r"^#\s+(?:第\s*\d+\s*章|\d+)\s*",
                f"# 第 {chapter} 章 ",
                line,
            )
            lines[index] = re.sub(r"\s+", " ", title).strip()
            break
    return lines


def _normalize_heading_style(lines: list[str]) -> list[str]:
    """Keep decorative icons out of content headings and remove local pseudo-numbering."""

    result: list[str] = []
    fence_char: str | None = None
    fence_length = 0
    for line in lines:
        if fence_char is not None:
            result.append(line)
            if re.match(rf"^\s{{0,3}}{re.escape(fence_char)}{{{fence_length},}}\s*$", line):
                fence_char = None
                fence_length = 0
            continue
        fence_match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)
            fence_char = marker[0]
            fence_length = len(marker)
            result.append(line)
            continue
        heading_match = re.match(r"^(#{2,6})\s+(.+)$", line)
        if heading_match is None:
            result.append(line)
            continue
        marker, title = heading_match.groups()
        if line not in CANONICAL_HEADINGS.values():
            title = title.replace("🆕", "").replace("🎯", "")
        if len(marker) >= 3:
            title = re.sub(r"^\s*\d+[.)]\s+", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        result.append(f"{marker} {title}")
    return result


def _normalize_editorial_labels(lines: list[str]) -> list[str]:
    return [
        line.replace("面试金句", "回答要点").replace("面试必考", "面试重点").replace("面试必问", "面试常见")
        for line in lines
    ]


def _renumber_main_sections(chapter: int, sections: list[list[str]]) -> list[list[str]]:
    result: list[list[str]] = []
    counter = 0
    for section in sections:
        heading = section[0]
        match = re.match(r"^##\s+(\d+\.\d+(?:\.\d+)?)\s+(.+)$", heading)
        if not match:
            result.append(section)
            continue
        counter += 1
        old_prefix = match.group(1)
        new_prefix = f"{chapter}.{counter}"
        title = match.group(2)
        updated: list[str] = []
        for line in section:
            updated.append(
                re.sub(
                    rf"^(#{{2,6}}\s+){re.escape(old_prefix)}(?=\.|\s)",
                    rf"\g<1>{new_prefix}",
                    line,
                )
            )
        updated[0] = f"## {new_prefix} {title}"
        result.append(updated)
    return result


def _generated_summary(meta: ChapterMeta) -> list[str]:
    return ["本章应形成以下可复述结论：", "", *[f"- {item}。" for item in meta.objectives]]


def _generated_selftest(meta: ChapterMeta) -> list[str]:
    return [
        "先合上正文，再回答以下问题；无法说明证据或边界时，回到对应小节复习。",
        "",
        *[f"{index}. 你能否{objective}？" for index, objective in enumerate(meta.objectives, start=1)],
    ]


def _generated_code(chapter: int, code_dir: str | None, tier: str | None) -> list[str]:
    if not code_dir or not tier:
        return [
            "本章暂无独立代码目录。验收时应完成正文中的推导或决策题，并能在自测中说明适用边界。",
            "",
            "成功标准：概念、输入输出、关键指标和失败条件能够相互对应，不用未经验证的性能数字代替结论。",
        ]
    timeout = 180 if tier in {"llm", "gpu"} else 60
    prefix = ['$env:LLM_MOCK = "1"'] if tier == "llm" else []
    return [
        f"配套目录：`{code_dir}/`。从 `code/` 目录运行：",
        "",
        "```powershell",
        *prefix,
        f"python scripts/run_all_examples.py --tier {tier} --chapter ch{chapter:02d} --parallel 1 --timeout {timeout}",
        "```",
        "",
        "成功标准：命令退出码为 0，示例输出 `OK`；缺少可选依赖时必须给出明确 `[SKIP]`，而不是 traceback。",
        "真实 API、GPU、模型下载和付费调用不属于默认离线验收，必须按示例 metadata 与章节说明单独确认。",
    ]


def _generated_interview(meta: ChapterMeta) -> list[str]:
    return [
        "1. 用 30 秒说明本章解决的问题、核心机制和一个适用边界。",
        f"2. 如果面试官要求落地，你会如何{meta.objectives[1]}？",
        f"3. 给出一个反例或失败场景，说明如何{meta.objectives[2]}。",
        "",
        "回答时采用“问题 → 机制 → 证据 → 取舍”的结构；没有线上数据时，明确区分离线实验、个人项目和生产结果。",
    ]


def _generated_quick(meta: ChapterMeta) -> list[str]:
    rows = ["| 学习主题 | 验收标准 |", "|---|---|"]
    rows.extend(f"| {objective} | 能复述方法、证据和适用边界 |" for objective in meta.objectives)
    return rows


def _generated_related(meta: ChapterMeta, paths: dict[int, Path]) -> list[str]:
    lines: list[str] = []
    for chapter in meta.prerequisites:
        lines.append(f"- {_wikilink(chapter, paths)}：提供本章依赖的前置概念。")
    for chapter in meta.related:
        lines.append(f"- {_wikilink(chapter, paths)}：承接本章方法并进入下一层应用或工程问题。")
    if not lines:
        lines.append("- [[00_目录索引]]：返回全书学习路线，按目标岗位选择下一章。")
    return lines


def _generated_sources() -> list[str]:
    return [
        "> 核验日期：2026-08-04。版本、价格、法规、模型能力和 benchmark 以链接页面当前状态为准。",
        "",
        "- [[docs/AUTHORITATIVE_SOURCES|章节权威来源索引]]：按章节维护的官方文档、标准、原论文和官方仓库。",
    ]


def _merge_existing(category: str, entries: list[tuple[str, list[str]]]) -> list[str]:
    if not entries:
        return []
    merged: list[str] = []
    multiple = len(entries) > 1
    for title, body in entries:
        clean = _clean_body(body)
        if not clean:
            continue
        if multiple:
            subtitle = re.sub(r"^\d+(?:\.\d+)*(?:\.x)?\s+", "", title).strip()
            subtitle = re.sub(r"^[📋📊📚🎯🧭🧪🔗📖🗺️]\s*", "", subtitle)
            merged.extend([f"### {subtitle}", ""])
        merged.extend([*clean, ""])
    return _clean_body(merged)


def normalize_chapter(path: Path, repo: Path = REPO) -> str:
    chapter = int(path.name[:2])
    meta = CHAPTER_META[chapter]
    paths = _chapter_map(repo)
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    lines = _remove_existing_navigation(lines)
    lines = _add_updated_frontmatter(lines)
    lines = _normalize_h1(lines, chapter)
    lines = _normalize_heading_style(lines)
    lines = _normalize_editorial_labels(lines)
    lines = _normalize_opening_context(lines)

    h2_indexes = _fence_aware_h2_indexes(lines)
    if not h2_indexes:
        raise ValueError(f"{path.name} has no H2 sections")
    prelude = lines[: h2_indexes[0]]
    sections = [
        lines[start : h2_indexes[index + 1] if index + 1 < len(h2_indexes) else len(lines)]
        for index, start in enumerate(h2_indexes)
    ]

    main_sections: list[list[str]] = []
    appendices: dict[str, list[tuple[str, list[str]]]] = {key: [] for key in APPENDIX_ORDER}
    for section in sections:
        title = section[0][3:].strip()
        category = _classify_heading(title)
        if category is None:
            main_sections.append(section)
        else:
            appendices[category].append((title, section[1:]))

    main_sections = _renumber_main_sections(chapter, main_sections)
    main_titles = [
        section[0][3:].strip() for section in main_sections if re.match(r"^##\s+\d+\.\d+", section[0])
    ]
    code_dir, tier = _code_directory(chapter, repo)

    try:
        h1_index = next(index for index, line in enumerate(prelude) if re.match(r"^#\s+", line))
    except StopIteration as exc:
        raise ValueError(f"{path.name} has no H1") from exc
    prelude = _neutralize_opening_prose(prelude)
    prelude = _ensure_opening_intro(prelude, h1_index, meta)
    navigation = _navigation_block(chapter, meta, paths, code_dir, main_titles)
    remainder = prelude[h1_index + 1 :]
    while remainder and not remainder[0].strip():
        remainder.pop(0)
    prelude = [*prelude[: h1_index + 1], "", *navigation, "", *remainder]
    while len(prelude) >= 2 and not prelude[-1].strip() and not prelude[-2].strip():
        prelude.pop()
    while prelude and not prelude[-1].strip():
        prelude.pop()
    if prelude and prelude[-1].strip() == "---":
        prelude.pop()
    while prelude and not prelude[-1].strip():
        prelude.pop()

    generated = {
        "summary": _generated_summary(meta),
        "selftest": _generated_selftest(meta),
        "code": _generated_code(chapter, code_dir, tier),
        "interview": _generated_interview(meta),
        "quick": _generated_quick(meta),
        "knowledge": [],
        "related": _generated_related(meta, paths),
        "sources": _generated_sources(),
    }

    output: list[str] = [*prelude]
    for section in main_sections:
        output.extend(["", *_clean_body(section)])

    for category in APPENDIX_ORDER:
        existing = _merge_existing(category, appendices[category])
        body = existing or generated[category]
        if category == "sources" and existing and "AUTHORITATIVE_SOURCES" not in "\n".join(existing):
            body = [*existing, "", *_generated_sources()]
        if category == "knowledge" and not body:
            continue
        output.extend(["", CANONICAL_HEADINGS[category], "", *body])

    normalized = "\n".join(output).rstrip() + "\n"
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report files that are not normalized")
    args = parser.parse_args()
    changed: list[str] = []
    for path in chapter_files():
        normalized = normalize_chapter(path)
        current = path.read_text(encoding="utf-8")
        if normalized == current:
            continue
        changed.append(path.name)
        if not args.check:
            path.write_text(normalized, encoding="utf-8", newline="\n")
    if changed:
        action = "would update" if args.check else "updated"
        print(f"{action}: {len(changed)} chapter(s)")
        for name in changed:
            print(f"- {name}")
        return 1 if args.check else 0
    print("OK: all 40 chapters use the normalized narrative structure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
