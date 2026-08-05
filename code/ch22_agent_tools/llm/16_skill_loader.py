# ---
# chapter: 23
# topic: MCP、A2A 与 Skills 协议生态
# topic_id: agent_tools.skill_loader
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: [pyyaml]
# run: python 16_skill_loader.py
# expected_runtime: <1s
# expected_output: 列出 Skills、按 tag 过滤、加载指定 Skill
# ---
# See: ../../../23_MCP_A2A与Skills协议生态.md
# Interview hooks:
#   1. SKILL.md 与传统 README.md / package.json 的关键差异？(声明式 vs 命令式)
#   2. Skills Marketplace 与 npm/PyPI 的本质区别？(由 LLM 解释执行 vs 解释器/编译器)
#   3. Skills 的"流程步骤"如何从 Markdown 中正则提取出来？有什么局限？

"""
Skills 加载器 - 从目录加载 SKILL.md 并提供给 Agent
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except Exception:  # pragma: no cover
    HAS_YAML = False


@dataclass
class Skill:
    """解析后的 Skill 对象"""

    name: str
    description: str
    version: str
    author: str
    tags: list
    inputs: list
    outputs: list
    tools: list = field(default_factory=list)
    flow_steps: list = field(default_factory=list)
    raw_markdown: str = ""
    file_path: str = ""


class SkillLoader:
    """
    Skills Marketplace 加载器

    使用示例：
        loader = SkillLoader(skills_dir="./skills")
        skill = loader.load("code-review")
        loader.list_by_tag("security")
    """

    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self._cache: dict[str, Skill] = {}

    def load(self, skill_name: str) -> Skill:
        """加载指定 Skill（带缓存）"""
        if skill_name in self._cache:
            return self._cache[skill_name]

        skill_file = self.skills_dir / skill_name / "SKILL.md"
        if not skill_file.exists():
            raise FileNotFoundError(f"Skill not found: {skill_file}")

        content = skill_file.read_text(encoding="utf-8")
        skill = self._parse(content)
        skill.file_path = str(skill_file)
        self._cache[skill_name] = skill
        return skill

    def list_all(self) -> list:
        """列出目录下所有 Skill"""
        skills = []
        if not self.skills_dir.exists():
            return skills
        for sub in self.skills_dir.iterdir():
            if sub.is_dir() and (sub / "SKILL.md").exists():
                try:
                    skills.append(self.load(sub.name))
                except Exception:
                    continue
        return skills

    def list_by_tag(self, tag: str) -> list:
        """按 tag 过滤"""
        return [s for s in self.list_all() if tag in s.tags]

    def _parse(self, content: str) -> Skill:
        """解析 SKILL.md（YAML Frontmatter + Markdown 正文）"""
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
        if not match:
            raise ValueError("SKILL.md 必须以 YAML frontmatter 开头")
        yaml_text, markdown = match.groups()
        if HAS_YAML:
            meta = yaml.safe_load(yaml_text) or {}
        else:
            # 极简 YAML 解析兜底（仅 key: value 形式）
            meta = {}
            for line in yaml_text.splitlines():
                if ":" in line and not line.strip().startswith("-"):
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()

        tools = re.findall(r"`([a-z_]+)`\s*[:：]", markdown)
        flow_steps = re.findall(r"^\d+\.\s+(.+)$", markdown, re.MULTILINE)

        return Skill(
            name=meta.get("name", ""),
            description=meta.get("description", ""),
            version=meta.get("version", "0.0.1"),
            author=meta.get("author", "unknown"),
            tags=meta.get("tags", []),
            inputs=meta.get("inputs", []),
            outputs=meta.get("outputs", []),
            tools=tools,
            flow_steps=flow_steps,
            raw_markdown=markdown,
        )


def _create_demo_skills():
    """在临时目录中创建几个示例 SKILL.md"""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    (tmp / "code-review" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp / "code-review" / "SKILL.md").write_text(
        """---
name: code-review
description: 对 Git diff 进行多维度代码审查，包括安全、性能、可读性
version: 1.0.0
author: community
tags:
  - code-review
  - security
  - performance
license: MIT
inputs:
  - name: diff
    type: string
    description: Git diff 内容
    required: true
outputs:
  - name: review_report
    type: object
---

# Code Review Skill

## 描述
对 Git diff 进行多维度代码审查，输出结构化报告。

## 工具依赖
- `read_file`: 读取 diff 文件
- `search_pattern`: 搜索可疑模式
- `language_detect`: 检测编程语言

## 执行流程
1. 解析 diff，识别变更的文件
2. 对每个文件进行语言检测
3. 加载对应语言的审查规则
4. 执行多维度检查
5. 汇总问题，输出结构化报告

## 回退策略
- diff 格式无法解析 → 报告错误并跳过审查
- 工具调用失败 → 重试 1 次 → 仍失败则返回降级报告
""",
        encoding="utf-8",
    )

    (tmp / "data-summary").mkdir(parents=True, exist_ok=True)
    (tmp / "data-summary" / "SKILL.md").write_text(
        """---
name: data-summary
description: 对 CSV 数据生成摘要
version: 0.1.0
author: alice
tags:
  - data
  - summary
---

# Data Summary Skill

## 工具依赖
- `read_csv`: 读取 CSV

## 执行流程
1. 加载 CSV 文件
2. 统计基本指标
3. 输出摘要报告
""",
        encoding="utf-8",
    )
    return tmp


def demo_skill_loader():
    skills_dir = _create_demo_skills()
    loader = SkillLoader(skills_dir=str(skills_dir))

    print("=== All Skills ===")
    for s in loader.list_all():
        print(f"- {s.name} v{s.version}: {s.description}")

    print("\n=== Security Skills ===")
    for s in loader.list_by_tag("security"):
        print(f"- {s.name}")

    skill = loader.load("code-review")
    print(f"\nLoaded: {skill.name}")
    print(f"Tools: {skill.tools}")
    print(f"Flow: {skill.flow_steps}")


if __name__ == "__main__":
    demo_skill_loader()
    print("OK")
