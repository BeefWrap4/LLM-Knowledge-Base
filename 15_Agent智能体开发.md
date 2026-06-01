---
chapter: 15
topic: Agent智能体开发
difficulty: 中高
interview_frequency: 5
created: 2026-06-01T00:00:00.000Z
tags:
  - Agent
  - 智能体
  - MCP
  - Function-Calling
  - 大模型应用
---
# 第15章 Agent 智能体开发 ⭐⭐⭐⭐⭐

> **面试频率**：极高（2025-2026年持续最热方向，几乎必考）| **技术热度**：★★★★★
>
> Agent（智能体）是大模型应用开发的下一个范式。从 ReAct 框架到 Function Calling，从 MCP 协议到 Multi-Agent 协作，Agent 正在重塑人机交互的边界。本章深入解析 Agent 的四大核心模块、手写 ReAct Agent、多工具调用、MCP 协议、A2A 协议、Agent Teams 等 2025-2026 年持续最热考点，助你在面试中从容应对。

---

## 15.1 Agent 基础概念 ⭐⭐⭐⭐⭐

### 15.1.1 什么是 AI Agent

AI Agent（人工智能智能体）是一个能够**感知环境、自主决策、执行行动**的系统。与大模型的单次调用不同，Agent 具备**持续交互**和**目标驱动**的能力。

**普通 LLM 调用 vs Agent**：

| 维度 | 普通 LLM 调用 | AI Agent |
|------|-------------|----------|
| **交互模式** | 单轮/多轮对话 | 持续的感知-思考-行动循环 |
| **外部工具** | 无 | 可调用 API、搜索、数据库等 |
| **记忆能力** | 仅对话历史 | 短期+长期记忆 |
| **目标导向** | 逐轮响应 | 自主规划步骤达成目标 |
| **反思能力** | 无 | 可评估行动效果并调整 |

```mermaid
graph LR
    subgraph "普通 LLM 调用"
        U1[用户] -->|"提问"| L1[LLM]
        L1 -->|"回答"| U1
        style L1 fill:#e1f5e1
    end
    
    subgraph "AI Agent"
        U2[用户] -->|"目标"| A[Agent]
        A -->|"Action"| T[工具/API]
        T -->|"Observation"| A
        A -->|"调用"| E[环境/数据库]
        E -->|"反馈"| A
        A -->|"结果"| U2
        style A fill:#fff3e0,stroke:#ff9800
    end
```

### 15.1.2 Agent 四大核心模块

```mermaid
graph TD
    subgraph "Agent 核心架构"
        direction TB
        
        P["🧠 规划 Planning<br/>拆解目标 → 制定步骤 → 选择策略"]
        M["💾 记忆 Memory<br/>短期记忆 + 长期记忆"]
        A2["🔧 执行 Action<br/>调用工具 → 影响环境 → 获取反馈"]
        R["🔄 反思 Reflection<br/>评估结果 → 修正计划 → 学习经验"]
        
        P --> A2
        A2 --> M
        M --> R
        R --> P
    end
    
    style P fill:#e3f2fd,stroke:#1976d2
    style M fill:#f3e5f5,stroke:#7b1fa2
    style A2 fill:#e8f5e9,stroke:#388e3c
    style R fill:#fff3e0,stroke:#f57c00
```

#### 模块1：感知（Perception）

Agent 接收外部输入的方式：
- **用户指令**：自然语言描述的目标
- **工具返回值**：API 调用结果、数据库查询结果
- **环境状态**：系统状态、错误信息、时间等

#### 模块2：规划（Planning）⭐⭐⭐⭐⭐

规划是 Agent 的"大脑"，将复杂目标拆解为可执行的步骤：

| 规划类型 | 说明 | 示例 |
|----------|------|------|
| **单路径规划** | 线性步骤序列 | Step1→Step2→Step3 |
| **多路径规划** | 生成多个候选计划，评估后选择 | ToT（Tree of Thoughts）|
| **动态规划** | 根据执行反馈调整计划 | ReAct、Reflexion |
| **分层规划** | 高层目标 → 子目标 → 具体行动 | Hierarchical Agent |

#### 模块3：执行（Action）

执行模块负责**调用工具**改变环境状态。常见工具类型：

```python
# Agent 工具定义示例
TOOLS = [
    {
        "name": "web_search",
        "description": "搜索引擎，用于获取最新信息",
        "parameters": {
            "query": {"type": "string", "description": "搜索关键词"}
        }
    },
    {
        "name": "calculator",
        "description": "计算器，执行数学运算",
        "parameters": {
            "expression": {"type": "string", "description": "数学表达式"}
        }
    },
    {
        "name": "database_query",
        "description": "数据库查询",
        "parameters": {
            "sql": {"type": "string", "description": "SQL 语句"}
        }
    },
    {
        "name": "send_email",
        "description": "发送邮件",
        "parameters": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"}
        }
    },
]
```

#### 模块4：记忆（Memory）

记忆让 Agent 能够"记住"过去的经验和知识：

- **短期记忆（Short-term Memory）**：当前对话上下文，在 Prompt 中维护
- **长期记忆（Long-term Memory）**：向量数据库存储的历史经验、知识图谱
- **工作记忆（Working Memory）**：当前任务相关的临时信息

#### 模块5：反思（Reflection）

反思模块让 Agent 具备**自我评估**和**错误修正**能力：

```python
# 反思 Prompt 模板
REFLECTION_PROMPT = """请评估刚才的行动结果：

原目标：{goal}
执行计划：{plan}
实际行动：{action}
观察结果：{observation}

请回答：
1. 目标是否达成？（是/部分/否）
2. 如果未达成，原因是什么？
3. 下一步计划如何调整？
4. 有哪些经验可以记录到长期记忆中？
"""
```

### 15.1.3 Agent 分类

```mermaid
graph TD
    A["AI Agent 分类"] --> B["按决策方式"]
    A --> C["按协作方式"]
    A --> D["按工具能力"]
    
    B --> B1["单步 Agent<br/>一次决策完成"]
    B --> B2["多步 Agent<br/>迭代决策<br/>ReAct/AutoGPT"]
    B --> B3["分层 Agent<br/>高层规划+底层执行"]
    
    C --> C1["单 Agent<br/>独立工作"]
    C --> C2["多 Agent<br/>协作/竞争<br/>AutoGen/CrewAI"]
    
    D --> D1["工具调用型<br/>Function Calling"]
    D --> D2["代码执行型<br/>Code Interpreter"]
    D --> D3["多模态型<br/>视觉+语言+动作"]
```

---

## 15.2 ReAct 框架 ⭐⭐⭐⭐⭐

### 15.2.1 ReAct 核心思想

ReAct（Reasoning + Acting）是 Agent 领域最具影响力的框架，核心思想是**将推理（Reasoning）和行动（Acting）交织在一起** —— 模型先思考（Thought），再决定行动（Action），然后观察结果（Observation），循环往复直到完成目标。

### 15.2.2 ReAct 决策流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as Agent<br/>(LLM)
    participant T as 工具/API
    participant E as 环境

    U->>A: 目标：查北京明天天气并算体温平均值
    
    loop ReAct 循环
        A->>A: Thought: 我需要先查天气，需要调用天气API
        A-->>U: Action: weather_api(city="北京", date="明天")
        U->>T: 执行工具调用
        T-->>U: Observation: {"temp": "22°C", "condition": "晴"}
        U->>A: 返回观察结果
        
        A->>A: Thought: 已获取天气，温度是22°C，<br/>现在需要计算体温平均值<br/>（36.5, 37.0, 36.8）
        A-->>U: Action: calculator(expression="(36.5+37.0+36.8)/3")
        U->>T: 执行计算
        T-->>U: Observation: 36.7666666667
        U->>A: 返回计算结果
        
        A->>A: Thought: 所有信息已获取，可以给出最终答案
        A-->>U: Final Answer: 北京明天晴，22°C。<br/>体温平均值 36.77°C。
    end
```

### 15.2.3 手写 ReAct Agent（完整实战）

```python
"""
从零实现 ReAct Agent - 完整可运行代码
"""
import json
import re
import os
from typing import Callable, Any

class Tool:
    """工具基类"""
    
    def __init__(self, name: str, description: str, func: Callable, params_schema: dict):
        self.name = name
        self.description = description
        self.func = func
        self.params_schema = params_schema
    
    def execute(self, **kwargs) -> str:
        """执行工具，返回字符串结果"""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"错误：{str(e)}"
    
    def to_prompt_format(self) -> str:
        """转换为 Prompt 中的工具描述"""
        params_desc = "\n".join([
            f"  - {k}: {v.get('description', v.get('type', 'string'))}"
            for k, v in self.params_schema.get("properties", {}).items()
        ])
        return f"- {self.name}: {self.description}\n参数：\n{params_desc}"


class ReActAgent:
    """
    ReAct Agent 完整实现
    
    核心循环：Thought → Action → Observation → ... → Final Answer
    """
    
    def __init__(self, llm_api_key: str = None):
        self.tools: dict[str, Tool] = {}
        self.memory: list[dict] = []  # 历史记录
        self.max_iterations = 10      # 最大迭代次数，防止无限循环
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        
        # ReAct Prompt 模板
        self.react_prompt_template = """你是一个智能助手，可以通过调用工具来完成任务。

可用工具：
{tools_description}

你必须按照以下格式思考和工作：
Thought: 你的思考过程，分析当前状况和下一步行动
Action: 工具名称(参数1="值1", 参数2="值2")
Observation: 工具返回的结果（由系统自动填入）
... （可以重复多轮 Thought/Action/Observation）
Thought: 任务已完成
Final Answer: 最终答案

---

开始任务！

{history}
Thought: """
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def _build_tools_description(self) -> str:
        """构建工具描述"""
        return "\n".join([t.to_prompt_format() for t in self.tools.values()])
    
    def _call_llm(self, prompt: str) -> str:
        """调用 LLM（支持 OpenAI 和模拟模式）"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.llm_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一个严格遵循 ReAct 格式的智能助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stop=["Observation:"],  # 在 Observation 前停止，等待工具执行
            )
            return response.choices[0].message.content
        except Exception:
            # 模拟模式（用于演示和测试）
            return self._simulate_llm(prompt)
    
    def _simulate_llm(self, prompt: str) -> str:
        """LLM 模拟器（用于无 API 时的测试）"""
        history = prompt.split("开始任务！")[-1] if "开始任务！" in prompt else ""
        
        # 简单规则匹配模拟决策
        if "weather" in history.lower() or "天气" in history or "温度" in history:
            if "Observation" not in history:
                return "我需要先查询天气信息。\nAction: weather_api(city=\"北京\")"
            elif "calculator" not in history and "average" in history.lower() or "平均" in history:
                return "现在计算体温平均值。\nAction: calculator(expression=\"(36.5+37.0+36.8)/3\")"
            else:
                return "所有信息已获取。\nFinal Answer: 北京今天气温为 25°C，天气晴朗。三人体温平均值为 36.77°C。"
        
        if "search" in history.lower() or "查" in history:
            if "Observation" not in history:
                return "我需要搜索相关信息。\nAction: search(query=\"Python GIL\")"
            else:
                return "已找到相关信息。\nFinal Answer: Python GIL（全局解释器锁）是 CPython 中防止多线程并发执行字节码的机制。"
        
        return "我需要分析当前情况。\nAction: search(query=\"一般信息\")"
    
    def _parse_action(self, text: str) -> tuple[str, dict] | None:
        """从 LLM 输出解析 Action"""
        # 匹配 Action: tool_name(param1="value", param2="value")
        action_pattern = r'Action:\s*(\w+)\((.*)\)'
        match = re.search(action_pattern, text)
        
        if not match:
            return None
        
        tool_name = match.group(1)
        params_str = match.group(2)
        
        # 解析参数
        params = {}
        # 匹配 key="value" 或 key='value' 或 key=value
        param_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s]*))'
        for pmatch in re.finditer(param_pattern, params_str):
            key = pmatch.group(1)
            value = pmatch.group(2) or pmatch.group(3) or pmatch.group(4)
            params[key] = value
        
        return tool_name, params
    
    def _extract_final_answer(self, text: str) -> str | None:
        """提取 Final Answer"""
        match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_thought(self, text: str) -> str:
        """提取 Thought"""
        match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def run(self, task: str) -> dict:
        """
        执行 ReAct 循环
        
        Returns:
            {
                "task": str,
                "final_answer": str,
                "steps": list[dict],
                "iterations": int,
            }
        """
        history = f"任务：{task}\n"
        steps = []
        
        for i in range(self.max_iterations):
            # 构建完整 Prompt
            prompt = self.react_prompt_template.format(
                tools_description=self._build_tools_description(),
                history=history
            )
            
            # 调用 LLM 生成 Thought + Action
            llm_output = self._call_llm(prompt)
            
            thought = self._extract_thought(llm_output)
            final_answer = self._extract_final_answer(llm_output)
            
            # 检查是否已有最终答案
            if final_answer:
                steps.append({"type": "final", "thought": thought, "answer": final_answer})
                return {
                    "task": task,
                    "final_answer": final_answer,
                    "steps": steps,
                    "iterations": i + 1,
                }
            
            # 解析 Action
            action_parsed = self._parse_action(llm_output)
            
            if not action_parsed:
                steps.append({"type": "error", "output": llm_output, "reason": "无法解析 Action"})
                break
            
            tool_name, params = action_parsed
            
            # 执行工具
            if tool_name not in self.tools:
                observation = f"错误：工具 '{tool_name}' 不存在。可用工具：{list(self.tools.keys())}"
            else:
                tool = self.tools[tool_name]
                observation = tool.execute(**params)
            
            # 记录步骤
            steps.append({
                "type": "action",
                "thought": thought,
                "action": f"{tool_name}({params})",
                "observation": observation,
            })
            
            # 更新历史
            history += f"{llm_output}\nObservation: {observation}\n"
        
        # 超过最大迭代次数
        return {
            "task": task,
            "final_answer": "未能完成任务（达到最大迭代次数）",
            "steps": steps,
            "iterations": self.max_iterations,
        }


# ============ 工具函数定义 ============

def weather_api(city: str, date: str = "今天") -> str:
    """模拟天气查询"""
    weather_db = {
        "北京": {"temp": 25, "condition": "晴", "humidity": "45%"},
        "上海": {"temp": 28, "condition": "多云", "humidity": "65%"},
        "深圳": {"temp": 30, "condition": "小雨", "humidity": "80%"},
    }
    info = weather_db.get(city, {"temp": 22, "condition": "未知", "humidity": "50%"})
    return f"{city}{date}天气：{info['condition']}，气温{info['temp']}°C，湿度{info['humidity']}"


def calculator(expression: str) -> str:
    """安全计算器"""
    # 只允许数字和基本运算符
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "错误：表达式包含非法字符"
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


def search(query: str) -> str:
    """模拟搜索引擎"""
    knowledge_base = {
        "Python GIL": "Python GIL（全局解释器锁）是 CPython 解释器的机制，确保同一时刻只有一个线程执行 Python 字节码。",
        "RAG": "RAG（检索增强生成）将外部知识检索与大语言模型结合，有效减少模型幻觉。",
        "LoRA": "LoRA（低秩适配）是一种参数高效微调方法，通过低秩矩阵微调大模型。",
    }
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return f"搜索结果：{value}"
    return f"搜索结果：找到关于 '{query}' 的 10 条相关网页..."


# ============ 使用示例 ============

def main():
    """主函数 - 运行 ReAct Agent"""
    agent = ReActAgent()
    
    # 注册工具
    agent.register_tool(Tool(
        name="weather_api",
        description="查询指定城市的天气信息",
        func=weather_api,
        params_schema={"properties": {
            "city": {"type": "string", "description": "城市名称"},
            "date": {"type": "string", "description": "日期，如'今天'、'明天'"}
        }}
    ))
    agent.register_tool(Tool(
        name="calculator",
        description="执行数学计算",
        func=calculator,
        params_schema={"properties": {
            "expression": {"type": "string", "description": "数学表达式，如(36.5+37.0)/2"}
        }}
    ))
    agent.register_tool(Tool(
        name="search",
        description="搜索引擎，查询一般知识",
        func=search,
        params_schema={"properties": {
            "query": {"type": "string", "description": "搜索关键词"}
        }}
    ))
    
    # 执行任务
    task = "查询北京今天天气，然后计算36.5、37.0、36.8的平均值"
    result = agent.run(task)
    
    print(f"任务：{result['task']}")
    print(f"最终答案：{result['final_answer']}")
    print(f"迭代次数：{result['iterations']}")
    print("\n详细步骤：")
    for i, step in enumerate(result['steps'], 1):
        print(f"\n--- 步骤 {i} ---")
        if step['type'] == 'action':
            print(f"Thought: {step['thought']}")
            print(f"Action: {step['action']}")
            print(f"Observation: {step['observation']}")
        elif step['type'] == 'final':
            print(f"Thought: {step['thought']}")
            print(f"Final Answer: {step['answer']}")


if __name__ == "__main__":
    main()
```

---

## 15.3 Function Calling ⭐⭐⭐⭐⭐

### 15.3.1 Function Calling 核心流程

Function Calling（函数调用）是大模型的一项核心能力，让模型能够**理解工具定义**、**判断何时调用**、**生成正确参数**，从而实现与外部世界的交互。

```mermaid
sequenceDiagram
    participant U as 用户
    participant LLM as 大模型
    participant App as 应用程序
    participant API as 外部 API

    U->>App: "帮我查北京天气"
    App->>LLM: 消息 + 工具定义（weather_api）
    
    Note over LLM: 模型内部决策：<br/>需要调用 weather_api<br/>参数：city="北京"
    
    LLM-->>App: function_call: {"name": "weather_api", "arguments": "{\"city\": \"北京\"}"}
    
    App->>App: 解析参数，执行函数
    App->>API: 调用天气 API(city="北京")
    API-->>App: {"temp": 25, "condition": "晴"}
    
    App->>LLM: function_result: 北京晴 25°C
    
    Note over LLM: 模型生成自然语言回答
    
    LLM-->>App: "北京今天天气晴朗，气温 25°C。"
    App-->>U: 展示结果
```

### 15.3.2 Function Calling vs ReAct 的本质区别

| 维度 | ReAct | Function Calling |
|------|-------|-----------------|
| **输出格式** | 文本格式（Thought + Action） | 结构化 JSON |
| **解析方式** | 正则/规则解析 | 原生 JSON 解析 |
| **思考过程** | 显式输出 Thought | 隐式（模型内部） |
| **模型支持** | 任何 LLM（通过 Prompt） | 需要模型原生支持（GPT-4、Claude、Qwen 等）|
| **可靠性** | 中（解析可能出错） | 高（结构化输出） |
| **灵活性** | 高（可自定义格式） | 中（遵循 API 格式） |

**核心关系**：Function Calling 是 ReAct 中 "Action" 步骤的**工程化、标准化实现**。ReAct 是思想，Function Calling 是工具。

### 15.3.3 多工具调用 Agent 实战

```python
"""
Function Calling 多工具 Agent - 完整实战
"""
import json
import openai

class FunctionCallingAgent:
    """
    基于 OpenAI Function Calling 的 Agent
    
    核心流程：
    1. 定义 tools（函数 schema）
    2. 发送用户消息 + tools 定义
    3. 如果模型返回 function_call，执行对应函数
    4. 将结果返回给模型，生成最终回答
    """
    
    def __init__(self, api_key: str = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4"
        self.tools = []
        self.tool_functions = {}
        self.conversation = []
    
    def register_tool(self, name: str, description: str, parameters: dict, func: callable):
        """注册工具"""
        self.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        })
        self.tool_functions[name] = func
    
    def execute(self, user_message: str, max_tool_calls: int = 5) -> str:
        """执行对话循环"""
        self.conversation = [{"role": "user", "content": user_message}]
        
        for _ in range(max_tool_calls):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                tools=self.tools if self.tools else None,
                tool_choice="auto",
            )
            
            message = response.choices[0].message
            
            # 检查是否需要调用工具
            if not message.tool_calls:
                # 不需要工具，直接返回回答
                return message.content
            
            # 记录助手消息（含 tool_calls）
            self.conversation.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # 执行所有工具调用
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name in self.tool_functions:
                    result = self.tool_functions[func_name](**func_args)
                else:
                    result = f"错误：工具 {func_name} 不存在"
                
                # 将工具结果加入对话
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
        
        # 达到最大工具调用次数，生成最终回答
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
        )
        return final_response.choices[0].message.content


# ============ 工具函数定义 ============

import requests

def get_weather(city: str) -> str:
    """获取天气（模拟）"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C",
        "深圳": "雷阵雨，29°C",
    }
    return weather_data.get(city, "未知城市")


def get_stock_price(symbol: str) -> str:
    """获取股票价格（模拟）"""
    stocks = {
        "AAPL": "182.50 USD",
        "GOOGL": "142.30 USD",
        "MSFT": "380.20 USD",
        "TSLA": "240.10 USD",
    }
    return stocks.get(symbol.upper(), "未知股票代码")


def search_knowledge(query: str) -> str:
    """知识库搜索（模拟）"""
    kb = {
        "年假": "员工每年享有 15 天带薪年假，入职满 1 年后可申请。",
        "报销": "差旅报销需在出差结束后 30 天内提交，附发票和行程单。",
        "加班": "加班需提前在 OA 系统申请，加班费按法定标准计算。",
    }
    for key, value in kb.items():
        if key in query:
            return value
    return f"未找到 '{query}' 的相关政策"


def send_notification(to: str, message: str) -> str:
    """发送通知（模拟）"""
    return f"通知已发送给 {to}: {message}"


# ============ 使用示例 ============

def main():
    agent = FunctionCallingAgent()
    
    # 注册工具
    agent.register_tool(
        name="get_weather",
        description="获取指定城市的当前天气",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，如北京、上海"}
            },
            "required": ["city"]
        },
        func=get_weather
    )
    
    agent.register_tool(
        name="get_stock_price",
        description="获取指定股票的当前价格",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如 AAPL、GOOGL"}
            },
            "required": ["symbol"]
        },
        func=get_stock_price
    )
    
    agent.register_tool(
        name="search_knowledge",
        description="搜索公司内部知识库",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        },
        func=search_knowledge
    )
    
    agent.register_tool(
        name="send_notification",
        description="发送通知给指定人员",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "接收人"},
                "message": {"type": "string", "description": "通知内容"}
            },
            "required": ["to", "message"]
        },
        func=send_notification
    )
    
    # 测试：需要调用多个工具的复杂查询
    query = "北京天气怎么样？顺便帮我查一下 AAPL 的股价。如果天气好，通知小王出门记得带伞。"
    result = agent.execute(query)
    print(f"查询：{query}")
    print(f"结果：{result}")


if __name__ == "__main__":
    main()
```

---

## 15.4 MCP 协议 ⭐⭐⭐⭐⭐

### 15.4.1 什么是 MCP

MCP（Model Context Protocol，模型上下文协议）是 Anthropic 于 2024 年底推出的**开放标准协议**，旨在为 AI 模型提供与外部工具和数据源连接的标准化方式。它被称为 "AI 领域的 USB-C 接口"。

```mermaid
graph TB
    subgraph "MCP 架构"
        direction TB
        
        Client["MCP Client<br/>（LLM 应用）"]
        Server1["MCP Server A<br/>（GitHub 工具集）"]
        Server2["MCP Server B<br/>（数据库工具集）"]
        Server3["MCP Server C<br/>（文件系统工具集）"]
        Server4["MCP Server D<br/>（Slack/邮件工具集）"]
        
        Client <-->|"JSON-RPC 2.0<br/>stdio / SSE"| Server1
        Client <-->|"JSON-RPC 2.0"| Server2
        Client <-->|"JSON-RPC 2.0"| Server3
        Client <-->|"JSON-RPC 2.0"| Server4
    end
    
    style Client fill:#e3f2fd,stroke:#1976d2
    style Server1 fill:#e8f5e9,stroke:#388e3c
    style Server2 fill:#e8f5e9,stroke:#388e3c
    style Server3 fill:#e8f5e9,stroke:#388e3c
    style Server4 fill:#e8f5e9,stroke:#388e3c
```

**MCP 的核心设计**：

| 组件 | 角色 | 说明 |
|------|------|------|
| **MCP Client** | 消费者 | LLM 应用（如 Claude Desktop、Cursor、自定义 Agent）|
| **MCP Server** | 提供者 | 暴露工具和数据源的服务端 |
| **Transport** | 传输层 | stdio（本地）或 SSE（远程）|
| **Protocol** | 协议层 | JSON-RPC 2.0 |

### 15.4.2 MCP 三大核心能力

MCP Server 可以向 Client 暴露三类能力：

#### 1. Tools（工具）⭐⭐⭐⭐⭐

模型可调用的函数，类似于 Function Calling 中的函数定义：

```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "读取文件内容",
      "inputSchema": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "文件路径"}
        },
        "required": ["path"]
      }
    },
    {
      "name": "write_file",
      "description": "写入文件内容",
      "inputSchema": {
        "type": "object",
        "properties": {
          "path": {"type": "string"},
          "content": {"type": "string"}
        },
        "required": ["path", "content"]
      }
    }
  ]
}
```

#### 2. Resources（资源）

只读的数据资源，模型可以读取但不可修改：

```json
{
  "resources": [
    {
      "uri": "file:///project/README.md",
      "mimeType": "text/markdown",
      "name": "项目 README"
    },
    {
      "uri": "db://users/schema",
      "mimeType": "application/json",
      "name": "用户表结构"
    }
  ]
}
```

#### 3. Prompts（提示模板）

预定义的提示词模板，Server 可以向 Client 提供标准化的交互模式：

```json
{
  "prompts": [
    {
      "name": "code_review",
      "description": "代码审查模板",
      "arguments": [
        {
          "name": "language",
          "description": "编程语言",
          "required": true
        }
      ]
    }
  ]
}
```

### 15.4.3 MCP 通信流程

```mermaid
sequenceDiagram
    participant Client as MCP Client
    participant Server as MCP Server

    %% 初始化
    Client->>Server: initialize (protocolVersion, capabilities)
    Server-->>Client: initialize (protocolVersion, capabilities)
    Client->>Server: initialized (notification)

    %% 工具发现
    Client->>Server: tools/list
    Server-->>Client: [{"name": "read_file", ...}, ...]

    %% 工具调用
    Client->>Server: tools/call(name="read_file", arguments={"path": "/tmp/test.py"})
    Server-->>Client: {"content": [{"type": "text", "text": "print('hello')"}], "isError": false}

    %% 资源读取
    Client->>Server: resources/read(uri="file:///tmp/test.py")
    Server-->>Client: {"contents": [{"uri": "...", "mimeType": "text/x-python", "text": "print('hello')"}]}
```

### 15.4.4 MCP 与 Function Calling 的本质区别 ⭐⭐⭐⭐⭐

这是 **2025 年面试最高频的问题之一**。

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| **定位** | 模型的**输出格式能力** | **连接协议/标准** |
| **层级** | 应用层（单个函数调用）| 协议层（客户端-服务端架构）|
| **工具发现** | 调用前静态定义 | 运行时动态发现（tools/list）|
| **工具来源** | 应用程序硬编码 | 独立的 MCP Server，可插拔 |
| **通信方式** | 函数调用 → 本地执行 | JSON-RPC 2.0（stdio/SSE）|
| **复用性** | 低（每个应用自己实现）| 高（一个 Server 服务多个 Client）|
| **生态** | 各平台独立 | 开放生态，社区共享 Server |

**一句话总结**：Function Calling 是**能力**（模型能输出函数调用指令），MCP 是**协议**（标准化地连接模型与工具生态）。

```mermaid
graph LR
    subgraph "Function Calling"
        A[应用代码] -->|"定义函数"| B[LLM]
        B -->|"输出调用指令"| A
        A -->|"执行函数"| C[本地函数]
    end
    
    subgraph "MCP"
        D[MCP Client] <-->|"JSON-RPC"| E[MCP Server]
        E -->|"暴露工具"| F[GitHub API]
        E -->|"暴露工具"| G[数据库]
        E -->|"暴露工具"| H[文件系统]
        D -->|"调用"| I[LLM]
    end
```

### 15.4.5 MCP Server 简单实现

```python
"""
MCP Server 简化实现示例
展示 MCP 协议的核心交互模式
"""
import json
import sys
from typing import Any

class MCPServer:
    """
    MCP Server 简化实现
    
    通信方式：stdio（标准输入输出上的 JSON-RPC）
    """
    
    def __init__(self):
        self.tools = {
            "read_file": self.read_file,
            "list_directory": self.list_directory,
        }
    
    def send(self, message: dict):
        """发送 JSON-RPC 消息"""
        print(json.dumps(message), flush=True)
    
    def recv(self) -> dict | None:
        """接收 JSON-RPC 消息"""
        try:
            line = sys.stdin.readline()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    
    def handle_initialize(self, request_id: Any, params: dict) -> dict:
        """处理初始化请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "simple-filesystem-server",
                    "version": "1.0.0"
                }
            }
        }
    
    def handle_tools_list(self, request_id: Any) -> dict:
        """处理工具列表请求"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "read_file",
                        "description": "读取文件内容",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "list_directory",
                        "description": "列出目录内容",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    }
                ]
            }
        }
    
    def handle_tools_call(self, request_id: Any, params: dict) -> dict:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](**arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": result}],
                        "isError": False
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": f"错误: {str(e)}"}],
                        "isError": True
                    }
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"工具 {tool_name} 不存在"}
            }
    
    def read_file(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def list_directory(self, path: str) -> str:
        import os
        entries = os.listdir(path)
        return "\n".join(entries)
    
    def run(self):
        """主事件循环"""
        while True:
            request = self.recv()
            if request is None:
                break
            
            method = request.get("method", "")
            request_id = request.get("id")
            params = request.get("params", {})
            
            if method == "initialize":
                response = self.handle_initialize(request_id, params)
            elif method == "tools/list":
                response = self.handle_tools_list(request_id)
            elif method == "tools/call":
                response = self.handle_tools_call(request_id, params)
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"未知方法: {method}"}
                }
            
            self.send(response)


if __name__ == "__main__":
    server = MCPServer()
    server.run()
```

---

### 15.4.6 MCP 工程化管理 🆕（2026年更新）

> 2026年，MCP 已从"新协议介绍"进入"工程化管理"阶段。面试中不再只问"什么是 MCP"，而是追问"你们怎么管理几百个 MCP Server？"

#### 1. 大量 MCP Server 的管理挑战

当生产环境中的 MCP Server 从 3-5 个增长到 50+ 甚至 100+ 时，面临以下挑战：

| 挑战 | 描述 | 解决方案 |
|------|------|---------|
| **服务发现** | 如何知道有哪些 Server 可用 | MCP Registry（注册中心） |
| **动态加载** | 运行时增删 Server 不重启 | 热插拔 + 健康检查 |
| **权限控制** | 不同用户能访问不同工具 | RBAC + Tool 级权限 |
| **审计追踪** | 谁调用了什么工具 | 全链路日志 + 调用链 |
| **版本管理** | Server 升级不影响 Client | 语义化版本 + 灰度发布 |
| **性能监控** | 哪些工具慢、失败率高 | 指标采集 + 告警 |

#### 2. MCP Server 动态加载架构

```mermaid
graph TB
    subgraph "MCP 工程化管理架构"
        direction TB

        Client["MCP Client
（LLM 应用）"]
        Registry["MCP Registry
（注册中心）"]
        HC["Health Checker
（健康检查）"]

        subgraph "MCP Server Pool"
            S1["Server A v1.2"]
            S2["Server B v2.0"]
            S3["Server C v1.5"]
            SN["... Server N"]
        end

        Auth["权限控制层
RBAC"]
        Audit["审计日志
全链路追踪"]

        Client -->|"1. 查询可用 Server"| Registry
        Client -->|"2. JSON-RPC 调用"| Auth
        Auth -->|"鉴权通过"| S1
        Auth -->|"鉴权通过"| S2
        Auth -->|"鉴权通过"| S3
        S1 -->|"3. 返回结果"| Audit
        Audit -->|"4. 记录日志"| Client
        HC -->|"心跳检测"| S1
        HC -->|"心跳检测"| S2
        HC -->|"心跳检测"| S3
        HC -->|"更新状态"| Registry
    end

    style Client fill:#e3f2fd,stroke:#1976d2
    style Registry fill:#fff3e0,stroke:#ff9800
    style Auth fill:#ffebee,stroke:#c62828
    style Audit fill:#f3e5f5,stroke:#7b1fa2
```

#### 3. 动态加载与权限控制代码示例

```python
"""
MCP 工程化管理 - Server 注册中心 + 动态加载 + 权限控制
"""
import json
import time
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class PermissionLevel(Enum):
    """工具权限级别"""
    DENY = 0      # 禁止访问
    READ = 1      # 只读访问
    WRITE = 2     # 读写访问
    ADMIN = 3     # 完全控制


@dataclass
class MCPServerInfo:
    """MCP Server 注册信息"""
    name: str
    version: str
    transport: str          # "stdio" | "sse"
    endpoint: str           # 路径或 URL
    tools: list[dict] = field(default_factory=list)
    permissions: dict[str, PermissionLevel] = field(default_factory=dict)
    last_heartbeat: float = 0.0
    health_status: str = "unknown"  # "healthy" | "unhealthy" | "unknown"
    metadata: dict = field(default_factory=dict)


class MCPRegistry:
    """
    MCP Server 注册中心

    功能：
    1. Server 注册与发现
    2. 健康检查
    3. 版本管理
    4. 工具级权限控制 (RBAC)
    """

    def __init__(self):
        self._servers: dict[str, MCPServerInfo] = {}
        self._user_roles: dict[str, list[str]] = {}  # user_id -> [role, ...]
        self._role_permissions: dict[str, dict[str, PermissionLevel]] = {}
        self._audit_log: list[dict] = []
        self._max_log_entries = 10000

    def register(self, server_info: MCPServerInfo) -> bool:
        """注册 MCP Server"""
        server_id = f"{server_info.name}@{server_info.version}"
        server_info.last_heartbeat = time.time()
        server_info.health_status = "healthy"
        self._servers[server_id] = server_info
        return True

    def discover(self, user_id: str) -> list[MCPServerInfo]:
        """
        为用户发现可用的 Server（根据权限过滤）

        Args:
            user_id: 用户ID

        Returns:
            用户有权限访问的 Server 列表
        """
        available = []
        user_perms = self._get_user_permissions(user_id)

        for server_id, server in self._servers.items():
            if server.health_status != "healthy":
                continue
            # 过滤用户有权限的工具
            allowed_tools = []
            for tool in server.tools:
                tool_name = tool.get("name", "")
                perm = user_perms.get(tool_name, PermissionLevel.DENY)
                if perm.value >= PermissionLevel.READ.value:
                    allowed_tools.append(tool)

            if allowed_tools:
                filtered_server = MCPServerInfo(
                    name=server.name,
                    version=server.version,
                    transport=server.transport,
                    endpoint=server.endpoint,
                    tools=allowed_tools,
                    permissions={k: v for k, v in user_perms.items() 
                               if v.value >= PermissionLevel.READ.value},
                    health_status=server.health_status,
                    metadata=server.metadata,
                )
                available.append(filtered_server)

        return available

    def check_permission(
        self, 
        user_id: str, 
        tool_name: str, 
        required_level: PermissionLevel = PermissionLevel.READ
    ) -> bool:
        """检查用户是否有权限调用指定工具"""
        user_perms = self._get_user_permissions(user_id)
        actual = user_perms.get(tool_name, PermissionLevel.DENY)
        return actual.value >= required_level.value

    def _get_user_permissions(self, user_id: str) -> dict[str, PermissionLevel]:
        """获取用户的所有工具权限"""
        roles = self._user_roles.get(user_id, ["default"])
        merged: dict[str, PermissionLevel] = {}

        for role in roles:
            role_perms = self._role_permissions.get(role, {})
            for tool, perm in role_perms.items():
                if tool not in merged or perm.value > merged[tool].value:
                    merged[tool] = perm

        return merged

    def log_tool_call(
        self, 
        user_id: str, 
        tool_name: str, 
        arguments: dict, 
        result: str,
        duration_ms: float,
        success: bool
    ):
        """记录工具调用审计日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self._hash_id(user_id),
            "tool_name": tool_name,
            "arguments": self._sanitize_args(arguments),
            "result_preview": result[:200] if result else "",
            "duration_ms": duration_ms,
            "success": success,
        }
        self._audit_log.append(entry)

        # 防止日志无限增长
        if len(self._audit_log) > self._max_log_entries:
            self._audit_log = self._audit_log[-self._max_log_entries // 2:]

    def health_check(self) -> dict[str, str]:
        """对所有 Server 执行健康检查"""
        now = time.time()
        for server_id, server in self._servers.items():
            if now - server.last_heartbeat > 60:  # 60秒无心跳视为不健康
                server.health_status = "unhealthy"
        return {sid: s.health_status for sid, s in self._servers.items()}

    @staticmethod
    def _hash_id(user_id: str) -> str:
        """对用户ID做哈希处理（隐私保护）"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    @staticmethod
    def _sanitize_args(args: dict) -> dict:
        """清理敏感参数（如密码、token）"""
        sensitive_keys = {"password", "token", "secret", "api_key", "auth"}
        sanitized = {}
        for k, v in args.items():
            if any(sk in k.lower() for sk in sensitive_keys):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = v
        return sanitized


# ============ 使用示例 ============

def demo_mcp_registry():
    """MCP Registry 使用演示"""
    registry = MCPRegistry()

    # 1. 定义角色权限：客服人员只能查询，管理员可以修改
    registry._role_permissions["customer_service"] = {
        "query_order": PermissionLevel.READ,
        "query_policy": PermissionLevel.READ,
        "search_knowledge": PermissionLevel.READ,
    }
    registry._role_permissions["admin"] = {
        "query_order": PermissionLevel.WRITE,
        "update_order": PermissionLevel.WRITE,
        "refund_request": PermissionLevel.ADMIN,
    }

    # 2. 分配角色
    registry._user_roles["alice"] = ["customer_service"]
    registry._user_roles["bob"] = ["admin"]

    # 3. 注册 Server
    registry.register(MCPServerInfo(
        name="order-system",
        version="1.2.0",
        transport="stdio",
        endpoint="/servers/order-mcp",
        tools=[
            {"name": "query_order", "description": "查询订单"},
            {"name": "update_order", "description": "更新订单"},
        ],
    ))

    # 4. 权限检查
    print(f"Alice 能否 query_order: {registry.check_permission('alice', 'query_order')}")
    print(f"Alice 能否 update_order: {registry.check_permission('alice', 'update_order')}")
    print(f"Bob 能否 update_order: {registry.check_permission('bob', 'update_order')}")

    # 5. 发现可用 Server
    alice_servers = registry.discover("alice")
    print(f"\nAlice 可用的 Server: {[s.name for s in alice_servers]}")
    print(f"Alice 可用的工具: {[t['name'] for s in alice_servers for t in s.tools]}")

    # 6. 记录审计日志
    registry.log_tool_call(
        user_id="alice",
        tool_name="query_order",
        arguments={"order_id": "#12345"},
        result="订单状态：已发货",
        duration_ms=45.2,
        success=True,
    )
    print(f"\n审计日志条目数: {len(registry._audit_log)}")


if __name__ == "__main__":
    demo_mcp_registry()
```

**面试追问**："MCP Server 多了之后，一个工具的调用链怎么追踪？" → 引入**调用链追踪（Trace ID）**，每个 Agent 任务生成唯一 Trace ID，贯穿所有 MCP 工具调用，便于故障排查和性能分析。
## 15.5 Agent 记忆管理 ⭐⭐⭐⭐

### 15.5.1 记忆分层架构

```mermaid
graph TB
    subgraph "Agent 记忆架构"
        direction TB
        
        STM["🧠 短期记忆<br/>Short-Term Memory<br/>当前对话上下文<br/>Sliding Window<br/>~4K-8K tokens"]
        
        WM["📝 工作记忆<br/>Working Memory<br/>当前任务关键信息<br/>结构化提取<br/>实体/意图/约束"]
        
        LTM["💾 长期记忆<br/>Long-Term Memory<br/>历史经验/知识<br/>向量数据库<br/>知识图谱"]
        
        STM --> WM
        WM --> LTM
        LTM -.->|"检索相关记忆"| STM
    end
    
    style STM fill:#e3f2fd,stroke:#1976d2
    style WM fill:#fff3e0,stroke:#ff9800
    style LTM fill:#f3e5f5,stroke:#7b1fa2
```

### 15.5.2 短期记忆：滑动窗口管理

```python
class SlidingWindowMemory:
    """滑动窗口短期记忆管理"""
    
    def __init__(self, max_tokens: int = 4000, tokenizer=None):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.tokenizer = tokenizer
    
    def add(self, role: str, content: str):
        """添加消息，超出窗口时移除最旧的消息"""
        self.messages.append({"role": role, "content": content})
        self._ensure_window_size()
    
    def _ensure_window_size(self):
        """确保不超过最大 token 数"""
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 2:
            # 保留 system prompt，移除最早的 user/assistant 对话
            if len(self.messages) > 1 and self.messages[1]["role"] != "system":
                self.messages.pop(1)
            else:
                self.messages.pop(2)
    
    def _estimate_tokens(self) -> int:
        """估算 token 数（粗略估计：1 token ≈ 0.75 中文字符）"""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            # 粗略估算
            total += len(content) // 3 * 2 + 4  # 每条消息 overhead
        return total
    
    def get_messages(self) -> list[dict]:
        return self.messages.copy()
    
    def clear(self):
        self.messages = []


# 使用示例
memory = SlidingWindowMemory(max_tokens=4000)
memory.add("system", "你是一个智能客服助手")
memory.add("user", "我想退货")
memory.add("assistant", "好的，请提供您的订单号")
memory.add("user", "订单号是 #12345")
# ... 更多对话
```

### 15.5.3 长期记忆：向量存储 + 知识图谱

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class LongTermMemory:
    """
    Agent 长期记忆系统
    
    包含两个存储：
    1. 经验记忆（向量数据库）：存储历史对话和经验
    2. 事实记忆（知识图谱）：存储实体关系
    """
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-zh-v1.5"):
        self.embedder = SentenceTransformer(embedding_model)
        
        # 经验记忆: [(text, embedding, metadata), ...]
        self.experiences: list[dict] = []
        
        # 事实记忆: {entity: {relation: target, ...}, ...}
        self.facts: dict[str, dict[str, str]] = {}
    
    def add_experience(self, text: str, experience_type: str = "conversation"):
        """添加经验记忆"""
        embedding = self.embedder.encode(text, normalize_embeddings=True)
        self.experiences.append({
            "text": text,
            "embedding": embedding,
            "metadata": {"type": experience_type, "timestamp": "now"}
        })
    
    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[str]:
        """检索相关经验"""
        if not self.experiences:
            return []
        
        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        
        # 计算余弦相似度
        experiences_array = np.array([e["embedding"] for e in self.experiences])
        similarities = np.dot(experiences_array, query_emb)
        
        # 取 Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.experiences[i]["text"] for i in top_indices]
    
    def add_fact(self, entity: str, relation: str, target: str):
        """添加事实记忆"""
        if entity not in self.facts:
            self.facts[entity] = {}
        self.facts[entity][relation] = target
    
    def query_fact(self, entity: str, relation: str = None) -> dict | str:
        """查询事实"""
        if entity not in self.facts:
            return {}
        if relation:
            return self.facts[entity].get(relation, "未知")
        return self.facts[entity]


# 完整记忆管理集成
class AgentMemory:
    """Agent 完整记忆系统"""
    
    def __init__(self, stm_max_tokens: int = 4000):
        self.stm = SlidingWindowMemory(max_tokens=stm_max_tokens)
        self.ltm = LongTermMemory()
        self.working_memory: dict = {}
    
    def memorize_interaction(self, user_msg: str, assistant_msg: str):
        """记录一次交互到短期记忆和长期记忆"""
        self.stm.add("user", user_msg)
        self.stm.add("assistant", assistant_msg)
        
        # 同时存入长期经验记忆
        self.ltm.add_experience(f"User: {user_msg}\nAssistant: {assistant_msg}")
    
    def get_context(self, current_query: str) -> list[dict]:
        """
        获取完整上下文：
        1. 短期记忆（对话历史）
        2. 从长期记忆中检索相关经验
        """
        messages = self.stm.get_messages()
        
        # 检索长期记忆中的相关经验
        relevant_experiences = self.ltm.retrieve_relevant(current_query, top_k=2)
        
        if relevant_experiences:
            # 将相关经验作为上下文注入
            context_msg = "相关历史经验：\n" + "\n---\n".join(relevant_experiences)
            messages.insert(1, {"role": "system", "content": context_msg})
        
        return messages
```

---

## 15.6 多 Agent 协作系统 ⭐⭐⭐⭐

### 15.6.1 多 Agent 架构模式

```mermaid
graph TB
    subgraph "多 Agent 协作架构"
        direction TB
        
        subgraph "模式1：层级协作"
            M[Manager Agent<br/>任务分配+结果汇总]
            W1[Worker Agent 1<br/>数据收集]
            W2[Worker Agent 2<br/>数据分析]
            W3[Worker Agent 3<br/>报告撰写]
            M --> W1
            M --> W2
            M --> W3
            W1 --> M
            W2 --> M
            W3 --> M
        end
        
        subgraph "模式2：流水线"
            P1[Agent A<br/>数据提取] --> P2[Agent B<br/>数据清洗]
            P2 --> P3[Agent C<br/>数据分析]
            P3 --> P4[Agent D<br/>报告生成]
        end
        
        subgraph "模式3：去中心化"
            A1[Agent 1] <-->|"消息总线"| H[Hub/消息队列]
            A2[Agent 2] <-->|"消息总线"| H
            A3[Agent 3] <-->|"消息总线"| H
            A4[Agent 4] <-->|"消息总线"| H
        end
    end
```

### 15.6.2 主流多 Agent 框架

| 框架 | 开发者 | 核心特点 | 适用场景 |
|------|--------|---------|---------|
| **AutoGen** | 微软 | Conversational Programming、多 Agent 对话、代码执行 | 复杂任务自动化、代码生成 |
| **MetaGPT** | 开源社区 | SOP（标准作业程序）驱动、角色专业化 | 软件开发、项目管理 |
| **CrewAI** | 开源社区 | 角色扮演、流程编排、工具共享 | 团队协作模拟、工作流自动化 |
| **A2A 协议** | Google | Agent-to-Agent 开放协议、标准化 Agent 间通信 | 跨平台 Agent 互操作 |

### 15.6.3 A2A 协议简介 ⭐⭐⭐⭐

A2A（Agent-to-Agent）是 Google 于 2025 年推出的**开放 Agent 间通信协议**，旨在解决不同框架、不同平台的 Agent 如何互相发现和协作的问题。

```mermaid
graph LR
    subgraph "A2A 协议架构"
        A1[Agent A<br/>客户Agent] -->|"1. 发现"| D[Agent Card<br/>能力描述]
        A1 -->|"2. 任务下发"| A2[Agent B<br/>远程Agent]
        A2 -->|"3. 状态更新"| A1
        A2 -->|"4. 结果返回"| A1
    end
    
    style D fill:#ffe6cc,stroke:#d79b00
```

**A2A 核心概念**：

| 概念 | 说明 |
|------|------|
| **Agent Card** | 描述 Agent 能力的 JSON 文件（类似 OpenAPI 文档） |
| **Task** | Agent 间传递的任务单元 |
| **Message** | 任务中的消息流（可包含文本、文件、结构化数据） |
| **Push Notification** | 异步任务状态更新机制 |

**MCP vs A2A 的区别**：

| 维度 | MCP | A2A |
|------|-----|-----|
| **连接对象** | Client ↔ Server（模型 ↔ 工具）| Agent ↔ Agent（智能体 ↔ 智能体）|
| **关系模式** | 一对多（一个 Client 连多个 Server）| 多对多（任意 Agent 间通信）|
| **能力描述** | tools/list + resources/list | Agent Card |
| **发起方** | Client 调用 Server | 任意 Agent 可向另一 Agent 下发任务 |
| **类比** | USB-C（设备连接标准）| 蓝牙（设备间通信协议）|

---

## 15.6.4 Agent Teams 架构 🆕（2026年更新）

> Agent Teams 是 Claude 4.6 引入的革命性架构，标志着多 Agent 协作从**串行流水线**进入**并行协作**时代。2026年面试高频考点。

#### 1. Agent Teams 核心概念

传统 Multi-Agent 是**串行**的：Manager 分配任务 → Worker 执行 → Manager 汇总。Agent Teams 是**并行协作**的：Team Lead 统筹，Teammates 各自有独立上下文，通过共享任务列表和邮箱系统通信。

```mermaid
graph TB
    subgraph "Agent Teams 架构（Claude 4.6）"
        direction TB

        TL["👤 Team Lead
统筹规划 + 任务分解
维护 Shared Task List"]

        subgraph "📬 Mailbox System"
            M1["Mailbox: Alice"]
            M2["Mailbox: Bob"]
            M3["Mailbox: Carol"]
        end

        subgraph "Teammates（并行执行）"
            T1["🤖 Alice
数据分析师
独立上下文"]
            T2["🤖 Bob
代码工程师
独立上下文"]
            T3["🤖 Carol
测试工程师
独立上下文"]
        end

        STL["📝 Shared Task List
共享任务状态板"]

        TL -->|"1. 分解任务"| STL
        TL -->|"2. 分配"| M1
        TL -->|"2. 分配"| M2
        TL -->|"2. 分配"| M3
        M1 -->|"3. 读取"| T1
        M2 -->|"3. 读取"| T2
        M3 -->|"3. 读取"| T3
        T1 -->|"4. 更新状态"| STL
        T2 -->|"4. 更新状态"| STL
        T3 -->|"4. 更新状态"| STL
        T1 -->|"5. 异步消息"| M2
        T2 -->|"5. 异步消息"| M3
        STL -->|"6. 监控进度"| TL
    end

    style TL fill:#fff3e0,stroke:#ff9800
    style STL fill:#e3f2fd,stroke:#1976d2
    style M1 fill:#f3e5f5,stroke:#7b1fa2
    style M2 fill:#f3e5f5,stroke:#7b1fa2
    style M3 fill:#f3e5f5,stroke:#7b1fa2
```

#### 2. Agent Teams vs 传统 Multi-Agent vs 子代理

| 维度 | 传统 Multi-Agent | Agent Teams (Claude 4.6) | 子代理 (Sub-agent) |
|------|-----------------|-------------------------|-------------------|
| **架构模式** | 串行流水线 | 并行协作 | 嵌套调用 |
| **上下文** | 共享/透传 | 独立上下文 | 继承父上下文 |
| **生命周期** | 随任务创建销毁 | 持久性独立实例 | 临时实例 |
| **通信方式** | 直接函数调用 | Mailbox + Shared Task List | 参数传递 |
| **协作关系** | Manager-Worker | Team Lead + Teammates | Parent-Child |
| **类比** | 工厂流水线 | 敏捷开发团队 | 函数嵌套调用 |

**核心区别**：Agent Teams 的 Teammates 是**持久性独立实例**，拥有各自的上下文和状态，不是临时创建销毁的子代理。

#### 3. Agent Teams 关键组件

| 组件 | 作用 | 类比 |
|------|------|------|
| **Team Lead** | 接收任务、分解子任务、分配给小组成员、监控进度 | 项目经理 |
| **Teammates** | 各自独立执行分配的任务，拥有独立上下文 | 团队成员 |
| **Shared Task List** | 共享的任务状态板，所有人可见当前进度 | Jira / Trello |
| **Mailbox System** | 异步消息通信，Teammates 之间通过邮箱交换信息 | 企业邮箱 |
| **Handoff Protocol** | 任务交接协议，确保任务在不同 Agent 间平滑转移 | 工作交接单 |

#### 4. Agent Teams 工作流程示例

```python
"""
Agent Teams 简化实现示例
展示 Team Lead + Teammates + Shared Task List + Mailbox 的核心交互
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """共享任务列表中的任务单元"""
    id: str
    description: str
    assignee: Optional[str] = None  # Agent 名称
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)  # 依赖的其他任务ID


@dataclass
class Message:
    """Mailbox 消息"""
    id: str
    from_agent: str
    to_agent: str
    content: str
    timestamp: str
    task_id: Optional[str] = None


class SharedTaskList:
    """共享任务状态板 - 所有 Agent 可见"""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._listeners: list[callable] = []

    def add_task(self, task: Task):
        self._tasks[task.id] = task
        self._notify_listeners("task_added", task)

    def update_status(self, task_id: str, status: TaskStatus, result: str = None):
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = status
            if result:
                task.result = result
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                task.completed_at = datetime.now().isoformat()
            self._notify_listeners("task_updated", task)

    def get_pending_tasks(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]

    def get_tasks_by_assignee(self, agent_name: str) -> list[Task]:
        return [t for t in self._tasks.values() if t.assignee == agent_name]

    def is_all_completed(self) -> bool:
        if not self._tasks:
            return False
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) 
                  for t in self._tasks.values())

    def summary(self) -> dict:
        statuses = {}
        for t in self._tasks.values():
            statuses[t.status.value] = statuses.get(t.status.value, 0) + 1
        return statuses

    def _notify_listeners(self, event: str, task: Task):
        for listener in self._listeners:
            try:
                listener(event, task)
            except Exception:
                pass


class MailboxSystem:
    """邮箱系统 - Agent 间异步通信"""

    def __init__(self):
        self._mailboxes: dict[str, list[Message]] = {}

    def register(self, agent_name: str):
        if agent_name not in self._mailboxes:
            self._mailboxes[agent_name] = []

    def send(self, message: Message):
        self.register(message.to_agent)
        self._mailboxes[message.to_agent].append(message)

    def receive(self, agent_name: str) -> list[Message]:
        """获取并清空某 Agent 的邮箱"""
        messages = self._mailboxes.get(agent_name, [])
        self._mailboxes[agent_name] = []
        return messages

    def peek(self, agent_name: str) -> list[Message]:
        """查看但不清空"""
        return self._mailboxes.get(agent_name, []).copy()


class AgentTeam:
    """
    Agent Team 协调器

    简化版实现，展示核心概念：
    - Team Lead 分解任务
    - Teammates 并行执行
    - Shared Task List 同步状态
    - Mailbox 异步通信
    """

    def __init__(self, team_lead_name: str = "lead"):
        self.team_lead = team_lead_name
        self.teammates: list[str] = []
        self.task_list = SharedTaskList()
        self.mailbox = MailboxSystem()
        self.mailbox.register(team_lead_name)

    def add_teammate(self, name: str, role: str = "worker"):
        """添加团队成员"""
        self.teammates.append(name)
        self.mailbox.register(name)

    def create_tasks(self, project_description: str) -> list[Task]:
        """
        Team Lead 分解任务
        （实际中这里会调用 LLM 进行任务分解）
        """
        # 模拟 LLM 分解结果
        tasks = [
            Task(
                id=str(uuid.uuid4())[:8],
                description=f"分析需求: {project_description}",
                assignee=self.teammates[0] if self.teammates else None,
                created_at=datetime.now().isoformat(),
            ),
            Task(
                id=str(uuid.uuid4())[:8],
                description="编写核心代码",
                assignee=self.teammates[1] if len(self.teammates) > 1 else None,
                created_at=datetime.now().isoformat(),
                dependencies=[],  # 依赖第一个任务
            ),
            Task(
                id=str(uuid.uuid4())[:8],
                description="编写测试用例",
                assignee=self.teammates[2] if len(self.teammates) > 2 else None,
                created_at=datetime.now().isoformat(),
                dependencies=[],
            ),
        ]

        # 设置依赖关系
        if len(tasks) >= 2 and tasks[1].assignee:
            tasks[1].dependencies.append(tasks[0].id)

        for task in tasks:
            self.task_list.add_task(task)

        return tasks

    def teammate_execute(self, teammate: str, task: Task) -> str:
        """
        模拟 Teammate 执行任务
        （实际中这里会调用独立的 LLM Agent 实例）
        """
        # 更新任务状态
        self.task_list.update_status(task.id, TaskStatus.IN_PROGRESS)

        # 模拟执行（实际中调用 LLM）
        result = f"[{teammate}] 完成任务: {task.description}"

        # 更新任务状态为完成
        self.task_list.update_status(task.id, TaskStatus.COMPLETED, result)

        # 发送通知到 Team Lead 的邮箱
        self.mailbox.send(Message(
            id=str(uuid.uuid4())[:8],
            from_agent=teammate,
            to_agent=self.team_lead,
            content=f"任务 {task.id} 已完成: {result}",
            timestamp=datetime.now().isoformat(),
            task_id=task.id,
        ))

        return result

    def get_progress(self) -> dict:
        """获取整体进度"""
        return {
            "team_lead": self.team_lead,
            "teammates": self.teammates,
            "task_summary": self.task_list.summary(),
            "all_completed": self.task_list.is_all_completed(),
        }


# ============ 使用示例 ============

def demo_agent_teams():
    """Agent Teams 演示"""
    # 1. 创建团队
    team = AgentTeam(team_lead_name="项目经理")
    team.add_teammate("Alice", "需求分析师")
    team.add_teammate("Bob", "代码工程师")
    team.add_teammate("Carol", "测试工程师")

    # 2. Team Lead 分解任务
    print("=== Team Lead 分解任务 ===")
    tasks = team.create_tasks("开发一个用户登录模块")
    for t in tasks:
        print(f"  任务 {t.id}: {t.description} -> {t.assignee}")

    # 3. Teammates 并行执行任务
    print("\n=== Teammates 并行执行 ===")
    for i, task in enumerate(tasks):
        if task.assignee:
            result = team.teammate_execute(task.assignee, task)
            print(f"  {result}")

    # 4. Team Lead 查看邮箱（异步通知）
    print("\n=== Team Lead 查看邮箱 ===")
    messages = team.mailbox.receive(team.team_lead)
    for m in messages:
        print(f"  来自 {m.from_agent}: {m.content}")

    # 5. 查看整体进度
    print("\n=== 整体进度 ===")
    progress = team.get_progress()
    print(f"  团队: {progress['team_lead']} + {progress['teammates']}")
    print(f"  任务统计: {progress['task_summary']}")
    print(f"  全部完成: {progress['all_completed']}")


if __name__ == "__main__":
    demo_agent_teams()
```

#### 5. Agent Teams 与 A2A 的协同

```mermaid
graph TB
    subgraph "Agent Teams + A2A 协同架构"
        TL1["Team Lead A
本公司"]
        T1["Teammate A1"]
        T2["Teammate A2"]

        TL2["Team Lead B
合作方"]
        T3["Teammate B1"]

        A2A["A2A Protocol
Agent 间通信"]

        TL1 --> T1
        TL1 --> T2
        TL2 --> T3

        TL1 <-->|"A2A Task 下发"| A2A
        A2A <-->|"A2A 结果返回"| TL2
    end

    style A2A fill:#ffe6cc,stroke:#d79b00
```

**关键理解**：Agent Teams 解决**团队内部**协作问题，A2A 解决**跨团队/跨组织**协作问题。两者可以叠加使用。

---

## 15.7 Agent 开发实战 ⭐⭐⭐⭐

### 15.7.1 智能客服 Agent 完整实现

```python
"""
智能客服 Agent - 完整实战
集成：ReAct + Function Calling + RAG + 记忆管理
"""
import json
import openai
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CustomerServiceAgent:
    """
    智能客服 Agent
    
    能力：
    - 公司政策问答（RAG 知识库）
    - 订单查询（数据库工具）
    - 情感分析与安抚
    - 工单创建与转人工
    """
    
    api_key: str
    model: str = "gpt-4"
    conversation: list = field(default_factory=list)
    escalation_threshold: float = 0.8  # 转人工阈值
    
    def __post_init__(self):
        self.client = openai.OpenAI(api_key=self.api_key)
        self.tools = self._define_tools()
        self.system_prompt = self._build_system_prompt()
    
    def _define_tools(self) -> list:
        """定义客服 Agent 可用的工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_policy",
                    "description": "查询公司政策（如退换货、运费、会员权益等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "政策主题"}
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_order",
                    "description": "查询订单信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "user_id": {"type": "string"}
                        },
                        "required": ["order_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "创建工单，转交人工客服",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "issue": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}
                        },
                        "required": ["user_id", "issue", "priority"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "refund_request",
                    "description": "处理退款申请",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"},
                            "amount": {"type": "number"}
                        },
                        "required": ["order_id", "reason"]
                    }
                }
            }
        ]
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是某电商平台的智能客服助手「小智」。

核心原则：
1. 专业：准确解答用户问题，不清楚时主动查询政策
2. 共情：用户情绪激动时先安抚，再解决问题
3. 边界：涉及敏感操作（大额退款、投诉）主动转人工
4. 效率：优先用工具查询，不要编造信息

工作流程：
1. 理解用户意图
2. 如需查询信息，调用对应工具
3. 基于查询结果给出准确回答
4. 判断是否需要转人工（用户情绪极差/问题超出能力范围）

当前时间：{time}""".format(time=datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    def analyze_sentiment(self, message: str) -> dict:
        """情感分析"""
        prompt = f"""分析以下用户消息的情感倾向，输出 JSON：
{{
    "sentiment": "positive/neutral/negative/angry",
    "intensity": 0-1,  // 情绪强度
    "key_concerns": ["用户关注的主要问题"],
    "needs_escalation": true/false  // 是否需要立即转人工
}}

消息：{message}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception:
            return {"sentiment": "neutral", "intensity": 0.5, "needs_escalation": False}
    
    def handle(self, user_message: str, user_id: str = "anonymous") -> dict:
        """
        处理用户消息
        
        Returns:
            {
                "response": str,          # 给用户的消息
                "actions": list,          # 执行的操作
                "escalated": bool,        # 是否已转人工
                "sentiment": dict,        # 情感分析结果
            }
        """
        # 情感分析
        sentiment = self.analyze_sentiment(user_message)
        
        # 情绪激烈，直接转人工
        if sentiment.get("needs_escalation") or sentiment.get("intensity", 0) > self.escalation_threshold:
            return {
                "response": "非常抱歉给您带来不好的体验，我立即为您转接人工客服 specialist。",
                "actions": [{"type": "escalate", "reason": "high_emotion"}],
                "escalated": True,
                "sentiment": sentiment,
            }
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation,
            {"role": "user", "content": user_message}
        ]
        
        actions = []
        max_rounds = 3
        
        for _ in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )
            
            message = response.choices[0].message
            
            # 无需工具调用
            if not message.tool_calls:
                # 保存对话历史
                self.conversation.append({"role": "user", "content": user_message})
                self.conversation.append({"role": "assistant", "content": message.content})
                
                # 保持对话历史长度
                if len(self.conversation) > 20:
                    self.conversation = self.conversation[-20:]
                
                return {
                    "response": message.content,
                    "actions": actions,
                    "escalated": False,
                    "sentiment": sentiment,
                }
            
            # 处理工具调用
            tool_results = []
            for tc in message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                
                # 执行工具
                result = self._execute_tool(func_name, func_args, user_id)
                tool_results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": str(result),
                })
                actions.append({"tool": func_name, "args": func_args, "result": result})
            
            # 更新消息列表
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in message.tool_calls
                ]
            })
            messages.extend(tool_results)
        
        return {
            "response": "抱歉，处理时间较长，我为您转接人工客服。",
            "actions": actions + [{"type": "escalate", "reason": "max_rounds"}],
            "escalated": True,
            "sentiment": sentiment,
        }
    
    def _execute_tool(self, name: str, args: dict, user_id: str) -> str:
        """执行工具调用（模拟实现）"""
        if name == "query_policy":
            policies = {
                "退换货": "7天无理由退货，15天换货。商品需保持原状。",
                "运费": "满99包邮，不满收取6元运费。偏远地区除外。",
                "会员": "VIP会员享95折，每月送运费券3张。",
            }
            topic = args.get("topic", "")
            for key, value in policies.items():
                if key in topic:
                    return value
            return f"关于'{topic}'的政策：请参考官网帮助中心。"
        
        elif name == "query_order":
            order_id = args.get("order_id", "")
            return f"订单 {order_id}：已发货，预计 2-3 天到达。"
        
        elif name == "create_ticket":
            return f"工单 #{hash(str(args)) % 10000} 已创建，专人将在 10 分钟内联系您。"
        
        elif name == "refund_request":
            order_id = args.get("order_id", "")
            return f"退款申请已提交（订单 {order_id}），审核需要 1-3 个工作日。"
        
        return "工具执行成功"


# ============ 使用示例 ============

def demo():
    """智能客服 Agent 演示"""
    agent = CustomerServiceAgent(api_key="your-api-key")
    
    # 场景1：普通政策咨询
    result1 = agent.handle("你们退换货政策是什么？")
    print(f"用户：你们退换货政策是什么？")
    print(f"客服：{result1['response']}")
    print(f"情感：{result1['sentiment']['sentiment']}, 强度：{result1['sentiment']['intensity']}")
    print()
    
    # 场景2：订单查询
    result2 = agent.handle("帮我查一下订单 #12345")
    print(f"用户：帮我查一下订单 #12345")
    print(f"客服：{result2['response']}")
    print(f"执行操作：{result2['actions']}")
    print()
    
    # 场景3：情绪激动的用户
    result3 = agent.handle("你们这是什么垃圾服务！我的货都丢了一周了！我要投诉！")
    print(f"用户：你们这是什么垃圾服务！我的货都丢了一周了！")
    print(f"客服：{result3['response']}")
    print(f"是否转人工：{result3['escalated']}")


if __name__ == "__main__":
    demo()
```

### 15.7.2 手搓 Agent vs 使用框架的选型

```mermaid
graph TD
    Q["Agent 开发选型"] --> A{"是否需要<br/>快速上线？"}
    A -->|是| B["使用框架"]
    A -->|否| C["评估复杂度"]
    
    C --> D{"场景复杂？"}
    D -->|简单<br/>单 Agent+少量工具| E["手搓 Agent<br/>（~500行代码）"]
    D -->|复杂<br/>多 Agent+多工具| F["框架+自定义组件"]
    
    B --> G["LangChain/LangGraph<br/>适合：快速原型"]
    B --> H["AutoGen<br/>适合：多 Agent 协作"]
    B --> I["CrewAI<br/>适合：角色扮演工作流"]
    
    E --> J["优势：可控、轻量、无依赖<br/>劣势：需自己维护"]
    F --> K["框架提供：状态管理、工具注册、<br/>错误处理、可视化调试"]

    style E fill:#ccffcc,stroke:#228b22
    style J fill:#ccffcc,stroke:#228b22
```

**选型决策表**：

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 学习/面试 | **手搓 ReAct Agent** | 深入理解原理，面试加分 |
| MVP 原型 | LangChain Agent | 快速搭建，生态丰富 |
| 多 Agent 协作 | AutoGen / CrewAI | 对话驱动，角色清晰 |
| 生产环境 | **手搓 + MCP** | 可控性高，MCP 接入工具生态 |
| 复杂工作流 | LangGraph | 状态图编排，可视化调试 |

**面试建议**：能清晰描述 ReAct 循环并手写简化版 Agent，面试印象分极高。框架只是工具，原理才是核心。

---

## 15.8 2026年 Agent 面试新增考点 🆕

> 2026年 Agent 面试已从"概念理解"深入为"工程化设计"。面试官不再满足于"知道是什么"，而是追问"怎么设计、怎么管控风险"。本节覆盖2026年最高频的新增考点。

---

### 15.8.1 Function Calling vs MCP vs Skills vs A2A 四者关系图解

这是 **2026 年面试最高频的辨析题**，许多候选人能分别说出每个是什么，但说不清楚它们之间的关系。

#### 一句话定义

| 概念 | 一句话定义 | 解决的问题 |
|------|----------|-----------|
| **Function Calling** | 模型"输出函数调用指令"的能力 | 模型怎么**表达**要调用工具 |
| **MCP** | 连接模型应用与工具生态的标准**协议** | 模型怎么**接入**外部工具 |
| **Skills** | 封装完整功能的可复用**单元**（代码+配置+推理逻辑） | Agent 怎么**拥有**某项专业能力 |
| **A2A** | Agent 之间协作通信的标准**协议** | 多个 Agent 怎么**互相协作** |

#### 四层架构图解

```mermaid
graph TB
    subgraph "Agent 技术栈四层模型（2026年）"
        direction TB

        subgraph "第4层：协作层"
            A2A["A2A Protocol
Agent ↔ Agent 通信
解决：多 Agent 怎么协作"]
        end

        subgraph "第3层：能力层"
            Skills["Skills
完整功能单元
代码 + 配置 + 推理逻辑
解决：Agent 有什么专业能力"]
        end

        subgraph "第2层：连接层"
            MCP["MCP Protocol
Client ↔ Server
JSON-RPC 2.0
解决：模型怎么接入工具"]
        end

        subgraph "第1层：基础层"
            FC["Function Calling
模型的输出格式能力
JSON Schema 描述工具
解决：模型怎么表达调用意图"]
        end

        Skills -->|"调用工具通过"| MCP
        MCP -->|"工具调用指令用"| FC
        A2A -->|"Agent 间传递"| Skills
        A2A -->|"跨 Agent 调用工具"| MCP
    end

    style FC fill:#e8f5e9,stroke:#388e3c
    style MCP fill:#e3f2fd,stroke:#1976d2
    style Skills fill:#fff3e0,stroke:#ff9800
    style A2A fill:#f3e5f5,stroke:#7b1fa2
```

#### 关键辨析

**Skills vs MCP**：
- **Skills** = "我会做什么"（能力单元，包含完整的推理逻辑 + 工具调用链）
- **MCP** = "我怎么连接工具"（连接协议，不关心工具里有什么业务逻辑）
- 一个 Skill 内部可能使用多个 MCP Server 提供的工具

**Skills vs Few-shot Prompting**：
- **Few-shot** = 教模型"格式"（给出输入输出示例，让模型模仿格式）
- **Skills** = 教模型"方法论"（完整的解题思路、工具组合、验证流程）
- 类比：Few-shot 是"照着例题做题"，Skills 是"掌握解题方法论"

**四者关系总结**：
> Function Calling 是**能力**，MCP 是**连接协议**，Skills 是**功能单元**，A2A 是**协作协议**。四层叠加，缺一不可。

---

### 15.8.2 Agent 工程化安全五道防线

生产环境部署 Agent 必须考虑的五大安全问题，2026年面试必问。

```mermaid
graph TB
    subgraph "Agent 工程化安全五道防线"
        direction LR

        D1["🛡️ 防线1
死循环防范
最大步数 + 相同动作检测"]
        D2["🛡️ 防线2
工具调用幻觉
Schema 校验 + 白名单"]
        D3["🛡️ 防线3
上下文污染
合理截断 + 任务重置"]
        D4["🛡️ 防线4
Token 爆炸
输出截断 + 分页"]
        D5["🛡️ 防线5
Prompt Injection
输入过滤 + 权限隔离"]

        D1 --> D2 --> D3 --> D4 --> D5
    end

    style D1 fill:#ffebee,stroke:#c62828
    style D2 fill:#fff3e0,stroke:#ef6c00
    style D3 fill:#fffde7,stroke:#f9a825
    style D4 fill:#e8f5e9,stroke:#388e3c
    style D5 fill:#e3f2fd,stroke:#1976d2
```

#### 防线1：死循环防范

Agent 可能因为"反复尝试同一动作"或"目标不可达"而陷入死循环。

```python
class LoopPrevention:
    """死循环防范机制"""

    def __init__(self, max_steps: int = 10, similarity_threshold: int = 3):
        self.max_steps = max_steps
        self.similarity_threshold = similarity_threshold
        self.action_history: list[str] = []
        self.step_count = 0

    def check(self, action: str) -> tuple[bool, str]:
        """
        检查是否可能陷入死循环

        Returns:
            (是否继续, 原因)
        """
        self.step_count += 1

        # 检查1：最大步数
        if self.step_count > self.max_steps:
            return False, f"超过最大步数限制 ({self.max_steps})"

        # 检查2：相同动作重复
        self.action_history.append(action)
        recent_actions = self.action_history[-self.similarity_threshold:]
        if len(recent_actions) >= self.similarity_threshold:
            if len(set(recent_actions)) == 1:
                return False, f"连续 {self.similarity_threshold} 次执行相同动作"

        # 检查3：动作震荡（A→B→A→B 模式）
        if len(self.action_history) >= 4:
            last4 = self.action_history[-4:]
            if last4[0] == last4[2] and last4[1] == last4[3]:
                return False, "检测到动作震荡模式 (A→B→A→B)"

        return True, "ok"
```

#### 防线2：工具调用幻觉

模型可能编造不存在的工具名称或参数。

```python
class ToolHallucinationGuard:
    """工具调用幻觉防护"""

    def __init__(self, allowed_tools: set[str], schema_registry: dict):
        self.allowed_tools = allowed_tools
        self.schema_registry = schema_registry

    def validate(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """
        严格校验工具调用

        1. 工具名白名单校验
        2. 参数 Schema 校验
        3. 必填参数检查
        """
        # 白名单校验
        if tool_name not in self.allowed_tools:
            return False, f"工具 '{tool_name}' 不在白名单中"

        schema = self.schema_registry.get(tool_name, {})
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # 必填参数检查
        for param in required:
            if param not in arguments:
                return False, f"缺少必填参数 '{param}'"

        # 参数类型检查
        for key, value in arguments.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._type_check(value, expected_type):
                    return False, f"参数 '{key}' 类型错误，期望 {expected_type}"

        return True, "校验通过"

    @staticmethod
    def _type_check(value, expected_type: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True
```

#### 防线3：上下文污染

多轮工具调用后，历史记录可能"污染"当前任务的判断。

```python
class ContextManager:
    """上下文管理 - 防止污染"""

    def __init__(self, max_context_turns: int = 6):
        self.max_context_turns = max_context_turns
        self.task_separator = "\n--- 新任务 ---\n"

    def build_prompt(self, current_task: str, history: list[dict]) -> str:
        """
        构建干净的 Prompt
        1. 只保留最近 N 轮对话
        2. 不同任务之间加明确分隔
        3. 定期总结历史，替代原始对话
        """
        # 保留最近 N 轮
        recent_history = history[-self.max_context_turns * 2:]

        # 如果历史很长，用摘要替代早期对话
        if len(history) > self.max_context_turns * 2:
            early_history = history[:-self.max_context_turns * 2]
            summary = self._summarize(early_history)
            context = [summary] + recent_history
        else:
            context = recent_history

        return self._format_prompt(current_task, context)

    def _summarize(self, history: list[dict]) -> dict:
        """对早期历史进行摘要（实际中调用 LLM）"""
        return {
            "role": "system",
            "content": f"[历史摘要] 已完成 {len(history)//2} 轮交互，关键结论：..."
        }

    def _format_prompt(self, task: str, context: list[dict]) -> str:
        parts = []
        for msg in context:
            parts.append(f"{msg['role']}: {msg['content']}")
        return self.task_separator + f"当前任务: {task}\n" + "\n".join(parts)
```

#### 防线4：Token 爆炸

Agent 可能产生超长输出导致 Token 消耗失控。

```python
class TokenLimiter:
    """Token 限制器"""

    def __init__(self, max_output_tokens: int = 2000, max_total_tokens: int = 8000):
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.total_consumed = 0

    def check_budget(self, estimated_tokens: int) -> tuple[bool, dict]:
        """检查 Token 预算是否充足"""
        if self.total_consumed + estimated_tokens > self.max_total_tokens:
            return False, {
                "status": "budget_exceeded",
                "consumed": self.total_consumed,
                "budget": self.max_total_tokens,
                "action": "触发任务终止或摘要降级",
            }
        self.total_consumed += estimated_tokens
        return True, {"status": "ok", "remaining": self.max_total_tokens - self.total_consumed}

    def truncate_output(self, text: str, max_length: int = None) -> str:
        """截断输出"""
        max_len = max_length or self.max_output_tokens
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n...[输出已截断]"
```

#### 防线5：Prompt Injection 防御

```python
class PromptInjectionGuard:
    """Prompt Injection 防护"""

    # 危险的注入模式
    DANGEROUS_PATTERNS = [
        "忽略之前的指令",
        "ignore previous instructions",
        "you are now",
        "system prompt",
        "\n\n---\n\n",  # 分隔符注入
        "<|im_start|>",   # 特殊token注入
        "<|im_end|>",
        "```system",      # 代码块注入
    ]

    def scan(self, user_input: str) -> tuple[bool, str]:
        """
        扫描用户输入是否包含注入攻击

        Returns:
            (是否安全, 原因)
        """
        lower_input = user_input.lower()

        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in lower_input:
                return False, f"检测到可疑注入模式: '{pattern}'"

        # 检查嵌套指令结构
        if lower_input.count("ignore") >= 2 and "instruction" in lower_input:
            return False, "检测到潜在的指令覆盖攻击"

        # 检查过长输入（可能隐藏注入）
        if len(user_input) > 10000:
            return False, "输入长度异常，可能隐藏注入内容"

        return True, "安全"
```

---

### 15.8.3 Skills 设计方法论

Skills 是 2026 年的重要概念，面试中经常要求"设计一个 Skill"。

#### Skills 设计三步法

```
┌─────────────────────────────────────────────────────┐
│  Step 1: 能力拆解                                     │
│  将"大能力"拆成"原子操作"                               │
│  例：客服 Skill → 查询订单 + 查询政策 + 情感分析 + 转人工  │
├─────────────────────────────────────────────────────┤
│  Step 2: 工具编排                                      │
│  定义原子操作的执行顺序、依赖关系、错误处理                │
│  例：先查订单 → 根据状态决定查政策还是转人工               │
├─────────────────────────────────────────────────────┤
│  Step 3: 验证闭环                                      │
│  每个 Skill 必须有输出验证 + 回退策略                     │
│  例：工具调用失败 → 重试 → 降级 → 人工接管                │
└─────────────────────────────────────────────────────┘
```

#### Skill 定义示例（AGENTS.md 开放标准格式）

```markdown
# Skill: 智能客服

## 描述
处理电商平台的用户咨询，包括订单查询、政策解答、退款处理、工单创建。

## 适用场景
- 用户查询订单状态
- 用户咨询退换货政策
- 用户申请退款
- 用户情绪激动需要安抚

## 工具依赖
- query_order: 查询订单信息
- query_policy: 查询公司政策
- refund_request: 处理退款申请
- create_ticket: 创建人工工单
- analyze_sentiment: 情感分析

## 执行流程
1. 接收用户消息 → 情感分析
2. 若情绪激烈（intensity > 0.8）→ 直接转人工（create_ticket）
3. 若情绪正常 → 识别意图
   - 意图=订单查询 → query_order → 给出结果
   - 意图=政策咨询 → query_policy → 给出结果
   - 意图=退款申请 → 校验条件 → refund_request → 给出结果
4. 任何步骤失败 → 重试1次 → 仍失败则 create_ticket

## 输出格式
{"response": "给用户的话", "actions": [], "escalated": false}

## 回退策略
- 工具调用失败: 重试1次 → 仍失败转人工
- 意图不明确: 反问用户澄清
- 超出能力范围: 诚恳告知 + 转人工
```

### 🎯 高频题1：AI Agent 和普通 LLM 调用的本质区别是什么？

**参考答案**：

Agent 和 LLM 调用的本质区别是**自主性（Autonomy）**：

- **普通 LLM 调用**：被动响应，输入 → 输出，单次交互，无状态（除对话历史外）
- **AI Agent**：主动规划，具备持续的**感知→规划→执行→反思**循环，能够调用工具改变环境，根据反馈动态调整策略

类比：LLM 是"会说话的百科全书"，Agent 是"能动手办事的助理"。

---

### 🎯 高频题2：ReAct 框架中 Thought、Action、Observation 的作用分别是什么？

**参考答案**：

- **Thought**：推理过程，分析当前状态、评估进展、决定下一步行动。是 Agent 的"内心独白"。
- **Action**：具体的工具调用指令，格式为 `工具名(参数)`。是 Agent 对环境的"输出"。
- **Observation**：工具执行后返回的结果。是环境对 Agent 的"反馈"。

三者构成闭环：Thought 决定 Action，Action 产生 Observation，Observation 影响下一轮 Thought。循环直到 Thought 判断目标达成。

---

### 🎯 高频题3：MCP 和 Function Calling 的本质区别？

**参考答案**（重点中的重点）：

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| **本质** | 模型的**输出能力** | **连接协议/标准** |
| **类比** | 一个人会说"请帮我拿杯水" | USB-C 接口标准 |
| **层级** | 应用层 | 协议层 |
| **工具来源** | 应用内硬编码 | 独立 Server，即插即用 |
| **发现机制** | 静态定义 | 运行时动态发现 |

**一句话**：Function Calling 是模型"能发出工具调用指令"的能力；MCP 是"标准化地连接模型应用与工具生态"的协议。两者互补 —— MCP Server 提供工具，Function Calling 让模型调用这些工具。

---

### 🎯 高频题4：Agent 的记忆管理是怎么做的？

**参考答案**：

Agent 记忆通常分三层：

1. **短期记忆（Short-term Memory）**：当前对话上下文，用滑动窗口管理（通常 4K-8K tokens），超出时丢弃最旧的消息
2. **工作记忆（Working Memory）**：从对话中提取的关键信息（如用户偏好、当前任务目标），以结构化方式临时存储
3. **长期记忆（Long-term Memory）**：向量数据库（存储历史经验，支持语义检索）+ 知识图谱（存储实体关系，支持多跳查询）

---

### 🎯 高频题5：手搓 Agent 和使用 LangChain 等框架，如何选择？

**参考答案**：

- **学习/面试**：手搓 Agent，深入理解 ReAct 循环和工具调用机制
- **快速原型**：LangChain（生态丰富，组件即插即用）
- **多 Agent 协作**：AutoGen 或 CrewAI
- **生产环境**：建议手搓核心框架 + MCP 接入工具生态，可控性更高、性能更好

关键考量：框架带来开发效率，但引入抽象层和依赖；手搓带来灵活性和性能，但需要更多工程投入。

---

### 🎯 高频题6：A2A 协议和 MCP 协议的区别？

**参考答案**：

- **MCP** 是 **Client-Server 架构**，连接"模型应用"和"工具服务"，解决的是"模型如何调用工具"的问题
- **A2A** 是 **Peer-to-Peer 架构**，连接"Agent"和"Agent"，解决的是"Agent 之间如何协作"的问题

类比：MCP 像 USB-C（连接设备与配件），A2A 像蓝牙（设备之间互相通信）。

---

## 15.9 面试题精讲 🎯

### 🎯🆕 高频题7（2026年新题）：Function Calling、MCP、Skills、A2A 四者的关系是什么？怎么区分？

**参考答案**：

四层模型从上到下：

1. **A2A（协作层）**：解决 Agent 之间如何通信协作，类比"蓝牙"（设备间通信）
2. **Skills（能力层）**：封装完整功能的可复用单元（代码+配置+推理逻辑），类比"App 应用"（完整功能）
3. **MCP（连接层）**：标准化连接模型与工具生态的协议，类比"USB-C"（设备连接标准）
4. **Function Calling（基础层）**：模型输出函数调用指令的能力，类比"电流"（底层能力）

**四者关系**：Function Calling 是能力，MCP 是连接协议，Skills 是功能单元，A2A 是协作协议。四层叠加，缺一不可。一个 Skill 内部可能通过 MCP 调用多个工具，多个 Agent 通过 A2A 协作时交换 Skills。

---

### 🎯🆕 高频题8（2026年新题）：生产环境部署 Agent，你会考虑哪些安全风险？怎么防范？

**参考答案**：

五大安全防线：

| 防线 | 风险 | 防范方案 |
|------|------|---------|
| **死循环** | Agent 反复尝试同一动作 | 最大步数限制 + 相同动作检测 + 震荡模式检测 |
| **工具幻觉** | 模型编造不存在的工具 | 工具名白名单 + Schema 严格校验 + 必填参数检查 |
| **上下文污染** | 历史记录干扰当前任务 | 滑动窗口 + 任务分隔符 + 历史摘要替代 |
| **Token 爆炸** | 输出超长导致成本失控 | 输出截断 + Token 预算 + 分页机制 |
| **Prompt Injection** | 用户输入覆盖系统指令 | 输入过滤（危险模式检测）+ 权限隔离 + 长度限制 |

**面试加分**：提到"全链路审计日志"——每个工具调用记录 who/what/when/result，便于事后追溯。

---

### 🎯🆕 高频题9（2026年新题）：如果让你设计一个电商客服 Agent，你会怎么设计？

**参考答案**（开放题，考察架构思维）：

```
1. 能力拆解（Skills 设计）
   - 订单查询 Skill：query_order + 状态解释
   - 政策咨询 Skill：query_policy + 多轮澄清
   - 退款处理 Skill：校验条件 + refund_request + 结果通知
   - 情感安抚 Skill：analyze_sentiment + 安抚话术 + 转人工判断

2. 工具层（MCP 接入）
   - order-mcp-server：订单相关工具
   - policy-mcp-server：政策相关工具
   - payment-mcp-server：退款相关工具

3. 安全机制
   - 死循环：最大 5 轮工具调用
   - 转人工：情感强度 > 0.8 或问题超出范围
   - 敏感操作：退款需二次确认 + 金额上限

4. 记忆设计
   - 短期：当前对话（最近 6 轮）
   - 工作：用户当前意图 + 订单号缓存
   - 长期：用户偏好 + 常见问题模式

5. 监控
   - 解决率、平均轮次、转人工率、用户满意度
```

---

### 🎯🆕 高频题10（2026年新题）：MCP Server 从 5 个增长到 100 个，怎么管理？

**参考答案**：

工程化管理五要素：

1. **注册中心（Registry）**：Server 启动时注册，Client 运行时查询
2. **健康检查（Health Check）**：心跳检测，自动剔除不可用 Server
3. **权限控制（RBAC）**：不同角色访问不同工具集（客服只读，管理员可写）
4. **版本管理**：语义化版本，灰度升级（先 10% 流量切到新版本）
5. **审计追踪**：每个工具调用记录 Trace ID、调用者、参数、结果、耗时

**代码要点**：动态加载（热插拔不停机）、工具发现（运行时拉取 tools/list）、权限过滤（根据用户角色过滤可见工具）。

---

### 🎯🆕 高频题11（2026年新题）：Agent Teams 和传统的 Multi-Agent 有什么区别？

**参考答案**：

| 维度 | 传统 Multi-Agent | Agent Teams (Claude 4.6) |
|------|-----------------|-------------------------|
| **执行模式** | 串行（Manager→Worker→Manager） | 并行（Team Lead + 多 Teammate 同时执行） |
| **上下文** | 共享/透传 | 每个 Teammate 独立上下文 |
| **生命周期** | 随任务创建销毁 | 持久性独立实例 |
| **通信** | 直接函数调用 | Mailbox + Shared Task List |
| **类比** | 工厂流水线 | 敏捷开发团队 |

**核心区别**：Agent Teams 的 Teammates 是**持久性独立实例**，不是临时创建销毁的子代理。通过 Shared Task List 同步状态，通过 Mailbox 异步通信。

---

### 🎯🆕 高频题12（2026年新题）：Skills 和 Few-shot Prompting 有什么区别？什么时候用 Skills？

**参考答案**：

| 维度 | Few-shot Prompting | Skills |
|------|-------------------|--------|
| **本质** | 教模型"格式" | 教模型"方法论" |
| **内容** | 输入输出示例对 | 完整推理逻辑 + 工具编排 + 验证闭环 |
| **复用性** | 低（每次都要带示例） | 高（封装后可复用） |
| **类比** | 照着例题做题 | 掌握解题方法论 |

**什么时候用 Skills**：任务需要多步推理、涉及工具调用链、需要错误处理和回退策略时。简单的格式模仿用 Few-shot，复杂的业务功能用 Skills。

---

### 🎯🆕 高频题13（2026年新题）：Agent 调用工具时模型"幻觉"了怎么办？（编造工具名或参数）

**参考答案**：

三层防护：

1. **Schema 校验**：工具调用前严格校验参数类型、必填字段、取值范围
2. **白名单机制**：工具名必须在预定义列表中，拒绝任何未知工具调用
3. **重试+降级**：校验失败时返回明确错误 + 让模型重试；连续失败则人工接管

```python
# 伪代码
def validate_tool_call(tool_name, args):
    if tool_name not in ALLOWED_TOOLS:
        return False, f"工具 '{tool_name}' 不存在"
    schema = TOOL_SCHEMAS[tool_name]
    for param in schema["required"]:
        if param not in args:
            return False, f"缺少必填参数 '{param}'"
    return True, "校验通过"
```

---

### 🎯🆕 高频题14（2026年新题）：AGENTS.md 是什么？为什么要制定这个开放标准？

**参考答案**：

**AGENTS.md** 是 2026 年提出的**开放标准**，用于描述 Agent 的能力、工具依赖、执行流程和回退策略，类比：
- `README.md` → 描述项目
- `API.md` → 描述接口  
- `AGENTS.md` → 描述 Agent 的能力和行为规范

**内容结构**：
1. **Skill 定义**：名称、描述、适用场景
2. **工具依赖**：需要的 MCP Server 和工具列表
3. **执行流程**：任务处理的工作流
4. **回退策略**：异常情况的处理方式

**为什么要制定**：不同团队开发的 Agent 需要互操作时，AGENTS.md 提供了统一的能力描述格式，降低集成成本。类似于 OpenAPI 对 API 文档标准化的作用。

```mermaid
graph TD
    subgraph "Agent 技术栈总结"
        A["Agent 基础"] --> B["ReAct 框架<br/>Thought-Action-Observation"]
        B --> C["Function Calling<br/>结构化工具调用"]
        C --> D["MCP 协议<br/>标准化工具接入"]
        D --> E["记忆管理<br/>STM + LTM"]
        E --> F["多 Agent 协作<br/>AutoGen / A2A"]
    end
```

| 知识点 | 面试频率 | 关键要点 |
|--------|---------|---------|
| Agent 四大模块 | ⭐⭐⭐⭐⭐ | 感知、规划、执行、反思 |
| ReAct 框架 | ⭐⭐⭐⭐⭐ | Thought→Action→Observation 循环 |
| Function Calling | ⭐⭐⭐⭐⭐ | 工具定义→模型判断→生成调用→执行 |
| MCP 协议 | ⭐⭐⭐⭐⭐ | Client-Server、Tools/Resources/Prompts |
| MCP vs Function Calling | ⭐⭐⭐⭐⭐ | 协议 vs 能力 |
| Agent 记忆管理 | ⭐⭐⭐⭐ | 短期+长期+工作记忆 |
| 多 Agent 协作 | ⭐⭐⭐⭐ | AutoGen、CrewAI、A2A、Agent Teams |
| Skills 设计 | ⭐⭐⭐⭐⭐ | 能力拆解、工具编排、验证闭环 |
| MCP 工程化管理 | ⭐⭐⭐⭐⭐ | Registry、RBAC、审计追踪、健康检查 |
| Agent 安全防护 | ⭐⭐⭐⭐⭐ | 死循环+幻觉+污染+Token+注入 五道防线 |

**下一步**：Agent 赋予了大模型"行动能力"，但要在生产环境高效运行，还需要掌握模型微调、推理优化和部署技术。

---

## 📚 相关章节

- [[12_Transformer与大模型原理]] — Agent Teams、大模型工具调用能力与涌现能力
- [[13_Prompt_Engineering]] — ReAct 模式、CoT 推理是 Agent 的核心思维框架
- [[14_RAG检索增强生成]] — Agent 通过 RAG 获取外部知识，RAG-as-a-Tool 架构
- [[16_模型微调与推理优化]] — Agent 专用模型微调与推理加速部署
