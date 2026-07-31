# Real Demos：历史样例与真实验收

> 核验日期：2026-07-31。本页不提供当前线上验收结论；它说明旧输出应如何解读，
> 以及怎样重新取得可审计的真实调用证据。

## 1. 历史输出的证据边界

旧版文档曾列出 13 段文本，并注明它们采集于 2026-06-07、来自 MiniMax Codin Plan。
但页面没有同时保存原始响应、请求 ID、确切模型 ID、代码提交、依赖版本和逐项日志，因此无法仅凭
这些文本复核它们是否来自真实 API，也无法把它们外推为当前模型的输出、时延、token 或能力。

本版不再把这些历史文本写成“实际输出”或“真实通过”。如需教学展示，可将其标注为
“历史示意，非验收证据”；如需验收，请按下文重新运行并保存证据。

## 2. wrapper 当前覆盖的脚本

`scripts/run_real_demos.sh` 的 `quick` 模式选择 3 个脚本，`all` 模式通常选择 10 个；只有
`provider=openai` 且 `mode=all` 时，才追加 2 个 OpenAI 专用脚本：

| 集合 | 脚本 |
|---|---|
| quick（3） | Ch13 `06_self_consistency_cot.py`、`09_compare_temperatures.py`；Ch15 `02_react_agent_from_scratch.py` |
| all 追加（7） | Ch17 `05_llm_as_judge.py`；Ch18 `02_llmchain_basic.py`、`03_sequential_chain.py`、`05_conversation_buffer_memory.py`、`09_chatbot_with_memory.py`、`13_llamaindex_vectorstore_index.py`、`14_llamaindex_summary_index.py` |
| 仅 OpenAI all 追加（2） | Ch13 `14_openai_auto_caching.py`、`20_openai_json_schema_strict.py` |

这些脚本的依赖和协议并不相同；某个 provider 能完成普通 chat，不代表它能完成 OpenAI 专用缓存、
严格 Schema、工具调用、LangChain 或 LlamaIndex 路径。

## 3. 运行方式

先在 `code/` 目录配置目标厂商 Key。真实运行必须显式关闭离线模式：

```bash
cd code/
LLM_MOCK=0 bash scripts/run_real_demos.sh --confirm-real all deepseek
```

快速子集：

```bash
LLM_MOCK=0 bash scripts/run_real_demos.sh --confirm-real quick deepseek
```

跑单个例子：

```bash
LLM_MOCK=0 LLM_PROVIDER=deepseek python ch13_prompt_engineering/llm/06_self_consistency_cot.py
```

PowerShell 中如需调用 Git Bash：

```powershell
$env:LLM_MOCK="0"; bash scripts/run_real_demos.sh --confirm-real all deepseek
```

离线回归使用：

```bash
LLM_MOCK=1 python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180
```

## 4. wrapper 不是“真实通过”门禁

当前 `run_real_demos.sh` 会要求 `--confirm-real`、核对目标 Key、强制导出 `LLM_MOCK=0`，并把
进程失败、可识别的 mock/离线输出和 `[SKIP]` 分开统计。它仍主要根据进程退出码与输出文本判定：
不会读取 Python 响应对象，也不会核对厂商 usage 或账单。因此：

- wrapper 的 `passed` 表示脚本正常退出且未命中它能识别的 mock/离线标记；
- `[SKIP]`、`[mock]` 或 `resp.mock is True` 均不是实时 API 通过；真实 SDK 异常应直接使脚本失败；
- 0 退出码仍可能包含单独列出的 skip；
- 没有打印可识别标记的兜底响应，仍可能被 wrapper 漏判；
- provider 参数会被写入 `LLM_PROVIDER`；验收时仍需核对实际 provider/model。

先用严格探针验证目标 provider：

```bash
LLM_MOCK=0 LLM_PROVIDER=deepseek python -c "from shared.llm_client import UnifiedClient; r=UnifiedClient().chat(prompt='Reply only OK', max_tokens=16); assert not r.mock, repr(r.raw); print(r.provider, r.model, r.usage)"
```

然后逐项检查脚本日志。只有满足业务断言且没有 mock/skip/error 标记的项目，才能记为真实通过。

## 5. 建议保存的证据

每次真实验收至少记录：

1. 时间、代码 commit、Python/SDK/框架版本；
2. 完整命令，包含 `LLM_MOCK=0` 和目标 provider；
3. 实际响应中的 provider/model、请求 ID（若提供）、token 与时延；
4. `resp.mock is False` 或等价的原始 SDK 成功证据；
5. 每个脚本的退出码、关键断言和脱敏日志；
6. Key 权限、限流、地区与网络错误单独报告。

不要保存 API Key，也不要把模型生成文本本身当成来源真实性证明。

## 6. 成本与模型时效性

不在教程中固定价格、赠送额度或“全套预计成本”。应使用本次响应的实际 token 分类，并按运行时
所选模型的官方当期计费规则计算。多次采样会增加调用量，但缓存、推理 token、工具调用和重试会让
实际成本偏离简单倍数。

当前仓库 provider 默认值见 [API_KEYS.md](API_KEYS.md)。运行前查询厂商 `/models` 或控制台，并参考：

- [DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)
- [Kimi 模型列表](https://platform.kimi.com/docs/models)
- [SiliconFlow 获取模型列表](https://docs.siliconflow.cn/cn/api-reference/models/get-model-list)
- [MiniMax 获取模型列表](https://platform.minimaxi.com/docs/api-reference/models/openai/list-models)
- [OpenAI Models](https://developers.openai.com/api/docs/models)
- [Claude 模型概览](https://platform.claude.com/docs/en/about-claude/models/overview)
