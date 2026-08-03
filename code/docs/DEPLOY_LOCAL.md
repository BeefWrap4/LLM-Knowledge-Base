# 本地环境部署与验收

> 核验日期：2026-07-31。本指南覆盖 Python、离线回归、可选本地模型、中间件和真实 LLM API。
> 各组件独立验收；“脚本能结束”不等于所有组件或真实 API 已通过。

## 1. 前置条件

- Python 3.10 或更高版本；
- 安装依赖和下载模型所需的磁盘、内存与网络；
- Docker（仅在需要 Redis/pgvector 时）；
- GPU 是 GPU 示例和较大本地模型的可选条件，不是 core/离线 LLM 回归的前提。

模型实际下载体积和运行内存会随文件修订、精度与缓存变化。先看清单，不使用固定“整套约 3 GB”
作为容量承诺：

```bash
cd code/
python scripts/download_models.py --help
```

## 2. 安装 Python 环境

从 `code/` 目录安装最小 core 依赖：

```bash
python -m pip install -r requirements-core.txt
```

需要 LLM/框架示例时：

```bash
python -m pip install -r requirements-llm.txt
```

先运行离线门禁；它不扫描或读取 `.env`，不返回或使用 LLM Key，也不联网调用厂商 API：

```bash
LLM_MOCK=1 make ci-quick
LLM_MOCK=1 python -m pytest tests/ -m "not gpu" -q
```

PowerShell 可用：

```powershell
$env:LLM_MOCK="1"; make ci-quick
```

## 3. 配置真实 LLM

复制模板并按需填写一家厂商：

```bash
cp .env.example .env
```

PowerShell：

```powershell
Copy-Item .env.example .env
```

当前 provider、模型、base URL、MiniMax 大小写限制与安全要求见 [API_KEYS.md](API_KEYS.md)。
价格、赠送额度、账户权限和模型能力以运行时厂商控制台为准。

真实探针必须显式关闭离线模式，并检查没有进入错误兜底：

```bash
LLM_MOCK=0 LLM_PROVIDER=deepseek python -c "from shared.llm_client import UnifiedClient; r=UnifiedClient().chat(prompt='Reply only OK', max_tokens=16); assert not r.mock, repr(r.raw); print(r.provider, r.model, r.usage)"
```

PowerShell：

```powershell
$env:LLM_MOCK="0"; $env:LLM_PROVIDER="deepseek"; python -c "from shared.llm_client import UnifiedClient; r=UnifiedClient().chat(prompt='Reply only OK', max_tokens=16); assert not r.mock, repr(r.raw); print(r.provider, r.model, r.usage)"
```

## 4. 下载可选本地模型

下载脚本当前默认选择三个 `required=True` 项，脚本元数据估算合计约 1.7 GB：

- `BAAI/bge-small-zh-v1.5`；
- `BAAI/bge-reranker-v2-m3`；
- `Qwen/Qwen2.5-0.5B-Instruct`。

执行：

```bash
python scripts/download_models.py --required-only
```

只下载单类：

```bash
python scripts/download_models.py --embedding
python scripts/download_models.py --reranker
python scripts/download_models.py --llm
```

模型写入仓库外的 `TUTORIAL_MODELS_DIR`（Windows 当前默认是
`E:\AI_Models\Projects\MyDocument\Python到大模型应用_面试教程_2026版\models`）。下载器会依次
尝试 ModelScope 与 Hugging Face 镜像；是否可下载、许可条件和文件大小仍应以模型仓库当前内容为准。

## 5. 启动 Redis 与 pgvector

下面的独立容器命令与 `scripts/test_integration.py` 当前硬编码的宿主端口和测试密码一致：

```bash
docker run -d --name llm-kb-redis -p 16379:6379 redis:7-alpine redis-server --save 60 1 --appendonly yes
docker run -d --name llm-kb-postgres -p 15432:5432 -e POSTGRES_USER=llmkb -e POSTGRES_PASSWORD=llmkb_test -e POSTGRES_DB=vectordb pgvector/pgvector:pg16
```

只读健康检查：

```bash
docker ps --filter name=llm-kb-redis --filter name=llm-kb-postgres
docker exec llm-kb-redis redis-cli ping
docker exec llm-kb-postgres pg_isready -U llmkb -d vectordb
```

### Compose 的当前边界

仓库根目录 `docker-compose.yml` 中：

- `llm` profile 启动 app + Redis，不含 pgvector；
- `gpu` profile 才包含 pgvector；
- postgres 没有映射宿主 `15432`，默认密码也不是集成脚本使用的 `llmkb_test`；
- compose 没有传入 `MINIMAX_API_KEY`。

因此，当前不能把 `docker compose --profile llm up -d` 写成宿主机四组件集成测试的等价替代。
若修改 compose，请同步端口、密码、环境变量与测试脚本后再验收。

## 6. 组件测试

在 `bge-small-zh-v1.5` 和两个中间件均就绪后：

```bash
RUN_REAL_INTEGRATION=1 LLM_MOCK=0 LLM_PROVIDER=deepseek python scripts/test_integration.py
```

当前 `test_integration.py` 会先要求 `RUN_REAL_INTEGRATION=1`、`LLM_MOCK=0`、显式 provider 与
对应 Key，然后检查 embedding、自相似检索、Redis、pgvector 和一个真实 LLM 请求。缺 Key、
`resp.mock is True`、空响应或任一组件异常都会使汇总失败。

它的 0 退出码只能作为这四项和本次所选 provider 的条件性组件 smoke 证据，不覆盖其他模型、
框架、GPU、Docker profile 或生产要求。业务上线仍需第 3 节的 provider 探针与业务输出断言。

## 7. 运行示例

离线 LLM tier：

```bash
LLM_MOCK=1 python scripts/run_all_examples.py --tier llm --parallel 4 --timeout 180
```

真实 demo wrapper：

```bash
LLM_MOCK=0 bash scripts/run_real_demos.sh --confirm-real all deepseek
```

wrapper 会拒绝缺少确认标志或目标 Key 的运行，并区分可识别的 mock、skip 与失败；它仍不能读取
每个示例的响应对象或账单。证据边界见 [REAL_DEMOS.md](REAL_DEMOS.md)。

## 8. 故障定位

### Python 依赖缺失

```bash
python -m pip install -r requirements-llm.txt
```

### Redis 或 pgvector 不可达

先检查容器状态、端口映射和健康状态：

```bash
docker ps --filter name=llm-kb-redis --filter name=llm-kb-postgres
docker logs llm-kb-redis
docker logs llm-kb-postgres
```

### 模型下载失败

列出目标并单独重试，以便定位来源或权限问题：

```bash
python scripts/download_models.py --help
python scripts/download_models.py --embedding
```

### LLM 返回 401、403、404、429 或 fallback

检查 Key 权限、实际模型列表、base URL、地区/网络与限流。`resp.mock is True` 一律按真实调用失败处理，
不要改报为离线成功。

## 9. 停止与清理

停止容器不会删除数据：

```bash
docker stop llm-kb-redis llm-kb-postgres
```

第 5 节的独立容器命令没有挂载持久化 volume，删除容器会同时删除其中的 Redis/PostgreSQL 数据。
先确认不再需要数据；如果自行添加了命名 volume，volume、宿主模型目录和 `.env` 不会因删除容器
自动消失：

```bash
docker rm llm-kb-redis llm-kb-postgres
```

模型目录和 `.env` 可能包含大量文件或密钥，应由使用者确认精确路径后另行删除，不在本指南提供递归
删除命令。

## 10. 最终验收记录

分别记录：

- 离线：`LLM_MOCK=1`、pytest/verify 命令、通过/跳过/失败数量；
- 本地模型：模型 ID、来源、文件完整性与实际设备；
- 中间件：镜像标签、健康检查、端口与持久化策略；
- 真实 API：`LLM_MOCK=0`、provider/model、`resp.mock is False`、请求 ID、token、时延和业务断言。

只有四类证据都存在时，才能声称对应的完整本地栈通过；不要用单个脚本的 0 退出码替代分项证据。
