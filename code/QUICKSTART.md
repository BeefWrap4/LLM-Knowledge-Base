# QUICKSTART — 5 分钟从 clone 到第一次运行

## 1. 进入目录 + 创建虚拟环境

```bash
cd code/
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

> 需要 Python 3.10+。推荐 3.11。

## 2. 安装 core tier（30 秒，CPU 即可）

```bash
make install-core
```

或手动：
```bash
pip install -r requirements-core.txt
```

## 3. 运行你的第一个例子

```bash
python ch12_transformer_architecture/core/01_scaled_dot_product_attention.py
```

预期输出：
```
output shape: torch.Size([2, 4, 6])
attn shape: torch.Size([2, 4, 4])
OK
```

🎉 成功！这就是教程 §12.2.5 讨论的 Scaled Dot-Product Attention。

## 4. 跑所有 core smoke tests

```bash
make test
```

应通过所有测试，耗时 < 2 分钟。

## 5. 升级到 llm tier（需要 5 分钟 + API key）

```bash
make install-llm

# 设置 OpenAI / Anthropic key（任一）
export OPENAI_API_KEY=sk-...
# 或
export ANTHROPIC_API_KEY=sk-ant-...

make test-llm
```

注：llm tier 默认用 `shared.mock_llm` 跑测试（无需真实 API key）。如想跑真实 API：
```bash
export LIVE_LLM=1
pytest tests/ -m "llm"
```

## 6. 升级到 gpu tier（需要 NVIDIA GPU + 30 分钟）

```bash
make install-gpu
nvidia-smi  # 确认有 GPU
make test-gpu
```

如果你没 GPU（Mac/笔记本），gpu tier 会输出清晰报错，不影响其他 tier。

## 7. 探索章节目录

```bash
ls ch01_python_basics/core/
ls ch18_llm_frameworks/llm/
ls ch25_inference_engines/gpu/
```

每个目录有 `README.md` 列出该章所有例子。

## 故障排查

### `ModuleNotFoundError: No module named 'torch'`
→ 你没装 gpu tier。Core tier 不需要 torch，但 Ch12+ 例子需要。

### `ImportError: No module named 'openai'`
→ 装 llm tier: `make install-llm`

### `RuntimeError: CUDA not available` (gpu tier)
→ 正常。Mac/笔记本会看到友好提示。在有 NVIDIA GPU 的机器上跑。

### `openai.AuthenticationError: No API key provided`
→ llm 测试默认用 mock_llm，无需 key。如要真实 API：
```bash
export OPENAI_API_KEY=sk-...
```

### 装包太慢
```bash
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-core.txt
```

## 卸载

```bash
deactivate
rm -rf .venv
```
