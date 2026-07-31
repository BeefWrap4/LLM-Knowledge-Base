# ---
# chapter: 9
# topic: Web开发与FastAPI
# section: 9.3.1 路由与请求处理
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: fastapi, pydantic
# run: python 02_routes_and_requests.py
# expected_runtime: <1s
# expected_output: 已注册 4 个路由: /health, /chat, /models/{model_id}, /search
# ---
# See: ../tutorial/09_Web开发与FastAPI.md (lines 152-216)
# Interview hooks:
#   1. Query / Path / Body 三类参数是如何被 FastAPI 区分的？
#   2. response_model 的作用是什么？和返回类型注解有何区别？
#   3. Pydantic Field 中的 ge/le/min_length 如何参与自动校验？

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

app = FastAPI(title="LLM Service API", version="1.0.0")


# ========== 数据模型定义 ==========
class ChatRequest(BaseModel):
    """聊天请求模型 - Pydantic 自动验证"""

    message: str = Field(min_length=1, max_length=2000, description="用户消息")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成长度")

    model_config = {
        "json_schema_extra": {
            "examples": [{"message": "你好，请介绍FastAPI", "temperature": 0.7, "max_tokens": 512}]
        }
    }


class ChatResponse(BaseModel):
    """聊天响应模型"""

    reply: str
    tokens_used: int
    model: str


# ========== 路由定义 ==========
@app.get("/health", summary="健康检查")
async def health_check():
    """最简单的端点，用于服务探活"""
    return {"status": "ok", "service": "llm-api"}


@app.post("/chat", response_model=ChatResponse, summary="对话接口")
async def chat(request: ChatRequest):
    """
    对话接口 - 完整展示请求体验证和响应模型
    - 请求体自动按 ChatRequest 模型校验
    - 响应自动按 ChatResponse 模型序列化
    """
    # 模拟 LLM 推理
    reply = f"收到消息：{request.message[:50]}..."
    return ChatResponse(reply=reply, tokens_used=42, model="demo-chat-v1")


@app.get("/models/{model_id}", summary="获取模型信息")
async def get_model(model_id: str = Path(description="模型ID", pattern=r"^[a-zA-Z0-9-_]+$")):
    """路径参数 + 正则校验"""
    models_db = {"demo-chat-v1": "Demo", "local-chat": "Self-hosted"}
    if model_id not in models_db:
        raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")
    return {"model_id": model_id, "provider": models_db[model_id]}


@app.get("/search", summary="搜索接口")
async def search(
    q: str = Query(min_length=2, description="搜索关键词"),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """查询参数 + 分页验证"""
    return {"query": q, "limit": limit, "offset": offset, "results": []}


if __name__ == "__main__":
    # 1. 验证 Pydantic 模型行为
    req = ChatRequest(message="hello world")
    assert req.temperature == 0.7  # default 值
    resp = ChatResponse(reply="hi", tokens_used=1, model="demo-chat-v1")
    assert resp.model_dump() == {"reply": "hi", "tokens_used": 1, "model": "demo-chat-v1"}

    # 2. 校验非法输入
    try:
        ChatRequest(message="", temperature=3.0)
    except Exception as e:
        assert "min_length" in str(e) or "temperature" in str(e) or "le" in str(e)

    # 3. 列出注册的路由
    paths = sorted({r.path for r in app.routes if hasattr(r, "path")})
    assert "/health" in paths and "/chat" in paths
    assert "/models/{model_id}" in paths
    assert "/search" in paths
    print(f"已注册 {len(paths)} 个路由: {', '.join(paths)}")
    print("OK")
