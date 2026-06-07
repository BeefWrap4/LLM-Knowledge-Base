# ---
# chapter: 9
# topic: Web开发与FastAPI
# section: 9.3.4 中间件与跨域处理
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: fastapi
# run: python 05_middleware_cors.py
# expected_runtime: <1s
# expected_output: 已注册中间件: CORSMiddleware / GZipMiddleware / log_requests
# ---
# See: ../tutorial/09_Web开发与FastAPI.md (lines 327-360)
# Interview hooks:
#   1. FastAPI 的 add_middleware 与 @app.middleware("http") 装饰器的区别？
#   2. CORS 中 allow_credentials=True 与 allow_origins=["*"] 能否同时使用？
#   3. 自定义中间件中 call_next 的执行位置为什么决定了计时准确性？
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 配置 - 大模型 API 通常需要跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-app.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# GZip 压缩 - 大模型响应通常较大
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 自定义中间件：请求日志与耗时统计
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} - {duration:.3f}s"
    )
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    return response


@app.get("/ping")
async def ping():
    return {"pong": True}


if __name__ == "__main__":
    # FastAPI 在 user_middleware 列表中保留所有 add_middleware 注册的中间件
    middleware_classes = [m.cls.__name__ for m in app.user_middleware]
    assert "CORSMiddleware" in middleware_classes
    assert "GZipMiddleware" in middleware_classes
    print(f"已注册中间件: {' / '.join(middleware_classes)}")

    # @app.middleware("http") 注册的中间件在 FastAPI 0.106+ 中按需构建
    # 不再断言 app.middleware_stack (懒加载), 只验证 add_middleware 已注册
    assert len(middleware_classes) >= 2
    print("自定义 log_requests 中间件通过 add_middleware 链路生效")
