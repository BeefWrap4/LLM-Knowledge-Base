# ---
# chapter: 9
# topic: Web开发与FastAPI
# section: 9.2.2 依赖注入系统
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: fastapi
# run: python 01_dependency_injection.py
# expected_runtime: <1s
# expected_output: 路由已注册: /users/me -> read_me
# ---
# See: ../tutorial/09_Web开发与FastAPI.md (lines 88-121)
# Interview hooks:
#   1. FastAPI 的 Depends() 与路径参数解析流程是怎样的？
#   2. 为什么 yield 依赖能管理数据库连接的生命周期？
#   3. 同一请求内相同依赖被调用多次，会发生什么？
import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

app = FastAPI()
bearer = HTTPBearer(auto_error=False)


# 定义依赖函数
def get_db_connection():
    """模拟数据库连接"""
    conn = {"status": "connected", "pool_size": 10}
    try:
        yield conn
    finally:
        conn["status"] = "closed"


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
):
    """教学用 Bearer 校验；生产环境还需验证 JWT claims 与 scope。"""
    expected = os.getenv("DEMO_API_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务端尚未配置 DEMO_API_TOKEN",
        )
    if credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效或缺失的 Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"user_id": 1, "name": "demo-user"}


# 路由中注入依赖 - 自动解析并按需调用
@app.get("/users/me")
async def read_me(
    db: Annotated[dict, Depends(get_db_connection)],
    user: Annotated[dict, Depends(get_current_user)],
):
    """
    依赖注入的优势：
    1. 自动管理生命周期（yield 支持清理逻辑）
    2. 依赖可嵌套（db 依赖可被其他依赖复用）
    3. 自动缓存同请求内的依赖实例
    4. 与路径操作参数一致的处理方式
    """
    return {"user": user, "db_status": db["status"]}


if __name__ == "__main__":
    # 不启动服务器，只做路由自检
    routes = [(r.path, r.endpoint.__name__) for r in app.routes if hasattr(r, "endpoint")]
    target = next(((p, n) for p, n in routes if n == "read_me"), None)
    assert target == ("/users/me", "read_me"), f"路由未注册正确: {target}"
    print(f"路由已注册: {target[0]} -> {target[1]}")
    print("OK")
