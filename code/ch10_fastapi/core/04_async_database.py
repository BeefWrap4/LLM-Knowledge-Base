# ---
# chapter: 10
# topic: FastAPI 与后端服务
# topic_id: fastapi.async_database
# difficulty: ⭐⭐⭐⭐
# tier: core
# deps: fastapi, sqlalchemy, aiosqlite
# run: python 04_async_database.py
# expected_runtime: <1s (无实际数据库写)
# expected_output: 模型字段已注册 / 异步引擎可创建
# ---
# See: ../../../10_FastAPI与后端服务.md
# Interview hooks:
#   1. create_async_engine 与 create_engine 在事件循环层面的区别？
#   2. 异步会话中的 yield 依赖如何处理 commit / rollback？
#   3. aiosqlite / asyncpg / aiomysql 三者性能与适用场景对比？
import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

# 异步数据库引擎
engine = create_async_engine("sqlite+aiosqlite:///./llm_service.db", echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class Conversation(Base):
    """对话记录表"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_message: Mapped[str] = mapped_column(String(2000))
    assistant_reply: Mapped[str] = mapped_column(String(4000))
    model: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())


# 依赖：获取数据库会话
async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# 单独演示 get_db 的生成器语义（无需真启动 FastAPI）
def _demo_dependency_generator():
    """快速观察 get_db 是异步生成器: 调用返回协程."""
    gen = get_db()
    import inspect

    assert inspect.isasyncgen(gen), "get_db 必须是 async generator"
    return gen


if __name__ == "__main__":
    # 1. 模型元数据已注册
    tables = list(Base.metadata.tables.keys())
    assert "conversations" in tables
    print(f"模型字段已注册: tables={tables}")

    # 2. 引擎创建成功（不实际连接）
    assert engine.url.drivername.endswith("aiosqlite")
    print(f"异步引擎可创建: {engine.url}")

    # 3. 验证 get_db 是异步生成器
    gen = _demo_dependency_generator()
    print(f"get_db 返回类型: async generator = {gen is not None}")
    print("OK")
