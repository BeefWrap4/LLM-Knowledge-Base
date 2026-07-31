# ---
# chapter: 05
# topic: Python并发编程
# section: 5.5.5 实战：用 asyncio 编写高并发 HTTP 客户端
# difficulty: ⭐⭐⭐
# tier: core
# deps: [aiohttp]
# run: python 15_async_http_client.py
# expected_runtime: <2s (local test server; no external network)
# expected_output: prints elapsed time, sample results, success ratio, OK
# ---
# See: ../tutorial/05_Python并发编程.md#555-实战用-asyncio-编写高并发-http-客户端
# Interview hooks:
#   - aiohttp 相比 requests 的优势？为什么事件循环里要共享 ClientSession？
#   - asyncio.Semaphore 在 HTTP 客户端中的用途？
#   - gather(return_exceptions=True) 配合 try/except 的替代方案？
import asyncio
import time

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer


async def fetch(session: aiohttp.ClientSession, url: str) -> dict:
    """异步 HTTP GET 请求"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            data = await response.json()
            return {"url": url, "status": response.status, "data": data}
    except Exception as e:
        return {"url": url, "error": str(e)}


async def fetch_all(urls: list[str], max_concurrent: int = 100) -> list[dict]:
    """
    高并发批量请求

    Args:
        urls: URL 列表
        max_concurrent: 最大并发数（防止目标服务器过载）
    """
    if max_concurrent < 1:
        raise ValueError("max_concurrent 必须大于 0")

    # 使用信号量限制并发数
    semaphore = asyncio.Semaphore(max_concurrent)

    # 共享 session（复用 TCP 连接，性能更好）
    async with aiohttp.ClientSession() as session:

        async def bounded_fetch(url):
            async with semaphore:
                return await fetch(session, url)

        # 使用 gather 并发执行所有请求
        tasks = [bounded_fetch(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        return [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]


async def local_post(request: web.Request) -> web.Response:
    """本地确定性测试端点；示例默认不访问公网。"""
    post_id = int(request.match_info["post_id"])
    await asyncio.sleep(0.01)
    return web.json_response({"id": post_id, "title": f"post-{post_id}"})


async def main():
    app = web.Application()
    app.router.add_get("/posts/{post_id}", local_post)

    async with TestServer(app) as server:
        urls = [str(server.make_url(f"/posts/{i}")) for i in range(1, 21)]
        start = time.perf_counter()
        results = await fetch_all(urls, max_concurrent=5)
        elapsed = time.perf_counter() - start

        print(f"请求 {len(urls)} 个本地 URL，耗时 {elapsed:.2f}s")
        print(f"前3个结果: {results[:3]}")

        success = sum(1 for result in results if 200 <= result.get("status", 0) < 300)
        print(f"成功率: {success}/{len(urls)}")
        assert success == len(urls)


if __name__ == "__main__":
    asyncio.run(main())
    print("OK")
