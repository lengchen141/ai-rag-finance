import asyncio
import time
import httpx

# ========== 1. 基础：async 函数 ==========
# async def 定义协程，相当于返回 CompletableFuture 的方法
# await 相当于 .join() / .get()，等待结果但不阻塞线程

async def fetch_stock_price(client, stock_name, delay):
    """模拟调用股票接口（带延迟）"""
    await asyncio.sleep(delay)   # 模拟网络IO，相当于 Thread.sleep 但不阻塞
    prices = {"茅台": 1820.5, "宁德时代": 210.3, "比亚迪": 245.8}
    return f"{stock_name}: {prices.get(stock_name, '未知')}"

# ========== 2. 串行 vs 并发对比 ==========

async def serial_version():
    """串行：一个一个等，总共 3 秒"""
    print("--- 串行执行 ---")
    start = time.time()
    result1 = await fetch_stock_price(None, "茅台", 1)
    result2 = await fetch_stock_price(None, "宁德时代", 1)
    result3 = await fetch_stock_price(None, "比亚迪", 1)
    print(f"{result1}\n{result2}\n{result3}")
    print(f"耗时: {time.time() - start:.1f}秒\n")

async def concurrent_version():
    """并发：同时发出去，总共 1 秒"""
    print("--- 并发执行（asyncio.gather）---")
    start = time.time()
    # gather 相当于 CompletableFuture.allOf，等所有任务完成
    results = await asyncio.gather(
        fetch_stock_price(None, "茅台", 1),
        fetch_stock_price(None, "宁德时代", 1),
        fetch_stock_price(None, "比亚迪", 1)
    )
    for r in results:
        print(r)
    print(f"耗时: {time.time() - start:.1f}秒\n")

# ========== 3. 实战：用 httpx 异步调用真实接口 ==========

async def fetch_real_api():
    """异步HTTP请求，对标 Java 的 WebClient / OkHttp 异步回调"""
    print("--- 真实异步HTTP请求 ---")
    start = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        # 同时发3个请求，并发等待
        tasks = [
            client.get("https://httpbin.org/delay/1"),
            client.get("https://httpbin.org/delay/1"),
            client.get("https://httpbin.org/delay/1"),
        ]
        responses = await asyncio.gather(*tasks)

        for i, resp in enumerate(responses, 1):
            print(f"请求{i}: 状态码 {resp.status_code}")

    print(f"3个请求总耗时: {time.time() - start:.1f}秒（串行需要3秒）\n")

# ========== 4. 对比：同步版本怎么写 ==========

def sync_version():
    """同步版本，用 requests 库（后面做项目时会看到区别）"""
    print("--- 同步HTTP请求（对比用）---")
    import requests
    start = time.time()
    for i in range(3):
        requests.get("https://httpbin.org/delay/1")
        print(f"请求{i+1} 完成")
    print(f"3个请求总耗时: {time.time() - start:.1f}秒\n")

# ========== 运行 ==========
async def main():
    await serial_version()
    await concurrent_version()
    await fetch_real_api()

# asyncio.run() 是入口，相当于启动事件循环
asyncio.run(main())

# 同步版本对比（取消注释可以跑，需要 pip install requests）
# sync_version()