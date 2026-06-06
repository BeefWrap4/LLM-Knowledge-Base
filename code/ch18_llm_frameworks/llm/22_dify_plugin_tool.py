# ---
# chapter: 18
# topic: LLM工程框架实战
# section: 18.5.4 插件与工具扩展
# difficulty: ⭐⭐⭐⭐
# tier: llm
# deps: yfinance (optional, mocked)
# run: python 22_dify_plugin_tool.py
# expected_runtime: <1s
# expected_output: stock price demo
# ---
# See: ../tutorial/18_LLM工程框架实战.md § 18.5.4
# Interview hooks:
#   1. Dify 插件系统的 Provider / Tool / Parameter 是如何组织的？
#   2. 工具调用节点与 LLM 节点在工作流中如何串联？
# Dify 插件 manifest + 工具实现（离线 mock）
# 文件：my_tool_provider.yaml
MANIFEST_YAML = """
identity:
  name: "股票查询工具"
  author: "your_team"
  label:
    en_US: "Stock Query Tool"
    zh_Hans: "股票查询工具"

tools:
  - name: "query_stock_price"
    description: "查询股票实时价格"
    parameters:
      - name: "symbol"
        type: "string"
        description: "股票代码，如 AAPL"
        required: true
      - name: "period"
        type: "string"
        description: "查询周期：1d/5d/1m/6m/1y"
        required: false
"""

def query_stock_price(symbol: str, period: str = "1d") -> str:
    """查询股票价格 - Mock 实现，避免对 yfinance 的硬依赖"""
    try:
        import yfinance as yf  # 可选依赖
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        current_price = hist['Close'].iloc[-1]
        return f"{symbol} 最新价格: ${current_price:.2f}"
    except ImportError:
        # 离线 mock：返回模拟价格
        mock_price = {"AAPL": 192.34, "TSLA": 248.50, "MSFT": 421.77}.get(symbol, 100.0)
        return f"{symbol} 最新价格 (mock): ${mock_price:.2f}"
    except Exception as e:
        return f"查询失败: {e}"

print("=== Dify Plugin Manifest ===")
print(MANIFEST_YAML)
print("=== 工具调用演示 ===")
print(query_stock_price("AAPL"))
print(query_stock_price("TSLA", "5d"))

if __name__ == "__main__":
    print("OK")
