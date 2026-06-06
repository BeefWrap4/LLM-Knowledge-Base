# ---
# chapter: 15
# topic: Agent智能体开发
# section: 15.9.6 ACI Design - 反模式与正模式对比
# difficulty: ⭐⭐⭐⭐⭐
# tier: llm
# deps: []
# run: python 21_aci_design_examples.py
# expected_runtime: <1s
# expected_output: 三组反模式 → 正模式对照；分页读文件示例
# ---
# See: ../tutorial/15_Agent智能体开发.md#15.9.6-ACI-Design
# Interview hooks:
#   1. ACI 和 API 设计最大的差异是什么？使用者不同（LLM vs 人类开发者）
#   2. 为什么"工具描述模糊"是 Agent 系统最常见的失败原因？
#   3. 大工具集（10+）vs 少而可组合的原子工具（2-3），哪种更适合 Agent？为什么？

# ============ 反模式 1：工具描述模糊 ============
# 不好的工具定义
bad_tool = {
    "name": "process_data",
    "description": "处理一些数据",
    "parameters": {
        "data": {"type": "object"},
        "options": {"type": "object"},
    }
}

# 改进后
good_tool = {
    "name": "filter_csv_rows",
    "description": """根据列名和值过滤 CSV 数据行。
    适用：用户说找出销售额大于 1000 的订单
    不适用：复杂 SQL 查询，用 query_database""",
    "parameters": {
        "csv_path": {
            "type": "string",
            "description": "CSV 文件绝对路径",
            "example": "/data/orders.csv",
        },
        "filter_column": {
            "type": "string",
            "description": "过滤列名",
            "example": "amount",
        },
        "filter_operator": {
            "type": "string",
            "enum": [">", "<", "==", "!=", "in"],
            "description": "比较操作符",
        },
        "filter_value": {
            "description": "比较值，支持数字、字符串、列表",
            "example": 1000,
        },
        "output_path": {
            "type": "string",
            "description": "过滤结果保存路径，可选，不传则返回内存数据",
        },
    },
    "returns": "JSON: row_count 整数, output_path 字符串, preview 列表",
    "errors": [
        {"code": "FILE_NOT_FOUND", "message": "文件不存在，建议检查路径"},
        {"code": "COLUMN_NOT_EXIST", "message": "列名不存在，可用 list_csv_columns 工具查询"},
    ],
}


# ============ 反模式 2：返回数据过大 ============
# 一次性返回整个 1GB 文件
def read_full_file_bad(path: str) -> str:
    return open(path).read()


# 改进：分页加引用
def read_file_with_pagination(path: str, start_line: int = 0,
                               line_count: int = 100) -> dict:
    """
    分页读取文件

    Returns:
        {
            "content": "前 100 行内容",
            "next_start_line": 100,
            "total_lines": 50000,
            "has_more": True,
        }
    """
    with open(path) as f:
        lines = f.readlines()
    return {
        "content": "".join(lines[start_line:start_line + line_count]),
        "next_start_line": start_line + line_count,
        "total_lines": len(lines),
        "has_more": start_line + line_count < len(lines),
    }


# ============ 反模式 3：工具膨胀 ============
# 10 个专用工具
bad_tools = [
    "get_user_by_id", "get_user_by_email", "get_user_by_phone",
    "get_active_users", "get_inactive_users", "get_recent_users",
    "get_user_count", "get_user_paginated", "get_user_summary",
    "search_users",
]

# 改进：少量可组合的原子工具
good_tools = [
    "query_users 带 filter 与 sort 与 page 与 page_size 参数",
    "get_user_by_id 接收 id 参数",
]


def main():
    print("=== 反模式 1：工具描述对比 ===")
    print(f"bad_tool  desc={bad_tool['description']!r}")
    print(f"good_tool desc={good_tool['description'][:60]!r}...")
    assert bad_tool["description"] == "处理一些数据"
    assert "适用" in good_tool["description"] and "不适用" in good_tool["description"]

    print("\n=== 反模式 2：分页读文件（演示） ===")
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for i in range(1, 251):
            f.write(f"line {i}\n")
        tmp_path = f.name

    page1 = read_file_with_pagination(tmp_path, start_line=0, line_count=100)
    print(f"page1: total_lines={page1['total_lines']}, has_more={page1['has_more']}, "
          f"next_start={page1['next_start_line']}")
    page3 = read_file_with_pagination(tmp_path, start_line=200, line_count=100)
    print(f"page3: has_more={page3['has_more']}, next_start={page3['next_start_line']}")

    print("\n=== 反模式 3：工具数量对比 ===")
    print(f"bad_tools count={len(bad_tools)} | good_tools count={len(good_tools)}")
    assert len(bad_tools) > len(good_tools)
    print("\nOK")


if __name__ == "__main__":
    main()
