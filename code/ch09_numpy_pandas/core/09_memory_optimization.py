# ---
# chapter: 9
# topic: NumPy 与 Pandas 数据处理
# topic_id: numpy_pandas.memory_optimization
# difficulty: 高
# tier: core
# deps: pandas, numpy, scipy
# run: python 09_memory_optimization.py
# expected_runtime: <10s
# expected_output: 优化前/后内存以及减少百分比；稀疏矩阵 csr 构造成功
# ---
# See: ../../../09_NumPy与Pandas数据处理.md
# Interview hooks:
#   1. 10GB 的 CSV 怎么读取和处理（分块 + 迭代器）？
#   2. int64 -> int8/16/32 的判定依据是什么？用 np.iinfo 检查什么？
#   3. Dask 与 Polars 各自相对 Pandas 的优势是什么？

import numpy as np
import pandas as pd


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    大数据集内存优化技巧

    面试常考：如何处理内存不足的 Pandas 操作？
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"优化前内存: {start_mem:.2f} MB")

    # 1. 数值类型下转换
    for col in df.select_dtypes(include=["int"]).columns:
        col_min, col_max = df[col].min(), df[col].max()

        if col_min >= 0:  # 无符号整数
            if col_max <= np.iinfo(np.uint8).max:
                df[col] = df[col].astype(np.uint8)
            elif col_max <= np.iinfo(np.uint16).max:
                df[col] = df[col].astype(np.uint16)
            elif col_max <= np.iinfo(np.uint32).max:
                df[col] = df[col].astype(np.uint32)
        else:  # 有符号整数
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

    # 2. 浮点类型下转换
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].astype(np.float32)

    # 3. 类别型数据用 category
    for col in df.select_dtypes(include=["object"]).columns:
        num_unique = df[col].nunique()
        num_total = len(df)
        # 当唯一值比例 < 50% 时使用 category
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"优化后内存: {end_mem:.2f} MB")
    print(f"减少: {(1 - end_mem / start_mem) * 100:.1f}%")

    return df


# ========== 分块读取大文件 ==========
def process_large_csv(filepath: str, chunksize: int = 100000):
    """
    分块处理大 CSV 文件（内存不足时的解决方案）

    面试常考：10GB 的 CSV 怎么读取和处理？
    """
    chunk_results = []

    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # 对每个 chunk 进行处理
        processed = chunk.groupby("category")["value"].sum()
        chunk_results.append(processed)

    # 合并所有 chunk 的结果
    final_result = pd.concat(chunk_results).groupby(level=0).sum()
    return final_result


# ========== 使用更高效的数据类型 ==========
# 时间序列数据用 datetime
df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "flag": [1, 0, 1, 0, 1]})
df["date"] = pd.to_datetime(df["date"])

# 布尔值
df["flag"] = df["flag"].astype(bool)

# 稀疏矩阵（大量零值）— 只取数值列避免 object dtype
from scipy.sparse import csr_matrix

numeric_df = df.select_dtypes(include=["int", "float", "bool"])
sparse_data = csr_matrix(numeric_df.values)  # 内存大幅减少
print(f"稀疏矩阵: shape={sparse_data.shape}, nnz={sparse_data.nnz}")


if __name__ == "__main__":
    # 构造一个可以用作优化演示的 DataFrame
    rng = np.random.default_rng(42)
    demo = pd.DataFrame(
        {
            "small_int": rng.integers(0, 200, size=1000),  # 可用 uint8
            "big_int": rng.integers(-(10**6), 10**6, size=1000),  # 可下转换为 int32
            "price": rng.random(1000) * 100,  # 可用 float32
            "category": rng.choice(["A", "B", "C", "D"], size=1000),  # 可用 category
        }
    )
    optimized = optimize_memory(demo)
    print(optimized.dtypes)
    print("OK")
