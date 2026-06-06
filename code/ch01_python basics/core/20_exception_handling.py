# ---
# chapter: 1
# topic: 异常处理最佳实践
# section: 1.5.1
# difficulty: ⭐⭐
# tier: core
# deps: []
# run: python 20_exception_handling.py
# expected_runtime: <1s
# expected_output: 异常处理示例
# ---
# See: ../tutorial/01_Python编程基础.md (lines 1204-1315)
# Interview hooks:
#   1. BaseException 与 Exception 的区别?为什么不要捕获 BaseException?
#   2. try-except-else-finally 各自何时执行?
#   3. 自定义异常为什么要继承 Exception 而非 BaseException?
"""
异常处理最佳实践
"""

# ─────────────────────────────────────────────────────────────
# 异常层级结构（面试常问：捕获顺序要从子类到父类）
# ─────────────────────────────────────────────────────────────

"""
BaseException
 ├── SystemExit           # sys.exit() 触发
 ├── KeyboardInterrupt    # Ctrl+C 触发
 └── Exception            # 所有普通异常的基类
      ├── ArithmeticError
      │    └── ZeroDivisionError
      ├── LookupError
      │    ├── IndexError      # 列表索引越界
      │    └── KeyError        # 字典键不存在
      ├── TypeError            # 类型错误
      ├── ValueError           # 值错误
      ├── AttributeError       # 属性不存在
      └── IOError
           └── FileNotFoundError
"""

# ─────────────────────────────────────────────────────────────
# try-except-else-finally 完整结构
# ─────────────────────────────────────────────────────────────

def safe_read_file(filepath: str) -> str:
    """
    完整的异常处理示例
    """
    content = ""
    try:
        f = open(filepath, "r", encoding="utf-8")
        content = f.read()
    except FileNotFoundError:
        print(f"文件不存在: {filepath}")
        return ""
    except PermissionError:
        print(f"无权限读取: {filepath}")
        return ""
    except Exception as e:           # 捕获其他所有异常
        print(f"未知错误: {e}")
        return ""
    else:
        # try 成功执行（无异常）时执行
        print("文件读取成功")
    finally:
        # 无论是否异常都会执行
        if 'f' in locals() and not f.closed:
            f.close()
            print("文件已关闭")

    return content

# 演示：文件不存在
result = safe_read_file("__nonexistent_file__.txt")

# ─────────────────────────────────────────────────────────────
# 自定义异常
# ─────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """参数校验异常基类"""
    pass

class AgeError(ValidationError):
    """年龄校验异常"""
    def __init__(self, age, message=None):
        self.age = age
        self.message = message or f"无效年龄: {age}"
        super().__init__(self.message)

def validate_age(age: int) -> None:
    if not isinstance(age, int):
        raise TypeError(f"年龄必须是整数，收到 {type(age).__name__}")
    if age < 0 or age > 150:
        raise AgeError(age)

# 演示：年龄校验
for age in [25, -1, 200, "abc"]:
    try:
        validate_age(age) if isinstance(age, int) else validate_age(age)
    except (AgeError, TypeError) as e:
        print(f"校验失败 ({age!r}): {e}")

# ─────────────────────────────────────────────────────────────
# 上下文管理器简化文件操作
# ─────────────────────────────────────────────────────────────

class FileReader:
    """自定义上下文管理器"""

    def __init__(self, filepath, mode="r"):
        self.filepath = filepath
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filepath, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        exc_type: 异常类型
        exc_val:  异常值
        exc_tb:   异常追踪信息
        返回 True 表示异常已被处理，不再向外传播
        """
        if self.file:
            self.file.close()
        if exc_type is not None:
            print(f"捕获异常: {exc_type.__name__}: {exc_val}")
        return False   # 不抑制异常

# 使用自定义上下文管理器
# with FileReader("test.txt") as f:
#     content = f.read()

if __name__ == "__main__":
    print("OK")
