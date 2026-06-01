---
chapter: 6
topic: 内存管理与垃圾回收
difficulty: 中
interview_frequency: 4
created: 2026-06-01T00:00:00.000Z
tags:
  - python
  - 内存管理
  - 垃圾回收
  - 引用计数
  - 分代回收
---
# 第6章 Python 内存管理与垃圾回收 ⭐⭐⭐⭐

> **面试频率**：高（中高级岗位常问） | **难度**：中 | **建议学习时长**：4-5 小时

内存管理是 Python 进阶面试的高频考点，尤其在中高级岗位和性能优化场景中。本章深入讲解 Python 的内存分配器、垃圾回收三驾马车（引用计数、标记-清除、分代回收），以及内存泄漏的排查方法。

---

## 6.1 内存管理机制

### 6.1.1 Python 内存分配器架构

Python 的内存管理采用**分层架构**，根据对象大小使用不同的分配策略：

```mermaid
flowchart TD
    subgraph "Python 内存分配器层级"
        A[Python 对象] -->|> 512 bytes| B[pymalloc<br/>对象分配器]
        A -->|<= 512 bytes| C[C malloc<br/>系统分配器]
        
        B --> D[内存池 Pool]<-->E[内存块 Block]
        
        C --> F[操作系统<br/>虚拟内存]
    end
    
    style B fill:#4A6FA5,color:#fff
    style C fill:#6B8CBB,color:#fff
```

**三层分配器详解**：

| 层级 | 名称 | 管理对象 | 功能 |
|------|------|---------|------|
| **Layer 3** | 对象分配器 (pymalloc) | `<= 512 bytes` 的小对象 | 高效的内存池，减少 malloc 调用 |
| **Layer 2** | C 标准分配器 | `> 512 bytes` 的大对象 | 直接调用 `malloc`/`free` |
| **Layer 1** | 操作系统 | 虚拟内存页 | 页级别的内存管理 |

**pymalloc 工作原理**：

```mermaid
flowchart LR
    subgraph "pymalloc 内存池结构"
        A[arena<br/>256KB] --> B[pool 1<br/>4KB]
        A --> C[pool 2<br/>4KB]
        A --> D[pool N<br/>4KB]
        
        B --> B1[block 1<br/>64B]
        B --> B2[block 2<br/>64B]
        B --> B3[block N<br/>64B]
        
        C --> C1[block 1<br/>256B]
        C --> C2[block 2<br/>256B]
    end
    
    style A fill:#4A6FA5,color:#fff
    style B fill:#6B8CBB,color:#fff
    style C fill:#6B8CBB,color:#fff
```

- **Arena**：`256KB` 的大块内存，由 `malloc` 分配
- **Pool**：`4KB`（1 个内存页），同一 Pool 中的 block 大小相同
- **Block**：实际分配给对象的最小单元，大小为 `8, 16, 32, ..., 512` 字节

```python
import sys

# 查看对象内存大小
print(sys.getsizeof(42))           # 28 bytes (int)
print(sys.getsizeof("hello"))      # 54 bytes (str)
print(sys.getsizeof([1, 2, 3]))    # 88 bytes (list)
print(sys.getsizeof({"a": 1}))     # 232 bytes (dict)

# 查看 pymalloc 是否启用
print(sys._pymem_in_use())  # 当前使用的内存（Python 3.13+）
```

### 6.1.2 对象的内存布局（PyObject 头部）

Python 中**所有对象**在内存中都有统一的头部结构：

```
┌─────────────────────────────────────────┐
│              PyObject 头部               │
├──────────────────┬──────────────────────┤
│  ob_refcnt       │      ob_type          │
│  (引用计数)       │    (类型指针)          │
│   8 bytes        │      8 bytes          │
├──────────────────┴──────────────────────┤
│              对象特定数据                 │
│    (如 int 的 ob_digit, list 的 ob_item) │
└─────────────────────────────────────────┘
```

```c
// CPython 源码：Include/object.h
typedef struct _object {
    _PyObject_HEAD_EXTRA  // 双向链表指针（调试时使用）
    Py_ssize_t ob_refcnt;  // 引用计数
    PyTypeObject *ob_type; // 指向类型对象的指针
} PyObject;
```

**示例：整数对象的内存布局**

```python
import sys

# int 对象在内存中的结构
# PyObject_HEAD + ob_digit 数组
# 小整数（-5 ~ 256）被缓存，不重复创建

a = 42
b = 42
c = 257
d = 257

print(a is b)      # True （小整数缓存）
print(c is d)      # False （大整数不缓存，CPython 实现细节）

# 列表存储的是指针数组
lst = [1, 2, 3]
# 内存布局：PyObject_HEAD + ob_size + allocated + ob_item[]
# ob_item 存储的是指向各个元素对象的指针

# 验证列表存储的是引用
inner = [10, 20]
lst2 = [inner, inner]  # 两个元素指向同一对象
lst2[0][0] = 99
print(lst2)  # [[99, 20], [99, 20]] — 同一对象的引用
```

### 6.1.3 引用计数的核心机制

引用计数是 Python 内存管理的**基石**，每个对象维护一个引用计数器：

```mermaid
flowchart TD
    subgraph "引用计数变化示例"
        A[对象 X<br/>refcnt = 0] 
        
        B["a = X"] --> A2[对象 X<br/>refcnt = 1]
        A2 --> C["b = a"]
        C --> A3[对象 X<br/>refcnt = 2]
        A3 --> D["del a"]
        D --> A4[对象 X<br/>refcnt = 1]
        A4 --> E["del b"]
        E --> A5[对象 X<br/>refcnt = 0]
        A5 --> F["立即回收内存"]
    end
    
    style A5 fill:#e74c3c,color:#fff
```

**引用计数增加的场景**：

| 操作 | 示例 | 引用计数变化 |
|------|------|------------|
| 赋值给变量 | `a = obj` | +1 |
| 添加至列表 | `lst.append(obj)` | +1 |
| 添加至字典 | `dct['key'] = obj` | +1 |
| 作为参数传递 | `func(obj)` | +1（函数内） |
| 作为元组元素 | `tup = (obj,)` | +1 |

**引用计数减少的场景**：

| 操作 | 示例 | 引用计数变化 |
|------|------|------------|
| 变量离开作用域 | 函数返回 | -1 |
| 变量被重新赋值 | `a = other` | -1 |
| 从容器中移除 | `lst.remove(obj)` | -1 |
| 容器被销毁 | `del lst` | 内部所有元素 -1 |
| 对象被 del | `del a` | -1 |

```python
import sys

# 查看对象的引用计数
a = [1, 2, 3]
print(f"初始引用计数: {sys.getrefcount(a) - 1}")  # 减1因为 getrefcount 本身会+1

b = a  # 引用计数 +1
print(f"赋值后: {sys.getrefcount(a) - 1}")

c = [a, a]  # 引用计数 +2
print(f"加入列表后: {sys.getrefcount(a) - 1}")

del b  # 引用计数 -1
print(f"del b 后: {sys.getrefcount(a) - 1}")
```

---

## 6.2 垃圾回收机制 ⭐⭐⭐⭐

Python 采用**引用计数为主、标记-清除为辅、分代回收为优化**的三层垃圾回收策略。

### 6.2.1 引用计数：即时回收

```mermaid
flowchart LR
    A[对象创建] --> B[引用计数 = 1]
    B --> C{引用计数?}
    C -->|> 0| D[继续使用]
    C -->|= 0| E["立即调用 __del__<br/>回收内存"]
    
    D --> F["新引用 +1"] --> C
    D --> G["引用消失 -1"] --> C
    
    style E fill:#4A6FA5,color:#fff
```

**优点**：
- **即时回收**：引用计数归零时立即释放内存
- **简单高效**：无需暂停整个程序（理论上）
- **局部性好**：回收操作分散在正常运行中

**缺点**：
- **无法处理循环引用**：两个对象相互引用时，引用计数永远不会归零
- **线程安全开销**：每次修改引用计数都需要原子操作
- **空间开销**：每个对象需要存储引用计数字段

### 6.2.2 循环引用问题与标记-清除

```python
# 循环引用示例
class Node:
    def __init__(self, name):
        self.name = name
        self.next = None
    
    def __del__(self):
        print(f"Node {self.name} 被销毁")

# 创建循环引用
a = Node("A")
b = Node("B")
a.next = b  # A -> B
b.next = a  # B -> A （循环引用！）

del a  # Node("A") 的引用计数 = 1（b.next 指向它）
del b  # Node("B") 的引用计数 = 1（a.next 指向它）

# 循环引用导致内存泄漏！
# 引用计数无法回收这两个对象
```

**循环引用内存布局**：

```mermaid
flowchart LR
    subgraph "循环引用"
        A[对象 A<br/>refcnt = 1<br/>外部引用已删除]
        B[对象 B<br/>refcnt = 1<br/>外部引用已删除]
        
        A -->|"B 是 A 的属性"| B
        B -->|"A 是 B 的属性"| A
    end
    
    style A fill:#e74c3c,color:#fff
    style B fill:#e74c3c,color:#fff
```

### 6.2.3 标记-清除算法（Mark-and-Sweep）

标记-清除算法专门用于解决**容器对象**之间的循环引用问题。

```mermaid
flowchart TD
    subgraph "标记-清除算法流程"
        A[开始 GC 循环] --> B[阶段1：标记 Mark]
        
        B --> C["从根对象出发<br/>(全局变量、栈变量)"]
        C --> D["DFS/BFS 遍历所有可达对象"]
        D --> E["标记可达对象: gc_refs > 0"]
        
        E --> F[阶段2：清除 Sweep]
        F --> G["遍历所有容器对象"]
        G --> H{对象被标记?}
        H -->|Yes| I[保留对象]
        H -->|No| J["回收对象内存<br/>调用 __del__"]
        
        I & J --> K[结束 GC 循环]
    end
    
    style E fill:#2ecc71,color:#fff
    style J fill:#e74c3c,color:#fff
```

**算法步骤详解**：

1. **标记阶段**：从根对象（全局命名空间、调用栈上的局部变量）出发，递归标记所有**外部可达**的对象
2. **清除阶段**：遍历所有容器对象，未被标记的对象就是**仅被循环引用**的对象，可以被安全回收

```python
import gc

# 手动触发标记-清除
class Node:
    def __init__(self, name):
        self.name = name
        self.next = None
    def __del__(self):
        print(f"  Node {self.name} 被销毁")

# 创建循环引用并删除外部引用
a = Node("A")
b = Node("B")
a.next = b
b.next = a

del a, b

print("删除外部引用后，循环引用对象仍存在")
print(f"不可达对象数量: {len(gc.garbage)}")

# 手动触发垃圾回收
print("手动触发 gc.collect()...")
collected = gc.collect()  # 返回回收的对象数量
print(f"回收了 {collected} 个对象")
```

### 6.2.4 分代回收（Generational GC）

分代回收基于**弱代假说（Weak Generational Hypothesis）**：

> 大多数对象的生命周期很短，而存活时间越长的对象，继续存活的可能性越大。

```mermaid
flowchart TD
    subgraph "分代回收模型"
        A[新创建对象] --> B["第0代 (young)<br/>存活阈值: 700"]
        
        B -->|"GC 后存活"| C["第1代 (middle)<br/>存活阈值: 10"]
        C -->|"GC 后存活"| D["第2代 (old)<br/>存活阈值: 10"]
        
        B -->|"GC 回收"| E[回收内存]
        C -->|"GC 回收"| E
        D -->|"GC 回收"| E
        
        B -.->|"第0代 GC 10 次<br/>触发第1代 GC"| C
        C -.->|"第1代 GC 10 次<br/>触发第2代 GC"| D
    end
    
    style B fill:#e74c3c,color:#fff
    style C fill:#f39c12,color:#fff
    style D fill:#2ecc71,color:#fff
```

**分代回收参数**：

| 代数 | 名称 | 对象特征 | 默认阈值 | 检查频率 |
|------|------|---------|---------|---------|
| 第 0 代 | 新生代 | 新创建的对象 | 700 次分配 | 最频繁 |
| 第 1 代 | 中年代 | 经历过 1 次 GC | 10 次 GC | 中等 |
| 第 2 代 | 老年代 | 经历过 2+ 次 GC | 10 次 GC | 最少 |

```python
import gc

# 查看分代回收的阈值
print("GC 阈值:", gc.get_threshold())  # (700, 10, 10)
# 含义：
# 700: 第0代分配 700 次新对象触发一次 GC
# 10: 第0代 GC 10 次触发一次第1代 GC
# 10: 第1代 GC 10 次触发一次第2代 GC

# 查看各代当前对象数量
print("各代对象数量:", gc.get_count())  # (count0, count1, count2)

# 设置阈值（不建议随意修改）
# gc.set_threshold(500, 5, 5)  # 更频繁地 GC

# 单独清理某一代
gc.collect(0)  # 只清理第0代
gc.collect(1)  # 清理第0、1代
gc.collect(2)  # 清理所有代（最彻底）
```

### 6.2.5 三种 GC 机制的工作流程汇总

```mermaid
flowchart TD
    subgraph "Python 垃圾回收全貌"
        A[对象创建] --> B{是否有循环引用风险?}
        B -->|否<br/>非容器对象| C["引用计数管理<br/>refcnt = 0 立即回收"]
        B -->|是<br/>容器对象| D["引用计数 +<br/>分代 GC 监控"]
        
        D --> E{引用计数归零?}
        E -->|是| F["立即回收"]
        E -->|否| G["等待分代 GC 触发"]
        
        G --> H{分代 GC 阈值到达?}
        H -->|否| D
        H -->|是| I["标记-清除算法"]
        
        I --> J["阶段1: 标记可达对象"]
        J --> K["阶段2: 清除循环引用"]
        K --> L["回收循环引用对象的内存"]
        
        F & C & L --> M[内存释放]
    end
    
    style C fill:#4A6FA5,color:#fff
    style I fill:#e74c3c,color:#fff
```

### 6.2.6 gc 模块的手动控制

```python
import gc

# ========== 基本控制 ==========

# 禁用自动垃圾回收（性能关键场景）
gc.disable()
print(f"GC 是否启用: {gc.isenabled()}")  # False

# 手动触发全量 GC
gc.collect()  # 返回回收的不可达对象数量

# 重新启用
gc.enable()

# ========== 调试功能 ==========

# 设置调试标志
gc.set_debug(gc.DEBUG_STATS)  # 打印 GC 统计信息
# gc.set_debug(gc.DEBUG_LEAK)   # 打印循环引用泄漏信息
# gc.set_debug(gc.DEBUG_SAVEALL) # 将被回收对象保存到 gc.garbage

# ========== 获取对象信息 ==========

class MyClass:
    pass

obj = MyClass()

# 获取引用该对象的所有对象
referrers = gc.get_referrers(obj)
print(f"引用 obj 的对象数: {len(referrers)}")

# 获取对象引用的所有对象
referents = gc.get_referents(obj)
print(f"obj 引用的对象: {referents}")  # MyClass 的属性字典等

# 判断对象是否被某个代追踪
print(f"obj 在几代中: {gc.get_generation(obj)}")  # 0, 1, 或 2

# ========== 弱引用（打破循环引用的设计模式）==========
import weakref

class Parent:
    def __init__(self, name):
        self.name = name
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
        # 使用弱引用，不增加引用计数
        child.parent = weakref.ref(self)

class Child:
    def __init__(self, name):
        self.name = name
        self.parent = None
    
    def get_parent(self):
        # 弱引用需要 () 来访问
        parent = self.parent() if self.parent else None
        return parent.name if parent else "已销毁"

parent = Parent("P1")
child = Child("C1")
parent.add_child(child)

print(f"child 的 parent: {child.get_parent()}")  # P1

# 即使存在父子关系，也可以正常回收
del parent
print(f"parent 销毁后: {child.get_parent()}")  # 已销毁
```

---

## 6.3 内存泄漏与调试

### 6.3.1 常见内存泄漏场景

| 场景 | 原因 | 解决方案 |
|------|------|---------|
| **循环引用中的 `__del__`** | 有 `__del__` 的对象无法被标记-清除回收 | 使用 `weakref` 或重写 `__del__` |
| **全局缓存无上限** | 字典/列表无限增长 | 使用 LRUCache、定期清理 |
| **事件监听器未注销** | 观察者模式中对象被持续引用 | 注销监听器或使用弱引用 |
| **ORM 会话未关闭** | 数据库查询缓存累积 | 使用上下文管理器确保关闭 |
| **大对象引用残留** | 变量未释放，仍被引用 | 及时 `del`，缩小作用域 |
| **模块级变量累积** | 全局变量持续累积数据 | 定期清理或使用局部变量 |

### 6.3.2 内存泄漏检测工具

```python
import tracemalloc
import gc

# ========== tracemalloc：跟踪内存分配 ==========

def trace_memory():
    """使用 tracemalloc 追踪内存分配"""
    # 启动内存跟踪
    tracemalloc.start()
    
    # 记录初始状态
    snapshot1 = tracemalloc.take_snapshot()
    
    # 执行可能泄漏的代码
    leak_list = []
    for i in range(10000):
        leak_list.append("x" * 1000)  # 分配大量内存
    
    # 记录之后的状态
    snapshot2 = tracemalloc.take_snapshot()
    
    # 对比差异
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("[内存分配 TOP 10]")
    for stat in top_stats[:10]:
        print(f"  {stat}")
    
    # 当前内存使用
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n当前内存: {current / 1024 / 1024:.2f} MB")
    print(f"峰值内存: {peak / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()

# ========== objgraph：可视化对象引用图 ==========

def find_growth():
    """查找持续增长的对象类型"""
    gc.collect()  # 先清理一次
    
    # 记录当前各类对象数量
    before = {}
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        before[obj_type] = before.get(obj_type, 0) + 1
    
    # 执行一些操作...
    # operation()
    
    gc.collect()
    
    # 再次记录
    after = {}
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        after[obj_type] = after.get(obj_type, 0) + 1
    
    # 找出增长最多的类型
    growth = {
        t: after.get(t, 0) - before.get(t, 0)
        for t in set(list(before.keys()) + list(after.keys()))
    }
    
    print("[对象增长 TOP 10]")
    for obj_type, count in sorted(growth.items(), key=lambda x: -x[1])[:10]:
        if count > 0:
            print(f"  {obj_type}: +{count}")


# ========== 手动排查循环引用 ==========

def find_cycle_refs():
    """查找循环引用的对象"""
    # 获取所有不可达但有循环引用的对象
    gc.set_debug(gc.DEBUG_SAVEALL)
    gc.garbage.clear()
    
    gc.collect()
    
    if gc.garbage:
        print(f"发现 {len(gc.garbage)} 个循环引用对象:")
        for obj in gc.garbage:
            print(f"  {type(obj).__name__}: {repr(obj)[:100]}")
    else:
        print("未发现循环引用")
    
    gc.set_debug(0)


# ========== 使用上下文管理器确保资源释放 ==========

from contextlib import contextmanager

@contextmanager
def managed_resource(name):
    """确保资源被正确释放"""
    resource = {"name": name, "data": []}
    try:
        yield resource
    finally:
        # 清理操作
        resource["data"].clear()
        print(f"资源 {name} 已清理")


def safe_operation():
    with managed_resource("db_connection") as res:
        res["data"].extend([1, 2, 3])
        # 即使发生异常，finally 块也会执行
        return res["data"]
```

### 6.3.3 内存优化最佳实践

```python
# ========== 1. 使用 __slots__ 减少内存占用 ==========

class RegularClass:
    """普通类：每个实例都有 __dict__"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

class SlotsClass:
    """使用 __slots__：固定属性，无 __dict__"""
    __slots__ = ['a', 'b', 'c']
    
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

import sys

r = RegularClass(1, 2, 3)
s = SlotsClass(1, 2, 3)

print(f"RegularClass 大小: {sys.getsizeof(r)} bytes")
print(f"SlotsClass 大小: {sys.getsizeof(s)} bytes")
# SlotsClass 节省约 50%+ 内存

# 大规模创建时的差异更明显
regular_objs = [RegularClass(i, i, i) for i in range(100000)]
slots_objs = [SlotsClass(i, i, i) for i in range(100000)]

import tracemalloc
tracemalloc.start()
# 对比内存使用...

# ========== 2. 使用生成器代替列表 ==========

# 内存占用大：一次性生成所有数据
def get_all_data_bad(n):
    return [i ** 2 for i in range(n)]  # 列表推导式

# 内存友好：惰性生成
def get_all_data_good(n):
    for i in range(n):
        yield i ** 2  # 生成器

# ========== 3. 使用弱引用避免循环引用 ==========

import weakref

cache = weakref.WeakValueDictionary()  # 值不会阻止垃圾回收

class Data:
    pass

data = Data()
cache["key"] = data

print("key" in cache)  # True
del data  # 唯一强引用被删除
gc.collect()
print("key" in cache)  # False（自动从缓存中移除）

# ========== 4. 及时释放大对象引用 ==========

def process_large_data():
    # 大数据处理
    large_data = list(range(10000000))
    result = sum(large_data)
    
    # 处理完成后立即释放
    del large_data  # 主动释放引用
    gc.collect()    # 提示 GC（可选）
    
    return result

# ========== 5. 使用 lru_cache 限制缓存大小 ==========

from functools import lru_cache

@lru_cache(maxsize=128)  # 最多缓存 128 个结果
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))
print(f"缓存信息: {fibonacci.cache_info()}")  # CacheInfo(hits=98, misses=102, maxsize=128, currsize=102)
fibonacci.cache_clear()  # 手动清空缓存
```

---

## 🎯 面试真题精讲

### 题目 1：Python 的垃圾回收机制是什么？

> **答案**：Python 采用**三层垃圾回收机制**：
> 1. **引用计数（Reference Counting）**：主要机制，每个对象维护引用计数器，为 0 时立即回收。优点是即时回收，缺点是无法处理循环引用。
> 2. **标记-清除（Mark-and-Sweep）**：辅助机制，专门解决容器对象间的循环引用问题。从根对象出发标记可达对象，清除未被标记的对象。
> 3. **分代回收（Generational GC）**：优化机制，基于弱代假说，将对象分为三代（0/1/2），新生对象检查频率高，老对象检查频率低，减少 GC 开销。

### 题目 2：引用计数的优缺点？如何解决循环引用？

> **答案**：
> **优点**：即时回收、简单高效、局部性好
> **缺点**：无法处理循环引用、线程安全开销、无法回收有 `__del__` 的循环引用
> 
> **解决方案**：
> - 标记-清除算法检测循环引用
> - 使用 `weakref` 弱引用打破循环（如父子关系）
> - 避免不必要的双向引用

### 题目 3：`gc.collect()` 什么时候会无法回收对象？

> **答案**：以下情况 `gc.collect()` 无法回收：
> 1. 对象仍有外部引用（引用计数 > 0）
> 2. 循环引用中的对象定义了 `__del__` 方法（CPython 不确定回收顺序，保守处理）
> 3. 被 C 扩展模块持有引用
> 4. 被调试工具持有引用（如 `gc.DEBUG_SAVEALL` 模式下会存到 `gc.garbage`）

### 题目 4：什么是弱引用（weakref）？使用场景？

> **答案**：弱引用是一种不增加引用计数的引用方式。被弱引用指向的对象可以正常被垃圾回收。
> **使用场景**：
> - **缓存实现**：`WeakValueDictionary`/`WeakKeyDictionary`，对象被回收时自动从缓存中移除
> - **打破循环引用**：如树结构中父节点对子节点用强引用，子节点对父节点用弱引用
> - **观察者模式**：被观察者用弱引用存储观察者，避免注销遗漏导致内存泄漏

### 题目 5：`__del__` 方法一定会在对象销毁时调用吗？

> **答案**：**不一定**。`__del__` 的调用有以下限制：
> 1. 如果对象存在循环引用且定义了 `__del__`，标记-清除可能无法确定回收顺序，`__del__` 不会被调用
> 2. 解释器退出时，不保证所有对象的 `__del__` 都被调用
> 3. 如果 `__del__` 内部引用了即将销毁的对象，行为未定义
> 
> **最佳实践**：使用上下文管理器（`__enter__`/`__exit__`）或 `weakref.finalize` 替代 `__del__` 进行资源清理。

---

## 本章小结

```
内存管理与 GC
├── 内存分配器 (pymalloc)
│   ├── arena 256KB → pool 4KB → block 可变
│   ├── 对象头部：ob_refcnt + ob_type
│   └── 大对象(>512B)直接 malloc
├── 垃圾回收三层机制
│   ├── 引用计数 — 立即回收，无法处理循环引用
│   ├── 标记-清除 — DFS 标记可达对象，清除循环引用
│   └── 分代回收 — 第0代(700)/第1代(10)/第2代(10)
├── 内存泄漏场景
│   ├── 循环引用（__del__ 阻止回收）
│   ├── 全局缓存无限增长
│   ├── 未注销的事件监听器
│   └── 解决：weakref 弱引用
└── 优化技巧
    ├── __slots__ 减少每实例内存
    ├── 生成器替代列表
    ├── lru_cache 控制缓存大小
    └── WeakValueDictionary 自动清理
```

| 知识点 | 面试频率 | 掌握要求 |
|--------|---------|---------|
| 引用计数原理 | ⭐⭐⭐⭐ | 理解增减场景和优缺点 |
| 标记-清除算法 | ⭐⭐⭐⭐ | 能描述标记和清除两个阶段 |
| 分代回收机制 | ⭐⭐⭐⭐ | 理解三代模型和弱代假说 |
| 循环引用处理 | ⭐⭐⭐⭐⭐ | 能写代码演示和解决 |
| weakref 使用 | ⭐⭐⭐⭐ | 知道使用场景和 API |
| `__del__` 的坑 | ⭐⭐⭐⭐ | 了解不调用的情况 |
| 内存泄漏排查 | ⭐⭐⭐⭐ | 熟悉 tracemalloc 和 gc 模块 |
| `__slots__` 优化 | ⭐⭐⭐ | 知道内存优化效果 |

---

## 📚 相关章节

- [[01_Python编程基础]] — 数据类型的内存布局与引用计数基础
- [[02_Python核心面试专题_可变性与拷贝]] — 深拷贝/浅拷贝与循环引用的实际案例
- [[05_Python并发编程]] — GIL 与多线程下的内存安全问题
