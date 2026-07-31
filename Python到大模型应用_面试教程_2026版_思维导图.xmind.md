# Python 到大模型应用 —— 面试教程 2026 版 (思维导图)

> **导入 XMind**: 文件 → 导入 → Markdown → 选择本文件
> **结构约定**: `#` 中心主题, `##` 一级分支, `###` 二级, `####` 三级, 列表项为叶子
> **覆盖范围**: 40 章教程 + 433 个配套代码示例 + CI workflow + Docker 部署

## 0. 库总览

### 0.1 元信息
- 库名: Python 到大模型应用 —— 面试教程 2026 版
- 章节数: 40 章正文 + 1 目录 + 1 健康报告
- 配套代码: 433 .py (158 core + 199 llm + 76 gpu)
- 面试题: 300+ 道
- 教程大小: ~2,200 KB
- 难度跨度: 入门 → 专家
- 时效基线: 2026-07-31
- 验收状态: 以 `code/scripts/verify_all.py` 和分层 runner 的当前输出为准

### 0.2 7 大板块
- Python 编程基础 (Ch01-06)
- 数据科学与算法 (Ch07-08)
- Web 开发与工程 (Ch09)
- 机器学习与深度学习 (Ch10-11)
- 大模型核心技术 (Ch12-16)
- 大模型工程实践 (Ch17-29)
- 前沿专题与岗位实战 (Ch30-40)

### 0.3 岗位学习路径
- 大模型算法工程师: Ch01-07 + Ch10-12 + Ch16 + Ch19 + Ch22
- 大模型应用开发工程师: Ch01-06 + Ch09 + Ch13-15 + Ch18 + Ch20
- 大模型推理部署工程师: Ch05 + Ch09 + Ch16 + Ch19 + Ch20 + Ch24 + Ch25
- 大模型 Agent 工程师 (2026): Ch12 + Ch14 + Ch15 + Ch18 + Ch25
- 推理平台工程师 (2026): Ch16 + Ch19 + Ch24 + Ch25
- 大模型评估工程师 (2026): Ch12 + Ch14 + Ch15 + Ch17 + Ch27

---

## 1. Python 编程基础 (Ch01-06)

### Ch01 Python 编程基础 (⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 1.1 Python 概览与新特性
- Python 之禅: 显式优于隐式
- Python 3.13 引入可选自由线程构建和新交互式 REPL
- Python 3.14: 自由线程进入正式支持阶段，仍非默认构建
- 虚拟环境: venv (纯 Python) / conda (数据科学) / poetry (生产)

#### 1.2 基础语法
- 数据类型: 不可变 (int/float/str/bool/tuple/frozenset/bytes) vs 可变 (list/dict/set/bytearray)
- 小整数复用: CPython 实现细节；不要依赖具体区间，也不要用 `is` 比较数值
- is vs ==: `is` 比地址 (id), `==` 比值 (`__eq__`); None 必须 `is None`
- 链式比较: `1 < x < 10` 等价于 `1 < x and x < 10`
- for-else: 循环未 break 才执行 else
- 海象运算符 `:=` (3.8+)
- 短路求值: `value or default`

#### 1.3 核心数据结构
- List: 动态数组; append O(1), pop(0) O(n)
- Dict: 哈希表 O(1); 3.6+ 保插入顺序; 键必须可哈希
- Set: 哈希集合, O(1) 成员判断; frozenset 不可变
- Tuple: 浅层不可变 (含可变对象时可改); 元组拆包; 单元素 `(x,)`
- 去重: `set()` (无序) / `dict.fromkeys()` (保序)
- 字典合并: 3.9+ `d1 | d2`

#### 1.4 函数与模块
- 传对象引用: 不可变不外溢, 可变外溢
- 默认参数陷阱: 函数定义时求值只创建一次; 用 `None` 哨兵避免
- `*args`/`**kwargs`: 位置→默认→*args→keyword-only→**kwargs
- Lambda + 高阶: map/filter/reduce; `sorted(key=lambda)`
- `lru_cache`: 函数结果缓存, 记忆化递归
- LEGB 规则: Local→Enclosing→Global→Built-in; `global` 改全局, `nonlocal` 改外层
- lambda 循环陷阱: 捕获自由变量; 用 `lambda x=i: x` 修复

#### 1.5 异常处理与 PEP 8
- 异常层级: BaseException→Exception→子类
- try-except-else-finally: else 无异常时执行; finally 必执行
- 上下文管理器: `__enter__`/`__exit__`; return True 吞掉异常
- PEP 8: 4 空格缩进, 79 字符行宽, 命名约定, import 分组

### Ch02 可变性与拷贝 (⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 2.1 可变 vs 不可变
- 对象三属性: identity (id) / type / value
- `type()` vs `isinstance()`: type 不含继承链, isinstance 含
- 小整数与字符串驻留: `sys.intern()` 强制驻留

#### 2.2 深拷贝 vs 浅拷贝
- 赋值 `=`: 仅绑定引用
- 浅拷贝: `lst[:]` / `list()` / `copy.copy()`; 只复制外层
- 深拷贝 `copy.deepcopy()`: 递归复制; `memo` 处理循环引用
- `[[0]*3]*3` 陷阱: 共享同一内层列表; 正确写法列表推导式
- 元组陷阱: `t[1].append(4)` 合法, `t[1] = [5,6]` 报错
- 自定义拷贝: `__copy__` + `__deepcopy__(memo)` 方法

#### 2.3 只读字典
- `MappingProxyType`: 字典的只读视图, 不复制数据

### Ch03 面向对象编程 (⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 3.1 类与对象
- 类属性 vs 实例属性: 可变类属性是陷阱
- 私有属性 Name Mangling: `__password` → `_ClassName__password`
- property 装饰器: getter/setter/deleter
- 类方法/静态方法: `@classmethod` 接 cls, `@staticmethod` 无 self/cls

#### 3.2 三大特性
- 封装: `_*` 约定私有, `@property` 暴露受控接口
- 继承: `super().__init__()`; 多继承需 MRO
- 多态: Python 鸭子类型
- ABC: `@abstractmethod` 强制接口; `Shape.register(Circle)` 虚拟子类

#### 3.3 多继承与 C3 线性化
- 钻石问题: D 继承 B,C 都继承 A; C3 保证 A 只调用一次
- C3 三原则: 子类优先, 声明顺序, 单调性
- MRO 查看: `Class.__mro__` / `Class.mro()`; `super()` 按 MRO 调用
- 协同调用: 所有 `__init__` 都需 `super()`, 否则 MRO 链断裂
- MRO 冲突: 抛 TypeError

#### 3.4 `__new__` vs `__init__`
- `__new__`: 实例创建前, cls 接收, 必须返回实例
- `__init__`: 实例创建后, self 接收, 不能返回值
- 单例模式: `__new__`+类属性+双重检查锁 / 装饰器 / 元类重写 `__call__`

#### 3.5 魔术方法
- 生命周期: `__new__`/`__init__`/`__del__`
- 字符串: `__repr__` (eval 可重建) / `__str__` (print) / `__format__`
- 比较: `__eq__`/`__lt__`/`__hash__` 必须一致
- 算术: `__add__`/`__radd__`/`__iadd__` (+=)
- 容器: `__len__`/`__getitem__`/`__iter__`/`__contains__`
- `__getattr__` vs `__getattribute__`: 前者仅属性不存在时, 后者所有访问
- `@functools.total_ordering`: 自动补全比较方法

#### 3.6 描述符协议
- 数据描述符: 实现 `__get__+__set__`; 优先级高于实例 `__dict__`
- 非数据描述符: 仅 `__get__`; 优先级低于实例字典
- 属性查找顺序: 数据描述符 → 实例字典 → 非数据描述符 → 类字典 → 父类 MRO
- `@property` 底层是数据描述符; `classmethod` 是非数据描述符

#### 3.7 元类
- 元类层级: type 是 type 的元类
- 动态创建类: `type(name, bases, namespace)`
- 自定义元类: 继承 type, 重写 `__new__`
- 应用: ORM 字段映射, API 框架自动注册, 单例模式

### Ch04 高级特性与函数式编程 (⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 4.1 闭包
- 三要素: 嵌套函数 + 引用外部变量 + 返回内部函数
- `__closure__`: 闭包变量存储在 cell 对象
- `nonlocal`: 声明修改外层非全局变量
- 循环闭包陷阱: `lambda: i` 共享 i; 用 `lambda x=i: x` 修复
- 应用: 函数工厂, 数据隐藏

#### 4.2 装饰器
- 本质: `decorator(func)` 接收函数返回新函数
- 无参数装饰器: 两层嵌套
- 带参数装饰器: 三层嵌套
- `@functools.wraps`: 保留 `__name__`/`__doc__` 元信息
- 多重装饰器: 从最靠近函数定义处开始执行 (自下而上)
- 类装饰器: `__init__` 接收 func, `__call__` 实现 wrapper
- 常见手写: 计时器, 重试, 统计调用, 登录认证

#### 4.3 生成器与迭代器
- 迭代器协议: `__iter__()` 返回自身 + `__next__()` 返回元素/抛 StopIteration
- 可迭代 vs 迭代器: 列表可多次遍历, 迭代器只能一次
- `yield`: 暂停函数执行并返回值
- `yield from`: 委托子生成器
- 生成器表达式: `(x for x in ...)` 惰性求值
- 大文件处理: `for line in f` 逐行 O(1) 内存

#### 4.4 上下文管理器
- 协议: `__enter__()` + `__exit__(exc_type, exc_val, exc_tb)`
- `@contextmanager`: yield 前=enter, yield 后=exit
- `__exit__` 返回 True: 吞掉异常不传播
- 多上下文: 3.10+ `with (a, b, c):`

#### 4.5 itertools 与 functools
- itertools: `islice`/`chain`/`groupby`/`product`/`combinations`/`cycle`/`zip_longest`
- functools: `lru_cache`/`reduce`/`partial`/`wraps`
- 三种风格: 命令式 vs 函数式 vs Pythonic (生成器表达式优先)

### Ch05 并发编程 (⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 5.1 GIL 全局解释器锁
- 定义: CPython 互斥锁, 同一时刻仅一个线程执行字节码
- 存在原因: 简化引用计数内存管理
- 释放时机: 时间片到期 (默认 5ms), IO 操作, `time.sleep`
- 影响: CPU 密集型多线程无效, IO 密集型有效
- 3.13+ 自由线程构建: biased RC 等机制；3.14 进入正式支持阶段

#### 5.2 进程/线程/协程对比
- 进程: 独立内存, 绕过 GIL 真并行, CPU 密集型
- 线程: 共享内存, GIL 限制, IO 密集低并发
- 协程: 用户态轻量级, 事件循环, 单线程数万+并发
- 切换开销: 进程 > 线程 > 协程
- 同步原语: Lock/RLock/Semaphore/Condition/Event

#### 5.3 asyncio 异步
- async/await: async def 定义协程; await 挂起让出控制权
- 事件循环: 单线程调度; Task 通过 await 挂起, IO 完成后回调
- create_task vs gather vs TaskGroup: 单独/批量 (3.11+ 结构化并发)
- 同步阻塞集成: `loop.run_in_executor()` / `asyncio.to_thread()` (3.9+) / ProcessPoolExecutor
- aiohttp: 高并发 HTTP 客户端 (100 请求 1s vs 同步 50s)
- 异步上下文管理器: `__aenter__`/`__aexit__` + `async with`
- 异步迭代器: `__aiter__`/`__anext__` + `async for`

#### 5.4 并发选型决策
- CPU 密集 → multiprocessing
- IO 高并发 → asyncio
- IO 低并发 → threading
- 混合 → asyncio + executor

### Ch06 内存管理与垃圾回收 (⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 6.1 内存分配器
- 三层架构: pymalloc (≤512B 小对象) → C malloc (>512B) → OS 虚拟内存
- pymalloc 结构: arena (256KB) → pool (4KB) → block (8/16/.../512B)
- PyObject 头部: ob_refcnt (8B) + ob_type (8B)
- 小整数复用: CPython 实现细节；具体区间不属于 Python 语言保证
- 列表存储: 指针数组

#### 6.2 引用计数
- 增减场景: 赋值/容器添加/参数传递 +1; del/重赋值/容器销毁 -1
- `sys.getrefcount()`: 调用本身 +1, 需减 1
- 优点: 即时回收, 简单高效
- 缺点: 无法处理循环引用, 线程安全开销

#### 6.3 标记-清除
- 作用: 解决容器对象循环引用
- 流程: 标记根可达 → 清除未标记
- `__del__`: Python 3.4+ 的纯 Python 循环通常可回收；调用时机/顺序、对象复活和 C 扩展边界仍需谨慎

#### 6.4 分代回收
- 弱代假说: 多数对象生命周期短
- 分代数量、阈值和 `collect(generation)` 语义随 CPython 小版本实现演进
- `gc.get_threshold()` / `gc.get_stats()`: 在目标解释器实测，不背固定默认值
- `gc.collect(generation)`: 按目标 CPython 版本文档选择 generation

#### 6.5 weakref 弱引用
- 不增加引用计数: `weakref.ref(obj)` 不阻止回收
- `WeakValueDictionary`: 值不阻止 GC; 常用于缓存
- 打破循环引用: 父子节点中子对父用弱引用

#### 6.6 内存优化
- `__slots__`: 可避免每实例自动创建 `__dict__`/`__weakref__`；收益受属性、继承和版本影响，需实测
- 生成器: 惰性求值
- `lru_cache(maxsize=)`: 限制缓存大小
- `tracemalloc`: 内存分配快照对比定位泄漏
- 常见泄漏/滞留: 长寿命根或全局容器持有、无界缓存、finalizer 对象复活、监听器/外部资源未释放

---

## 2. 数据科学与算法 (Ch07-08)

### Ch07 数据结构与算法 (⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 7.1 链表
- 反转链表: 迭代 prev/curr/next 三指针 O(n) O(1)
- 环形链表检测: 快慢指针
- 找环入口: 相遇后一个回 head
- 合并有序链表: 哨兵 dummy + 双指针
- LRU Cache: 哈希表 + 双向链表; get/put O(1)

#### 7.2 栈与队列
- 栈: list 实现 LIFO
- 队列: `collections.deque` 实现 FIFO; `popleft()` O(1)
- 用栈实现队列: 入队栈 + 出队栈; pop 均摊 O(1)
- 最小栈: 辅助栈存当前最小值
- 单调栈: 下一个更大元素 O(n)

#### 7.3 哈希表
- 拉链法处理冲突; 质数取模 base=769
- 平均 O(1), 最坏 O(n)
- 应用: 两数之和, 字母异位词, Counter 字符频率

#### 7.4 树与图
- 二叉树遍历: 前序/中序/后序递归+迭代; 层序 BFS
- BST 验证: 中序遍历有序
- 平衡二叉树: 自底向上递归 + 剪枝
- BFS vs DFS: 队列求最短路径 vs 栈做回溯
- 图的表示: 邻接表 `defaultdict(list)`

#### 7.5 排序
- 快排: 平均 O(nlogn), 最坏 O(n²), 不稳定
- 归并: 稳定 O(nlogn), O(n) 空间; 适合链表/需要稳定
- Python `sort()`: TimSort (归并+插入混合)

#### 7.6 二分查找
- 模板: `mid = left + (right - left) // 2` 防溢出
- 变体: 第一个/最后一个等于 target; 旋转排序数组最小值

#### 7.7 双指针与滑动窗口
- 双指针: 盛最多水 (移动短线), 三数之和 (排序+剪枝+去重)
- 滑动窗口: 子串/子数组 O(n); 无重复字符最长子串, 最小覆盖子串

#### 7.8 动态规划
- 五步框架: 状态定义, 状态转移, 初始化, 遍历顺序, 返回结果
- 经典题: 爬楼梯, 零钱兑换, LIS, LCS, 0/1 背包, 编辑距离

### Ch08 数据科学核心库 (⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 8.1 NumPy
- ndarray 核心: 同质连续内存; 支持 SIMD 向量化
- 属性: ndim/shape/size/dtype/itemsize/nbytes/strides
- 广播机制: 从右往左匹配; 维度相等, 为 1, 或缺失才可广播
- 矩阵运算: `@` 矩阵乘法; `np.linalg` 解方程组/SVD

#### 8.2 Pandas
- 核心结构: Series/DataFrame/Index
- loc vs iloc: 标签索引 (含结束) vs 整数位置 (不含结束)
- groupby 三剑客: agg 聚合 / transform 同形 / filter 整组
- apply vs map vs applymap: map Series 逐元素; apply 按行/列; applymap 逐元素
- 缺失值: isnull/dropna/fillna/interpolate
- merge vs join vs concat: 按列 / 索引 / 沿轴拼接
- 大数据集优化: int64→int32/float64→float32/object→category; chunksize 分块
- 数据清洗: 缺失值→异常值→重复值→类型转换→类别编码→标准化

#### 8.3 数据可视化
- Matplotlib: `plt.subplots()` + plot/scatter/bar/hist
- Seaborn: boxplot (5 数概括), heatmap (相关矩阵), histplot
- 箱线图异常值: Q1-1.5×IQR 或 Q3+1.5×IQR

---

## 3. Web 开发与工程 (Ch09)

### Ch09 Web 开发与 FastAPI (⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 9.1 框架对比
- Django: 全栈, 同步, 电池全包
- Flask: 微框架, 同步, 灵活
- FastAPI: 现代 API 优先, 异步, 类型驱动, OpenAPI 原生

#### 9.2 FastAPI 核心技术
- 技术栈: Starlette (ASGI) + Pydantic v2 + Uvicorn (uvloop)
- Pydantic v2: Rust 重写, 速度 5-50x; `model_dump()`; `@field_validator`/`@model_validator`
- 依赖注入: `Depends()`; 支持 yield 清理; 可嵌套
- 自动文档: Swagger UI (/docs) / ReDoc (/redoc) / OpenAPI JSON

#### 9.3 路由与请求处理
- 数据模型: `class Model(BaseModel): field: type = Field(...)`
- 路径参数: `Path(pattern=...)` 正则校验
- 查询参数: `Query(min_length=, ge=, le=)`
- 响应模型: `response_model=ResponseModel`
- 异步数据库: `create_async_engine` + `AsyncSession`

#### 9.4 中间件
- CORS: `CORSMiddleware`
- GZip: `GZipMiddleware` 压缩大响应
- 自定义: `@app.middleware("http")` 日志/计时

#### 9.5 流式响应 SSE
- LLM 场景核心: `StreamingResponse(generator, media_type="text/event-stream")`
- SSE vs WebSocket: 单向/基于 HTTP vs 全双工/需协议升级
- 数据格式: `data: {json}\n\n`

#### 9.6 Pydantic v1 vs v2
- API 变化: `dict()`→`model_dump()`; `@validator`→`@field_validator`
- 类型注解: v2 更严格, 需显式 `Optional`

---

## 4. 机器学习与深度学习 (Ch10-11)

### Ch10 机器学习基础 (⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 10.1 三大范式
- 监督学习: 标注数据 (X, y)
- 无监督学习: 无标注数据; 聚类/降维
- 强化学习: 环境交互; 最大化累积奖励

#### 10.2 偏差-方差
- 公式: `E[(y-f̂)²] = Bias² + Variance + σ²`
- 过拟合: 低偏差高方差; 缓解 L1/L2/Dropout/早停/数据增强/集成
- 欠拟合: 高偏差低方差; 增加复杂度/特征

#### 10.3 经典算法
- 线性回归: `y = wᵀx + b`; MSE 损失; 解析解; L1/L2 正则 (Ridge/Lasso)
- 逻辑回归: Sigmoid 映射概率; 交叉熵损失; 决策边界 `wᵀx=0`
- 决策树: 信息熵/信息增益 (ID3) / 基尼系数 (CART)
- 随机森林: Bagging + 特征随机 (√m); 降低方差
- SVM: 最大间隔; 硬/软间隔; 核函数 (线性/多项式/RBF/Sigmoid)
- 朴素贝叶斯: 贝叶斯定理 + 条件独立; 高斯/多项式/伯努利

#### 10.4 集成学习
- Bagging vs Boosting: 并行降方差 vs 串行降偏差
- XGBoost 核心: 二阶泰勒展开 + 正则化项 `Ω(f) = γT + ½λΣw²` + 列采样 + 缺失值自动处理
- XGBoost vs LightGBM: Level-wise+预排序 vs Leaf-wise+直方图; LightGBM 快 2-10 倍

#### 10.5 无监督学习
- K-Means: 最小化簇内平方和; K-Means++; 肘部法则/轮廓系数
- PCA: 协方差矩阵特征值分解; 保留最大方差; `n_components=0.95`

#### 10.6 模型评估
- 指标: Accuracy/Precision/Recall/F1
- ROC-AUC: FPR vs TPR 曲线; AUC 统计意义
- 交叉验证: K-Fold / StratifiedKFold (分类首选)
- 超参数调优: Grid/Random/Bayesian

#### 10.7 Sklearn 实战
- Pipeline: 防止数据泄漏; `StandardScaler + Classifier`

### Ch11 深度学习与 PyTorch (⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 11.1 神经网络基础
- 感知机 → MLP → DNN
- 激活函数: Sigmoid/Tanh/ReLU/Leaky ReLU/GELU (Transformer 标配)

#### 11.2 反向传播
- 核心: 链式法则
- 前向: `z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾+b⁽ˡ⁾`; `a⁽ˡ⁾ = g(z⁽ˡ⁾)`
- 反向: `δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ g'(z⁽ˡ⁾)`; `∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ`

#### 11.3 PyTorch 核心
- Tensor: 创建/运算/维度操作; 与 NumPy 共享内存
- Autograd: `requires_grad=True` + `.backward()` + `.grad`; `no_grad()`/`inference_mode()`
- nn.Module: 自定义模型继承; `__init__`/`forward()`/`parameters()`
- 权重初始化: He (ReLU) / Xavier (Tanh)
- DataLoader: `Dataset` + `DataLoader(batch_size, shuffle)`

#### 11.4 CNN
- 卷积操作: `Output[i,j] = ΣΣInput[i+m,j+n]·Kernel[m,n]`
- 输出尺寸: `H_out = ⌊(H+2P-K)/S⌋ + 1`
- 架构演进: LeNet(5)→AlexNet(8)→VGG(16)→ResNet(152+, 残差)

#### 11.5 RNN/LSTM
- RNN 公式: `h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b)`
- 梯度问题: 隐藏状态连乘导致指数级消失/爆炸
- LSTM 三门: 遗忘门 f_t / 输入门 i_t / 输出门 o_t
- LSTM vs GRU: 3 门/C+h vs 2 门/仅 h
- Attention 引入: RNN 串行不可并行; Transformer 解决

#### 11.6 训练技巧
- 权重初始化: Xavier/He/Orthogonal
- 学习率调度: Step/Cosine/ReduceLROnPlateau/Warmup+Cosine (Transformer 标配)
- 正则化: L2/Dropout/BatchNorm/LayerNorm (Transformer 标配)/Label Smoothing
- BatchNorm vs LayerNorm: 跨 batch vs 跨 feature; CNN vs Transformer
- 混合精度: FP16 (需 Loss Scaling) / BF16 (A100+ 推荐)
- 优化器: SGD+Momentum/Adam/AdamW (Transformer 标配)

---

## 5. 大模型核心技术 (Ch12-16)

### Ch12 Transformer 与大模型原理 (⭐⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 12.1 Self-Attention 机制
- 公式: `Attention(Q,K,V) = softmax(QK^T / √d_k) V`
- Q/K/V 来源: 同一输入经三个不同的 W_Q/W_K/W_V 矩阵
- 缩放因子 √d_k: 防止点积过大进入 softmax 饱和区
- 计算复杂度: O(n²d) — 长序列瓶颈
- Multi-Head Attention: 多组 Q/K/V 并行, 头数 h, 每头 d_k=d/h

#### 12.2 Transformer 架构
- Encoder-Decoder vs Decoder-only (GPT 路线)
- Positional Encoding: 绝对位置 (sin/cos) / 相对位置 (RoPE, ALiBi)
- Pre-LN vs Post-LN: 现代用 Pre-LN, 训练更稳
- Causal Mask: 上三角遮罩, 保证自回归
- 关键组件: RMSNorm/SwiGLU/RoPE (LLaMA 路线)

#### 12.3 GPT vs BERT
- GPT: Decoder-only, 单向, Causal LM
- BERT: Encoder-only, 双向, MLM
- 二者预训练目标不同 → 下游任务不同

#### 12.4 大模型对齐三阶段
- SFT (监督微调): 高质量 (prompt, response) 对
- RLHF: 训 RM (奖励模型) → PPO 优化
- DPO: 直接偏好优化, 无需 RM
- GRPO (DeepSeek): 去 Critic 化, 组内优势
- SFT/RLHF/DPO/GRPO 演进: 2017-2026 全谱

#### 12.5 MoE 架构
- 专家并行: 多专家网络, Top-K 路由
- 优点: 模型容量大, 激活参数少
- 挑战: 专家不均衡, 通信开销
- 代表: Mixtral 8x7B；DeepSeek-V3 的规模与激活参数以官方技术报告为准

#### 12.6 涌现能力与 Scaling Law
- 涌现: 某些离散指标会呈现突变外观；不等于所有能力存在统一物理阈值
- Chinchilla 类拟合: `L(N,D)=E+A/N^α+B/D^β`，指数是特定实验拟合值而非常数
- 2026 趋势: Test-Time Compute、工具调用与可验证 Agent 工作流

#### 12.7 2026 LLM 主流模型
- OpenAI: GPT-5.6 family；模型 ID、窗口与工具能力查官方模型目录
- Anthropic: Claude 4 Opus/Sonnet/Haiku 4.6 (200K-1M)
- Google: Gemini 2.5 Pro/Deep Think (1M-2M)
- DeepSeek: V3, R1 (开源)
- 阿里: Qwen3.6 / Qwen3-VL（以当前开放平台与权重页为准）
- Meta: Llama 3.3/4
- Mistral: Large 2

### Ch13 Prompt Engineering (⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 13.1 Few-shot
- 模板: `examples = [{"input": ..., "output": ...}]`
- 数量: 由上下文预算、任务覆盖与验证集结果决定，不存在通用最佳个数
- 多样性: 覆盖不同场景

#### 13.2 Chain-of-Thought (CoT)
- Zero-shot CoT: "Let's think step by step"
- Few-shot CoT: 推理示例
- Self-Consistency: 采样 K 个, 多数投票
- Tree of Thoughts (ToT): 树状搜索 + 评估

#### 13.3 ReAct 框架
- 交替 Reason + Act
- 格式: Thought → Action → Observation 循环
- 与 Function Calling 结合

#### 13.4 采样参数
- Temperature / Top-p / Top-k: 支持范围、默认值和交互因提供方/模型而异
- Temperature 设为 0 也不等于跨硬件、版本和服务请求绝对可复现
- Frequency/Presence Penalty: 先确认模型支持，再用目标任务评估重复率与质量

#### 13.5 Extended Thinking（按 Claude 当前模型能力）
- thinking 配置、预算约束与工具交错能力按所选 Claude 模型当前文档核验
- 不假设内部推理轨迹会完整返回；记录可见响应、工具事件、usage 与停止原因

#### 13.6 Prompt Caching
- Anthropic / OpenAI / Gemini: 自动或显式机制、TTL、支持模型和计价规则均以当前官方文档为准
- 验证: 同一稳定前缀重复请求，检查 usage 缓存字段与实际账单，不预设固定节省比例
- 最佳实践: 稳定 system prompt + few-shot 放前缀

#### 13.7 Prompt 注入防御
- 指令隔离: XML 标签 `<user_input>...</user_input>`
- 输入清洗: 检测越狱
- 输出过滤: 校验格式
- 权限最小化: 工具调用白名单

### Ch14 RAG 检索增强生成 (⭐⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 14.1 分块策略
- 固定长度: 简单但可能切断语义
- 按句子/段落: 保持语义完整
- Recursive: 递归分割
- Semantic: 嵌入相似度切分
- 滑动窗口: 重叠避免边界丢失
- 结构化: 按 Markdown/HTML 标签
- Chunk 大小/重叠: 由 tokenizer、语料结构、retriever、上下文预算和 Golden Dataset sweep 决定

#### 14.2 Embedding 模型
- 中文: BAAI/bge-small-zh (0.1GB), bge-large-zh-v1.5
- 英文: bge-large-en-v1.5, mteb
- 多模态: BGE-VL, ColPali
- 维度: 384/768/1024/1536/3072
- 评估: MTEB benchmark

#### 14.3 检索算法
- 稀疏: BM25, TF-IDF
- 稠密: 向量最近邻 (cosine/dot)
- 索引: HNSW (O(log N) 图), IVF (倒排), PQ (乘积量化)
- 混合搜索: 稀疏 + 稠密加权
- 过滤: metadata pre-filter / post-filter

#### 14.4 Re-ranking
- Bi-Encoder: 双塔, 独立编码, 快速粗排
- Cross-Encoder: 交叉编码, 精确精排
- 代表: bge-reranker-v2-m3, cohere-rerank
- 流程: Bi-Encoder Top-100 → Cross-Encoder Top-10

#### 14.5 高级 RAG 技术
- Self-RAG: 自检 + 检索
- Graph RAG: 知识图谱 + LLM 提取 (Microsoft)
- Agentic RAG: Agent 决策何时检索
- Multi-hop RAG: 多跳推理
- RAG-as-a-Tool: Agent 把 RAG 当工具
- 多模态 RAG: ColPali/ColQwen 处理图表

#### 14.6 RAG 评估
- RAGAS: Faithfulness / Answer Relevancy / Context Precision/Recall
- TruLens: 反馈函数
- TruEra: 偏见检测
- ARES: 自动化 RAG 评估

#### 14.7 RAG 优化清单
- 文档质量: 清洗, 去重, 标准化
- 分块: 大小, 重叠, 结构感知
- Embedding: 选对模型, 微调
- 检索: HNSW, 混合, 多路
- Re-rank: bge-reranker
- Prompt: 引用来源, 限制幻觉

### Ch15 Agent 智能体开发 (⭐⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 15.1 ReAct 框架
- Thought → Action → Observation 循环
- 与 Chain-of-Thought 区别: ReAct 有外部交互
- 实现: LangChain ReAct agent

#### 15.2 Function Calling
- OpenAI tool_calls API
- JSON Schema 描述参数
- 多工具并发
- 错误处理: 重试, 降级

#### 15.3 MCP 协议 (Model Context Protocol)
- 2024-11 Anthropic 开源
- 架构: Host (Claude/Cursor) + MCP Client + MCP Server
- 三大原语: Tools/Resources/Prompts
- 传输: stdio (本地) / HTTP+SSE (远程)
- 优势: 一次开发, 多端复用

#### 15.4 A2A 协议 (Agent-to-Agent)
- 2025 Google 开源
- Agent Card: 能力描述
- Task Lifecycle: submitted→working→completed
- 通信: JSON-RPC over HTTP

#### 15.5 Agent 记忆管理
- 短期: 对话历史, in-context
- 长期: 向量库 (user facts/preferences)
- 情景: 过去事件, 结构化
- 程序: 技能/工具使用模式

#### 15.6 多 Agent 协作
- CrewAI: 角色 (Agent) + 任务 (Task) + 流程 (Crew)
- AutoGen: ConversableAgent + GroupChat
- LangGraph: 图编排, 节点 (Agent/Tool) + 边 (条件路由)
- Agent Teams (Claude Code): 多个独立 Agent 协同

#### 15.7 Skills vs MCP vs FC vs A2A
- FC: 单次工具调用, 同步
- MCP: 协议, 跨平台, 工具 + 资源 + prompt
- Skills: Claude 私有, 高层能力组合
- A2A: Agent 互联, 异步任务委派
- 层次: FC (原子) < MCP (标准化) < Skills (产品化) < A2A (生态)

#### 15.8 Agent 安全五道防线
- 工具白名单
- 输入清洗 (Prompt Injection)
- 输出过滤 (敏感信息)
- 权限沙箱
- 审计日志

### Ch16 模型微调与推理优化 (⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 16.1 LoRA / QLoRA
- LoRA: 低秩分解 ΔW = BA, B∈R^(d×r), A∈R^(r×k), r≪d
- 优点: 可显著减少可训练参数，适配器可按实现选择合并；比例取决于目标层与 rank
- QLoRA: 4-bit 量化 + LoRA + NF4 + Double Quant
- 显存: 受基座、序列长度、batch、优化器、激活与 offload 影响，先用目标配置实测

#### 16.2 推理优化关键技术
- KV Cache: 缓存历史 K/V, 避免重算
- Flash Attention: O(N) 内存, IO 感知
- Paged Attention (vLLM): 借鉴 OS 虚拟内存, 块表管理
- Continuous Batching: 动态插入新请求
- Speculative Decoding: 小模型 draft, 大模型 verify
- 量化: FP32→FP16/BF16→FP8→INT8→INT4
- KV Cache 量化: FP8/INT4

#### 16.3 端云协同部署
- 云端主力: 按质量、吞吐、延迟、数据边界和可用算力选模型/集群
- 端侧: 按可用内存、上下文、功耗与后端实测选择量化模型
- 路由: 简单任务端侧, 复杂任务云端
- Secure Minions: 远程证明 + 加密传输 + 受信执行环境（研究原型）

#### 16.4 Test-Time Compute (与 Ch27 配合)
- CoT 扩展: 让模型多思考
- 采样 + 投票: K 次采样, 多数投票
- 树搜索: MCTS + PRM
- Budget Forcing: 强制思考预算
- Reasoning Effort: 先在当前模型目录核对是否支持、参数名与允许档位

#### 16.5 强化学习训练
- PPO: 经典, 需 Critic 网络
- DPO: 直接偏好, 无 RM
- GRPO (DeepSeek-R1): 去 Critic, 组内优势
- RLVR: 可验证奖励 (数学答案/代码测试)

#### 16.6 RLHF 核心算法对比
- PPO: 常见组件含 Policy / Value / Ref / RM；可共享、冻结或卸载
- DPO: 直接偏好目标；reference log-prob 可在线计算或预计算
- GRPO: 去 Critic, 组内归一化
- 训练成本: 取决于在线采样、模型副本、序列长度与基础设施，不能只按算法名固定排序

---

## 6. 大模型工程实践 (Ch17-24, 含 Ch25-29 2026 新)

### Ch17 大模型评估体系 (⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 17.1 传统指标
- BLEU: n-gram 精度 (机器翻译)
- ROUGE: n-gram 召回 (摘要)
- METEOR: 同义词 + 词干
- CIDEr: TF-IDF 加权 (图像描述)
- BERTScore: 上下文嵌入余弦
- Perplexity: 语言模型困惑度

#### 17.2 LLM-as-Judge
- 思路: LLM 评估 LLM 输出
- 代表: GPT-4 judge, Claude judge
- 评估维度: 准确性, 相关性, 流畅性, 安全性
- 局限性: 偏置, 自我偏好
- MT-Bench / Chatbot Arena: 人类 + LLM 双重评估

#### 17.3 RAG 评估
- RAGAS: Faithfulness / Answer Relevancy / Context Precision/Recall
- TruLens: 反馈函数
- ARES: 自动化

#### 17.4 Agent 评估
- Trajectory Eval: 路径质量
- Tool-use Eval: 工具选择与参数
- Task Success Rate: 最终任务完成度

#### 17.5 红队测试
- 目的: 主动发现 LLM 漏洞
- 方法: 越狱, 提示注入, 偏见诱导
- 工具: Garak, PyRIT, deepteam

#### 17.6 OpenTelemetry GenAI 语义约定
- 2025 OTel 标准
- span/attributes: `gen_ai.*` 命名空间
- 追踪: prompt, completion, tool calls, latency, cost

### Ch18 LLM 工程框架实战 (⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 18.1 LangChain / LCEL
- Chain: 顺序组合
- LCEL (LangChain Expression Language): `prompt | model | parser`
- Runnable: 统一接口 (invoke/batch/stream)
- 工具: Tools, Retrievers, Memory
- Agent: ReAct, OpenAI Tools

#### 18.2 LangGraph
- 图编排: 节点 (Agent/Tool) + 边 (条件路由)
- State: TypedDict, 跨节点传递
- Checkpointer: MemorySaver (持久化)
- Human-in-the-loop: interrupt_before
- 多 Agent 协作: 子图

#### 18.3 LlamaIndex
- RAG 框架: 文档加载 → 切分 → 索引 → 查询
- 索引: VectorStoreIndex, SummaryIndex, KnowledgeGraphIndex
- 引擎: RetrieverQueryEngine, SubQuestionQueryEngine
- 优势: RAG 场景最专业

#### 18.4 LLaMA-Factory
- 一站式微调: LoRA, QLoRA, 全量
- 模型支持: 100+ 开源 LLM
- 训练器: SFT, DPO, PPO, KTO, RM
- WebUI + CLI

#### 18.5 Dify
- LLMOps 平台: 可视化构建 AI 应用
- 工作流: 拖拽式编排
- RAG: 内置, 易用
- 部署: 自托管, 云服务

#### 18.6 AutoGen / CrewAI
- AutoGen: ConversableAgent + GroupChat, Microsoft
- CrewAI: 角色 + 任务 + 流程, Crew 编排
- Magentic-One: 4-Agent 系统 (Orchestrator/WebSurfer/Coder/ComputerTerminal)
- AG2: AutoGen 继任

#### 18.7 选型决策树
- 快速原型: LangChain + LCEL
- 复杂 Agent: LangGraph
- RAG 优先: LlamaIndex
- 微调: LLaMA-Factory
- 零代码: Dify
- 多 Agent: CrewAI/AutoGen
- 2026 趋势: Pydantic AI (类型安全), Strands, OpenAI Agents SDK

### Ch19 分布式训练系统 (⭐⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 19.1 数据并行
- DP (Data Parallel): 单进程多线程, 复制模型
- DDP (DistributedDataParallel): 多进程, NCCL 通信
- DDP vs DP: 避免 GIL, 通信效率高
- 启动: `torchrun --nproc_per_node=N script.py`

#### 19.2 模型并行
- 张量并行 (TP): 切分权重到多卡, AllReduce
- 流水线并行 (PP): 切分层到多卡, 微批次
- 专家并行 (EP): MoE 专用, 路由 + All-to-All
- 3D 并行: TP × PP × DP

#### 19.3 内存优化
- ZeRO-1: 优化器状态分片
- ZeRO-2: + 梯度分片
- ZeRO-3: + 参数分片
- FSDP: PyTorch 原生分片数据并行

#### 19.4 DeepSpeed
- 集成 ZeRO + 各种优化
- 启动: `deepspeed --num_gpus=N script.py`
- 配置文件: `ds_config.json`

#### 19.5 Megatron-LM
- NVIDIA 张量并行 + 流水线并行
- Transformer Engine: FP8 训练
- 适用: 100B+ 模型

#### 19.6 NCCL 通信
- NVIDIA 集合通信库
- AllReduce / AllGather / ReduceScatter / Broadcast
- NVLink (机内) / IB (机间) 拓扑

#### 19.7 2026 训练前沿
- Muon Optimizer: 新优化器, 收敛快 2x
- GaLore: 低秩梯度投影, 内存节省
- Schedule-Free AdamW: 无需学习率调度
- TokenSpeed: Token 级并行
- TLX Block Attention: 块注意力加速

### Ch20 LLMOps 与模型可观测性 (⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 20.1 实验追踪
- MLflow: 开源, 模型注册
- W&B: 商业, UI 优秀
- TensorBoard: 经典
- 追踪: 参数, 指标, artifact, 代码版本

#### 20.2 LLM 专用追踪
- LangSmith: LangChain 官方
- Langfuse v3: 开源, OpenTelemetry 兼容
- Phoenix (Arize): 开源
- Helicone: API 代理 + 追踪
- Trace: prompt, completion, latency, cost, feedback

#### 20.3 Prompt 版本管理
- Git LFS 存 prompt + 评估
- 平台: Langfuse / PromptLayer / Helicone
- A/B 测试: 流量切分, 效果对比

#### 20.4 LLM 评估 Pipeline
- 离线: 测试集 + LLM-as-Judge
- 在线: 用户反馈, A/B
- CI: 每次 prompt 变更自动评估

#### 20.5 成本监控
- Token 消耗: 输入/输出分开计费
- 按 vendor/model 维度聚合
- 异常告警: 单价激增, 重试风暴

### Ch21 多模态大模型 (⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 21.1 CLIP
- 双塔: 图像编码器 + 文本编码器
- InfoNCE Loss: 对比学习
- 零样本分类: text prompt 引导

#### 21.2 ViT
- 图像切分为 patch (16x16)
- patch embedding + position embedding
- Transformer encoder

#### 21.3 LLaVA / 多模态 LLM
- 视觉编码器 (CLIP ViT) + Projector + LLM
- Projector: 视觉特征映射到 LLM 词空间
- 训练: 预训练 + SFT

#### 21.4 扩散模型
- DDPM: 加噪 → 去噪 UNet
- DiT (Diffusion Transformer): UNet 换 Transformer
- Stable Diffusion: 文本编码 + UNet + VAE
- 2026 实时: 4-step distillation (SD-Turbo)

#### 21.5 多模态 LoRA
- 只训练 Projector + LoRA 适配
- 显存友好, 适合消费级 GPU

#### 21.6 多模态 RAG
- ColPali: 视觉文档检索
- ColQwen: 多向量检索
- 图表理解, PDF 检索

### Ch22 大模型数据工程 (⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 22.1 数据清洗
- 去重: MinHash LSH
- 质量过滤: FastText 分类, perplexity 阈值
- PII 脱敏: 邮箱, 电话, 身份证
- 格式标准化: 统一编码, 标点

#### 22.2 合成数据
- Self-Instruct: 种子 → LLM 生成 → 过滤
- Evol-Instruct: 难度进化 (加深, 加约束, 具体化)
- WizardLM: Evol-Instruct 进化
- 风险: 模型坍缩, 偏见放大, IP 问题

#### 22.3 SFT 数据构建
- 高质量 (prompt, response) 对
- 多样性: 任务类型, 长度, 难度
- 配比: SFT 数据按任务/难度分层

#### 22.4 LIMA 现象
- 65 条高质量数据 SFT 超过百万条
- "Less is More" 数据质量 > 数量
- 启示: 严格过滤, 拒绝低质

#### 22.5 数据配比
- Domain mixing: 通用 + 代码 + 数学 + 多模态
- Chinchilla 比例: 20 tokens/param
- 增量训练: 通用 → 行业微调

#### 22.6 RLHF/RLAIF 数据
- 偏好对: (chosen, rejected)
- 标注: 人类反馈或 AI 反馈 (RLAIF)
- Agent 轨迹: 完整 (input, action, observation, reward)

### Ch23 AI 安全与伦理 (⭐⭐⭐, 面试 ⭐⭐⭐)

#### 23.1 Prompt Injection
- 直接注入: 覆盖系统提示
- 间接注入: 文档/网页藏恶意
- 防御: 输入清洗, 指令隔离, 沙箱

#### 23.2 越狱攻防
- 越狱方法: DAN, role-play, payload splitting
- 防御: Constitutional AI, 自我批评
- 评估: HarmBench, AdvBench

#### 23.3 OWASP LLM Top 10 (2025)
- LLM01: Prompt Injection
- LLM02: Sensitive Information Disclosure
- LLM03: Supply Chain
- LLM04: Data Poisoning
- LLM05: Improper Output Handling
- LLM06: Excessive Agency
- LLM07: System Prompt Leakage
- LLM08: Vector and Embedding Weaknesses
- LLM09: Misinformation
- LLM10: Unbounded Consumption

#### 23.4 EU AI Act
- 2024 通过, 2026 全面生效
- 风险分级: 不可接受 / 高 / 有限 / 最小
- 高风险 AI: 严格审计, 数据治理, 人工监督
- 通用 AI 模型 (GPAI): 透明度, 版权, 训练数据摘要

#### 23.5 红队测试
- 主动对抗: Garak, PyRIT, deepteam
- 持续集成: 每次模型/PR 跑红队
- 报告: 漏洞类型, 严重度, 修复建议

#### 23.6 Agentic Misalignment
- 风险: Agent 越权, 工具滥用, 目标偏离
- 防御: 工具白名单, 权限沙箱, 审计日志
- Anthropic Agentic Misalignment 研究 (2025)

### Ch24 云原生部署与工程化 (⭐⭐, 面试 ⭐⭐⭐)

#### 24.1 Docker
- 多阶段构建: builder + runtime, 减小镜像
- ARG vs ENV: 构建时 vs 运行时
- layer 缓存: requirements 单独 COPY

#### 24.2 Kubernetes
- Deployment/StatefulSet/DaemonSet
- Service/Ingress
- GPU 调度: nvidia.com/gpu resource
- Pod 拓扑: 亲和性, 反亲和性

#### 24.3 gRPC vs REST
- gRPC: 二进制, 高性能, 流式
- REST: JSON, 通用, 易调试
- LLM 服务: gRPC 内部 + REST 对外

#### 24.4 模型网关
- LiteLLM: 统一多厂商 API
- Kong/APISIX: 限流, 鉴权
- AI Gateway: 路由 + 缓存 + 监控

#### 24.5 GPU 调度
- K8s + NVIDIA Device Plugin
- MIG: A100 多实例 GPU
- Time-slicing: 共享 GPU
- vGPU: 虚拟化

#### 24.6 GitOps
- ArgoCD / Flux: Git 声明式部署
- 模型版本 + 镜像版本 一致
- 自动回滚

---

## 7. 2026 新增章节 (Ch25-29)

### Ch25 推理引擎与高性能服务 (⭐⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 25.1 推理核心原理
- 自回归两阶段: Prefill (计算密集, GEMM 主导) vs Decode (访存密集, KV Cache, HBM 带宽瓶颈)
- KV Cache 公式: `2 × n_layers × seq_len × n_kv_heads × head_dim × precision_bytes × batch_size`
- 三大指标: TTFT (Time To First Token) / TPOT (Time Per Output Token) / Throughput
- 2026 核心: 系统级调度 + 量化 + 异构

#### 25.2 五大推理引擎对比
- vLLM (PagedAttention): 借鉴 OS 虚拟内存分页、块表管理；显存利用率按模型与负载实测
- SGLang (RadixAttention): Radix Tree 自动识别共享前缀, 适用 system prompt 共享
- TensorRT-LLM: 提前编译, 硬件优化, In-flight batching, EAGLE-3/Medusa
- MLC-LLM: 跨平台编译 (Apache TVM)
- llama.cpp (GGUF): Q4_K_M 4-bit, 端侧事实标准

#### 25.3 关键优化技术
- Continuous Batching: 动态合批提高设备利用率；吞吐/尾延迟收益依工作负载而变
- PD-Disaggregation: Prefill/Decode 分离, NVLink/IB 传输
- Speculative Decoding: 小模型 draft + 大模型 verify, 数学证明保分布
- 量化阶梯: FP32→FP16/BF16→FP8→INT8→MXFP4/NVFP4→INT4
- MoE 专家并行: Top-K Gate + All-to-All
- 稀疏注意力: 1M+ 长上下文

#### 25.4 硬件可移植性
- NVIDIA H200/B200: vLLM/SGLang/TRT-LLM, FP4 + PD
- NVIDIA A100/H100: vLLM/SGLang, FP8
- AMD MI300X: vLLM (ROCm)
- Intel Gaudi 2/3: vLLM-Habana
- Apple Silicon: llama.cpp (MLX), Metal + ANE
- Ascend NPU 910B: vLLM-Ascend (CANN)

#### 25.5 选型决策树
- CLOUD + NVIDIA 专用优化候选 → TensorRT-LLM（以目标模型/硬件 benchmark 决定）
- CLOUD + 前缀共享/结构化生成候选 → SGLang（按相同 SLO 与流量压测）
- CLOUD + 通用 OpenAI-compatible serving 候选 → vLLM
- EDGE_MAC → MLX/llama.cpp
- EDGE_NVIDIA → TRT-LLM RTX
- EDGE_CPU → llama.cpp Q4

### Ch26 世界模型与具身AI (⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 26.1 具身智能全景
- 四要素循环: 感知 → 决策 → 行动 → 世界模型 → 决策
- 五层分级: L0 感知 (DINO/SAM) → L1 行动 (RT-1/RT-2) → L2 VLA (Pi0/GR00T) → L3 世界模型 (Genie 3/Cosmos) → L4 通用具身

#### 26.2 VLA 模型
- 核心思想: 视觉、语言与机器人状态共同条件化动作；动作可为 token、连续 head、diffusion 或 flow matching
- π0/π0.5 (Physical Intelligence): flow-matching action expert 路线
- GR00T N1.7 (NVIDIA): 当前公开主线；Cosmos-Reason2-2B backbone + 3B base checkpoint
- SmolVLA (Hugging Face): 官方 450M base；资源需求按当前模型卡/任务实测

#### 26.3 LeRobot 框架
- Hugging Face 开源机器人学习生态之一，不宣称“事实标准”
- 硬件、完整 BOM 与价格以当前官方文档/地区报价为准
- 训练: LeRobotDataset v3 + ACT/Diffusion/SmolVLA/π0 等当前策略目录

#### 26.4 模仿学习算法
- BC (基础监督)
- ACT (Transformer + 时间集成, Aloha)
- Diffusion Policy (Stanford, 多模态)
- VQ-BeT (Toyota, 矢量量化)
- Multitask DiT (DeepMind)

#### 26.5 RL 在机器人
- HIL-SERL: 人类示范 + RL 微调 (Stanford)
- TDMPC: 时间差分 MPC
- QC-FQL: Q-加权 FQL 研究路线（结论受论文设置约束）

#### 26.6 Sim-to-Real
- Domain Randomization
- OpenX-Embodiment 跨机器人数据
- DAgger 策略自己 rollout

#### 26.7 世界模型
- Genie 3 (DeepMind), Veo 3 (Google), Cosmos (NVIDIA), World Labs Marble
- 智能体对物理世界的内部预测模型
- "想象" 动作后果再决策

### Ch27 推理模型与 Test-Time Compute (⭐⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐⭐)

#### 27.1 推理模型范式
- Scaling Law 演进: 2017-2023 训练时扩展 → 2024-2026 推理时扩展 → 2026+ 自适应推理
- 核心洞察: 在最终答复前分配额外推理计算；内部轨迹是否暴露以及实现机制因模型而异
- Reasoning vs Standard: 可能消耗更多输出 token、延迟与费用；须在同模型、同任务和同 SLA 下实测
- 2026 当前候选: GPT-5.6 Sol、DeepSeek-V4-Pro、Claude Fable 5、Gemini 3.6、Qwen3.6、Kimi K2.5（调用前核对官方目录）

#### 27.2 Reasoning Effort API
- OpenAI: 按当前模型目录与 API 指南核对推理控制字段及允许档位
- 不跨模型复制 `budget_tokens`；按所选 Claude 模型的当前 thinking 配置核验
- 对比方法: 固定数据集与并发，记录质量、输出 token、TTFT/总时延、错误率和实际费用

#### 27.3 Test-Time Compute Scaling
- 三大方法: CoT 扩展, 采样+投票, 树状搜索
- 研究边界: Snell 等人的结果来自特定模型与数学任务，不是通用准确率曲线
- s1/s1.1 (Stanford 2025): 1000 高质量长 CoT SFT 32B, budget forcing, Wait token
- 阶梯: Zero-shot → CoT → Self-Consistency → Best-of-N → MCTS+PRM → Budget Forcing → Reasoning Effort

#### 27.4 PRM (Process Reward Model)
- PRM vs ORM: 逐步标注 vs 最终答案
- 数据: PRM800K, Math-Shepherd, rStar-Math, OmegaPRM
- 训练: MCTS 收集 step-level 标注 → 训练 step→correct/incorrect 分类器
- MCTS+PRM (AlphaProof 风格): UCB 平衡探索利用

#### 27.5 推理模型训练
- SFT 阶段: 强推理模型 → 生成大量长 CoT → SFT 学生
- RL 阶段: GRPO / RLOO / RLVR
- GRPO (DeepSeek-R1): 去 Critic, 组内优势 `A_i = (r_i - mean) / std`
- RLVR: 2026 主流, 可验证奖励 (数学/代码)
- R1-Zero vs R1: 纯 RL 涌现 vs SFT 冷启动 + RL

#### 27.6 部署挑战
- 输出长度波动 → 设置预算、流式返回并监控截断率
- 延迟波动 → 异步执行、超时与按难度路由
- 成本波动 → 结合质量收益、usage 和当前计费做分级
- 可解释性差 → PRM 验证

### Ch28 端侧与边缘 LLM (⭐⭐⭐, 面试 ⭐⭐⭐)

#### 28.1 端侧 LLM 全景
- 核心驱动: 隐私, 低延迟, 成本, 离线
- 部署目标: 手机, PC, 嵌入式, 浏览器

#### 28.2 GGUF 量化
- 格式: llama.cpp 标准, 单文件部署
- 量化等级: Q2_K / Q4_K_M / Q5_K_M / Q6_K / Q8_0；文件大小取决于模型参数与量化元数据
- 端侧选型: 预留 KV cache 和运行时开销后，再按设备内存、上下文、并发、温控实测

#### 28.3 Apple MLX
- vs CoreML/PyTorch MPS: 统一内存 (✅) / 动态图 (✅) / Apple 优化 (⭐⭐⭐⭐⭐)
- MLX 核心: Apple Silicon 专为设计, 统一内存无数据复制
- 代码: `mlx_lm.load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")`

#### 28.4 llama.cpp 多平台
- 后端: Metal/CUDA/Vulkan/OpenCL/Hexagon/CANN/MUSA/CPU
- Ollama: 一键部署, Modelfile, OpenAI 兼容 HTTP API

#### 28.5 WebGPU / WASM 浏览器推理
- WebLLM / MLC-LLM: OpenAI 兼容 chat.completions API
- WebGPU: 浏览器/驱动支持与可运行模型规模持续变化，需在目标设备做兼容和内存测试
- WASM/CPU: 作为兼容路径；速度、线程与内存限制按浏览器安全策略实测

#### 28.6 Secure Minions
- Stanford Hazy Research 的研究原型
- 远程证明 + 加密传输 + confidential CPU/GPU TEE
- 明文在受证明的 enclave 内解密推理；不等同于“云端永远看不到明文”
- 尚未经完整第三方安全审计，不应描述为成熟生产协议

#### 28.7 端云协同模式
- 纯端侧 / 云端主力 / 路由分流 / Secure Minions / 端侧缓存

### Ch29 Context Engineering (⭐⭐⭐⭐, 面试 ⭐⭐⭐⭐)

#### 29.1 从 Prompt 到 Context
- Context = Prompt + History + Tools + RAG + Memory + State
- Anthropic: "Context is the new code"
- Prompt Engineering → Context Engineering: 管每步推理时所见所有信息

#### 29.2 Context 四大组成
- Instructions: System prompt, Few-shot, Tool definitions, Output format
- Knowledge: RAG, 上传文档, DB 查询, Web 搜索
- Tools: MCP servers, Function schemas, 工具状态
- State: 对话历史, 长期记忆, 结构化 state, Sub-agent 结果

#### 29.3 上下文窗口经济学
- 公式: 总成本 = 输入tokens × 输入价 + 输出tokens × 输出价
- 上下文上限、价格与缓存规则按具体模型版本和官方页面核验
- Context Rot 是任务、位置、噪声和模型共同作用的现象，不使用固定百分比泛化
- 用长上下文基准和本业务 Golden Dataset 测量有效上下文，而非只看标称窗口

#### 29.4 压缩与裁剪
- Summarization: LLM 摘要
- Sliding Window: 保留最近 K 轮
- Compaction: 关键事实抽取
- LangGraph 持久化: MemorySaver checkpointer

#### 29.5 记忆系统
- 短期 → 长期 → 情景 → 程序 4 层
- 原则: 分层, 选择性, 可检索, 可更新, 隐私

#### 29.6 Sub-Agent 模式
- Main → Sub1 (搜索) / Sub2 (代码) / Sub3 (数据) → 聚合
- 优点: 干净 context, 并行, 隔离
- 缺点: 协调开销, 调试复杂
- 代表: Claude Code, Cursor Agent, Devin

#### 29.7 Prompt Caching
- 自动/显式缓存、TTL 与价格均随厂商和模型变化
- 只缓存稳定前缀；以官方计费页和实际 usage 字段验证命中

#### 29.8 Haystack 2.x Context-Engineered Pipelines
- vs LangChain/LlamaIndex: 组件化, production-ready
- 代码: `Pipeline() + add_component() + pipe.connect()`

---

### Ch30 高效序列架构 SSM 与 Mamba
- S4/Selective SSM/Mamba/Mamba-2 的状态空间视角
- SSD 以半可分矩阵与标量恒等 SSM 连接注意力和 SSM
- Jamba: Attention、Mamba 与 MoE FFN 的混合，而非“MoE Attention”

### Ch31 知识编辑与模型记忆
- ROME: 中层 FFN、subject 最后 token、协方差加权秩一更新
- MEMIT: 批量编辑；论文实验边界不外推到闭源 GPT-4 级模型
- TOFU 是机器遗忘 benchmark，不是已经解决遗忘问题的算法

### Ch32 DeepSeek 风格 MoE 与 MLA
- V3: 512 维 KV latent + 64 维 decoupled RoPE cache
- 1 shared + 256 routed experts，top-8 routed experts
- 路由 bias 由负载反馈更新；MTP 模块按顺序预测未来 token

### Ch33 训练稳定性与诊断
- AMP 溢出: GradScaler backoff，而非增大 loss scale
- AGC: 按参数单元比较梯度范数与参数范数
- 完整恢复: 模型/优化器/scaler/RNG/sampler/dataloader/step
- Muon: momentum orthogonalization + Newton-Schulz

### Ch34 Tokenizer 设计与词表工程
- SentencePiece 同时支持 BPE 与 Unigram，空格标记为 `▁`
- Llama 3/Qwen tokenizer 以各自官方 tokenizer 配置为准
- 迁移门禁: special tokens、chat template、归一化与压缩率回归

### Ch35 生产级 Agent 记忆框架
- 用户作用域与 metadata 分离
- 检索分数先归一化，再融合相关性、重要性和时间
- 写入冲突、删除、审计与隐私是生产门禁

### Ch36 JAX 与 TPU 大规模预训练
- jit/grad/vmap/sharding 的显式组合
- Pallas 支持 TPU 与 GPU 后端，具体能力以官方文档为准
- Pathways 不会自动把任意单设备程序无条件扩展到千卡

### Ch37 PD 分离推理架构与 KV 池化
- Prefill/Decode 独立扩缩容与 SLO
- 首个 decode 必须等待完整 prompt KV；可重叠的是 KV 传输与后续 prefill
- SGLang 使用官方 disaggregation server/router 接口

### Ch38 模型合并 MergeKit
- Linear/SLERP/Task Arithmetic/TIES/DARE
- TIES = Trim, Elect Sign, Merge
- 合并前验证 base、架构、tokenizer、chat template 与 license

### Ch39 Computer Use 与 GUI Agent 训练
- OSWorld/WebArena 环境与成功率评估
- ComputerRL/AutoGLM-OS-9B 的论文归属与公开结果
- 沙箱、权限、确认、审计和可恢复执行

### Ch40 国内大模型岗位面试实战
- 六张证据卡 + 90 秒项目陈述
- RAG 漏斗、Agent 故障语义、推理性能与系统设计
- 幂等键必须配合工具侧去重、唯一意图记录和状态查询

---

## 8. 配套代码 (code/) — 433 .py

### 8.1 目录结构
- code/README.md + Makefile + pyproject.toml
- code/requirements-{core,llm,gpu}.txt — 三层依赖
- code/shared/ — 跨章节工具 (8 文件: gpu_guard, env, llm_client, provider_registry, chatmodel_factory, vllm_compat, _error_helper, _mock_fallback)
- code/ch01-ch29/ — 29 章目录, 每章含 core/llm/gpu 三子目录
- code/scripts/ — verify_all, run_all_examples, download_models, test_integration, test_real_api_smoke
- code/tests/ — pytest smoke (test_pilots.py)
- code/docs/ — API_KEYS.md, MODELS.md, DEPLOY.md, MIGRATE_*.md
- code/tutorial/ — Windows junction → ..

### 8.2 三层依赖策略
- core (30秒, 50MB): pydantic/httpx/fastapi/typer/pytest — 任意电脑
- llm (+5min, 500MB): + openai/anthropic/langchain/haystack/pydantic-ai — 需 API Key
- gpu (+30min, 8GB): + torch/transformers/peft/vllm/mlx — 需 NVIDIA/Apple

### 8.3 shared/ 核心模块
- gpu_guard.py: require_cuda / require_nvidia_gpu / require_apple_silicon / require_ollama
- env.py: 12 厂商 API Key 自动加载 (dotenv 5 层搜索)
- provider_registry.py: 7 厂商注册表 (deepseek/kimi/siliconflow/MiniMax/openai/anthropic/mock), CN 优先
- llm_client.py: UnifiedClient 统一 OpenAI/Anthropic SDK 协议, 缺 Key 抛错 (不静默降级)
- chatmodel_factory.py: LangChain/LlamaIndex 工厂
- vllm_compat.py: 3 模式自动调度 (Docker 端点 / 真 vllm / 友好抛错)
- _error_helper.py: 统一 [ERROR]/[HELP] 格式

### 8.4 Makefile (51 个 .PHONY 目标, 5 模块)
- 安装 (4): install-core/llm/gpu/all
- 测试 (4): test/test-llm/test-llm-mock/test-gpu
- 同步验证 (5): verify/verify-xrefs/sync-links/llm-doctor/download-models
- CI 集成 (4): ci/ci-quick/ci-core/ci-llm
- Docker (7): docker-build/up/llm/gpu/bash/down/ci
- 一键部署 (3): setup-local/setup-local-quick/integration-test
- vLLM server (3): vllm-server-start/status/stop

### 8.5 文件头 Frontmatter 约定
- 字段: chapter / topic / section / difficulty / tier / deps / run / runtime
- 反向链接: `# See: ../tutorial/ChNN_*.md §X.Y`
- 跨章节引用 + 面试 hook

### 8.6 pytest Markers
- core: 158 个 core/.py
- llm: 199 个 llm/.py (mock 或 API)
- gpu: 76 个 gpu/.py (NVIDIA/Apple)
- slow: >10s
- `pytest -m "not gpu"` 跑 core+llm

### 8.7 7 厂商 LLM 接入
- DeepSeek: 端点与模型 ID 查官方 API 文档
- Kimi/Moonshot: 端点、模型 ID 和上下文窗口查官方 API 文档
- SiliconFlow: api.siliconflow.cn/v1；模型动态上下线，调用前查询 /v1/models
- MiniMax: api.minimaxi.com (无 s), MiniMax-M2.7
- OpenAI: api.openai.com/v1, gpt-5.6
- Anthropic: api.anthropic.com, claude-fable-5
- Mock: 离线, 无 API Key

---

## 9. CI/CD — 5 个 GitHub Actions Workflows

### 9.1 verify.yml (PR 必跑, ~10 min)
- 4 jobs: quick-check / core-suite / llm-suite / summary
- 默认 LLM_MOCK=1 (mock-first, 无需 API Key)
- 失败上传 artifact 诊断
- 触发: push / PR / 手动

### 9.2 integration-test.yml (真实组件集成, 60 min)
- services: redis:7-alpine (16379) + pgvector/pgvector:pg16 (15432)
- 4 项测试: Embedding (bge-small-zh) + Redis + pgvector + LLM (DeepSeek)
- 触发: push / PR / 手动 / 周日 04:00 UTC
- v1.0.11: --embedding 标志 (只下 1 个 ~0.1GB 模型, ~30s)
- v1.0.5: timeout 30→60 min

### 9.3 docker-build.yml (Docker 镜像, 30 min)
- 触发: push / tag v* / 手动
- 多阶段: builder + runtime
- v1.0.6: Dockerfile core-only install (跳过 5GB llm 依赖)
- v1.0.9: 移除 smoke test
- 镜像推送 ghcr.io/BeefWrap4/LLM-Knowledge-Base

### 9.4 ci-llm-doctor.yml (真实 API 健康, 15 min)
- 触发: 手动 / 周一 14:00
- 6 个 Secrets: DEEPSEEK/KIMI/SILICONFLOW/MINIMAX/OPENAI/ANTHROPIC_API_KEY
- LLM_MOCK=0 (真实 API)
- 缺 Key 时 ::warning:: 跳过

### 9.5 gpu-verify.yml (self-hosted, 30 min)
- 触发: 手动 only
- `runs-on: [self-hosted, gpu]`
- 需 NVIDIA GPU 标签的 self-hosted runner
- parallel=1 避免 GPU 争用

### 9.6 CI 演进历程 (v1.0.0 → v1.0.13)
- v1.0.0: 历史 “8-Wave” 重构标签（不能作为当前真实集成验收证据）
- v1.0.1: verify.yml mock-first 修复
- v1.0.2-v1.0.4: integration-test + docker-build 包修复
- v1.0.5: 删 Tsinghua + 60min timeout
- v1.0.6: Dockerfile core-only
- v1.0.7-v1.0.8: smoke test 各种尝试
- v1.0.9: 移除 smoke test, docker-build 6/6 green
- v1.0.10: integration timeout 30→60min
- v1.0.11: 仅下 bge-small-zh (1 模型)
- v1.0.12: real API smoke skip missing-key 文件
- v1.0.13: 移除 real-api-smoke job, 5/5 workflow 全绿

---

## 10. Docker 部署

### 10.1 Dockerfile (Multi-stage)
- 阶段 1 (builder): python:3.12-slim + 装 core + modelscope + huggingface_hub
- 阶段 2 (runtime): 精简镜像 + tini + user app + HEALTHCHECK
- ENV: PYTHONUNBUFFERED=1, LLM_PROVIDER=deepseek
- EXPOSE 8000 8888

### 10.2 docker-compose.yml (3 profile)
- core: 仅 app
- llm: app + redis:7-alpine (16379)
- gpu: app + redis + pgvector/pgvector:pg16 (15432, 需 NVIDIA)
- volumes: app-cache, redis-data, pg-data
- networks: llm-net

### 10.3 vLLM Docker escape hatch
- 解决 vLLM Windows 不支持 (无 _C wheel)
- 启 vLLM Docker → OpenAI 协议 → 7 个 ch25 例子零代码改动
- `make vllm-server-start` / status / stop
- `from openai import OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")`

### 10.4 .dockerignore (37 行)
- 排除: .git, .github, __pycache__, code/models/, .env*.local, *.png/jpg
- 例外: code/docs/**/*.md, code/models/README.md

---

## 11. 关键架构决策

### 11.1 历史 Wave 重构与当前验收边界
- Wave 是维护阶段标签，不代表当时或当前已完成真实 API/GPU/外部服务验收
- core: 本地确定性示例，由 runner 与 pytest 验收
- llm: 默认 `LLM_MOCK=1`；真实 API 只有显式 `LLM_MOCK=0`、指定 provider/Key 后逐项验收
- gpu: 默认 `--mock`；模型下载、训练、服务、浏览器和硬件路径必须在兼容主机上单独验收
- 当前文件数、PASS/SKIP/FAIL 与链接状态只引用 `99_库健康检查报告.md` 的本次精确结果

### 11.2 关键设计原则
- 三层依赖: 30s/core → +5min/llm → +30min/gpu
- mock-first CI: LLM_MOCK=1 唯一显式触发
- UnifiedClient 缺 Key 抛错 (不静默降级, 暴露配置错误)
- provider_registry 单一来源, 新厂商 +1 行
- vllm_compat 3 模式自动调度
- 多 profile Docker (core/llm/gpu 渐进)
- 国内源优先: PIP_INDEX_URL 清华, HF_ENDPOINT hf-mirror
- 0 mock 残留: grep 守卫

### 11.3 mock 残留检查
- `is_mock / USE_REAL_API / MockLLM() / fake_llm / FakeListChatModel`
- 在 `ch*/{core,llm,gpu}/*.py` 全 0 匹配

### 11.4 验收原则
- 历史评分只作记录，不代表当前工作区
- 以 Ruff、pytest、core/LLM mock runner、引用与统计门禁的当前输出为准

---

## 12. 面试高频考点速查

### 12.1 A-C
- `*args`/`**kwargs` (Ch04) ⭐⭐⭐⭐
- `__new__` vs `__init__` (Ch03) ⭐⭐⭐⭐⭐
- A2A 协议 (Ch15) ⭐⭐⭐⭐
- AdamW vs Adam (Ch11) ⭐⭐⭐⭐
- Agent 记忆管理 (Ch15) ⭐⭐⭐⭐
- Agent Teams (Ch15) ⭐⭐⭐⭐
- Agentic RAG (Ch14) ⭐⭐⭐⭐⭐
- asyncio / await (Ch05) ⭐⭐⭐⭐⭐
- Attention 机制 (Ch12) ⭐⭐⭐⭐⭐
- BatchNorm vs LayerNorm (Ch11/12) ⭐⭐⭐⭐⭐
- BERT vs GPT (Ch12) ⭐⭐⭐⭐⭐
- C3 线性化 (Ch03) ⭐⭐⭐⭐⭐
- Causal Mask (Ch12) ⭐⭐⭐⭐⭐
- CI/CD (LLM) (Ch20) ⭐⭐⭐⭐
- CLIP (Ch21) ⭐⭐⭐⭐
- Continuous Batching (Ch16) ⭐⭐⭐⭐⭐
- CoT (Ch13) ⭐⭐⭐⭐⭐

### 12.2 D-G
- DDP vs DP (Ch19) ⭐⭐⭐⭐
- Decoder-only (Ch12) ⭐⭐⭐⭐⭐
- DeepSpeed (Ch19) ⭐⭐⭐⭐⭐
- DPO (Ch12/16) ⭐⭐⭐⭐⭐
- Embedding 选型 (Ch14) ⭐⭐⭐⭐
- Flash Attention (Ch16) ⭐⭐⭐⭐⭐
- FSDP (Ch19) ⭐⭐⭐⭐
- Function Calling (Ch15) ⭐⭐⭐⭐⭐
- GIL (Ch05) ⭐⭐⭐⭐⭐
- GPT 架构 (Ch12) ⭐⭐⭐⭐⭐
- GRPO (Ch12/16) ⭐⭐⭐⭐⭐
- gRPC vs REST (Ch24) ⭐⭐⭐

### 12.3 H-M
- HNSW 索引 (Ch14) ⭐⭐⭐⭐⭐
- KV Cache (Ch16) ⭐⭐⭐⭐⭐
- LangChain vs LangGraph (Ch18) ⭐⭐⭐⭐⭐
- LangSmith/LangFuse (Ch20) ⭐⭐⭐⭐⭐
- LEGB (Ch01/04) ⭐⭐⭐⭐
- LIMA 现象 (Ch22) ⭐⭐⭐⭐
- LLM-as-Judge (Ch17) ⭐⭐⭐⭐⭐
- LlamaIndex (Ch18) ⭐⭐⭐⭐⭐
- LoRA/QLoRA (Ch16) ⭐⭐⭐⭐⭐
- LSTM/GRU (Ch11) ⭐⭐⭐⭐
- MCP 协议 (Ch15) ⭐⭐⭐⭐⭐
- MLOps vs LLMOps (Ch20) ⭐⭐⭐⭐
- MoE 架构 (Ch12) ⭐⭐⭐⭐
- Multi-Head Attention (Ch12) ⭐⭐⭐⭐⭐

### 12.4 N-S
- free-threaded Python 3.13/3.14 (Ch05) ⭐⭐⭐⭐
- NumPy 广播 (Ch08) ⭐⭐⭐⭐⭐
- Paged Attention (Ch16) ⭐⭐⭐⭐⭐
- Pandas loc vs iloc (Ch08) ⭐⭐⭐⭐⭐
- Positional Encoding (Ch12) ⭐⭐⭐⭐⭐
- Pre-LN vs Post-LN (Ch12) ⭐⭐⭐⭐⭐
- Prompt Injection 防御 (Ch13/23) ⭐⭐⭐⭐
- Pydantic v2 (Ch09) ⭐⭐⭐
- Q/K/V 机制 (Ch12) ⭐⭐⭐⭐⭐
- RAG 分块 (Ch14) ⭐⭐⭐⭐⭐
- RAG 评估 RAGAS (Ch17) ⭐⭐⭐⭐⭐
- ReAct (Ch13/15) ⭐⭐⭐⭐⭐
- Re-ranking (Ch14) ⭐⭐⭐⭐⭐
- RLHF (Ch12/16) ⭐⭐⭐⭐⭐
- Self-Attention 公式 (Ch12) ⭐⭐⭐⭐⭐
- SFT (Ch12) ⭐⭐⭐⭐⭐
- Speculative Decoding (Ch16) ⭐⭐⭐⭐
- SSE (Ch09) ⭐⭐⭐⭐
- `super()` (Ch03) ⭐⭐⭐⭐⭐

### 12.5 T-Z
- Temperature/Top-p (Ch13) ⭐⭐⭐⭐
- Test-Time Compute (Ch12/16) ⭐⭐⭐⭐⭐
- Transformer 架构 (Ch12) ⭐⭐⭐⭐⭐
- vLLM 部署 (Ch16) ⭐⭐⭐⭐⭐
- weakref (Ch06) ⭐⭐⭐
- XGBoost/LightGBM (Ch10) ⭐⭐⭐⭐
- ZeRO-1/2/3 (Ch19) ⭐⭐⭐⭐⭐
- 分代回收 GC (Ch06) ⭐⭐⭐⭐
- 可变/不可变 (Ch02) ⭐⭐⭐⭐⭐
- 多 Agent 协作 (Ch15) ⭐⭐⭐⭐
- 多模态 RAG (Ch14/21) ⭐⭐⭐⭐
- 引用计数 (Ch06) ⭐⭐⭐⭐
- 异步编程 (Ch05) ⭐⭐⭐⭐⭐
- 深拷贝/浅拷贝 (Ch02) ⭐⭐⭐⭐⭐
- 混合搜索 (Ch14) ⭐⭐⭐⭐⭐
- 混合精度 AMP (Ch11/19) ⭐⭐⭐⭐
- 涌现能力 (Ch12) ⭐⭐⭐⭐
- 生成器 yield (Ch04) ⭐⭐⭐⭐⭐
- 端云协同 (Ch16) ⭐⭐⭐⭐⭐
- 装饰器 (Ch04) ⭐⭐⭐⭐⭐
- 量子化 GPTQ/AWQ/GGUF (Ch16) ⭐⭐⭐
- 闭包 (Ch04) ⭐⭐⭐⭐

### 12.6 2026 趋势考点
- Reasoning Effort API (Ch27) ⭐⭐⭐⭐⭐
- GRPO 去 Critic (Ch16/27) ⭐⭐⭐⭐⭐
- PRM/MCTS (Ch27) ⭐⭐⭐⭐⭐
- MoE 专家并行 (Ch12/25) ⭐⭐⭐⭐⭐
- PD-Disaggregation (Ch25) ⭐⭐⭐⭐⭐
- Speculative Decoding (Ch16/25) ⭐⭐⭐⭐
- Paged Attention (Ch16/25) ⭐⭐⭐⭐⭐
- GGUF Q4_K_M (Ch28) ⭐⭐⭐⭐
- MLX Apple Silicon (Ch28) ⭐⭐⭐⭐
- WebLLM/WebGPU (Ch28) ⭐⭐⭐
- Secure Minions (Ch28) ⭐⭐⭐
- VLA 模型 (Ch26) ⭐⭐⭐⭐
- Diffusion Policy (Ch26) ⭐⭐⭐⭐
- HIL-SERL (Ch26) ⭐⭐⭐⭐
- 世界模型 Genie 3 (Ch26) ⭐⭐⭐
- Haystack 2.x (Ch29) ⭐⭐⭐⭐
- Sub-Agent 模式 (Ch29) ⭐⭐⭐⭐
- Prompt Caching (Ch13/29) ⭐⭐⭐⭐
- Context Rot (Ch29) ⭐⭐⭐⭐
- Test-Time Compute Scaling (Ch27) ⭐⭐⭐⭐⭐

---

## 13. 2026 新增 5 大主题

### 13.1 推理引擎 (Ch25) — 2026 最热
- vLLM/SGLang/TensorRT-LLM 选型
- PagedAttention / RadixAttention 核心
- PD-Disaggregation
- FP4/MXFP4 量化 (Blackwell 原生)
- MoE 推理优化

### 13.2 世界模型 (Ch26) — 具身 AI
- VLA (Vision-Language-Action)
- Pi0/GR00T/SmolVLA
- LeRobot 框架
- Diffusion Policy
- 5 级具身智能

### 13.3 推理模型 (Ch27) — Test-Time Compute
- 推理时计算 / R1 等公开研究范式
- Reasoning Effort 支持范围与档位以模型目录为准
- PRM/MCTS
- GRPO/RLVR
- s1 budget forcing

### 13.4 端侧 LLM (Ch28) — 边缘 AI
- Apple MLX 统一内存
- GGUF Q4_K_M 量化
- Ollama 一键部署
- WebGPU/WebLLM 浏览器
- Secure Minions 隐私

### 13.5 Context Engineering (Ch29) — Prompt 进化
- Context = Prompt + History + Tools + RAG + Memory + State
- Context Rot 现象
- 4 层记忆系统
- Sub-Agent 模式
- Prompt Caching 收益需按 TTL、命中率、usage 与当前计费验证

---

## 14. 使用建议

### 14.1 高效使用本库
- 系统学习: 按难度分级从入门到高级
- 面试突击: 速查索引定位薄弱考点
- 项目参考: 第 9/14/15/18 章的完整实战代码
- 查漏补缺: 面试前对照岗位路径检查

### 14.2 硬件 × 章节矩阵
- 任意笔记本: Ch1-11 + Ch13/14/15/17/18/20/22/29 (用 API Key)
- Apple Silicon: Ch28 MLX/Ollama 以统一内存余量和示例 guard 为准
- NVIDIA GPU: 小型教学 demo、推理服务和 VLA/世界模型的需求差异很大
- 执行前读取示例 metadata/guard，并按模型权重、KV cache、激活、batch 与并发核算显存

### 14.3 学习节奏
- 第 1 周: Ch1-4 Python 基础
- 第 2 周: Ch5-6 并发与内存
- 第 3 周: Ch7 LeetCode 刷题
- 第 4 周: Ch8-9 数据科学 + Web
- 第 5-6 周: Ch10-11 ML + DL
- 第 7-8 周: Ch12 Transformer 原理
- 第 9-10 周: Ch13-16 大模型技术栈
- 第 11-12 周: Ch17-29 工程实践
- 第 13+ 周: Ch30-40 前沿专题与岗位实战

### 14.4 API 与本地模型
- 按网络可达性、合规、质量、延迟和成本实测选择提供方
- 促销额度和模型窗口属于动态信息，不写入长期教程结论
- 密钥只放环境变量或未提交的本地 `.env`

---

*思维导图版本: 2026-07-31 | 覆盖 40 章 + 433 个代码示例 + CI workflow + Docker 部署*
*导入 XMind: 文件 → 导入 → Markdown → 选本文件*
*维护: 与教程章节同步更新*
