---
title: 'LangGraph 源码学习（00）：全局架构地图与一次 invoke 的完整链路'
description: '从 6 层架构、Monorepo 依赖倒置，到 Pregel BSP 三阶段执行模型与 Channel/Checkpoint 底层结构，拆解一次 invoke 的完整运行链路。'
publishDate: '2026-11-18 18:00:00'
tags:
  - AI Agent
  - LangGraph
  - Python
  - 源码学习
category: 源码学习
---

## 1. 引言：从“会调 API”到“看清运行时”

平时用 LangGraph 写 Agent，大多数时候停留在声明式 API 层：定义 `TypedDict` 状态、调用 `add_node` 注册函数、用 `add_edge` / `add_conditional_edges` 串联路由，最后配一个 `checkpointer` 跑通 ReAct 循环或人机交互（Human-in-the-loop）。

但在排查一些深水区问题时，仅靠 API 直觉很容易碰壁：

- 两个并行分支同时更新 State，为什么有的时候相安无事，有的时候直接抛出 `InvalidUpdateError`？
- 节点执行中途发生异常或触发 `interrupt()`，恢复执行时代码到底是从断点那一行继续，还是整个节点从头再跑一遍？
- 为什么图里明明没有死循环，却在某个条件边之后意外停机或漏触发下游节点？
- `langgraph-prebuilt` 里的 `create_react_agent` 需要构建 `StateGraph`，为什么它的 `pyproject.toml` 却不依赖 `langgraph` 核心包？

要搞清楚这些问题，必须掀开声明式语法糖，直接去看 LangGraph 的底层引擎——**Pregel 运行时**。

在正式逐行啃源码之前，这篇笔记先梳理出 LangGraph 的整体代码版图：从生态定位、6 层架构、Monorepo 包依赖设计，一直追踪到单次 `app.invoke()` 在内存中经历的 12 个原子步骤。

---

## 2. 生态定位与内部 6 层架构

### 2.1 为什么“和模型对话”与“让 Agent 跑起来”必须解耦？

早期 LangChain 试图在同一个框架内同时解决 Prompt/Model 编排与 Agent 循环控制，结果导致状态流转隐式化、循环难以中断、并发与持久化几乎不可控。

LangGraph 独立出来的核心工程判断非常简单：**模型调用（无状态/短状态转换）与 Agent 编排（有环图状态机运行时）是两个完全不同维度的问题。**

```
+-----------------------------------------------------------------------+
| 应用层 (Application)     Deep Agents / 业务多 Agent 流水线              |
+-----------------------------------------------------------------------+
                                   | 构建于其上
                                   v
+-----------------------------------------------------------------------+
| 编排运行时 (Runtime)     LangGraph (有状态有环图执行引擎 / BSP 调度)    |
+-----------------------------------------------------------------------+
            | 复用 Runnable 协议与消息类型                | 状态追踪与调试
            v                                           v
+-----------------------------------+   +-------------------------------+
| 模型与组件层 (Components)          |   | 可观测与托管层 (Observability) |
| LangChain Core (Messages / Tools) |   | LangSmith / LangGraph Studio  |
+-----------------------------------+   +-------------------------------+
```

LangGraph 只复用 `langchain-core` 的基础协议（`Runnable` 接口、`BaseMessage` 消息结构、`RunnableConfig` 上下文配置），而将全部精力放在**有状态图的并发调度、版本控制与快照持久化**上。

### 2.2 内部 6 层架构拆解

如果将 LangGraph 仓库的代码按职责分层，自顶向下可以清晰地切分为 6 层，遵循严格的**单向依赖原则（上层依赖下层，下层对上层无感知）**：

```
+=======================================================================+
| L6 用户 API 层       StateGraph / MessageGraph / @entrypoint & @task  |
|                      Prebuilt (create_react_agent, ToolNode)          |
+-----------------------------------------------------------------------+
| L5 图编译与校验层    StateNodeSpec -> CompiledStateGraph               |
|                      静态图验证 (validate) / trigger_to_nodes 索引构建 |
+-----------------------------------------------------------------------+
| L4 核心执行引擎层    Pregel / PregelLoop (Sync & Async)                |
|    (Runtime Heart)   BSP 三阶段: prepare_next_tasks -> Runner -> apply |
+-----------------------------------------------------------------------+
| L3 状态与持久化层    Channels (LastValue, BinaryOperatorAggregate...) |
|                      BaseCheckpointSaver / BaseStore / Serializer     |
+-----------------------------------------------------------------------+
| L2 流式传输与通信层  StreamModes (values/updates/messages/custom)     |
|                      Callback & Event Stream / RemoteGraph SDK        |
+-----------------------------------------------------------------------+
| L1 基础设施底座      Executor (ThreadPool / Asyncio) / RetryPolicy    |
|                      上下文传播 (RunnableConfig) / 内部控制常量        |
+=======================================================================+
```

翻看源码时最核心的一个发现是：**第 4 层（Pregel 执行引擎层）是整个系统的绝对心脏，且它完全不知道 `StateGraph` 的存在。**

在 `libs/langgraph/langgraph/graph/state.py` 中，调用 `builder.compile()` 生成的 `CompiledStateGraph`，本质上直接继承自 `Pregel`：

```python
# libs/langgraph/langgraph/graph/state.py

class CompiledStateGraph(Pregel[StateT, ContextT, InputT, OutputT]):
    """编译后的 StateGraph，本质上是一个预配置好 Channel 与 Node 触发关系的 Pregel 实例。"""

    builder: StateGraph[StateT, ContextT, InputT, OutputT]
    schema_to_mapper: dict[type[Any], Callable[[Any], Any]]
    ...
```

也就是说，`StateGraph`（第 6 层）与 `compile()`（第 5 层）所做的全部工作，仅仅是把人类易于理解的“节点 + 边”，翻译成 `Pregel` 引擎（第 4 层）能够识别的 **`PregelNode` + `BaseChannel` 订阅映射表**。

---

## 3. Monorepo 拆包与“依赖倒置”巧思

LangGraph 采用 Monorepo 组织（位于仓库 `libs/` 目录下），拆分为多个独立发版的 Python 包：

| 包名 | 源码路径 | 核心职责 | 关键依赖关系 |
| :--- | :--- | :--- | :--- |
| `langgraph-checkpoint` | `libs/checkpoint/` | 持久化抽象底座：`BaseCheckpointSaver`、`BaseStore`、`JsonPlusSerializer`、`InMemorySaver` | 仅依赖 `langchain-core`、`ormsgpack` |
| `langgraph-checkpoint-sqlite` | `libs/checkpoint-sqlite/` | 基于 SQLite 的本地持久化实现（`SqliteSaver` / `AsyncSqliteSaver`） | 依赖 `langgraph-checkpoint` |
| `langgraph-checkpoint-postgres` | `libs/checkpoint-postgres/` | 生产级 PostgreSQL 持久化实现（`PostgresSaver` / `PostgresStore`） | 依赖 `langgraph-checkpoint` |
| `langgraph-prebuilt` | `libs/prebuilt/` | 开箱即用高层组件：`create_react_agent`、`ToolNode`、`InjectedState` | 依赖 `langgraph-checkpoint`、`langchain-core` |
| `langgraph` | `libs/langgraph/` | 核心运行时：`StateGraph`、`Pregel` 引擎、`Channels`、Functional API | 依赖 `checkpoint`、`prebuilt`、`sdk` |
| `langgraph-sdk` / `cli` | `libs/sdk-py/`、`libs/cli/` | 远程 API Server 客户端 SDK 与本地调试/部署 CLI | 独立客户端 / CLI 包装 |

### 为什么 `langgraph-prebuilt` 不依赖 `langgraph` 核心包？

按常理，`prebuilt`（预构建高层组件，如 `create_react_agent`）位于架构最顶层，内部必须使用 `StateGraph` 来拼装图，理应是 `prebuilt -> langgraph`。但实际上，依赖箭头是反过来的：**`langgraph` 核心包依赖 `langgraph-prebuilt`**。

原因在于**消除循环依赖（Circular Dependency）与保持平滑演进**：

1. 历史与兼容性上，许多用户只安装 `pip install langgraph`，期望直接从 `langgraph.prebuilt` 导入 `ToolNode` 或 `create_react_agent`。因此 `langgraph` 必须在顶层重导出 `prebuilt`。
2. 如果 `langgraph` 依赖 `prebuilt`，同时 `prebuilt` 又在 `pyproject.toml` 中声明依赖 `langgraph`，两个独立发版的包就会陷入死锁的循环依赖。
3. **源码中的解法**：`langgraph-prebuilt` 在打包层只声明依赖底层的 `langgraph-checkpoint` 和 `langchain-core`；而在运行时实现（如 `libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py`）中，通过按需导入或依赖底层的 `checkpoint` 接口工作。两个模块共同锚定到底层抽象 `langgraph-checkpoint`，这是非常典型的**依赖倒置原则（DIP）**在 Python Monorepo 工程中的应用。

---

## 4. 核心执行引擎：Pregel BSP 三阶段模型

为什么 LangGraph 的核心引擎目录叫 `pregel/`？

2010 年 Google 发表了著名的大规模图计算论文 *"Pregel: A System for Large-Scale Graph Processing"*，其理论根基是 **BSP（Bulk Synchronous Parallel，整体同步并行）** 模型。LangGraph 将这套分布式图计算模型搬到了单机/协程级的 Agent 编排中。

### 4.1 三个核心原语的映射

在 LangGraph 的 `pregel` 引擎中，一切执行逻辑都被抽象为三个原语：

1. **`Channel`（状态通道）**：状态的唯一载体，也是节点之间通信的唯一媒介。在 `State` 中定义的每一个字段，在底层都是一个独立的 `Channel` 实例。节点之间**绝不直接传递返回值**，只通过读写共享黑板（Channel）通信。
2. **`PregelNode`（执行节点）**：每个节点声明自己 **订阅（triggers）** 哪些 Channel、**读取（channels）** 哪些 Channel，以及执行后将结果 **写入（writers）** 哪些 Channel。
3. **`Super-step`（超级步 / Tick）**：引擎推进的最小原子节拍。每一轮 Super-step 都包含严格的 **Plan → Execute → Update** 三个阶段，中间由 **同步屏障（Synchronization Barrier）** 隔开。

### 4.2 Super-step 三阶段与同步屏障

```
====================== Super-step N 开始 ======================

  [ 阶段 1: Plan (计划) ]  --> pregel/_algo.py: prepare_next_tasks()
   对比 channel_versions 与 versions_seen，找出有新数据的 Channel，
   通过 trigger_to_nodes 索引唤醒订阅节点，构造 PregelExecutableTask。
         |
         v
  [ 阶段 2: Execute (并发执行) ]  --> pregel/_runner.py: PregelRunner.tick()
   +---------------------+       +---------------------+
   | Task A (只读状态快照) |       | Task B (只读状态快照) |
   | 写入私有 task.writes |       | 写入私有 task.writes |
   +---------------------+       +---------------------+
         |                                 |
         +----------------+----------------+
                          |
        ==================v==================  <-- 同步屏障 (Barrier)
        所有并发节点必须全部执行完毕（快节点等慢节点）
        ==================|==================
                          v
  [ 阶段 3: Update (确定序合并) ] --> pregel/_algo.py: apply_writes()
   按确定的 task.path 顺序遍历所有 task.writes，调用 Channel.update()，
   递增被修改的 channel_versions，最后生成当前步的 Checkpoint 快照。

====================== Super-step N 结束 ======================
```

### 4.3 为什么并行节点绝对不会发生数据竞争（Race Condition）？

在多线程或异步协程并发跑多个 Agent 节点时，最头疼的就是共享字典的读写冲突。LangGraph 的解法**不是加锁（Mutex），而是“快照隔离 + 缓冲写入 + 确定序合并”**：

1. **读隔离（Input Snapshot）**：在 Plan 阶段生成 `PregelExecutableTask` 时，引擎会从 Channel 中拷贝一份当前时刻的只读快照传给节点。Execute 阶段无论其他并发节点怎么跑，当前节点读到的永远是本轮 Super-step 启动瞬间的冻结视图。
2. **写隔离（Buffered Writes）**：节点的返回值不会直接写入全局 `Channel`，而是先追加到该任务私有的 `task.writes`（一个 `deque` 缓冲区）中。
3. **确定序合并（Deterministic Apply）**：等所有并发节点越过同步屏障后，单线程运行的 `apply_writes()` 会将所有任务的写入按确定的任务路径顺序逐一合并进 Channel，并统一递增版本号。

这种设计的工程代价是**不能在同一个 Super-step 内实现流式数据依赖**（如果 B 依赖 A 的输出，必须拆成两个 Super-step），且存在**木桶效应**（同一 Super-step 内的快节点必须等待慢节点完成才能进入 Update 阶段）；但它换来了 Agent 系统最宝贵的两样东西：**100% 的状态并发安全**与**跨越任意机器的确定性重放（Deterministic Replay）**。

---

## 5. 追踪一次 `app.invoke()` 的 12 步完整链路

为了把上述概念串起来，以一个最简单的顺序图 `START -> node_a -> node_b -> END` 为例，假设 `State` 包含 `a_out: str` 和 `b_out: str` 两个字段，追踪调用 `app.invoke({"a_out": "init"}, config)` 时底层的 12 个执行步骤：

```
用户调用 app.invoke(input, config)
 │
 ├─ 1. [入口] CompiledStateGraph.invoke() 转调 Pregel.stream(stream_mode="values")
 ├─ 2. [恢复/初始化] SyncPregelLoop.__enter__()：
 │       从 checkpointer 加载历史快照（若无则创建空 Checkpoint，step = -1）
 ├─ 3. [写入输入] map_input() 将用户输入生成写入缓冲，调用 apply_writes() 写入通道，
 │       触发 __start__ / 输入通道版本递增（version = 1），保存 Input Checkpoint (step = -1)
 │
 ├─ === Super-step 0 ===
 │   ├─ 4. [Plan] loop.tick() -> prepare_next_tasks()：
 │   │       发现输入通道 version(1) > node_a 看到的 seen(0)，生成 Task(node_a)
 │   ├─ 5. [Execute] runner.tick()：
 │   │       执行 node_a(state_snapshot)，返回值写入 Task(node_a).writes 缓冲区，
 │   │       并立即调用 loop.put_writes() 将节点级结果落盘到 writes 表
 │   └─ 6. [Update] loop.after_tick() -> apply_writes()：
 │           合并 Task(node_a).writes 到 Channel[a_out] 和触发通道 branch:to:node_b，
 │           对应通道 version 递增为 2，更新 node_a 的 versions_seen，
 │           保存 Loop Checkpoint (step = 0)
 │
 ├─ === Super-step 1 ===
 │   ├─ 7. [Plan] loop.tick() -> prepare_next_tasks()：
 │   │       发现 branch:to:node_b 的 version > node_b 的 seen，生成 Task(node_b)
 │   ├─ 8. [Execute] runner.tick()：
 │   │       执行 node_b(state_snapshot)，输出写入缓冲区并调用 put_writes()
 │   └─ 9. [Update] loop.after_tick() -> apply_writes()：
 │           合并写入到 Channel[b_out]（version 递增），因为 node_b 连向 END，
 │           不再写入任何下游节点的触发通道，保存 Loop Checkpoint (step = 1)
 │
 ├─ === Super-step 2 (终止判定) ===
 │   ├─ 10. [Plan] loop.tick() -> prepare_next_tasks()：
 │   │        对比所有节点的 triggers 通道版本与 versions_seen，全部相等，返回空任务集 {}
 │   └─ 11. [退出循环] loop.tick() 返回 False，while 主循环正常终止
 │
 └─ 12. [输出提取] read_channels() 从最终 Output Channels 提取状态字典返回给调用方
```

从这 12 步链路中，可以提炼出两个极其重要的底层事实：

1. **没有真正的 `END` 节点**：`END`（常量值 `"__end__"`）在运行时并不对应任何可执行函数。图的终止条件不是“走到了 `END` 节点”，而是 **`prepare_next_tasks()` 发现没有任何节点的触发通道版本号大于该节点已记录的 `versions_seen`**。
2. **Checkpoint 的生成频率与边界**：除了最初的输入快照（`step = -1`），每完成一轮完整的 Super-step（在 `after_tick` 的末尾），引擎都会生成一个全新的 Checkpoint。

---

## 6. 状态与持久化的两大基石：Channel 与 Checkpoint

在深入后续具体模块之前，先对第 3 层的两个核心数据结构建立准确认知。

### 6.1 Channel：为什么不用普通 Python `dict` 存状态？

如果只用普通字典存 State，当两个并行节点同时返回 `{"messages": [msg1]}` 和 `{"messages": [msg2]}` 时，字典的 `.update()` 只会粗暴地用后者覆盖前者。

LangGraph 在 `libs/langgraph/langgraph/channels/` 下设计了多态的 `BaseChannel` 体系：

```python
# libs/langgraph/langgraph/channels/base.py (核心抽象简化)

class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    @abstractmethod
    def get(self) -> Value: ...

    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool: ...

    @abstractmethod
    def checkpoint(self) -> Checkpoint: ...

    @abstractmethod
    def from_checkpoint(self, checkpoint: Checkpoint) -> Self: ...
```

不同状态字段根据类型注解被编译为不同的 Channel 子类：

| Channel 子类 | 触发方式 / 映射规则 | `update(values)` 合并语义 | 典型场景 |
| :--- | :--- | :--- | :--- |
| `LastValue` | `count: int`（无 `Annotated` 注解） | 仅允许单次写入，新值覆盖旧值；同一 Super-step 内收到多个值直接抛 `InvalidUpdateError` | 普通标量状态字段 |
| `BinaryOperatorAggregate` | `messages: Annotated[list, add_messages]` | 使用用户指定的二元 `reducer` 函数，按确定序折叠合并同一批写入：`val = reducer(val, update)` | 消息列表追加、计数器累加 |
| `Topic` | 内部动态路由 / `Send` API 队列 | 发布-订阅列表，收集本轮所有写入，消费后在下一轮自动清空（或累加） | 动态 Map-Reduce 分支 |
| `EphemeralValue` | 节点触发通道（如 `branch:to:node_b`） | 仅存活一个 Super-step，消费后立刻清空，不跨步保留 | 边与条件边的内部信号传递 |

### 6.2 Checkpoint：为什么只存 `channel_values` 恢复不了执行？

如果在工程中尝试自己手写 Agent 状态持久化（比如直接把 `state` 字典 JSON 序列化存进数据库），会发现在面对**循环图**或**并行分支中断恢复**时很容易卡壳：恢复出了状态字典，但引擎怎么知道下一步该唤醒哪个节点？

来看 `libs/checkpoint/langgraph/checkpoint/base/__init__.py` 中 `Checkpoint` 的真实定义：

```python
# libs/checkpoint/langgraph/checkpoint/base/__init__.py

class Checkpoint(TypedDict):
    v: int
    id: str
    ts: str
    channel_values: dict[str, Any]
    channel_versions: ChannelVersions
    versions_seen: dict[str, ChannelVersions]
    pending_sends: list[Send]
```

除了业务状态 `channel_values`，LangGraph 额外持久化了三个决定执行流向的元数据：

1. **`channel_versions`（全局水位线）**：记录每个 Channel 当前递增到了第几版（例如 `{"messages": 4, "branch:to:tools": 3}`）。
2. **`versions_seen`（节点消费游标）**：记录每个节点**上一次被触发执行时**看到的各个 Channel 版本（例如 `{"agent": {"messages": 4}, "tools": {"branch:to:tools": 2}}`）。
3. **`pending_sends`（动态任务缓冲）**：记录通过 `Send` API 动态派发但尚未执行的 Map-Reduce 分支任务。

此外，在存储层通过 `checkpointer.get_tuple(config)` 取出时，返回的是外层包装对象 `CheckpointTuple`（包含 `config`、`checkpoint`、`metadata`、`parent_config` 以及记录同一步内已完成节点输出的 `pending_writes`）。

正是因为同时保存了**全局版本水位线（`channel_versions`）**、**节点消费游标（`versions_seen`）**和**半成品写入（`pending_writes`）**，LangGraph 才能在任意进程崩溃或人工 `interrupt` 之后，换一台机器精确推算出：*“哪些并行节点刚才已经跑完了不用重跑，哪些节点因为 `channel_versions > versions_seen` 需要立即唤醒”*。

## 7. 小结

把 LangGraph 的全局架构拆开来看，底层的核心设计脉络其实非常克制：
1. **职责边界清晰**：复用 `langchain-core` 的消息与 `Runnable` 协议，但将“有环图状态机运行时”完全独立出来，专注于状态流转、并发调度与快照持久化。
2. **编译期与运行期解耦**：顶层的 `StateGraph` 只是一层声明式构建器，`compile()` 将节点与边翻译为底层 `Pregel` 引擎的 `PregelNode` 与 `BaseChannel` 订阅网络。
3. **BSP 三阶段保证并发与确定性**：通过 **Plan（版本比对唤醒）→ Execute（快照只读 + 私有缓冲写入）→ Update（确定序合并 + 版本递增）** 的同步屏障循环，在不引入重型锁机制的前提下，消除了并行节点的数据竞争。
4. **版本水位线驱动恢复**：`Checkpoint` 不仅保存业务状态值（`channel_values`），更通过全局通道版本（`channel_versions`）、节点消费游标（`versions_seen`）与节点级落盘缓冲（`pending_writes`）的配合，支撑起任意 Super-step 边界的断点自愈与时间旅行。
