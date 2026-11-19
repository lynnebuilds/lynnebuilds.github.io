---
title: 'LangGraph 源码学习01：Pregel BSP 引擎与 PregelLoop 主循环拆解'
description: '深入 langgraph/pregel 核心目录，逐行拆解 Pregel.stream 主循环、prepare_next_tasks 节点触发算法与 apply_writes 确定序状态合并。'
publishDate: '2026-11-19 11:00:00'
tags:
  - AI Agent
  - LangGraph
  - Python
  - 源码学习
category: 源码学习
---

## 1. 引言：为什么 Agent 编排需要一个图计算引擎？

每当调用 `app.invoke()` 或 `app.stream()` 时，底层真正干活的是位于 `libs/langgraph/langgraph/pregel/` 目录下的 **Pregel 执行引擎**。

刚开始接触 LangGraph 时很容易产生一个疑问：*编排一个 Agent 不就是按顺序或分支调几个 Python 函数吗？为什么不直接用普通的 DAG（有向无环图）拓扑排序执行器，或者用异步 Actor 消息模型，非要搬出 Google 2010 年的 Pregel（BSP 整体同步并行）模型？*

如果对比一下另外两种模型在复杂 Agent 场景下的表现，就能看出它们各自的短板：

1. **普通 DAG 工作流执行器（如 Airflow / 静态流水线）**：核心假设是“无环（Acyclic）”，靠入度为 0 的拓扑排序推进。而 ReAct Agent 的“思考（LLM）→ 行动（Tool）→ 观察 → 再思考”天然是一个**有向有环图（Cyclic Graph）**，静态拓扑排序直接失效。
2. **Actor 模型（如 Akka / 异步消息组）**：虽然天然支持环路和完全异步并发，但 Actor 之间没有**全局一致的状态边界（Global Synchronization Point）**。当一个包含多个并发节点的 Agent 跑了一半需要暂停等待人工审批（Human-in-the-loop），或者进程崩溃需要断点续跑时，很难对正在乱序飞行的异步消息拍下一张“全局一致的快照”，也无法保证重放（Replay）时的执行顺序与崩溃前 100% 一致。

**BSP（Bulk Synchronous Parallel）模型恰好卡在“环路支持”、“并发安全”与“全局一致性快照”的交集点上。**

这篇笔记深入 `langgraph/pregel/` 目录，结合核心源码，拆解 `PregelLoop` 的主循环、`prepare_next_tasks()` 的节点唤醒算法，以及 `apply_writes()` 的确定序合并机制。

## 2. `langgraph/pregel/` 源码目录地图

在切入代码细节前，先梳理 `libs/langgraph/langgraph/pregel/` 下 9 个核心源文件的职责分工：

```
libs/langgraph/langgraph/pregel/
├── main.py          # Pregel 主类：定义 stream/invoke 顶层入口，管理 nodes 与 channels 规范
├── _loop.py         # BSP 状态机核心：PregelLoop 基类及 SyncPregelLoop / AsyncPregelLoop
├── _algo.py         # 核心纯算法：prepare_next_tasks (Plan阶段) 与 apply_writes (Update阶段)
├── _runner.py       # 并发执行器：PregelRunner，驱动 Execute 阶段的并发提交、重试与 commit
├── _checkpoint.py   # 快照辅助：create_checkpoint 与 channels_from_checkpoint
├── _read.py         # 通道读取包装：PregelNode 及其从 Channel 读取快照的 ChannelRead 逻辑
├── _write.py        # 通道写入包装：ChannelWrite，负责将节点返回值组装为待写入元组
├── _retry.py        # 节点级重试：run_with_retry / arun_with_retry
└── _io.py           # 输入输出映射：map_input 与 read_channels
```

读这套源码最清晰的抓手，就是顺着 **`main.py` → `_loop.py` → `_algo.py` → `_runner.py`** 这条主干调用链往下跟。

## 3. 主循环全景：`Pregel.stream()` 的三段式节拍

无论是调用 `app.invoke()` 还是 `app.stream()`，最终都会汇聚到 `pregel/main.py` 中的 `Pregel.stream()`（异步对应 `astream()`）。`invoke()` 本质上只是以 `stream_mode="values"` 消费 `stream()` 并取最后一个输出值。

将 `Pregel.stream()` 中处理监控和流式模式分发的样板代码剥离后，整个执行引擎的主循环浓缩为非常清晰的**三段式结构**：

```python
# libs/langgraph/langgraph/pregel/main.py (核心主循环精简)

def stream(
    self,
    input: Union[dict[str, Any], Any],
    config: Optional[RunnableConfig] = None,
    *,
    stream_mode: Optional[Union[StreamMode, list[StreamMode]]] = None,
    interrupt_before: Optional[Union[All, Sequence[str]]] = None,
    interrupt_after: Optional[Union[All, Sequence[str]]] = None,
    ...
) -> Iterator[Union[dict[str, Any], Any]]:
    # 初始化同步 BSP 循环上下文与并发执行器
    with SyncPregelLoop(
        input,
        stream=StreamProtocol(stream.put, stream_modes),
        config=config,
        store=store,
        checkpointer=checkpointer,
        nodes=self.nodes,
        specs=self.channels,
        trigger_to_nodes=self.trigger_to_nodes,
        ...
    ) as loop:
        runner = PregelRunner(
            submit=loop.submit,
            put_writes=loop.put_writes,
            node_finished=config.get(CONF, {}).get(CONFIG_KEY_NODE_FINISHED),
        )
        # === BSP Super-step 主循环 ===
        # 1. [Plan 阶段]：loop.tick() 准备本轮待执行任务，若无任务或触发中断则退出循环
        while loop.tick(input_keys=self.input_channels):
            # 2. [Execute 阶段]：runner.tick() 并发执行本轮所有就绪 Task，边完成边输出流信号
            for _ in runner.tick(
                loop.tasks.values(),
                timeout=self.step_timeout,
                retry_policy=self.retry_policy,
                get_waiter=get_waiter,
            ):
                yield from output()
            # 3. [Update 阶段]：loop.after_tick() 统一合并写入、递增 Channel 版本、落盘 Checkpoint
            loop.after_tick()
            yield from output()
```

可以用一张时序流转图来直观理解这三行核心调用的关系：

```
+-------------------------------------------------------------------------+
| while loop.tick():                      [ Plan 阶段 ]                   |
|   1. 首次进入时将用户输入写入 Input Channel，创建初始 Checkpoint             |
|   2. 调用 prepare_next_tasks() 扫描版本变化，生成本轮 loop.tasks 字典       |
|   3. 检查 interrupt_before；若 loop.tasks 为空，返回 False 终止整个图       |
+---------------------------------------------------------------------- --+
                                    |
                                    | 返回 True (存在待执行任务)
                                    v
+-------------------------------------------------------------------------+
| for _ in runner.tick(loop.tasks.values()):  [ Execute 阶段 ]            |
|   1. 并发提交所有 Task（每个 Task 持有冻结的 input 快照与私有 writes）        |
|   2. 单个 Task 完成后调用 commit() -> loop.put_writes() 写入缓冲/落表       |
|   3. 同步屏障 (Barrier)：等待本轮所有并发 Task 全部执行结束                   |
+------------------------------------------------------------------------+
                                    |
                                    | 所有 Task 执行完毕
                                    v
+-------------------------------------------------------------------------+
| loop.after_tick():                      [ Update 阶段 ]                 |
|   1. 调用 apply_writes() 按确定顺序合并所有 Task 的 writes 到 Channels      |
|   2. 被更新的 Channel 版本号 +1，更新各节点的 versions_seen 水位线           |
|   3. 调用 _put_checkpoint() 保存当前 Super-step 的完整状态快照              |
|   4. 检查 interrupt_after，随后进入下一轮 while loop.tick()                |
+-------------------------------------------------------------------------+
```

**简单概括主循环：`loop.tick()` 是“排班计划”，`runner.tick()` 是“并发干活”，`loop.after_tick()` 是“统一交账”。**

此外，`PregelLoop` 内部维护了 `self.step`（当前步数）和 `self.stop`（由 `config["recursion_limit"]` 决定，默认值为 `25`）。如果 `self.step >= self.stop` 时仍有未完成的任务，引擎会直接抛出 `GraphRecursionError`。注意：**`recursion_limit` 限制的是 Super-step 的轮数，而不是执行的节点总数**——如果同一轮 Super-step 内并行跑了 10 个节点，`step` 只加 1。

## 4. Plan 阶段源码深潜：`prepare_next_tasks()` 如何唤醒节点？

在状态图里写的是 `builder.add_edge("node_a", "node_b")`，但在运行时，`node_a` 执行完后**并不会直接调用** `node_b()`。那么 `node_b` 是怎么被唤醒的？

答案在 `pregel/_algo.py` 的 `prepare_next_tasks()` 函数中。

### 4.1 触发的核心判据：`channel_versions` vs `versions_seen`

编译阶段（`compile`）会把图的边关系编译成一个反查索引字典 `trigger_to_nodes: Mapping[str, Sequence[str]]`，记录“当某个 Channel 更新时，哪些节点应该被检查触发”。

到了运行时的 Plan 阶段，`prepare_next_tasks()` 会执行以下核心判断逻辑：

```python
# libs/langgraph/langgraph/pregel/_algo.py (核心触发逻辑拆解)

def _triggers(
    channels: Mapping[str, BaseChannel],
    versions: ChannelVersions,
    seen: Optional[ChannelVersions],
    null_version: V,
    proc: PregelNode,
) -> bool:
    """判断节点 proc 在当前 Super-step 是否应当被触发执行。"""
    if seen is None:
        # 节点从未执行过：只要它订阅的任意 trigger 通道版本号 > 初始空版本，即触发
        for chan in proc.triggers:
            if versions.get(chan, null_version) > null_version:
                return True
    else:
        # 节点之前执行过：只要它订阅的任意 trigger 通道当前版本号 > 该节点上次看到的 seen 版本，即触发
        for chan in proc.triggers:
            if versions.get(chan, null_version) > seen.get(
                chan, null_version
            ):
                return True
    return False
```

紧接着，在 `prepare_next_tasks()` 中，引擎利用 `trigger_to_nodes` 做了一层高效的**增量候选集过滤**，避免每次都全量扫描图中所有节点：

```python
# libs/langgraph/langgraph/pregel/_algo.py (prepare_next_tasks 核心片段)

def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: Sequence[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    *,
    for_execution: bool,
    trigger_to_nodes: Optional[Mapping[str, Sequence[str]]] = None,
    updated_channels: Optional[set[str]] = None,
    ...
) -> dict[str, PregelExecutableTask]:
    tasks: dict[str, PregelExecutableTask] = {}

    # 1. 优先处理 pending_sends（通过 Send API 动态派发的并行任务）
    for packet in checkpoint["pending_sends"]:
        ...

    # 2. 确定本轮需要检查哪些候选节点：
    # 如果知道上一轮哪些 Channel 更新了 (updated_channels)，直接通过 trigger_to_nodes 索引反查候选节点
    if updated_channels and trigger_to_nodes:
        triggered_nodes: set[str] = set()
        for channel in updated_channels:
            if node_ids := trigger_to_nodes.get(channel):
                triggered_nodes.update(node_ids)
        candidate_nodes = sorted(triggered_nodes)
    elif not checkpoint["channel_versions"]:
        candidate_nodes = ()
    else:
        candidate_nodes = processes.keys()

    # 3. 遍历候选节点，校验 _triggers 条件并构造 PregelExecutableTask
    for name in candidate_nodes:
        proc = processes[name]
        if _triggers(
            channels,
            checkpoint["channel_versions"],
            checkpoint["versions_seen"].get(name),
            null_version,
            proc,
        ):
            # 读取该节点所需的所有 Channel 当前值的快照副本 (Snapshot)
            val = _proc_input(proc, managed, channels, for_execution=for_execution)
            if val is SKIP:
                continue

            # 生成确定性的 task_id，并构造任务对象
            task = PregelExecutableTask(
                name=name,
                input=val,
                proc=proc.node,
                writes=deque(),  # 独立的私有写入缓冲区！
                config=patch_config(config, run_name=name, ...),
                triggers=tuple(proc.triggers),
                retry_policy=proc.retry_policy,
                cache_key=...,
                id=task_id,
                path=(PULL, name),
            )
            tasks[task_id] = task

    return tasks
```

### 4.2 `PregelExecutableTask`：读写隔离的物理载体

仔细观察上面第 3 步构造的 `PregelExecutableTask`（定义在 `libs/langgraph/langgraph/types.py` 中的 `NamedTuple`）：

```python
# libs/langgraph/langgraph/types.py

class PregelExecutableTask(NamedTuple):
    name: str
    input: Any                                   # 冻结的输入状态快照
    proc: Runnable                               # 包装后的节点可执行对象
    writes: deque[tuple[str, Any]]               # 线程私有的输出写入缓冲队列
    config: RunnableConfig                       # 节点级运行配置
    triggers: Sequence[str]                      # 触发本节点的 Channel 列表
    retry_policy: Sequence[RetryPolicy]          # 重试策略
    cache_key: Optional[CacheKey]                # 节点缓存 Key
    id: str                                      # 确定性任务唯一 ID (UUID5)
    path: tuple[Union[str, int, tuple], ...]     # 任务执行路径元组，用于确定序排序！
    scheduled: bool = False
```

这里藏着两个极其关键的工程细节：

1. **`input=val` 是深拷贝/独立构造的快照**：`_proc_input()` 从 `channels` 中读取当前值并组装成字典（或 Pydantic/DataClass 实例）赋值给 `task.input`。一旦进入 Execute 阶段，节点函数只跟 `task.input` 打交道，与外层的 `channels` 完全解耦。
2. **`writes=deque()` 是每个 Task 独享的双端队列**：节点跑完后，通过 `ChannelWrite` 组装出的 `(channel_name, value)` 元组只会 `extend` 到自己这个 `task.writes` 队列里。两个并行节点哪怕同时改同一个字段，也是各写各的 `deque`，因此在 Execute 阶段**连一把互斥锁（Lock）都不需要**。

## 5. Execute 阶段核心：`PregelRunner.commit()` 与缓冲落盘

当 `loop.tick()` 准备好 `loop.tasks` 后，`PregelRunner.tick()` 开始并发调度这些任务。

当某一个节点函数执行完毕返回结果时，`PregelRunner.commit()` 会立刻被调用：

```python
# libs/langgraph/langgraph/pregel/_runner.py (commit 核心逻辑精简)

def commit(
    self,
    task: PregelExecutableTask,
    exception: Optional[BaseException],
    writes: Optional[Sequence[tuple[str, Any]]] = None,
) -> None:
    if exception is not None:
        # 若节点抛出异常，向该 task 的 writes 写入 (ERROR, exception) 标记
        task.writes.append((ERROR, exception))
        self.put_writes(task.id, task.writes)
        return

    if writes is not None:
        task.writes.extend(writes)

    # 关键：节点一执行完，立刻调用 loop.put_writes 将本节点的 writes 持久化到 Checkpointer 的 writes 表
    self.put_writes(task.id, task.writes)
```

注意这里的**两层状态提交设计**：
- **第一层（Task 完成时立即触发）**：`commit()` 调用 `loop.put_writes(task.id, task.writes)`。它**不会修改内存里的 `Channel`**，但会立刻把该节点的输出写入持久化数据库（Checkpointer 的 `checkpoint_writes` 表）。如果同一 Super-step 里 A 节点成功了、B 节点因为网络超时崩溃了，下次恢复时，`prepare_next_tasks` 会直接把 A 在 `pending_writes` 里的结果挂载回 `task.writes`，跳过 A 的执行，只重跑 B。
- **第二层（所有 Task 完成后统一触发）**：所有并发任务越过同步屏障后，进入 `loop.after_tick()`，才真正调用 `apply_writes()` 修改内存 `Channel` 并生成新的 `Checkpoint`。

## 6. Update 阶段源码深潜：`apply_writes()` 如何保证 100% 确定序合并？

现在来到了最核心的问题：**并发执行的节点完成先后顺序是随机的（取决于网络 I/O 快慢或线程调度），LangGraph 怎么保证最终合并进 `Channel` 的状态顺序永远是 100% 确定的？**

直接看 `pregel/_algo.py` 中的 `apply_writes()` 源码：

```python
# libs/langgraph/langgraph/pregel/_algo.py (apply_writes 核心源码拆解)

def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: Optional[GetNextVersion],
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> set[str]:
    # === 第一步：按 task.path 确定序排序！ ===
    # 无论哪个节点先执行完、哪个后执行完，这里统一按照 task.path 升序排列
    tasks = sorted(tasks, key=attrgetter("path"))

    # === 第二步：更新每个节点的消费水位线 (versions_seen) ===
    # 记录本轮执行的每个节点已经“看过”了触发它的那些 Channel 的当前版本号，防止下一轮重复触发
    for task in tasks:
        checkpoint["versions_seen"].setdefault(task.name, {}).update(
            {
                chan: checkpoint["channel_versions"][chan]
                for chan in task.triggers
                if chan in checkpoint["channel_versions"]
            }
        )

    # 计算下一个递增版本号
    if get_next_version is None:
        next_version = None
    else:
        next_version = get_next_version(
            max(checkpoint["channel_versions"].values(), default=None),
            None,
        )

    # === 第三步：消费上一轮已触发的临时 Channel ===
    # 将本轮已被消费的 Channel 传入空更新 []，让 EphemeralValue / Topic 等通道有机会自我清空
    for chan in {
        chan
        for task in tasks
        for chan in task.triggers
        if chan not in RESERVED and chan in channels
    }:
        if channels[chan].consume() and next_version is not None:
            checkpoint["channel_versions"][chan] = next_version

    # === 第四步：按 Channel 聚合所有 Task 的写入（保持第一步排好的确定序） ===
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        for chan, val in task.writes:
            if chan in (NO_WRITES, PUSH, RESUME, INTERRUPT, RETURN, ERROR):
                pass
            elif chan == TASKS:
                checkpoint["pending_sends"].append(val)
            else:
                pending_writes_by_channel[chan].append(val)

    # === 第五步：批量调用 Channel.update() 合并状态，并递增版本号 ===
    updated_channels: set[str] = set()
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            # 调用具体 Channel 的 update(vals) 方法（如 LastValue 或 BinaryOperatorAggregate）
            if channels[chan].update(vals) and next_version is not None:
                checkpoint["channel_versions"][chan] = next_version
                # 只有当 Channel 真正更新时，才加入 updated_channels，用于下一轮 Plan 阶段唤醒下游
                updated_channels.add(chan)

    return updated_channels
```

读完这段不到 60 行的核心代码，整个 BSP 引擎的拼图就彻底清晰了：

### 6.1 确定性的来源：`tasks = sorted(tasks, key=attrgetter("path"))`

在第 1 步中，不管并发线程里是 `node_b` 先跑完还是 `node_a` 先跑完，`apply_writes` 都会强行按照 `task.path`（对于普通拉取节点，其值为 `('__pregel_pull', 'node_name')`）做字典序排序。

因此，当 `node_a` 和 `node_b` 在同一个 Super-step 中同时往 `messages` 通道追加消息时，`pending_writes_by_channel["messages"]` 收到的列表永远是 `[msg_from_a, msg_from_b]`，绝不会因为网络抖动变成 `[msg_from_b, msg_from_a]`。

### 6.2 并发冲突是怎么报错的？

在第 5 步中，`channels[chan].update(vals)` 接收的是一个列表 `vals`（包含了本轮所有节点对该 Channel 的写入）。

- 如果 `chan` 是带 `reducer` 的 `BinaryOperatorAggregate`（比如 `messages`），它会遍历 `vals` 依次调用 `reducer(current_val, v)` 合并。
- 如果 `chan` 是普通的 `LastValue`（比如 `count: int`，没有声明 `reducer`），看 `libs/langgraph/langgraph/channels/last_value.py` 的 `update` 实现：

```python
# libs/langgraph/langgraph/channels/last_value.py

class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            return False
        if len(values) != 1:
            msg = create_error_message(
                message=f"At key '{self.key}': Can receive only one value per step. Use an Annotated key to handle multiple values.",
                error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
            )
            raise InvalidUpdateError(msg)

        self.value = values[-1]
        return True
```

当两个并行节点在同一个 Super-step 同时修改一个没有声明 `Annotated[..., reducer]` 的普通状态字段时，`pending_writes_by_channel[chan]` 的长度 `len(values)` 就会等于 2。`LastValue.update()` 拒绝静默覆盖其中一个节点的数据，而是直接抛出 `InvalidUpdateError`，强制显式指定合并策略。

## 7. `after_tick()` 的收尾顺序为什么不能乱？

在 `apply_writes()` 返回了本轮被更新的通道集合 `updated_channels` 后，外层的 `PregelLoop.after_tick()`（位于 `pregel/_loop.py`）还要按严格顺序完成收尾工作：

```python
# libs/langgraph/langgraph/pregel/_loop.py (after_tick 逻辑精简)

def after_tick(self) -> None:
    # 1. 调用 apply_writes 合并所有 task.writes，更新 channel_versions 与 versions_seen
    self.updated_channels = apply_writes(
        self.checkpoint,
        self.channels,
        self.tasks.values(),
        self.checkpointer_get_next_version,
        self.trigger_to_nodes,
    )
    # 2. 若更新的通道涉及输出通道，向 stream 发射 "values" 模式事件
    if not self.updated_channels.isdisjoint(self.output_keys):
        self._emit("values", map_output_values, self.output_keys, True, self.channels)

    # 3. 清空本轮已合并的 pending_writes 缓存
    self.checkpoint_pending_writes.clear()

    # 4. 生成并持久化当前 Super-step 结束后的全新 Checkpoint
    self._put_checkpoint({"source": "loop"})

    # 5. 检查是否触发 interrupt_after（必须在保存完 Checkpoint 之后才能中断！）
    if self.interrupt_after and should_interrupt(
        self.checkpoint, self.interrupt_after, self.tasks.values()
    ):
        raise GraphInterrupt()
```

注意这里的第 4 步和第 5 步顺序：
**必须先调用 `self._put_checkpoint({"source": "loop"})` 保存新快照，然后才检查 `interrupt_after` 并抛出 `GraphInterrupt`。**
因为一旦抛出 `GraphInterrupt`，整个 `stream()` 循环就会立即退出；如果先抛中断再存 Checkpoint，本轮刚合并完的状态就会丢失，后续调用 `Command(resume=...)` 时就找不到中断那一刻的最新快照了。

## 8. 生产排障定位：从现象直接找源码

掌握了 `PregelLoop` 的主循环与 `_algo.py` 的核心函数后，平时开发中遇到的诡异现象都可以直接按图索骥定位到源码行：

| 故障现象 | 第一断点位置 | 第二断点位置 | 根因排查方向 |
| :--- | :--- | :--- | :--- |
| **某节点死活不执行** | `pregel/_algo.py`<br>`prepare_next_tasks()` | `pregel/_algo.py`<br>`_triggers()` | 对比该节点的 `proc.triggers` 在 `checkpoint["channel_versions"]` 中的版本是否大于 `versions_seen[node_name]` |
| **并行报错 `InvalidUpdateError`** | `pregel/_algo.py`<br>`apply_writes()` | `channels/last_value.py`<br>`LastValue.update()` | 检查报错的 `self.key` 字段被哪两个并行 `task` 同时写入了 `task.writes`，为该字段补充 `Annotated[..., reducer]` |
| **并行节点总耗时变成串行之和** | `pregel/main.py`<br>`while loop.tick()` | `pregel/_runner.py`<br>`PregelRunner.tick()` | 观察两个节点是在同一个 `step` 内并发提交，还是因为边依赖被分到了两个连续的 Super-step 中；或同步节点触发了 GIL/阻塞 I/O |
| **抛出 `GraphRecursionError`** | `pregel/_loop.py`<br>`PregelLoop.tick()` | `pregel/_algo.py`<br>`apply_writes()` | 观察每轮 `after_tick` 返回的 `updated_channels`，看是哪两个节点在互相触发对方的 trigger channel 形成死循环 |
| **中断恢复后节点重复执行了外部副作用** | `pregel/_runner.py`<br>`PregelRunner.commit()` | `pregel/_algo.py`<br>`prepare_next_tasks()` | 同一个节点内部若前半段调了外部 API、后半段抛异常或触发 `interrupt()`，恢复时整个节点函数会从头重跑，需将副作用拆为独立节点或保证幂等 |

## 9. 小结

回顾 `langgraph/pregel/` 的核心设计，可以用三个工程准则来概括：

1. **三段式 BSP 循环（Plan → Execute → Update）**：通过 `loop.tick()`、`runner.tick()` 和 `loop.after_tick()` 的严格分离，将“状态读取”、“并发计算”与“状态修改”彻底解耦。
2. **无锁并发（Lock-free Concurrency via Isolation）**：Execute 阶段每个 `PregelExecutableTask` 持有冻结的输入快照与私有的 `writes: deque` 缓冲区，Update 阶段通过 `sorted(tasks, key=attrgetter("path"))` 实现 100% 确定序合并。
3. **版本差值驱动（Version-driven Scheduling）**：节点的唤醒不靠函数直接调用，而是靠 `channel_versions[chan] > versions_seen[node][chan]` 的水位线差值驱动。
