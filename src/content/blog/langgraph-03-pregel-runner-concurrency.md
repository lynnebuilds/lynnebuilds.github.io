---
title: 'LangGraph 源码学习03：并发调度器 PregelRunner 与 Sync/Async 性能边界'
description: '深入 pregel/_runner.py 与 executor 源码，拆解 PregelRunner.tick 的 Fast-path 优化、FIRST_COMPLETED 流式等待机制，以及同步线程池与异步事件循环的性能差异。'
publishDate: '2026-11-21 17:00:00'
tags:
  - AI Agent
  - LangGraph
  - Python
  - 源码学习
category: 源码学习
---

## 1. 引言：为什么配了并行分支，Agent 跑起来还是串行的？

在构建多分支 Agent（例如同时执行“检索知识库”与“调用联网搜索”两个独立节点）时，经常会遇到几个与性能预期不符的现象：

1. 图里明明把两个节点配成了并行分支，但观察日志和总耗时，发现耗时依然接近两个节点之和。
2. 以为只要把调用方式从 `app.invoke()` 换成 `await app.ainvoke()`，并发性能就会自动起飞，结果发现节点函数如果还是普通的 `def`，底层依然在跑线程池。
3. 当同一轮 Super-step 里有 3 个并行节点，其中两个 1 秒就跑完了，第 3 个调用慢速大模型花了 25 秒——为什么前两个节点的流式事件（`stream_mode="updates"`）能提前出来，但下游汇总节点却必须死等第 25 秒结束才能启动？

要搞清楚 LangGraph 并发执行的真实行为与性能边界，必须深入 `libs/langgraph/langgraph/pregel/_runner.py` 与 `_loop.py`，看看负责 **Execute（执行）阶段** 的核心组件 **`PregelRunner`** 到底是怎么工作的。


## 2. `PregelRunner` 的定位：只管并发跑，绝不碰全局状态

在 BSP 三阶段模型中，`PregelRunner` 的职责非常纯粹：
它接收 Plan 阶段（`loop.tick()`）筛选出来的本轮就绪任务集合 `tasks: Iterable[PregelExecutableTask]`，把它们丢进并发执行器里跑，并在每个任务跑完的瞬间调用 `commit()` 将输出塞进任务私有的 `task.writes` 缓冲区。

它的整体调度链路如下：

```
                 loop.tasks.values() (本轮就绪任务集)
                                 |
                                 v
                +---------------------------------+
                |       PregelRunner.tick()       |
                +----------------+----------------+
                                 |
          +----------------------+----------------------+
          | len(tasks) == 1                             | len(tasks) > 1 (或配了 timeout)
          | (无超时、无流式 waiter)                       |
          v                                             v
+-----------------------+             +-----------------------------------+
|   [ Fast-path 优化 ]   |             |     [ 并发提交与流式等待路径 ]     |
| 跳过 Future 包装与池化 |             | 1. loop.submit(run_with_retry)    |
| 当前线程直接同步调用    |             |    放入 FuturesDict 映射表         |
| run_with_retry(task)  |             | 2. wait(FIRST_COMPLETED) 循环等待 |
| 完成后直接 commit()    |             | 3. 完成一个 -> callback commit()  |
+-----------------------+             |    -> yield 触发外部实时流式输出   |
                                      +-----------------+-----------------+
                                                        |
                                                        v
                                      +-----------------------------------+
                                      |   [ 同步屏障 (Barrier) 与收尾 ]    |
                                      | 等待所有 Future 结束               |
                                      | _panic_or_proceed() 统一结算异常   |
                                      +-----------------------------------+
```


## 3. 源码拆解：`PregelRunner.tick()` 的 5 个关键步骤

打开 `libs/langgraph/langgraph/pregel/_runner.py`，直接看同步版 `tick()`（异步版 `atick()` 结构完全对称）的核心源码：

```python
# libs/langgraph/langgraph/pregel/_runner.py (tick 核心实现精简)

class PregelRunner:
    def __init__(
        self,
        *,
        submit: Weakref[Submit],
        put_writes: Weakref[ Callable[[str, Sequence[tuple[str, Any]]], None]],
        node_finished: Optional[Callable[[str], None]] = None,
        ...
    ) -> None:
        self.submit = submit
        self.put_writes = put_writes
        self.node_finished = node_finished

    def tick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: Optional[float] = None,
        retry_policy: Optional[RetryPolicy] = None,
        get_waiter: Optional[Callable[[], concurrent.futures.Future[None]]] = None,
    ) -> Iterator[None]:
        tasks = tuple(tasks)
        futures = FuturesDict(
            callback=weakref.WeakMethod(self.commit),
            event=threading.Event(),
            future_type=concurrent.futures.Future,
        )

        # === 步骤 1：单任务 Fast-path（快速通道优化） ===
        # 当本轮只有 1 个待执行节点，且未开启 step_timeout 和外部流式 waiter 时：
        # 直接在当前主线程执行，彻底省去 ThreadPoolExecutor 提交与 Future 调度的上下文切换开销！
        if len(tasks) == 1 and timeout is None and get_waiter is None:
            t = tasks[0]
            try:
                run_with_retry(
                    t,
                    retry_policy,
                    configurable={
                        CONFIG_KEY_SEND: partial(writer, t),
                        CONFIG_KEY_CALL: partial(call, t),
                    },
                )
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, exc)
            if reraise and futures:
                _panic_or_proceed(futures.done, panic=reraise)
            yield
            return

        # === 步骤 2：多任务并发提交 ===
        for t in tasks:
            if not t.writes:  # 仅提交尚未完成（无缓存/恢复 writes）的任务
                futures[
                    self.submit()(
                        run_with_retry,
                        t,
                        retry_policy,
                        ...
                    )
                ] = t

        # === 步骤 3：FIRST_COMPLETED 循环等待（支撑边跑边流式输出） ===
        all_tasks = tasks
        end_time = timeout + time.monotonic() if timeout else None
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=(max(0, end_time - time.monotonic()) if end_time else None),
            )
            if not done:
                break  # 触发超时跳出
            for fut in done:
                futures.pop(fut)
            # 只要有一个任务完成（或流式队列收到新 chunk），立刻 yield 让外层 stream() 把数据推给客户端！
            yield

        # === 步骤 4：超时清理与异常记录 ===
        for fut, task in list(futures.items()):
            if not fut.done():
                fut.cancel()
                futures.pop(fut)
                self.commit(task, TimeoutError())

        # === 步骤 5：同步屏障后的统一异常结算 ===
        _panic_or_proceed(
            futures.done,
            timeout_exc_cls=TimeoutError,
            panic=reraise,
        )
```

这段源码里藏着三个非常值得细品的工程设计：

### 3.1 设计亮点一：单任务 Fast-path 优化

很多人担心引入图编排框架会带来额外的线程调度损耗。看上面**步骤 1**的代码：
在绝大多数顺序图（`A -> B -> C`）中，每一轮 Super-step 只有 1 个就绪任务（`len(tasks) == 1`）。此时如果没配 `step_timeout`，`PregelRunner` 根本不会去调 `self.submit()` 把任务扔进线程池，而是直接在当前调用栈里调 `run_with_retry(t)`！
这意味着对于普通顺序流程，LangGraph 的节点调度开销几乎等同于直接调用 Python 函数。

### 3.2 设计亮点二：`FIRST_COMPLETED` 与实时流式输出

既然 BSP 模型在 Super-step 末尾有同步屏障（必须等所有并行节点都跑完才能进入 `after_tick` 合并状态），那为什么在跑并行节点时，客户端依然能实时收到每个节点吐出的 LLM Token 或节点完成事件？

秘密就在**步骤 3** 的 `return_when=concurrent.futures.FIRST_COMPLETED` 以及 `get_waiter` 机制：
1. `FuturesDict` 绑定了完成回调 `callback=weakref.WeakMethod(self.commit)`。当并行任务 A 率先完成时，`Future` 的回调会立刻触发 `self.commit(task_a)`，将 `"updates"` 流事件推入 `stream` 队列。
2. 紧接着 `concurrent.futures.wait(..., FIRST_COMPLETED)` 立即返回，循环体执行 `yield`，控制权交回 `Pregel.stream()` 主循环的 `yield from output()`，客户端立刻就能拿到节点 A 的输出，**不需要干等慢节点 B**。
3. 如果开启了 Token 级流式（`stream_mode="messages"`），`get_waiter()` 还会往 `futures` 集合里塞入一个监听流队列的哨兵 `Future`——哪怕节点 A 和 B 都还没跑完，只要其中一个节点的 LLM Callback 往流队列里塞了一个 Token，哨兵 `Future` 就会完成并触发 `yield`，立刻把 Token 吐给调用方。

### 3.3 设计亮点三：`_panic_or_proceed` 的屏障后异常结算

当同一轮 Super-step 里的并行节点 A 成功了、节点 B 抛出了未捕获的异常（比如 `ValueError`），会发生什么？
在步骤 2 和步骤 3 中，节点 B 抛出异常时并不会立刻中断正在运行的节点 A，而是先通过 `self.commit(task_b, exc)` 把 `(ERROR, exc)` 记录到 `task_b.writes` 并落盘，同时等待节点 A 正常跑完并把结果持久化到 `checkpoint_writes` 表。
直到所有并发任务都结束（越过同步屏障），在**步骤 5** 调用 `_panic_or_proceed()` 时，引擎才会把节点 B 的异常重新 `raise` 出来终止运行。这样既保证了节点 A 的工作成果不会白费（下次恢复时直接跳过 A），又保证了异常栈的清晰抛出。


## 4. `SyncPregelLoop` vs `AsyncPregelLoop`：两条截然不同的并发底座

在 LangGraph 中，调用 `app.invoke()` / `app.stream()` 走的是 **`SyncPregelLoop`**；而调用 `await app.ainvoke()` / `app.astream()` 走的是 **`AsyncPregelLoop`**。

这两套循环在底层使用的并发原语完全不同：

| 维度 | `SyncPregelLoop` (`invoke` / `stream`) | `AsyncPregelLoop` (`ainvoke` / `astream`) |
| :--- | :--- | :--- |
| **底层并发执行器** | `BackgroundExecutor`（基于 `ThreadPoolExecutor` 线程池，携带 `ContextVars`） | `AsyncBackgroundExecutor`（基于 `asyncio.TaskGroup` / `asyncio.create_task` 事件循环） |
| **节点调用方法** | 调用 `proc.invoke(input, config)` | 调用 `await proc.ainvoke(input, config)` |
| **等待原语** | `concurrent.futures.wait(..., FIRST_COMPLETED)` | `asyncio.wait(..., FIRST_COMPLETED)` |
| **对同步 `def` 节点的处理** | 多任务时在线程池的工作线程中并行执行 | LangChain `RunnableLambda.ainvoke` 默认通过 `run_in_executor` 丢回线程池执行 |
| **任务取消 (Cancel) 能力** | **极弱**：Python 线程无法被外部安全强杀，`fut.cancel()` 只能取消尚未启动的排队任务 | **强**：直接调用 `task.cancel()` 向协程注入 `CancelledError` 立即中断等待 |

从这张对比表中，可以解释生产环境里最容易踩的三个并发性能坑：

### 坑 1：同步模式下跑 CPU 密集型任务，受限于 GIL 无法真正并行

如果使用 `app.invoke()`，底层通过 `ThreadPoolExecutor` 跑并行节点：
- 当节点内部是 **I/O 密集型操作**（如 `httpx.post` 请求大模型、查数据库），工作线程在等待网络响应时会释放 Python GIL（全局解释器锁），此时多线程确实能重叠等待时间，实现并发提速。
- 但如果节点内部包含 **CPU 密集型计算**（如解析巨型 JSON/Docx AST、本地 Numpy/Pandas 大矩阵运算），所有工作线程会争抢同一把 GIL，导致并行分支退化为单核时间片轮转，总耗时等于各节点之和。

### 坑 2：用了 `ainvoke()`，但节点函数写成同步阻塞 I/O

很多人在外层改用了 `await app.ainvoke()`，但定义节点时写的还是同步函数，且内部用了同步阻塞库（如 `requests.post` 或 `time.sleep`）：

```python
# 错误示范：在 async 图中混入阻塞操作或误以为 def 就是原生协程
def my_node(state: State):
    resp = requests.post(...)  # 同步阻塞调用
    return {"res": resp.json()}
```

虽然 LangChain 的 `RunnableLambda` 在 `ainvoke` 时会把普通 `def` 函数通过 `loop.run_in_executor()` 放到默认线程池里跑以避免直接卡死事件循环，但这不仅受限于默认线程池大小（`min(32, os.cpu_count() + 4)`），还失去了 `asyncio` 单线程高并发协程以及随时 `cancel()` 超时任务的全部优势。
更糟糕的是，如果你把节点声明成了 `async def my_node(state)`，却在里面调了同步的 `requests.post()` 或 `time.sleep()`，那就会**直接阻塞整个 `asyncio` 事件循环线程**，导致所有并行分支瞬间变成绝对串行。

### 坑 3：为什么同步节点无法真正中止超时任务？

在 `PregelRunner.tick()` 的步骤 4 中，当 `timeout` 到期时，代码执行了 `fut.cancel()`。
但在 Python 的 `concurrent.futures.Future` 中，一旦工作线程已经开始执行你的函数（处于 `RUNNING` 状态），`fut.cancel()` 会直接返回 `False`——**Python 标准库没有提供安全强杀一个正在运行的线程的机制**。主循环虽然抛出了 `TimeoutError` 退出了，但那个卡死的同步线程依然会在后台僵尸般地继续跑，直到 I/O 自己超时。
而在 `AsyncPregelLoop` 中，`fut.cancel()` 对应的是 `asyncio.Task.cancel()`，它会在协程当前的 `await` 挂起点直接注入 `asyncio.CancelledError`，干净利落地终止底层网络请求。

**因此，生产级 LangGraph 的黄金法则是：只要节点涉及 LLM 调用或外部网络 I/O，一律定义为 `async def`，内部使用异步客户端（如 `AsyncOpenAI`、`httpx.AsyncClient`），并在入口统一使用 `ainvoke()` / `astream()`。**


## 5. BSP 同步屏障的固有代价：慢节点（Straggler）木桶效应

即便全链路都改成了标准的 `async def` + `ainvoke()`，BSP 模型依然有一个架构层面的固有代价——**同步屏障（Synchronization Barrier）导致的慢节点拖尾问题**。

假设图中设计了如下并行拓扑：

```
          +--> [ 节点 A: 意图分类 (耗时 0.8s) ] --+--> [ 节点 D: 快速回复组装 ]
          |                                     |
[ START ] +--> [ 节点 B: 关键词提取 (耗时 0.5s) ] -+
          |
          +--> [ 节点 C: 深度长文检索 (耗时 25s) ] ---> [ 节点 E: 深度报告归档 ]
```

在直觉上，节点 D 只依赖 A 和 B 的输出，不依赖 C；既然 A 和 B 在第 1 秒内就跑完了，节点 D 应该在第 1 秒末就立刻启动才对。

但在 LangGraph 默认的 Super-step 调度下：
1. **Super-step 0**：A（0.8s）、B（0.5s）、C（25s）同时在 `START` 后被唤醒，进入同一个 `runner.tick()`。
2. 尽管 A 和 B 在 0.8 秒内就已经执行完毕并调用了 `commit()`，但 `runner.tick()` 的 `while` 循环必须等本轮所有的 `futures`（包括长达 25 秒的节点 C）全部完成，才能退出 `runner.tick()` 进入 `loop.after_tick()`！
3. 只有 `after_tick()` 执行完 `apply_writes()`，A 和 B 写入的通道版本号才会递增；到了第 25 秒后的 **Super-step 1**，节点 D 和节点 E 才会一起被唤醒。

也就是说，**同一个 Super-step 内的耗时永远等于最慢的那个节点（$\max(T_A, T_B, T_C)$）**。

如果业务中确实存在“快链路（A+B->D）完全独立于慢链路（C->E），且不希望快链路被慢链路阻塞”的场景，工程上有两种拆解方式：
1. **子图隔离或异步后台任务**：将慢链路 C->E 剥离为独立的异步后台任务，不在主图的同一个 Super-step 屏障内同步等待。
2. **控制单步颗粒度**：为慢速外部 I/O 节点配置合理的超时阈值与降级补偿策略，防止单个长尾请求无限制挂起整轮 Super-step。


## 6. 小结

回顾 `PregelRunner` 与并发执行层的源码，核心工程结论如下：

1. **职责解耦**：`PregelRunner` 只负责并发驱动任务并通过 `commit()` 将输出写入任务私有的 `writes` 缓冲与持久化表，绝不在多线程/多协程中直接修改 `Channel`。
2. **单任务零池化损耗**：当本轮只有一个就绪任务且无超时配置时，`PregelRunner.tick()` 直接走同步 Fast-path 执行，避免不必要的 `Future` 与线程池调度开销。
3. **流式输出突破屏障等待**：借助 `wait(..., FIRST_COMPLETED)` 与 `get_waiter` 哨兵机制，即使本轮 Super-step 尚未全部结束，已完成的节点更新和生成的 LLM Token 也能实时 `yield` 推送给客户端。
4. **生产首选原生 Async**：`SyncPregelLoop` 受限于 GIL 与线程不可取消性；生产环境应对 I/O 节点全面采用 `async def` + `ainvoke()` / `astream()`，并注意规避在 `async def` 内部混用同步阻塞调用。
