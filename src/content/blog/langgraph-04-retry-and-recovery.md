---
title: 'LangGraph 源码学习04：RetryPolicy、pending_writes 恢复与节点幂等性'
description: '深入 pregel/_retry.py 与 _loop.py 源码，拆解节点级指数退避重试、两层持久化下的 pending_writes 部分失败恢复机制，以及为什么节点必须满足幂等性。'
publishDate: '2026-11-24 12:00:00'
tags:
  - AI Agent
  - LangGraph
  - Python
  - 源码学习
category: 源码学习
---

## 1. 引言：当并发节点跑崩了一个，已算完的结果会丢吗？

在生产环境运行长流程 Agent 时，外部依赖（LLM 接口 429/502、搜索引擎超时、数据库连接抖动）的偶发失败是常态。

假设在同一个 Super-step 中有 A、B、C 三个并行执行的节点：
- 节点 A 和节点 B 各自消耗了数千 Token 成功跑完了；
- 节点 C 在请求第三方 API 时遇到网络超时崩溃，导致整个进程退出。

此时面临两个极其关键的工程问题：
1. **已成功的工作成果会不会白费？** 前一篇文章提到，Checkpoint 是在整个 Super-step 结束后的 `after_tick()` 阶段才生成的。既然节点 C 崩了，当前这一步就没有生成新的 Checkpoint——那下次带着同一个 `thread_id` 恢复运行时，已经跑成功的节点 A 和节点 B 难道要重新再跑一遍、再花一次 Token 吗？
2. **失败的节点 C 从哪里恢复？** 恢复执行时，节点 C 是从抛出异常的那一行代码继续往下走，还是从节点函数 `def node_c(state):` 的第一行重新执行？

要彻底搞懂 LangGraph 的容错体系，需要深入 `libs/langgraph/langgraph/pregel/_retry.py`、`_loop.py` 和 `_algo.py`，拆解它的三道防线：**节点内重试（`run_with_retry`）**、**节点级半成品持久化（`pending_writes`）**与**重放幂等性约束**。


## 2. 第一道防线：`run_with_retry` 节点级原位重试

当节点抛出偶发异常时，最轻量级的恢复不是走 Checkpoint 重放，而是在当前线程/协程内直接重试该节点。

在 `add_node` 或 `compile` 时传入的 `RetryPolicy`（定义在 `libs/langgraph/langgraph/types.py`），其默认结构如下：

```python
# libs/langgraph/langgraph/types.py

class RetryPolicy(NamedTuple):
    initial_interval: float = 0.5                      # 首次重试等待时间 (0.5s)
    backoff_factor: float = 2.0                        # 指数退避乘数
    max_interval: float = 128.0                        # 单次最大等待上限 (128s)
    max_attempts: int = 3                              # 最大尝试次数（含首次执行）
    jitter: bool = True                                # 是否开启随机抖动（防惊群）
    retry_on: Union[
        type[Exception], Sequence[type[Exception]], Callable[[Exception], bool]
    ] = default_retry_on                               # 异常过滤函数
```

### 2.1 `default_retry_on`：哪些异常重试，哪些直接放行？

并不是所有报错都值得重试（例如代码写错了触发 `KeyError` 或 `TypeError`，重试 100 次结果也一样）。看 `libs/langgraph/langgraph/pregel/_retry.py` 中的 `default_retry_on` 实现：

```python
# libs/langgraph/langgraph/pregel/_retry.py

def default_retry_on(exc: Exception) -> bool:
    import httpx
    import requests

    # 1. 连接失败与超时异常：一律重试
    if isinstance(
        exc,
        (
            ConnectionError,
            httpx.ConnectError,
            httpx.TimeoutException,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ),
    ):
        return True
    # 2. HTTP 状态码异常：仅重试 5xx 服务端错误（4xx 客户端错误不重试，但可根据需要扩展 429）
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, requests.exceptions.HTTPError):
        return (
            500 <= exc.response.status_code < 600
            if exc.response is not None
            else True
        )
    # 3. 明确属于编程逻辑或本地不可恢复的内置异常：一律不重试
    return not isinstance(
        exc,
        (
            ValueError,
            TypeError,
            ArithmeticError,
            ImportError,
            LookupError,
            NameError,
            SyntaxError,
            RuntimeError,
            ReferenceError,
            StopIteration,
            StopAsyncIteration,
            OSError,
        ),
    )
```

### 2.2 `run_with_retry()` 源码拆解：为什么重试前必须清空 `task.writes`？

来看 `pregel/_retry.py` 中驱动节点执行与重试的核心循环 `run_with_retry()`（异步对应 `arun_with_retry()`）：

```python
# libs/langgraph/langgraph/pregel/_retry.py (核心重试循环精简)

def run_with_retry(
    task: PregelExecutableTask,
    retry_policy: Optional[Sequence[RetryPolicy]],
    configurable: Optional[dict[str, Any]] = None,
) -> None:
    retry_policy = task.retry_policy or retry_policy
    attempts = 0
    config = task.config
    if configurable is not None:
        config = patch_config(config, configurable=configurable)

    while True:
        try:
            # === 关键动作：每次尝试（或重试）开始前，必须先清空 task.writes 缓冲区！ ===
            task.writes.clear()
            # 执行包装后的节点 Runnable，节点的返回值会通过 ChannelWrite 写入 task.writes
            return task.proc.invoke(task.input, config)
        except ParentCommand as exc:
            raise
        except GraphBubbleUp:
            # 注意：GraphInterrupt（人机交互中断）继承自 GraphBubbleUp，绝对不能当作异常被重试吞掉！
            raise
        except Exception as exc:
            if not retry_policy:
                raise

            # 匹配适用的 RetryPolicy
            matching_policy = None
            for policy in retry_policy:
                if _should_retry_on(policy, exc):
                    matching_policy = policy
                    break
            if not matching_policy:
                raise

            attempts += 1
            if attempts >= matching_policy.max_attempts:
                raise

            # 计算指数退避 + 随机 Jitter 等待时长
            interval = matching_policy.initial_interval * (
                matching_policy.backoff_factor ** (attempts - 1)
            )
            sleep_time = min(
                matching_policy.max_interval,
                interval + random.uniform(0, 1) if matching_policy.jitter else interval,
            )
            time.sleep(sleep_time)
```

这段源码中有两个非常重要的细节：

1. **`task.writes.clear()` 的必要性**：
   一个复杂的节点（或是嵌套的子图/Runnable 链）在跑崩之前，可能已经通过内部回调或 `CONFIG_KEY_SEND` 往 `task.writes` 缓冲区里写入了一半的脏数据。如果不执行 `task.writes.clear()` 就进入下一次重试，最终提交时就会把上一次失败尝试残留的半截写入和本次成功的写入混在一起，造成状态污染。
2. **`except GraphBubbleUp: raise` 的优先级**：
   当节点内部调用 `interrupt("请人工确认")` 时，底层实际上是抛出了一个 `GraphInterrupt` 异常（继承自 `GraphBubbleUp`）。代码专门在 `except Exception` 之前捕获并直接向上抛出 `GraphBubbleUp`，防止人工中断信号被 `RetryPolicy` 误判为普通故障而原地死循环重试。


## 3. 第二道防线：两层持久化与 `pending_writes` 部分失败恢复

如果节点 C 的重试次数耗尽（`attempts >= max_attempts`），异常最终抛出了 `run_with_retry()`，导致当前 Super-step 失败退出。这时候，同在一个 Super-step 里已经跑完的节点 A 和节点 B 怎么办？

这就引出了 LangGraph 持久化层最精妙的**两层落盘架构**：

```
==================== Super-step N (并行跑 Task A, B, C) ====================

  [ Task A 执行成功 ] (t = 1s)
       └──> runner.commit(Task A)
             └──> loop.put_writes(task_a.id, writes)
                   └──> 【第 1 层持久化】立即写入数据库 checkpoint_writes 表！

  [ Task B 执行成功 ] (t = 2s)
       └──> runner.commit(Task B)
             └──> loop.put_writes(task_b.id, writes)
                   └──> 【第 1 层持久化】立即写入数据库 checkpoint_writes 表！

  [ Task C 重试耗尽，抛出异常 ] (t = 5s)
       └──> runner.commit(Task C, exc)
             └──> loop.put_writes(task_c.id, [(ERROR, exc)])
       └──> _panic_or_proceed() 抛出异常，进程中断退出！
            （此时尚未走到 loop.after_tick()，【第 2 层持久化】Checkpoint N 未生成）

==================== 稍后使用相同 thread_id 恢复执行 ====================

  1. SyncPregelLoop.__enter__() 调用 checkpointer.get_tuple(config)
     ├── 加载上一个完整快照 Checkpoint (N-1)
     └── 同时加载挂在该快照名下的 pending_writes 列表（含 Task A、B 的成功写入）

  2. loop.tick() -> prepare_next_tasks() 重新规划 Super-step N 的任务：
     ├── 构造出 Task A, Task B, Task C（因 task.id 是由路径确定性生成的 UUID5，ID 与崩溃前完全一致！）
     ├── 匹配 pending_writes：直接把数据库中已保存的 writes 塞回 Task A.writes 和 Task B.writes
     └── Task C 在 pending_writes 中只有 ERROR，故 Task C.writes 保持为空

  3. runner.tick(tasks)：
     ├── 检查 `if not t.writes:` -> Task A 和 Task B 已有 writes，【直接跳过不执行】！
     └── 仅将 Task C 提交到执行器重新运行！

  4. Task C 本次运行成功 -> 进入 loop.after_tick()：
     └── apply_writes() 将【恢复出来的 Task A、B writes】与【新跑出的 Task C writes】一起合并，
         【第 2 层持久化】生成全新的完整快照 Checkpoint N！
```

### 3.1 源码验证：`prepare_next_tasks` 是如何挂载 `pending_writes` 的？

看 `libs/langgraph/langgraph/pregel/_algo.py` 中 `prepare_next_tasks()` 在构造完 `PregelExecutableTask` 之后做的恢复匹配逻辑：

```python
# libs/langgraph/langgraph/pregel/_algo.py (pending_writes 恢复逻辑精简)

# 1. 任务 ID 采用确定性的 UUID5 算法生成：由 checkpoint_id + 节点路径 + step 共同决定
# 因此崩溃前后，同一个 Super-step 里的同一个节点生成的 task_id 100% 相同！
task_id = str(
    uuid5(
        UUID(checkpoint["id"]),
        "|".join((str(p) for p in path)),
    )
)

task = PregelExecutableTask(
    name=name,
    input=val,
    proc=proc.node,
    writes=deque(),
    config=...,
    id=task_id,
    path=path,
)

# 2. 如果存在崩溃前落盘的 pending_writes，按 (task_id, channel, value) 进行匹配回填
if pending_writes:
    writes: list[tuple[str, Any]] = [
        (c, v)
        for tid, c, v in pending_writes
        if tid == task_id and c not in (ERROR, INTERRUPT, RESUME)
    ]
    if writes:
        # 将崩溃前已成功持久化的输出直接装入 task.writes 缓冲区！
        task.writes.extend(writes)
```

再对照上一篇看过的 `PregelRunner.tick()` 第 2 步：

```python
# libs/langgraph/langgraph/pregel/_runner.py
for t in tasks:
    if not t.writes:  # <-- Task A 和 Task B 的 writes 非空，直接跳过提交！
        futures[self.submit()(run_with_retry, t, ...)] = t
```

这两段代码配合得天衣无缝：
1. **确定性 UUID5（`uuid5(UUID(checkpoint["id"]), path)`）**：保证了从同一个历史 `Checkpoint` 恢复时，重新规划出的 `Task A` 的 `task_id` 与崩溃前分毫不差。
2. **过滤控制信号（`c not in (ERROR, INTERRUPT, RESUME)`）**：崩溃前 `Task C` 记录的 `(ERROR, exc)` 被过滤掉，所以 `Task C.writes` 为空，得以重新进入线程池执行；而 `Task A` 和 `Task B` 的正常业务写入被回填到 `task.writes` 中，在 `runner.tick()` 时被当作“已完成任务”直接跳过。


## 4. 三种持久化时机模式（`durability`）：性能与防崩的权衡

既然每次节点跑完都调 `put_writes` 落库、每次 Super-step 结束都调 `put` 存快照，高频数据库 I/O 会不会拖慢整个 Agent 的响应速度？

为了在不同业务场景下平衡 I/O 延迟与崩溃恢复能力，LangGraph 在 `Pregel` 中提供了三种持久化模式（`durability: Literal["sync", "async", "exit"]`）：

| 持久化模式 (`durability`) | 数据库写入时机 | 性能开销 | 崩溃恢复能力 | 典型适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **`"sync"`** | 每次 `put_writes` 和 `put_checkpoint` 都同步阻塞等待 DB 写入落盘后，才进入下一步 | 最高（每步都有同步 DB 往返延迟） | **最强**：任意时刻进程被 `kill -9`，已完成节点和历史步状态零丢失 | 涉及金融交易、高成本长耗时推理、不可逆外部调用的核心流水线 |
| **`"async"`**（默认） | 将 `put_writes` 和 `put_checkpoint` 提交给后台线程/协程异步写库，主循环立即推进下一轮 Super-step | 极低（DB I/O 与下一轮 LLM 推理并行重叠） | **强**：仅在极端断电/强杀瞬间可能丢失尚未刷盘的最后半步 | 绝大多数生产环境在线对话与 Multi-Agent 编排 |
| **`"exit"`** | 运行期间全在内存流转，仅在整张图正常结束、抛未捕获异常或触发 `interrupt` 退出时，一次性将最终状态落库 | 最低（中间步骤零 DB 写入） | **弱**：若中间进程直接崩溃（OOM / SIGKILL），中间所有步骤全部丢失 | 短平快的轻量级图、仅需跨轮对话记忆而无需步骤级断点续跑的场景 |


## 5. 核心工程铁律：为什么节点函数必须满足“幂等性”？

理解了 `run_with_retry` 和 `pending_writes` 之后，回到引言的第二个问题：**当节点 C 因为超时或 `interrupt()` 恢复重跑时，它是从函数内部崩溃的那一行继续，还是从函数第一行重跑？**

答案是：**永远从节点函数的第一行（`def node_c(state):` 开头）重新执行。**

原因在于 Python 语言本身的运行时特性：LangGraph 持久化的是**图级别（Channel）的数据快照**，而不是 Python 解释器栈帧（Stack Frame / 局部变量字节码指针）。

看一个非常容易在生产环境引发事故的反例：

```python
# 危险反例：在单个节点内混写“非幂等外部副作用”与“易失败/会中断的操作”
async def process_order_node(state: OrderState):
    # 步骤 1：调用外部支付网关扣款（产生了不可逆的外部副作用！）
    payment_res = await payment_client.charge(user_id=state["user_id"], amount=100)

    # 步骤 2：调用大模型生成订单回执摘要（假设此处遇到 502 超时触发了 RetryPolicy，或触发了 interrupt）
    summary = await llm.ainvoke(f"生成回执: {payment_res}")

    return {"receipt": summary}
```

在这段代码里：
1. 当步骤 1 扣款成功、步骤 2 调用大模型超时抛出异常时，`run_with_retry` 会捕获异常并在 0.5 秒后发起第 2 次重试（或者整个图崩溃后下次恢复重跑 `process_order_node`）。
2. 由于步骤 2 没跑完，`process_order_node` 根本没有走到 `return`，所以 `checkpoint_writes` 表里没有任何成功记录。
3. 当重试或恢复再次调用 `process_order_node(state)` 时，**步骤 1 的 `payment_client.charge()` 会被完完全全地再执行一次，导致用户被重复扣款！**

为了避免这种因重放导致的重复副作用，工程上编写 LangGraph 节点有三条铁律：

### 铁律 1：将“非幂等副作用”与“高风险 I/O / 人工中断”拆分为独立节点

不要在一个节点函数里干两件事。把上面的逻辑拆成 `charge_node -> summarize_node` 两个节点：
一旦 `charge_node` 跑完返回，它的输出就立即通过 `put_writes` / `Checkpoint` 焊死在数据库里了；后续无论 `summarize_node` 重试多少次、中断多少次，引擎都只会重跑 `summarize_node`，绝不会再回退去调 `charge_node`。

### 铁律 2：外部写操作携带确定性幂等键（Idempotency Key）

如果节点内部必须调用外部写接口（发邮件、创建工单、扣减库存），不要在节点里用 `uuid.uuid4()` 随机生成请求号，而是利用 `thread_id` + 业务单号（或从 `RunnableConfig` 中读取确定性的 `run_id` / `task_id`）构造幂等键传给下游接口。这样即便节点重跑，下游服务也能识别出重复请求并幂等返回。

### 铁律 3：在单节点内使用 `@task` 包装子步骤（Functional API 粒度）

如果确实不想把图拆得太碎，可以在节点内部将具有副作用的子函数用 `@task` 装饰器包装——每个 `@task` 拥有自己独立的 `task_id` 和 `pending_writes` 缓存，恢复时已完成的 `@task` 会直接命中缓存结果，跳过重复执行。


## 6. 小结

把 LangGraph 执行引擎的容错与恢复源码串联起来，核心机制可以归纳为 4 点：

1. **原位指数退避重试（`run_with_retry`）**：单节点遇到网络或 5xx 异常时优先在当前线程/协程内按指数退避 + Jitter 重试；每次重试前强制执行 `task.writes.clear()` 清理脏缓冲，且对 `GraphBubbleUp`（如人工中断）直接放行。
2. **两层持久化架构**：`put_writes()` 在每个节点完成的瞬间做**节点级落盘**，`put()` 在每个 Super-step 结束时做**全局快照落盘**。
3. **`pending_writes` 增量恢复**：依靠确定性的 `uuid5(checkpoint_id, path)` 生成 `task_id`，崩溃恢复时 `prepare_next_tasks()` 会将同一步内已成功节点的写入直接挂载回 `task.writes`，在 `runner.tick()` 中自动跳过它们，仅重跑失败节点。
4. **重放边界与幂等要求**：LangGraph 保证的是“Super-step 状态恰好合并一次”，但节点函数的执行语义是 **“至少一次（At-least-once，从函数开头重跑）”** 。拆分副作用节点或引入幂等键是生产落地的必备前提。
