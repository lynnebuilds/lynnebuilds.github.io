---
title: 'LangGraph 源码学习06：条件边 Branch 的真实执行时机与 Command 路由机制'
description: '深入 langgraph/graph/branch.py 源码，拆解条件边路由函数何时被调用、如何读取刚更新的节点状态，以及它与 Command(goto=...) 动态路由的底层差异。'
publishDate: '2026-11-27 15:00:00'
tags:
  - AI Agent
  - LangGraph
  - Python
  - 源码学习
category: 源码学习
---

## 1. 引言：关于条件边（Conditional Edges）的三个常见误区

在编写 ReAct Agent 或复杂分支工作流时，`add_conditional_edges(source, path, path_map)` 是使用频率最高的 API 之一：

```python
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
```

代码看起来非常直观，但在排查复杂路由不生效或状态读取异常时，很多开发者对条件边的底层运行机制存在三个典型的误区：

1. **误区一（关于执行时机）**：以为 `should_continue` 是在下一轮 Super-step 的 Plan 阶段被调度器独立调用的。实际上，**条件边的路由函数在 `source` 节点（如 `"agent"`）执行的同一个 Task 内部、紧挨着节点函数返回之后就立刻被调用了**！
2. **误区二（关于状态快照）**：既然在 Execute 阶段全局 `Channel` 还没更新（要等 `after_tick` 的 `apply_writes` 才合并），那紧跟在节点后面跑的 `should_continue(state)` 为什么能读到 `"agent"` 节点刚刚生成的 `last_message`？
3. **误区三（关于边共存与覆盖）**：以为给节点配了 `add_conditional_edges` 就会覆盖掉 `add_edge`，或者以为可以在节点内部返回 `Command(goto=...)` 的同时又在外层配条件边。

这篇笔记通过 `libs/langgraph/langgraph/graph/branch.py` 与 `state.py`，把条件边和 `Command` 动态路由的真实源码拆解清楚。


## 2. 源码拆解：`BranchSpec` 是怎么挂载到起点节点上的？

在上一篇拆解 `compile()` 时，可以看到普通边是通过 `attach_edge` 往起点节点的 `writers` 列表里塞入一个静态的 `ChannelWriteEntry("branch:to:B", None)`。

那么条件边呢？当调用 `builder.add_conditional_edges("agent", should_continue, path_map)` 时，它先在 `self.branches["agent"]` 里登记了一个 `BranchSpec` 对象；等到 `compile()` 执行 `attach_branch()` 时，发生了非常巧妙的组装。

来看 `libs/langgraph/langgraph/graph/branch.py` 与 `state.py` 的核心实现：

```python
# libs/langgraph/langgraph/graph/branch.py (BranchSpec 核心结构精简)

class BranchSpec(NamedTuple):
    path: Runnable[Any, Hashable | list[Hashable]]   # 包装后的路由函数 (如 should_continue)
    ends: Optional[dict[Hashable, str]]              # path_map 映射字典
    input_schema: Optional[type[Any]] = None

    def run(
        self,
        writer: Callable[[Sequence[str | Send], RunnableConfig], Any],
        reader: Optional[Callable[[RunnableConfig], Any]] = None,
    ) -> RunnableCallable:
        # 将 "读取最新状态 (reader) -> 执行路由函数 (self.path) -> 写入目标触发通道 (writer)"
        # 串联成一个挂载在 source 节点尾部的 Runnable！
        return ChannelWrite.register_writer(
            RunnableCallable(
                func=self._route,
                afunc=self._aroute,
                writer=writer,
                reader=reader,
                name=None,
                trace=False,
            )
        )
```

再看 `_route()` 内部在运行时到底做了什么：

```python
# libs/langgraph/langgraph/graph/branch.py (_route 核心执行逻辑拆解)

def _route(
    self,
    input: Any,
    config: RunnableConfig,
    *,
    reader: Optional[Callable[[RunnableConfig], Any]],
    writer: Callable[[Sequence[str | Send], RunnableConfig], Any],
) -> Runnable:
    if reader is not None:
        # === 关键点 1：通过 reader 获取融合了当前节点最新输出的状态视图！ ===
        value = reader(config)
        # 如果当前节点返回了字典更新，将其合并到读取到的状态副本上
        if isinstance(input, dict) and isinstance(value, dict):
            value = {**value, **input}
    else:
        value = input

    # === 关键点 2：调用用户编写的路由函数 should_continue(value) ===
    result = self.path.invoke(value, config)

    # === 关键点 3：根据返回值和 path_map (self.ends)，解析出目标节点列表 ===
    if isinstance(result, (list, tuple)):
        destinations = [
            r if isinstance(r, Send) else (self.ends[r] if self.ends else r)
            for r in result
        ]
    else:
        destinations = [
            result
            if isinstance(result, Send)
            else (self.ends[result] if self.ends else result)
        ]

    # === 关键点 4：调用 writer 生成向 `branch:to:{dest}` 通道的写入项！ ===
    return writer(destinations, config)
```

最后看 `state.py` 中的 `attach_branch()` 是怎么把这个 `writer` 接到目标节点的 `triggers` 上的：

```python
# libs/langgraph/langgraph/graph/state.py (attach_branch 核心实现精简)

def attach_branch(self, start: str, name: str, branch: BranchSpec) -> None:
    def branch_writer(
        packets: Sequence[str | Send], config: RunnableConfig
    ) -> Optional[ChannelWrite]:
        writes = [
            # 若路由函数返回的是普通节点名 dest（且非 END），则向 `branch:to:{dest}` 通道写入信号
            # 若返回的是 Send 动态任务对象，则向 TASKS ('__pregel_tasks') 通道写入 Send 包
            ChannelWriteEntry(f"branch:to:{p}" if not isinstance(p, Send) else TASKS, p)
            for p in packets
            if p != END
        ]
        return ChannelWrite(writes) if writes else None

    # 为所有可能的目标节点创建 `branch:to:{dest}` 通道并注册订阅 (triggers)
    for dest in (branch.ends.values() if branch.ends else self.builder.nodes):
        if dest != END:
            chan = f"branch:to:{dest}"
            if chan not in self.channels:
                self.channels[chan] = EphemeralValue(Any, guard=False)
                self.nodes[dest].triggers.append(chan)

    # 将包装好的 branch.run(...) 追加到起点节点 start 的 writers 管道末尾！
    self.nodes[start].writers.append(
        branch.run(branch_writer, self._read_channels(start))
    )
```


## 3. 真相还原：条件边在 Super-step 中的精确时序

把上面三段源码拼在一起，条件边在运行时的完整生命周期就水落石出了：

```
=============== Super-step N 的 Execute 阶段 (在 Task["agent"] 内部) ===============

  1. 执行节点主函数：
     output = agent_node(task.input)   # 返回 {"messages": [AIMessage(tool_calls=[...])]}
         |
         v
  2. 执行 writers[0] (attach_node 挂载的状态写入器)：
     将 ("messages", [AIMessage(...)]) 追加进 task.writes 缓冲区
     （注意：此时通过 ChannelRead.do_read 可以结合 task.writes 本地重算出合并后的最新 messages！）
         |
         v
  3. 执行 writers[1] (attach_branch 挂载的 BranchSpec._route)：
     ├── reader(config) 读取状态，并融合当前节点的输出（包含刚生成的 AIMessage）
     ├── 调用路由函数 should_continue(state) -> 返回字符串 "tools"
     ├── 查表 branch.ends["tools"] -> 得到目标节点名 "tools"
     └── branch_writer 将 ("branch:to:tools", "tools") 追加进同一个 task.writes 缓冲区！
         |
         v
  4. Task["agent"] 执行完毕，调用 runner.commit()：
     此时 task.writes 队列里同时装着：
       - ("messages", [AIMessage(...)])
       - ("branch:to:tools", "tools")
     两条记录作为同一个原子批次落盘到 checkpoint_writes 表！

=============== Super-step N 的 Update 阶段 (loop.after_tick) ===============

  5. apply_writes() 合并状态：
     ├── Channel["messages"] 合并新消息，版本号 +1
     └── Channel["branch:to:tools"] 写入信号，版本号 +1

=============== Super-step N+1 的 Plan 阶段 (loop.tick) ===============

  6. prepare_next_tasks() 扫描 updated_channels：
     发现 "branch:to:tools" 版本号更新，通过 trigger_to_nodes 立即唤醒 "tools" 节点！
```

这个时序揭示了三个极其重要的底层事实：

### 3.1 为什么路由函数能读到刚更新的 `messages`？

在 `attach_branch` 中传给 `branch.run` 的 `reader`，底层调用了 `PregelNode` 的本地状态读取器（结合了当前 `task.writes` 中的本地未提交写入，并通过 Channel 的副本合并逻辑计算出最新值）。
因此，尽管此时还没有走到 Super-step 末尾的 `apply_writes()`，紧跟在 `"agent"` 节点函数屁股后面运行的 `should_continue(state)`，拿到的 `state["messages"]` 已经是合并了当前节点输出之后的最新列表！

### 3.2 为什么路由函数绝对不能写有副作用的逻辑？

因为 `BranchSpec._route` 是作为起点节点 `Task` 的一部分（`writers` 管道的一环）在 `run_with_retry` 内部执行的！
如果路由函数自己抛出了异常，或者后续步骤触发了重试，路由函数会跟着起点节点一起重跑。所以**路由函数必须保持为轻量、快速、确定性的纯函数**——千万不要在 `should_continue` 里面再去调一次大模型 API 或写数据库；所有的 LLM 决策都应该在节点函数内完成并把决策标识写进 `State`，路由函数只负责读 `State` 字段做分支转发。

### 3.3 普通边与条件边共存时会发生什么？

假设既写了 `builder.add_edge("A", "B")`，又写了 `builder.add_conditional_edges("A", router_fn, {"C": "C", "D": "D"})`：
回看 `attach_edge` 和 `attach_branch` 的代码——它们都只是往 `self.nodes["A"].writers` 列表里 `.append(...)` 了一个 `ChannelWrite`！
- `writers[1]`（来自普通边）无条件往 `task.writes` 里写 `("branch:to:B", None)`；
- `writers[2]`（来自条件边）根据 `router_fn` 返回值往 `task.writes` 里写 `("branch:to:C", "C")`。

结果就是：**下一轮 Super-step 中，节点 `B` 和节点 `C` 会被同时唤醒并行执行！** 两者不是“覆盖”关系，而是“并集（Union）”关系。


## 4. 一对多扇出（Fan-out）：条件边返回列表或 `Send`

在上面的 `_route()` 源码中可以看到，路由函数 `self.path.invoke(value, config)` 的返回值不仅可以是单个字符串，还可以是 **`list[str]`** 或 **`list[Send]`**：

### 4.1 静态多目标并行（返回 `list[str]`）

```python
def route_parallel(state: State) -> list[str]:
    targets = [" compulsory_log"]
    if state["need_web_search"]:
        targets.append("web_search")
    if state["need_kb_search"]:
        targets.append("kb_search")
    return targets
```

当返回 `["compulsory_log", "web_search"]` 时，`branch_writer` 会在 `task.writes` 中同时生成 `("branch:to:compulsory_log", ...)` 和 `("branch:to:web_search", ...)` 两条写入，下一轮这两个节点自动并发执行。

### 4.2 动态 Map-Reduce 扇出（返回 `list[Send]`）

如果事前连“要并发跑多少个分支”都不知道（例如 Agent 拆解出了 $K$ 个子查询，需要对每个子查询并发跑同一个 `"search_worker"` 节点，且每个分支传入不同的局部参数），返回节点名字符串就不够用了。

此时路由函数可以返回 `[Send("search_worker", {"query": q}) for q in state["queries"]]`：
在 `branch_writer` 中，`isinstance(p, Send)` 的分支会将它写入特殊的保留通道 `TASKS`（常量值 `"__pregel_tasks"`），并在 `apply_writes()` 中存入 `checkpoint["pending_sends"]`。下一轮 `prepare_next_tasks()` 会遍历 `pending_sends`，为每一个 `Send` 包动态实例化一个独立的 `PregelExecutableTask(name="search_worker", input={"query": q}, path=('__pregel_push', idx))` 并发执行！


## 5. `add_conditional_edges` vs `Command`：两种动态路由范式的取舍

在较新版本的 LangGraph 中，除了在图外层用 `add_conditional_edges` 声明条件边，还支持在节点函数内部直接返回一个 `Command` 对象，同时完成**“修改状态（`update`）”**与**“指定下一跳节点（`goto`）”**：

```python
from langgraph.types import Command

def agent_node(state: AgentState) -> Command[Literal["tools", "__end__"]]:
    response = model.invoke(state["messages"])
    if response.tool_calls:
        return Command(update={"messages": [response]}, goto="tools")
    return Command(update={"messages": [response]}, goto=END)
```

在底层源码（`libs/langgraph/langgraph/graph/state.py` 的 `_get_updates` 与 `ChannelWrite`）中，当节点返回 `Command` 对象时：
1. `Command.update` 字典会被拆解为普通的 `(channel_name, val)` 状态写入；
2. `Command.goto` 指定的一组目标节点名 `dest`，会被直接转换成 `ChannelWriteEntry(f"branch:to:{dest}", None)`（若为 `Send` 则写入 `TASKS` 通道）；
3. 若设置了 `Command(graph=Command.PARENT, ...)`，还会通过抛出 `ParentCommand` 异常穿透子图边界，直接驱动父图的状态更新与路由跳转！

那么在工程实践中，什么时候用 `add_conditional_edges`，什么时候用 `Command`？

| 对比维度 | `add_conditional_edges`（外置条件边） | `Command(update=..., goto=...)`（内置指令） |
| :--- | :--- | :--- |
| **关注点分离** | **高**：节点只管算业务状态，拓扑流转逻辑全部收敛在图构建层（`builder`） | **低**：节点内部既管业务计算又管下一跳叫什么名字，节点与图拓扑耦合 |
| **可复用性** | 同一个节点函数可以挂载到不同的图中，配合不同的条件边复用 | 节点函数内部硬编码了 `goto="tools"`，难以脱离特定图结构复用 |
| **状态更新与路由联动** | 必须先把路由依据写进 `State` 字段，再由路由函数从 `State` 读出来判断 | 可以在节点内部根据局部临时变量直接决定 `goto`，无需污染全局 `State` 字段 |
| **跨子图跳转 (`PARENT`)** | 不支持直接跳出子图控制父图路由 | 原生支持 `Command(graph=Command.PARENT, goto=...)` 跨层级移交控制权 |
| **静态可视化支持** | 传入 `path_map`即可画出完整分支 | 需在节点函数返回值上显式标注类型注解 `-> Command[Literal["a", "b"]]`（或 `add_node(..., destinations=...)`）才能被编译器提取出虚线边 |

**工程选型建议**：
- 构建主干清晰、节点需复用的标准工作流时，优先使用 **`add_conditional_edges` + 显式 `path_map`**，让图结构一目了然。
- 在构建**多智能体移交（Multi-Agent Handoffs / Swarm）**、**子图向父图汇报跳转**，或者路由判断极度依赖节点内部局部变量时，使用 **`Command`（配合 `Command[Literal[...]]` 返回类型注解）** 会精简大量样板代码。
- **切忌混用**：不要在同一个节点内部返回 `Command(goto="B")` 的同时，又在外部给它绑一条 `add_conditional_edges`——因为底层两者都会往 `task.writes` 里塞 `branch:to:xxx` 信号，导致两个路由目标在下一轮被同时并行触发。


## 6. 小结

把条件边与动态路由的源码机制串起来，核心结论有 4 条：

1. **条件边是起点节点的尾部写入器**：`attach_branch` 将 `BranchSpec` 编译为起点节点 `PregelNode.writers` 末尾的一环；路由函数在起点节点执行完的**同一个 Task 内立即运行**，而非等到下一轮 Plan 阶段。
2. **本地合并视图读取**：路由函数通过 `reader` 拿到的 `state` 已经融合了当前节点刚刚在内存中产出的输出字典，因此可以直接读取当前节点的最新产物做路由判断。
3. **多边共存取并集**：同一个起点节点挂载的多条普通边、条件边或返回列表，其生成的 `branch:to:xxx` 信号会在 `task.writes` 中取并集，在下一轮 Super-step 中并发唤醒所有命中的目标节点。
4. **`Command` 与外置条件边同源**：节点内返回 `Command(update=..., goto=...)` 在底层同样被翻译为状态通道写入 + `branch:to:xxx` 触发信号，并额外支持 `Command.PARENT` 跨子图路由控制。
