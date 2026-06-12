# book_search 改造为带 judge_query 的多步 workflow

> ⚠️ **已演进 / 部分过时（2026-06-11）**
> 本 spec 描述的是 step1 的**早期单层形态**，已被后续设计取代两次：
> 1. **3 步 + 五类 category**：judge 从「2 轮收窄」演进为「规范化 / 降噪 / 判定明确性」+
>    `retrievable/pending_split/missing_info/ambiguous/other`（见本文末「增补」与 `core/workflow/query_preprocess.py`）。
> 2. **Router 拆分**：step1 进一步沿"通用净化 vs 检索专属"劈开——规范化+指代上提到
>    Intent Router，降噪+难度分类留在 QA capability。**以
>    [2026-06-11 拆分设计](2026-06-11-intent-router-and-preprocess-split-design.md) 与
>    [项目架构](../../ARCHITECTURE.md) 为准。**
>
> 同时：本 spec 提到的 `core/workflow/book_rag.py` 已被 `doc_workflow.py` 取代、将退役；
> "不把会话历史传进 workflow"这一**非目标已被推翻**（Router 顶层持有并贯穿会话 memory）。
> 下文保留作历史记录。

**日期:** 2026-06-04
**状态:** 已实现并演进 → 见上方提示，以新文档为准
**分支:** refactor/book-consolidation

## 背景

当前 `core/tools/book_tools.py` 的 `book_search` 是内联检索:`retriever.aretrieve(query)` →
`synthesizer.asynthesize(query, nodes)`。synthesizer 只看到 `query + nodes`,没有会话上下文——
这是正确的分层(会话上下文由 Agent 层 `ChatMemoryBuffer` 持有)。

但存在两类「query 不明确」会导致回答质量下降:

1. **指代/缺上下文**:如「它的索引结构呢?」。query 含指代词,脱离会话历史无法解析。
   只有 Agent 层(持有历史)能修,工具层修不了。
2. **太宽泛/太模糊**:如「讲讲数据库」。query 能独立读懂,但作为检索词太泛,top_k 命中很杂。
   工具层可以判定并收窄。

## 目标

- 在工具内加 `judge_query` 步骤:宽泛 query 自动改写,直到明确才进入检索,最多 2 轮。
- 在 Agent 层补强 system_prompt:调用工具前先把指代词改写成自包含 query。

## 非目标(YAGNI)

- 不做交互式反问用户(judge 不明确时不返回「请补充」,而是自动改写)。
- 不把会话历史传进 workflow(保持工具无状态;指代问题由 Agent 层 prompt 解决)。
- 不改工具名 / 工具签名 / 装配层代码。

## 架构

两个独立改动点:

| 改动点 | 位置 | 解决 |
|---|---|---|
| A. 指代改写 | `core/agent/agent.py` 的 `BOOK_SYSTEM_PROMPT` 追加一条规则 | 指代/缺上下文 |
| B. judge_query workflow | 新建 `core/workflow/book_rag.py`;`book_tools.py` 委托 workflow | 宽泛 query |

### A. system_prompt 补强

在 `BOOK_SYSTEM_PROMPT` 追加(放在「回答规则」区):

> 调用 book_search 前:若用户问题含指代词(它/这个/上面说的/前面提到的),
> 先根据会话历史把 query 改写为不依赖上文、能独立成立的句子,再传参。

依赖 Agent 层 LLM 执行,不保证 100%,但零架构改动、最便宜见效。

### B. BookRagWorkflow

`core/workflow/book_rag.py`,基于 LlamaIndex `Workflow` + 自定义事件实现循环:

```
StartEvent(query, book_title)
      ↓
  [start]  → 发出 JudgeEvent(query, round=0)
      ↓
  [judge]  ── LLM 判定 query 是否够明确 ──┐
      │  不明确 且 round < 2:             │
      │    改写 query → JudgeEvent(round+1)┘  (自循环)
      │  明确 或 round 达 2:
      ↓ RetrieveEvent(query, book_title)
  [retrieve]  → index.as_retriever(top_k, filters).aretrieve(query)
      ↓ SynthesizeEvent(query, nodes)
  [synthesize]  → get_response_synthesizer(llm).asynthesize(query, nodes)
      ↓
  StopEvent(result=Response)   # Response.source_nodes 供工具层取出
```

**事件定义**(`llama_index.core.workflow.Event` 子类):
- `JudgeEvent(query: str, round: int)`
- `RetrieveEvent(query: str, book_title: str | None)`
- `SynthesizeEvent(query: str, nodes: list[NodeWithScore])`

round 计数随事件携带,不用 Context 存状态。

**judge step 逻辑**:
- 用注入的同一个 `llm` 判定,要求返回结构化结果 `{clear: bool, rewritten_query: str}`(JSON)。
- judge prompt 约束:**只在原 query 语义范围内收窄,禁止新增用户没提到的约束**(压住「脑补答非所问」风险)。
- 解析失败 → 当作 `clear=True`,用当前 query 进入检索(防御性 fallback,绝不阻塞)。

**MAX_ROUNDS = 2**:judge → 不明确则改写 → 再 judge,最多改 2 次;达上限用最后一版 query 检索。

### 工具层改造

`create_book_search_tool` 工厂名、`book_search(query, book_title)` 签名**保持不变**。
内部从内联检索改为:

```python
workflow = BookRagWorkflow(index_manager=index_manager, llm=llm,
                           similarity_top_k=similarity_top_k)
result = await workflow.run(query=query, book_title=book_title)
add_sources([node_to_source_ref(n) for n in result.source_nodes])
return str(result)
```

`api/main.py` 与根 `main.py` 的装配代码**完全不动**。

空库 / 空检索的早返回文案沿用现状。

## 错误处理

| 情况 | 处理 |
|---|---|
| judge LLM 返回非法/解析失败 | 当作明确,用当前 query 直接检索,不阻塞 |
| 改写脱缰 | judge prompt 约束「只收窄、禁止脑补」 |
| round 达 2 仍不明确 | 用最后一版 query 检索 |
| query 非字符串(LLM 偶返 dict) | 沿用现有防御:取 title/text 字段或 str() |
| 检索为空 | 沿用现状,返回「没检索到」文案 |
| 知识库空(index is None) | 沿用现状,返回「知识库为空」文案 |

## 测试(TDD)

`tests/test_book_rag_workflow.py`(mock LLM 控制 judge 输出):

1. 明确 query → judge 直接判明确 → 不改写 → 用原 query 检索
2. 宽泛 query → judge 第 1 轮不明确改写 → 第 2 轮判明确 → 用改写后 query 检索
3. 一直宽泛 → 2 轮封顶 → 用最后一版 query 检索(验证不无限循环)
4. judge 输出非法 JSON → fallback 当明确,用当前 query 检索
5. 检索结果的 source_nodes 正确传到 `add_sources`
6. 空检索 → 返回「没检索到」文案

`tests/` 现有 book_tools 相关测试保持通过(工具签名未变)。

## 影响面

- 新增:`core/workflow/book_rag.py`、`tests/test_book_rag_workflow.py`
- 修改:`core/tools/book_tools.py`(book_search 内部)、`core/agent/agent.py`(system_prompt)
- 不变:`api/main.py`、`main.py`、前端、DB、source_context 钩子

---

# 增补:用户手动查询范围(硬 scope 通道)

**日期:** 2026-06-07
**状态:** 已实现

> 说明:上文初版的 judge(2 步、宽泛则「收窄改写」)已演进——现为 3 步(规范化 / 降噪 / 判定明确性)
> + 五类 category(`clear / too_broad / missing_info / ambiguous / unclear`),宽泛不再盲目收窄而是
> 分流(详见 `book_rag.py` 的 `_JUDGE_PROMPT`)。本增补只记录「查询范围」这一独立特性。

## 背景:旧 book_title 只是软提示,不是硬过滤

前端选书原走 `chat.py:_wrap_message`,把书名拼成「（请在《X》中查找）」塞进消息文本,
是给 Agent 的**提示**。后续是否过滤完全由 LLM 决定(它填不填 `book_search(book_title=…)`)。
即 **LLM 可以无视用户的选择全库搜**——「选书」并不能真正确定范围。

## 决策:新开一条 LLM 碰不到的请求级硬 scope 通道

用户手动选择 = 人下的硬约束,**不能被 LLM 推翻**。故不复用软提示路径,新增请求级通道
(复用 `source_context` 的 contextvar 模式,与 sources / clarify 预算同源)。

**权威性规则:** `用户选定 scope > LLM 的 book_title > 全库`。
v1 取最简:选了就权威,LLM 的 book_title 在范围内忽略,不让两通道暗中相互作用。

**软提示去留:** 已选范围 → 不拼软提示(范围由 scope 硬过滤);未选 → 保留单本 `book_title` 软提示。

## 数据流

```
前端选书 → req.book_titles → chat handler: set_scope(req.book_titles)   # 请求级，权威
                                              ↓
book_search: get_scope() 优先 > LLM book_title > None
                                              ↓
workflow.run(book_titles=…) → _make_filters(FilterOperator.IN) → 硬过滤检索
未选 → scope=None → 回落现有 category/clarify/too_broad 逻辑
```

scope 在父(handler)`set`、子任务(工具)只 `get`,纯读继承,无跨任务写回问题
(对比 clarify 预算需「原地改容器」是因为它在子任务里写)。

## 影响面(本增补)

| 文件 | 改动 |
|---|---|
| `core/agent/source_context.py` | `_scope_books` contextvar + `set_scope/get_scope`;`begin_collection` 一并重置 |
| `api/schemas.py` | `ChatRequest.book_titles: Optional[List[str]]`(硬范围多选);保留 `book_title`(软提示) |
| `api/routers/chat.py` | `_wrap_message` 按是否选范围决定拼不拼软提示;两处 handler 注入 `set_scope` |
| `core/workflow/book_rag.py` | `book_title`(单) → `book_titles`(多):事件字段 / `_make_filters`(IN) / `_retrieve_nodes` / start / preprocess |
| `core/tools/book_tools.py` | scope 权威性解析出 `effective_books`,传 `workflow.run(book_titles=…)` |

## 前向衔接:跨文档自动路由

本特性是**跨文档自动路由的手动版**。后续上自动路由时,退化为「**用户没选时,路由器自动往同一个
scope 通道写值**」(`set_scope(自动判定的相关书)`)——通道复用,无需重构。设计上一脉相承。
