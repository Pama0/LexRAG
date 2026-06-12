# Intent Router 引入 + step1 预处理拆分设计

**日期:** 2026-06-11
**状态:** step1 拆分 + 装配层切到 DocQueryWorkflow 已落地（2026-06-11，TDD）；book_rag/book_tools 退役待办
**关联:** [项目架构](../../ARCHITECTURE.md) · 取代 [2026-06-04 judge-query-workflow](2026-06-04-judge-query-workflow-design.md) 的单层 step1 设计

## 进度（2026-06-11）

step1 拆分已按本 spec 实现（TDD）：
- ✅ `core/workflow/intent_router.py`（新建）：消指代 + 规范化 → `clean_query`，意图分类 `qa | study_plan`（taxonomy v1，study_plan 为占位）。持有 `format_history` + `MAX_HISTORY_MSGS`。降级：解析失败 → `qa` + 原 query。
- ✅ `core/workflow/query_preprocess.py`（瘦身）：去掉规范化/指代/意图，只留降噪 + 难度分类；`run(clean_query)` 不再收 memory、不再消指代。
- ✅ `core/workflow/doc_workflow.py`（接线）：`start → route(Router) → preprocess(QA 降噪+难度) → 分支`；`route` 按 intent dispatch，`study_plan` 走占位分支。`clean_query`/`intent` 走 `ctx.store`。
- 测试：`tests/test_intent_router.py`、`tests/test_query_preprocess.py`、`tests/test_doc_workflow.py`（共 19 个）。

**装配层切换（2026-06-11 续）：** 顶层由 agent+工具改为 `DocQueryWorkflow`。
- ✅ `core/workflow/doc_workflow.py`：QA 分支改【直接检索 + 流式合成】（绕开 agent/book_tools），`source_nodes` 随结果带出；检索/合成进度通过 `write_event_to_stream` 推流（`RetrievalStart/RetrievalDone/AnswerDelta`）。
- ✅ `core/workflow/doc_query_service.py`（新建）：装配服务（会话锁 + `build_memory` + 每请求起 workflow），取代 `BookAgent` 在装配层的位置。
- ✅ `api/routers/chat.py`：`/chat`、`/chat/stream` 改驱动 `DocQueryService`；`_format_event` 把 workflow 流式事件映射成前端已有 SSE 词汇（`tool_call/tool_result/delta`），**前端零改动**；source 从 `result.source_nodes` 直接转换去重（不再走 `source_context` contextvar）。
- ✅ `api/routers/sessions.py`、`api/main.py`、`main.py`：注入 `DocQueryService`。
- 测试：`tests/test_doc_query_service.py`、`tests/test_chat_router.py`；doc_workflow 测试扩充到直接检索+流式。新链路单测合计 34 个绿；全量 `pytest --continue-on-collection-errors` → 59 passed。

**未做 / 已知问题：**
- `book_rag.py` 仍坏（`assume` 空体语法错）、`book_tools.py` 仍 import 它 → `tests/test_book_rag_workflow.py`、`tests/test_book_search_tool.py` 采集失败（2 errors），且会阻断不带 `--continue-on-collection-errors` 的 `pytest`。按约定本次不动，**退役/修复 book_rag + book_tools 是下一步**。
- 真实端到端（DeepSeek + chroma）未离线验证：token 流式用 `AsyncStreamingResponse.async_response_gen()`（已确认是 async 生成器），但未实跑；建议起服务跑一轮确认流式与 source 回传。
- `core/agent/agent.py`（BookAgent）现已无人引用，留存未删。

## 背景

当前 `core/workflow/query_preprocess.py` 的 `QueryPreprocessor`（step1）在**一次 LLM 调用**里打包了四件事：
1. **规范化**（纠错别字、统一全半角、展无歧义缩写）
2. **指代消解**（它/这个/上面说的 → 自包含问句，读会话历史）
3. **降噪**（去口语/礼貌词，留检索关键词）
4. **难度分类**（`retrievable / pending_split / missing_info / ambiguous / other`）

项目要升级为「Intent Router + 多 capability」架构（见 [架构文档](../../ARCHITECTURE.md)）。届时 step1 升维成 **意图分类 router**（qa / study_plan / life_plan / …）。问题随之而来：**这四件事不该再绑在一起——它们的"受众"不同。**

## 决策：沿"谁需要这一步的产物"把 step1 劈开

判据：**该步骤的产出，是不是所有 capability 都要。**

| 现有子步骤 | 谁需要 | 拆分后归属 |
|---|---|---|
| 规范化 | 所有 capability（干净 query 谁都受益） | **Router 门口（通用净化）** |
| 指代消解 | 所有 capability（life_plan 也有"把它用到我人生里"） | **Router 门口（通用净化）** |
| 降噪（检索向） | 仅 RAG —— life_plan 是推理不是检索 | **QA capability 内部** |
| 难度分类 | 仅 QA 内部路由 | **QA capability 内部** |

一句话：**"产出一个干净、自包含、规范的 query"是横切的 → 上提到 Router 门口；"为检索而降噪 + QA 内部难度路由"是检索专属 → 留在 QA。**

## 目标结构

```
门口（Router，所有请求都过）:
  规范化 + 指代消解  →  clean_query（自包含）
        ↓
  意图分类（在 clean_query 上）→ qa / study_plan / life_plan / ...
        ↓ 确定性 dispatch
┌────────────────────────────────────────────┐
│ QA capability (doc_workflow)                 │
│   内部 preprocess: 降噪 + 难度分类            │  ← 现有逻辑原地保留
│   → retrievable/split/missing_info/ambiguous │
├────────────────────────────────────────────┤
│ life_plan capability (agent，后续)            │
│   拿 clean_query，自己抽目标/约束，不做检索降噪  │
└────────────────────────────────────────────┘
```

**RAG 预处理没有消失**，它降级为「QA capability 内部的第一步」。Router 是新长出的、更薄的上层。

## 文件改动（拟）

| 文件 | 改动 |
|---|---|
| `core/workflow/intent_router.py`（**新建**） | 规范化 + 指代消解 + 意图分类。产出 `{intent, clean_query}`。持有 `format_history`（从 query_preprocess 迁入）。 |
| `core/workflow/query_preprocess.py`（**瘦身**） | 去掉规范化/指代/意图，只留**降噪 + 难度分类**，输入改为接收 `clean_query`（不再自己消指代）。`QueryJudgment` 的 prompt 同步删掉前两步。 |
| `core/workflow/doc_workflow.py` | `start` 之前先过 Router；`preprocess` step 接收 `clean_query`。Router 决定 dispatch 到哪个 capability（v1 只有 QA）。 |
| `core/workflow/book_rag.py` | 退役（逻辑已被取代），不再维护。 |

> 具体 import / 函数签名在开发时按 TDD 定；本 spec 只锁结构与切缝。

## 约束（沿用既有铁律）

1. **指代只消一次**：门口消解后产出 `clean_query`；QA preprocess 与 life_plan agent **都收 `clean_query`，不准再消**。
2. **意图分类在指代消解之后**：顺序固定 `规范化 → 指代 → 意图`（"它怎么帮我做职业选择"须先解"它"才能判 qa vs life_plan）。
3. **两层记忆**：会话记忆只存「用户原话 + 最终答案」，供门口指代消解读取；改写/分类/中间产物只走 `Context`，不入会话记忆。
4. **绝不阻塞**：Router 任一步 LLM 解析失败 → 降级（意图默认 qa、clean_query 用原 query），沿用 `QueryPreprocessor` 的 fallback 原则。
5. **结构化输出**：沿用 json_object + Pydantic 校验（DeepSeek 稳定端点；不依赖 strict schema）。

## 成本 / 性能

- 一条 QA 请求在检索前会有 **2 次 LLM 调用**：门口（规范化+指代+意图）+ QA 内（降噪+难度）。
- **可选优化（暂不做）**：门口在 `intent=qa` 时顺带吐出难度分类，省一次调用——但会让 Router prompt 耦合 QA 内部细节。**先保持两步分离**（模块清晰、各自可测），真测出延迟瓶颈再合并。

## 非目标（本次）

- 不实现 study_plan / life_plan capability（仅留 Router 扩展位）。
- 不建用户长期记忆系统（仅复用现有会话记忆）。
- 不改装配层（`api/main.py`、`main.py`）的对外接口。

## 开发启动前待定（明天确认）

1. **单实例 agent vs 按分支配 agent**：QA 内若仍要"不同分支不同工具子集"，是给一个 agent 全套工具（软路由）还是每分支换工具集？（影响 `_run_agent`）
2. **Router 产物载体**：`{intent, clean_query}` 通过 `StartEvent` 字段还是 `Context` 传入 capability？
3. **意图分类的 taxonomy v1**：先开几类（建议 `qa` + 一个占位，避免过度设计）。
4. **`book_rag.py` 退役方式**：直接删，还是移 `legacy/` 留参考。

## 测试（TDD，开发时展开）

- Router：规范化/指代/意图分类各自的单测（mock LLM）；指代消解读历史；解析失败降级。
- QA preprocess（瘦身后）：接收 clean_query，不再消指代；降噪 + 五类难度分类；fallback。
- 接线：Router → QA capability 的 dispatch；clean_query 不被二次消解。
