# 设计稿：book 业务整合进 core，剥离法条遗留

- 日期：2026-05-26
- 状态：已批准（待转 writing-plans）
- 范围：项目结构重构（梳理项目），不含功能新增

## 1. 背景与问题

项目从"法律条文 RAG"演进为"技术书籍知识库助手（上传 PDF + 前端问答）"。
book 业务代码堆在 `api/`，与 `core/` 主项目脱节。核心病灶有三：

1. **依赖方向倒置（最严重）**：`core/tools/book_tools.py` 第 17 行
   `from api.source_context import ...` —— 领域层反向 import Web 层，导致 book
   领域工具被焊死在 Web 请求机制上，无法被主 agent / CLI / eval 独立复用。
2. **两套并存的 agent**：
   - `core/agent/agent.py`（`MyAgent` + 旧 `ReActAgent`，CLI，无流式）——被指定为"主 agent"但实际被架空。
   - `api/agent_service.py`（`AgentService` + `FunctionAgent`，流式，前端依赖）——book 业务真正在跑的。
3. **领域逻辑散落在 api/**：`AgentService`（agent 封装 + 会话锁 + memory 构造）、
   book `SYSTEM_PROMPT`、`source_context`（来源收集）都是领域逻辑却住在 Web 层。

法条相关代码（citation 引用链、ArticleSplitter、legal AutoRetriever 等）当前 book 业务
**一个都没用到**。

## 2. 已确认决策

| # | 决策 | 结论 |
|---|---|---|
| 1 | 主 agent 统一 | 把 `AgentService` 能力（FunctionAgent + 会话锁 + memory 构造 + book SYSTEM_PROMPT）合并进 `core/agent/agent.py`；保留 CLI `chat()`；`ReActAgent` 退场 |
| 2 | 持久化归属 | SQLite 持久化层移进 core 新建子模块 `core/persistence/` |
| 3 | source_context 去耦 | 走最小方案：`source_context` 整体搬进 core（contextvar 机制不变），`book_tools` 改 import core；彻底重构（工具直接 return sources）留作以后 |
| 4 | 法条代码处理 | 移到 `legacy/` 目录冻结（不删，便于将来重启法条业务） |
| 5 | workflow→tool 扩展口子 | 保留 `core/workflow/` 包与工厂约定，供用户后续自写 book workflow 并封装为 tool（详见 §5） |

## 3. 目标架构

### 分层原则
依赖方向单向：`api/`（Web 适配）→ `core/`（领域）→ `configs/`（基础设施）。
**`core` 不再 import 任何 `api`。**

### 目录布局
```
core/                         ← book 领域，全部业务逻辑
  agent/
    agent.py                  ★重建：统一主 agent（FunctionAgent + 会话锁 + memory 构造
                                 + book SYSTEM_PROMPT + CLI chat()）；吸收 AgentService
    source_context.py         ←从 api/ 搬入；SourceRef 值对象也定义在此
  tools/
    book_tools.py             改 import → core.agent.source_context（去耦）
  rag/
    data_loader.py            ★瘦身：RAGIndexManager 仅 book（默认 book_knowledge，
                                 删 ArticleSplitter/CitationGraph/add_documents）
    pdf_parser.py             BookPDFParser（保留）
  persistence/                ★新建子模块
    __init__.py
    db.py                     ←从 api/ 搬入
    repositories.py           ←从 api/ 搬入
  workflow/                   保留为待命包（见 §5）；仅法条 workflow 迁出
    __init__.py
    README.md                 ★新增：workflow→tool 接线约定

api/                          ← 瘦成 HTTP 适配层
  main.py                     组装 core 组件 + 注册路由（基本不变）
  schemas.py                  Pydantic DTO；SourceRef 改为从 core 导入
  routers/{chat,documents,sessions}.py   调用 core（更新 import）
  agent_service.py            ✗删除（并入 core/agent/agent.py）
  db.py, repositories.py, source_context.py   ✗迁出

main.py (根)                   ★改为 book CLI 启动器（组装 book 工具 + core agent + chat）

legacy/                       ★新建：冻结法条相关，主干不再引用
  app.py, init_index.py
  tools.py                    (原 core/tools/tools.py)
  workflow/                   simple_rag, citation_rag, multi_strategy_rag,
                              query_engine_workflow, eval_workflow/
  rag/                        citation_extractor, citation_graph, parser,
                              vector_store_info, auto_retriever_prompt, data_loo, indexer
  evals/                      (整套评测)
  citation_graph.json
```

> `data/legal_documents/` 与 chroma 的 `documents` 集合（26889 块）是**数据非代码**，
> 留在原地不影响运行；清理走 `manage_chroma.py`，不在本次范围。

## 4. 模块去向清单

**保留在 core（book 主干）**：`core/agent/agent.py`(重建)、`core/agent/source_context.py`(迁入)、
`core/tools/book_tools.py`、`core/rag/data_loader.py`(瘦身)、`core/rag/pdf_parser.py`、
`core/persistence/*`(迁入)、`core/workflow/`(待命包)、`configs/*`。

**迁入 core 的（来自 api）**：`source_context.py`+`SourceRef` → `core/agent/`；
`db.py`、`repositories.py` → `core/persistence/`。

**移入 legacy/**：根 `app.py`、`init_index.py`；`core/tools/tools.py`；
`core/workflow/{simple_rag,citation_rag,multi_strategy_rag,query_engine_workflow,eval_workflow}`；
`core/rag/{citation_extractor,citation_graph,parser,vector_store_info,auto_retriever_prompt,data_loo,indexer}`；
`evals/`；`citation_graph.json`。

**删除**：`api/agent_service.py`（能力并入 core agent）。

## 5. workflow → tool 扩展口子（为后续自写 workflow 预留）

沿用 legacy `create_simple_rag_tool(SimpleRagWorkflow)` 的封装模式。约定：

1. 用户新写 `core/workflow/book_rag.py`：`class BookRagWorkflow(Workflow)` 自定义检索/合成流程。
2. 在 `core/tools/book_tools.py` 加工厂：
   ```python
   def create_book_rag_tool(index_manager, llm) -> FunctionTool:
       workflow = BookRagWorkflow(index_manager=index_manager, llm=llm)
       async def book_rag_search(query: str, book_title: str | None = None) -> str:
           result = await workflow.run(query=query, book_title=book_title)
           # 合成阶段把 source nodes 回传前端：
           add_sources([node_to_source_ref(n) for n in result.source_nodes])
           return str(result)
       return FunctionTool.from_defaults(fn=book_rag_search, name="book_rag", description="...")
   ```
3. 在 `api/main.py`（及根 `main.py`）的工具装配处，把 `create_book_rag_tool(...)` 加入 `tools` 列表。

口子三要素：保留 `core/workflow/` 包 + `book_tools` 工厂约定 + `core.agent.source_context.add_sources`
作为 core 内公开钩子。`core/workflow/README.md` 记录上述约定。

## 6. 迁移分阶段（每阶段结束保证 `api/main.py` 可启动）

| 阶段 | 内容 | 顺序理由 |
|---|---|---|
| 1 | `data_loader.py` 瘦身（去 ArticleSplitter/CitationGraph/add_documents/_update_citation_graph/_fetch_all_nodes，默认集合→book_knowledge） | 先切断 data_loader 对法条 rag 模块的 import，才能安全移走它们 |
| 2 | `source_context.py`+`SourceRef` 搬进 `core/agent/`；rewire `book_tools.py`；`api/schemas` 改从 core 导入 SourceRef | 消除 core→api 反向依赖（病灶） |
| 3 | 统一 agent：新 `core/agent/agent.py` 吸收 `AgentService`；删 `api/agent_service.py`；`routers/chat.py`、`sessions.py` 改用 core agent | 主 agent 落位 |
| 4 | `db.py`+`repositories.py` 迁 `core/persistence/`；修 `PROJECT_ROOT` 路径深度；更新各处 import | 持久化进 core |
| 5 | 法条代码移入 `legacy/`；保留 `core/workflow/` 包 + 新增 README；修残留引用 | core 主干清干净 |
| 6 | 根 `main.py`→book CLI；更新 `CLAUDE.md`（去掉法条/run_eval，写新跑法） | 入口与文档对齐 |

## 7. 关键风险与注意点

1. **SourceRef 位置**：现位于 `api/schemas.py`，被即将进 core 的 `source_context` 依赖。
   故 `SourceRef` 作为领域值对象移到 core，`api/schemas.py` 反过来 import 它（符合 api→core）。
   `repositories.py`（也迁入 core）同样从 core 取 `SourceRef`。
2. **`bookkb.db` 路径**：`db.py` 现靠 `dirname(dirname(...))` 算项目根；搬到 `core/persistence/`
   后层级深一层，须改为往上三级，否则 DB 落错位置、丢失现有会话历史。
3. **`build_memory` 解耦**：core agent 的 `build_memory` 接收 DB 行，采用鸭子类型读 `.role`/`.content`，
   避免 `core/agent` 反向 import `core/persistence`（同层依赖可接受，但鸭子类型更松）。
4. **每阶段冒烟**：阶段末执行 `python -c "import api.main"`（或启动 uvicorn）确认可加载。

## 8. 明确不在本次范围
- 评测（eval）对 book 的适配 —— 已暂缓。
- 法条数据 / chroma `documents` 集合的物理清理。
- `source_context` 的彻底重构（工具直接 return sources）。
- 任何 book 功能新增（含 book workflow 本身，由用户后续自写）。
