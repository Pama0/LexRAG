# core/workflow/ —— Agent workflow 包

本包用于放置自定义 RAG workflow（LlamaIndex `Workflow`），再封装为 Agent 工具。
当前 book 业务的 `book_search` 走 `core/tools/book_tools.py` 内联检索；
如需更复杂的多步检索流程，按以下约定新增：

## 新增一个 workflow 并封装为 tool

1. 在本目录写 `book_rag.py`：

   ```python
   from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent

   class BookRagWorkflow(Workflow):
       def __init__(self, index_manager, llm, **kw):
           super().__init__(**kw)
           self.index_manager = index_manager
           self.llm = llm
       @step
       async def run_step(self, ev: StartEvent) -> StopEvent:
           ...  # 自定义检索 + 合成，返回带 source_nodes 的 Response
   ```

2. 在 `core/tools/book_tools.py` 加工厂：

   ```python
   from core.agent.source_context import add_sources, node_to_source_ref
   from core.workflow.book_rag import BookRagWorkflow

   def create_book_rag_tool(index_manager, llm) -> FunctionTool:
       workflow = BookRagWorkflow(index_manager=index_manager, llm=llm)
       async def book_rag_search(query: str, book_title: str | None = None) -> str:
           result = await workflow.run(query=query, book_title=book_title)
           add_sources([node_to_source_ref(n) for n in result.source_nodes])
           return str(result)
       return FunctionTool.from_defaults(
           fn=book_rag_search, name="book_rag", description="...")
   ```

3. 在 `api/main.py` 与根 `main.py` 的工具装配处，把 `create_book_rag_tool(index_manager, llm)`
   加入 `tools` 列表。

口子三要素：本包常驻 + `book_tools` 工厂约定 + `core.agent.source_context.add_sources` 公开钩子。
