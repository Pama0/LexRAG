"""被测系统（SUT）抽象：协议 + DocQueryWorkflow 适配器。

map_doc_result 把 DocQueryWorkflow.run() 的返回值归一成 RagOutput，是纯函数便于单测；
DocQueryWorkflowSystem 负责实际运行与异常兜底（按决策 flag 构造，供 ablation 用）。
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RagOutput:
    response: str
    retrieved_contexts: list[str]
    outcome: str  # answered | empty | error


@runtime_checkable
class RagSystem(Protocol):
    async def answer(self, query: str) -> RagOutput: ...


# ── 当前系统（DocQueryWorkflow）适配器 ──────────────────────────────
def map_doc_result(result, response_cls=None) -> RagOutput:
    """DocQueryWorkflow.run() 的 Response → RagOutput（有 nodes→answered，无→empty）。"""
    if response_cls is None:
        from llama_index.core.base.response.schema import Response as response_cls  # noqa: N813
    if isinstance(result, response_cls):
        text = (getattr(result, "response", None) or "").strip()
        nodes = getattr(result, "source_nodes", None) or []
        if not text or not nodes:
            return RagOutput(text, [], "empty")
        contexts = [n.node.get_content() for n in nodes]
        return RagOutput(text, contexts, "answered")
    return RagOutput(str(result), [], "empty")


def _node_text(n) -> str:
    """从 NodeWithScore / Node 取正文（镜像 BookSearchTool 的提取逻辑）。"""
    return n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")


def map_agent_result(answer: str, sources: list) -> RagOutput:
    """AutoAgent.run() 的 (answer, source_nodes) → RagOutput。"""
    text = (answer or "").strip()
    if not text or not sources:
        return RagOutput(text, [], "empty")
    contexts = [_node_text(n) for n in sources]
    return RagOutput(text, contexts, "answered")


class DocQueryWorkflowSystem:
    """包装当前 DocQueryWorkflow，按决策 flag 构造，实现 RagSystem（评测 ablation 用）。"""

    def __init__(self, index_manager, llm, flags: dict | None = None,
                 similarity_top_k: int = 5, timeout: float = 120.0):
        self._index_manager = index_manager
        self._llm = llm
        self._flags = flags or {}
        self._similarity_top_k = similarity_top_k
        self._timeout = timeout

    async def answer(self, query: str, book_titles=None) -> RagOutput:
        from core.workflow.doc_workflow import DocQueryWorkflow

        wf = DocQueryWorkflow(
            index_manager=self._index_manager, llm=self._llm,
            similarity_top_k=self._similarity_top_k, timeout=self._timeout,
            retriever="hybrid",
            **self._flags,
        )
        try:
            result = await wf.run(query=query, book_titles=book_titles)
        except Exception as e:  # noqa: BLE001 — 单条异常记 error 不中断
            return RagOutput(f"{type(e).__name__}: {e}", [], "error")
        return map_doc_result(result)


# ── agent 自主规划路线（AutoAgent，绕过 DocQueryWorkflow 决策路由）──────
class _NullCtx:
    """AutoAgent.run 需要带 write_event_to_stream 的 ctx 推前端流式事件；
    评测无 workflow ctx，用 no-op 替身。最终答案来自 await handler，与 ctx 无关。"""

    def write_event_to_stream(self, event) -> None:  # noqa: D401 — no-op
        pass


class AgentSystem:
    """被测系统：每条 query 直接喂有界 AutoAgent 自主规划检索，实现 RagSystem。"""

    def __init__(self, index_manager, llm,
                 similarity_top_k: int = 5, max_iterations: int = 6):
        self._index_manager = index_manager
        self._llm = llm
        self._similarity_top_k = similarity_top_k
        self._max_iterations = max_iterations

    async def answer(self, query: str, book_titles=None) -> RagOutput:
        from core.agent.auto_agent import AutoAgent

        agent = AutoAgent(
            self._index_manager, self._llm,
            similarity_top_k=self._similarity_top_k,
            max_iterations=self._max_iterations,
        )
        try:
            answer, sources = await agent.run(_NullCtx(), query, book_titles)
        except Exception as e:  # noqa: BLE001 — 单条异常记 error 不中断
            return RagOutput(f"{type(e).__name__}: {e}", [], "error")
        return map_agent_result(answer, sources)


# ── 朴素 RAG 基线（对照组）───────────────────────────────────────────
NAIVE_PROMPT = """基于以下检索片段回答问题。只依据片段，片段没有的不要编造；
若片段与问题无关，如实说明知识库中暂无相关内容。

检索片段：
{context}

问题：{question}

中文作答，结构清晰。"""


class NaiveRagSystem:
    """朴素 RAG 基线：vector 检索 top-k → 拼 prompt → 单次 LLM 作答。

    作对照组：无路由 / 拆分 / agent / probe，凸显 workflow、agent 相对它的增益。
    LLM 调用走传入的 llm 实例（与其它 SUT 同一被挂表实例）→ token/时耗照常计入。
    """

    def __init__(self, index_manager, llm, similarity_top_k: int = 5):
        self._index_manager = index_manager
        self._llm = llm
        self._similarity_top_k = similarity_top_k

    async def answer(self, query: str, book_titles=None) -> RagOutput:
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        from core.retrieval.retrieve import make_retriever

        try:
            nodes = await make_retriever("vector").retrieve(
                query, index_manager=self._index_manager,
                book_titles=book_titles, top_k=self._similarity_top_k,
            )
            if not nodes:
                return RagOutput("", [], "empty")
            context = "\n---\n".join(_node_text(n) for n in nodes)
            prompt = NAIVE_PROMPT.format(context=context, question=query)
            resp = await self._llm.achat(
                [ChatMessage(role=MessageRole.USER, content=prompt)]
            )
            text = (resp.message.content or "").strip()
            contexts = [_node_text(n) for n in nodes]
            return RagOutput(text, contexts, "answered" if text else "empty")
        except Exception as e:  # noqa: BLE001 — 单条异常记 error 不中断
            return RagOutput(f"{type(e).__name__}: {e}", [], "error")
