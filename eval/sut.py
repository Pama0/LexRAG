"""被测系统（SUT）抽象：协议 + BookRagWorkflow 适配器。

map_workflow_result 把 workflow 返回值（Response / ClarifyResult）归一成 RagOutput，
是纯函数便于单测；BookRagWorkflowSystem 负责实际运行与异常兜底。
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class RagOutput:
    response: str
    retrieved_contexts: list[str]
    outcome: str  # answered | clarify | split | empty | error


@runtime_checkable
class RagSystem(Protocol):
    async def answer(self, query: str) -> RagOutput: ...


def map_workflow_result(result, response_cls=None) -> RagOutput:
    """把 BookRagWorkflow.run() 的返回值映射为 RagOutput。

    response_cls 仅供测试注入伪 Response；生产默认用 llama-index Response。
    """
    if response_cls is None:
        from llama_index.core.base.response.schema import Response as response_cls  # noqa: N813

    # clarify / split 分支统一返回 ClarifyResult
    if result.__class__.__name__ == "ClarifyResult":
        return RagOutput(response="", retrieved_contexts=[], outcome="clarify")

    if isinstance(result, response_cls):
        text = (getattr(result, "response", None) or "").strip()
        nodes = getattr(result, "source_nodes", None) or []
        if not text or not nodes:
            return RagOutput(response=text, retrieved_contexts=[], outcome="empty")
        contexts = [n.node.get_content() for n in nodes]
        return RagOutput(response=text, retrieved_contexts=contexts, outcome="answered")

    return RagOutput(response=str(result), retrieved_contexts=[], outcome="empty")


class BookRagWorkflowSystem:
    """包装 core.workflow.book_rag.BookRagWorkflow，实现 RagSystem 协议。"""

    def __init__(self, index_manager, llm, similarity_top_k: int = 5, timeout: float = 120.0):
        from core.workflow.book_rag import BookRagWorkflow

        self._workflow = BookRagWorkflow(
            index_manager=index_manager,
            llm=llm,
            similarity_top_k=similarity_top_k,
            timeout=timeout,
        )

    async def answer(self, query: str) -> RagOutput:
        try:
            result = await self._workflow.run(query=query)
        except Exception as e:  # noqa: BLE001 — eval 需吞掉单条异常，记 error 不中断
            return RagOutput(response=f"{type(e).__name__}: {e}",
                             retrieved_contexts=[], outcome="error")
        return map_workflow_result(result)
