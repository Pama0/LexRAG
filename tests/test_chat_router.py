"""chat 路由纯函数：workflow 流式事件 → 前端 SSE payload 映射；source 转换去重。

把 DocQueryWorkflow 的检索/合成进度映射到前端已有的状态机：
RetrievalStart→tool_call、RetrievalDone→tool_result、AnswerDelta→delta，前端零改动。
"""
from api.routers.chat import _format_event, _nodes_to_sources
from core.workflow.doc_workflow import (
    AnswerDeltaEvent,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)


def test_format_retrieval_start_maps_to_tool_call():
    payload = _format_event(RetrievalStartEvent(query="B+树索引"))
    assert payload == {
        "type": "tool_call",
        "tool_name": "book_search",
        "tool_kwargs": {"query": "B+树索引"},
    }


def test_format_retrieval_done_maps_to_tool_result():
    payload = _format_event(RetrievalDoneEvent(count=3))
    assert payload["type"] == "tool_result"
    assert "3" in payload["preview"]


def test_format_answer_delta_maps_to_delta():
    assert _format_event(AnswerDeltaEvent(delta="片段")) == {"type": "delta", "data": "片段"}


def test_format_unknown_event_returns_none():
    assert _format_event(object()) is None


class _FakeNode:
    def __init__(self, title, page, text):
        self.metadata = {"book_title": title, "page": page}
        self._t = text

    def get_content(self):
        return self._t


def test_nodes_to_sources_converts_and_dedups():
    n1 = _FakeNode("高性能MySQL", 10, "B+树是一种平衡多路查找树")
    dup = _FakeNode("高性能MySQL", 10, "B+树是一种平衡多路查找树")
    n2 = _FakeNode("Redis设计与实现", 22, "跳表用于有序集合")

    sources = _nodes_to_sources([n1, dup, n2])
    assert len(sources) == 2                       # 完全相同的去重
    titles = {s.book_title for s in sources}
    assert titles == {"高性能MySQL", "Redis设计与实现"}


def test_nodes_to_sources_empty_returns_empty():
    assert _nodes_to_sources([]) == []
