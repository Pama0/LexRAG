import pytest

import core.tools.book_tools as book_tools
from core.tools.book_tools import create_book_search_tool


class _FakeNode:
    pass


class FakeResponse:
    def __init__(self, text, source_nodes):
        self._t = text
        self.source_nodes = source_nodes
    def __str__(self):
        return self._t


class FakeIndexManager:
    """get_index 非 None 表示库非空。"""
    def __init__(self, index=object()):
        self._index = index
    def get_index(self):
        return self._index


async def test_book_search_propagates_sources(monkeypatch):
    captured = {}
    monkeypatch.setattr(book_tools, "add_sources", lambda refs: captured.setdefault("refs", refs))
    monkeypatch.setattr(book_tools, "node_to_source_ref", lambda n: f"ref:{id(n)}")

    nodes = [_FakeNode(), _FakeNode()]

    class StubWorkflow:
        def __init__(self, **kw):
            pass
        async def run(self, **kw):
            return FakeResponse("合成答案", source_nodes=nodes)

    monkeypatch.setattr(book_tools, "BookRagWorkflow", StubWorkflow)

    tool = create_book_search_tool(FakeIndexManager(), llm=object())
    result = await tool.async_fn(query="B+树的索引结构")

    assert result == "合成答案"
    assert len(captured["refs"]) == 2


async def test_book_search_empty_index_returns_hint():
    tool = create_book_search_tool(
        FakeIndexManager(index=None), llm=object()
    )
    result = await tool.async_fn(query="B+树")
    assert "知识库为空" in result


async def test_book_search_no_nodes_returns_scope_hint(monkeypatch):
    monkeypatch.setattr(book_tools, "add_sources", lambda refs: None)

    class StubWorkflow:
        def __init__(self, **kw):
            pass
        async def run(self, **kw):
            return FakeResponse("", source_nodes=[])

    monkeypatch.setattr(book_tools, "BookRagWorkflow", StubWorkflow)

    tool = create_book_search_tool(FakeIndexManager(), llm=object())
    result = await tool.async_fn(query="不存在的内容")
    assert "没有检索到" in result
    assert "知识库" in result
