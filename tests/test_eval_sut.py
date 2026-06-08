from eval.sut import map_workflow_result, RagOutput


class _Node:
    def __init__(self, text): self._t = text
    def get_content(self): return self._t


class _NodeWithScore:
    def __init__(self, text): self.node = _Node(text)


class _Response:
    """模拟 llama-index Response。"""
    def __init__(self, response, source_nodes):
        self.response = response
        self.source_nodes = source_nodes


class _ClarifyResult:
    """类名须为 ClarifyResult 以触发分流分支。"""
    def __init__(self, query, clarify_reason):
        self.query = query
        self.clarify_reason = clarify_reason


# 让伪类的类名匹配映射逻辑
_ClarifyResult.__name__ = "ClarifyResult"


def test_answered_extracts_text_and_contexts():
    resp = _Response("MVCC 通过 undo log 实现", [_NodeWithScore("片段A"), _NodeWithScore("片段B")])
    out = map_workflow_result(resp, response_cls=_Response)
    assert out.outcome == "answered"
    assert out.response == "MVCC 通过 undo log 实现"
    assert out.retrieved_contexts == ["片段A", "片段B"]


def test_empty_when_no_nodes():
    resp = _Response("", [])
    out = map_workflow_result(resp, response_cls=_Response)
    assert out.outcome == "empty"
    assert out.retrieved_contexts == []


def test_clarify_branch():
    cr = _ClarifyResult("这个索引", "指代不明")
    out = map_workflow_result(cr, response_cls=_Response)
    assert out.outcome == "clarify"
    assert out.response == ""
