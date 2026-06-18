from eval.harness.sut import map_doc_result, RagOutput


class _Node:
    def __init__(self, text): self._t = text
    def get_content(self): return self._t


class _NodeWithScore:
    def __init__(self, text): self.node = _Node(text)


# ── map_doc_result：当前 DocQueryWorkflow 的 Response（带 metadata.category）──
class _RespMeta:
    """模拟带 metadata 的 DocQueryWorkflow Response。"""
    def __init__(self, response, source_nodes, metadata):
        self.response = response
        self.source_nodes = source_nodes
        self.metadata = metadata


def test_doc_answered_with_category():
    r = _RespMeta("答案", [_NodeWithScore("片段")], {"category": "retrievable", "intent": "qa"})
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "answered"
    assert out.category == "retrievable"
    assert out.retrieved_contexts == ["片段"]


def test_doc_empty_when_no_nodes():
    r = _RespMeta("反问句", [], {"category": "missing_info", "intent": "qa"})
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "empty"
    assert out.category == "missing_info"


def test_doc_handles_missing_metadata():
    r = _RespMeta("答案", [_NodeWithScore("片段")], None)
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "answered"
    assert out.category == ""   # 无 metadata → category 空，不报错


# ── map_agent_result：自主 Agent 的 (answer, sources) ──
from eval.harness.sut import map_agent_result


def test_agent_answered_with_sources():
    out = map_agent_result("综合答案", [_Node("片段A"), _Node("片段B")])
    assert out.outcome == "answered"
    assert out.response == "综合答案"
    assert out.retrieved_contexts == ["片段A", "片段B"]
    assert out.category == ""          # agent 不产分类


def test_agent_empty_when_no_sources():
    out = map_agent_result("答案", [])
    assert out.outcome == "empty"
    assert out.category == ""


def test_agent_empty_when_blank_answer():
    out = map_agent_result("   ", [_Node("片段A")])
    assert out.outcome == "empty"
