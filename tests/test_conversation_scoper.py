"""ConversationScoper 单测：拼接 probe 文本 + 主导书判据 + 降级。"""
from llama_index.core.schema import NodeWithScore, TextNode

from core.workflow.conversation_scoper import ConversationScoper, ScopeDecision


class _Msg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeMemory:
    def __init__(self, msgs=None):
        self._msgs = list(msgs or [])

    def get(self):
        return self._msgs


class _FakeProbe:
    """按给定 book_title 序列造命中 node；记录最后一次 probe 文本。"""
    def __init__(self, books):
        self._books = books
        self.last_query = None

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        self.last_query = query
        return [
            NodeWithScore(node=TextNode(text="x", id_=str(i), metadata={"book_title": b}))
            for i, b in enumerate(self._books)
        ]


class _BoomProbe:
    async def retrieve(self, *a, **k):
        raise RuntimeError("probe down")


def test_probe_text_appends_recent_user_turns_only():
    sc = ConversationScoper(index_manager=None, probe_retriever=_FakeProbe([]), n_history_turns=2)
    mem = FakeMemory([
        _Msg("user", "讲讲openclaw"),
        _Msg("assistant", "……其中 gateway 是……"),
        _Msg("user", "讲一下A"),
    ])
    text = sc._probe_text("讲一下gateway", mem)
    assert "讲讲openclaw" in text                 # 历史用户问被拼进
    assert text.endswith("讲一下gateway")          # 本轮 clean_query 在末尾
    assert "其中 gateway 是" not in text           # assistant 内容被过滤


async def test_run_locks_single_dominant_book_and_uses_augmented_probe():
    probe = _FakeProbe(["openclaw"] * 6 + ["X"] * 2)
    sc = ConversationScoper(index_manager=None, probe_retriever=probe)
    d = await sc.run("讲一下gateway", None, FakeMemory([_Msg("user", "讲讲openclaw")]))
    assert d.effective_book_titles == ["openclaw"]
    assert "openclaw" in d.note
    assert "讲讲openclaw" in probe.last_query       # probe 文本带上了历史主体


async def test_run_locks_two_books_when_concept_spans():
    probe = _FakeProbe(["A"] * 4 + ["B"] * 3 + ["C"] * 1)
    sc = ConversationScoper(index_manager=None, probe_retriever=probe)
    d = await sc.run("q", None, FakeMemory())
    assert d.effective_book_titles == ["A", "B"]


async def test_run_no_narrow_when_diffuse():
    probe = _FakeProbe(["A"] * 3 + ["B"] * 3 + ["C"] * 2)
    sc = ConversationScoper(index_manager=None, probe_retriever=probe)
    d = await sc.run("q", None, FakeMemory())
    assert d.effective_book_titles is None
    assert d.note == ""


async def test_run_no_narrow_when_probe_empty():
    sc = ConversationScoper(index_manager=None, probe_retriever=_FakeProbe([]))
    d = await sc.run("q", None, FakeMemory())
    assert d.effective_book_titles is None


async def test_run_degrades_to_full_library_on_probe_error():
    sc = ConversationScoper(index_manager=None, probe_retriever=_BoomProbe())
    d = await sc.run("q", None, FakeMemory())
    assert d.effective_book_titles is None
    assert d.note == ""


async def test_run_noop_when_user_selected_books():
    probe = _FakeProbe(["A"] * 8)              # 若被咨询会收窄，但手选时不应 probe
    sc = ConversationScoper(index_manager=None, probe_retriever=probe)
    d = await sc.run("q", ["高性能MySQL"], FakeMemory())
    assert d.effective_book_titles == ["高性能MySQL"]
    assert d.note == ""
    assert probe.last_query is None
