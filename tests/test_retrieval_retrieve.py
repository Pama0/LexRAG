"""core/retrieval/retrieve.py 单测：纯工具 + VectorRetriever + make_retriever。"""
import pytest

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilters

from core.retrieval.retrieve import (
    Retriever,
    VectorRetriever,
    build_book_filters,
    bm25_tokenize,
    rrf_fuse,
    make_retriever,
)


# ── build_book_filters ────────────────────────────────────────────────
def test_build_book_filters_empty_returns_none():
    assert build_book_filters(None) is None
    assert build_book_filters([]) is None


def test_build_book_filters_builds_in_filter():
    f = build_book_filters(["《A》", "《B》"])
    assert isinstance(f, MetadataFilters)
    assert f.filters[0].key == "book_title"
    assert f.filters[0].value == ["《A》", "《B》"]


# ── bm25_tokenize：清洗（小写 + 丢空白/纯标点）──────────────────────────
def test_bm25_tokenize_drops_whitespace_and_punct_and_lowercases():
    toks = bm25_tokenize("ACID 隔离性！！")
    assert " " not in toks
    assert "！！" not in toks and "！" not in toks
    assert "acid" in toks          # 小写
    # jieba 默认把「隔离性」切成「隔离」+「性」——对 BM25 无碍（query 与语料同序分词即匹配）
    assert "隔离" in toks
    # 清洗：结果里不含空白/纯标点 token
    assert all(t.strip() and not __import__("re").fullmatch(r"[\W_]+", t) for t in toks)


# ── rrf_fuse：按 node id 融合两列表，去重排序截断 ──────────────────────
def _nws(nid, text="x"):
    return NodeWithScore(node=TextNode(text=text, id_=nid), score=1.0)


def test_rrf_fuse_combines_and_dedups_and_truncates():
    a = [_nws("n1"), _nws("n2"), _nws("n3")]   # dense
    b = [_nws("n3"), _nws("n1")]               # sparse：n3 居首
    out = rrf_fuse([a, b], top_k=2)
    ids = [o.node.node_id for o in out]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert "n1" in ids


# ── VectorRetriever：等价当前 as_retriever 路径 ────────────────────────
class _FakeRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    async def aretrieve(self, query):
        return self._nodes


class _FakeIndex:
    def __init__(self, nodes):
        self._nodes = nodes
        self.last_kw = None

    def as_retriever(self, **kw):
        self.last_kw = kw
        return _FakeRetriever(self._nodes)


class _FakeIndexManager:
    def __init__(self, nodes):
        self._index = _FakeIndex(nodes)

    def get_index(self):
        return self._index


async def test_vector_retriever_uses_as_retriever_with_topk_and_filters():
    im = _FakeIndexManager([_nws("a"), _nws("b")])
    out = await VectorRetriever().retrieve(
        "q", index_manager=im, book_titles=["《A》"], top_k=3)
    assert [o.node.node_id for o in out] == ["a", "b"]
    assert im._index.last_kw["similarity_top_k"] == 3
    assert isinstance(im._index.last_kw["filters"], MetadataFilters)


# ── make_retriever ────────────────────────────────────────────────────
def test_make_retriever_vector_and_none():
    assert isinstance(make_retriever(None), VectorRetriever)
    assert isinstance(make_retriever("vector"), VectorRetriever)


def test_make_retriever_unknown_raises():
    with pytest.raises(ValueError):
        make_retriever("no-such")


def test_make_retriever_memoizes_by_name(monkeypatch):
    import core.retrieval.retrieve as mod
    monkeypatch.setattr(mod, "_INSTANCES", {})
    first = mod.make_retriever("vector")
    second = mod.make_retriever("vector")
    assert first is second
