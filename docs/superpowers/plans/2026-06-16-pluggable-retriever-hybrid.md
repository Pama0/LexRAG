# 可插拔 Retriever（Hybrid dense+BM25）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 RAG 检索加一个装配时注入的 `Retriever` 策略组件——不传/`"vector"`=基线（当前向量检索），`"hybrid"`=dense + BM25（jieba 分词、RRF 融合、scope 后过滤），并让 eval ablation 量化召回侧增益。

**Architecture:** 沿用 reranker 那条已验证链路：`make_retriever(name)` 工厂（按名 memoize）→ `DocQueryWorkflow` 解析注入 → `QaCapability` 持有 → `_retrieve_nodes` 委托。`Retriever` 是数据源（`retrieve(query, *, index_manager, book_titles, top_k)`），依赖 call 时传入，策略对象自身无依赖。默认 `VectorRetriever`，现有行为与测试零变化。

**Tech Stack:** Python 3.12 async、LlamaIndex（dense 检索 + `MetadataFilters`、`TextNode`/`NodeWithScore`）、`rank_bm25`（纯 Python BM25Okapi）+ `jieba`（中文分词），pytest。`rank_bm25`/`jieba` 已安装。

---

## 文件结构

| 文件 | 职责 | 动作 |
|------|------|------|
| `core/retrieval/retrieve.py` | `Retriever` 协议 + `build_book_filters`/`bm25_tokenize`/`rrf_fuse` 工具 + `VectorRetriever` + `HybridRetriever` + `make_retriever` | 创建 |
| `core/workflow/qa_capability.py` | `__init__` 加 retriever；`_retrieve_nodes` 委托策略；删 `_make_filters`（上移）+ 清未用 imports | 修改 |
| `core/workflow/doc_workflow.py` | `__init__` 加 `retriever: str\|None`，`make_retriever` 解析注入 | 修改 |
| `eval/harness/compare.py` | `VARIANTS` 加 hybrid 变体 | 修改 |
| `requirements.txt` | 加 `rank_bm25`、`jieba` | 修改 |
| `tests/test_retrieval_retrieve.py` | 工具 + 策略 + 工厂单测 | 创建 |
| `tests/test_qa_capability.py` | retriever 接入 + 基线不变 | 修改 |
| `tests/test_doc_workflow.py` | retriever 注入路径 | 修改 |

---

## Task 1: retrieve.py 基础——协议、纯工具、VectorRetriever、make_retriever

**Files:**
- Create: `core/retrieval/retrieve.py`
- Test: `tests/test_retrieval_retrieve.py`

- [ ] **Step 1: 写失败测试 `tests/test_retrieval_retrieve.py`**

```python
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
    assert "隔离性" in toks


# ── rrf_fuse：按 node id 融合两列表，去重排序截断 ──────────────────────
def _nws(nid, text="x"):
    return NodeWithScore(node=TextNode(text=text, id_=nid), score=1.0)


def test_rrf_fuse_combines_and_dedups_and_truncates():
    a = [_nws("n1"), _nws("n2"), _nws("n3")]   # dense
    b = [_nws("n3"), _nws("n1")]               # sparse：n3 居首
    out = rrf_fuse([a, b], top_k=2)
    ids = [o.node.node_id for o in out]
    # n1（两列表都高）与 n3（一列表居首+一列表中游）融合后居前；去重
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_retrieval_retrieve.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'core.retrieval.retrieve'`

- [ ] **Step 3: 写实现 `core/retrieval/retrieve.py`（不含 HybridRetriever，Task 2 加）**

```python
"""可插拔 Retriever：装配时注入的检索数据源策略。

不传 / "vector" = 基线（向量检索）；"hybrid" = dense + BM25（Task 2 加）。
名字→对象在本模块（core）解析，eval 只传名字。Retriever 是数据源（不像 reranker 是
变换），依赖在 retrieve() 调用时传入，策略对象自身无依赖、由 make_retriever 零参构造。
"""
import re
from typing import Protocol, runtime_checkable

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

RRF_K = 60  # RRF 平滑常数（经验值）


@runtime_checkable
class Retriever(Protocol):
    """检索数据源：query → 候选 NodeWithScore 列表（已按相关度排序）。"""

    async def retrieve(
        self, query: str, *, index_manager, book_titles, top_k: int
    ) -> list: ...


def build_book_filters(book_titles):
    """scope 硬约束 → chroma 元数据过滤器；空范围 → None（全库）。"""
    if not book_titles:
        return None
    return MetadataFilters(filters=[
        MetadataFilter(
            key="book_title",
            operator=FilterOperator.IN,
            value=list(book_titles),
        ),
    ])


def bm25_tokenize(text: str) -> list[str]:
    """中文 BM25 分词：jieba 切词 + 小写 + 丢空白/纯标点 token（否则噪声毁排序）。"""
    import jieba

    out: list[str] = []
    for t in jieba.lcut(text.lower()):
        t = t.strip()
        if not t or re.fullmatch(r"[\W_]+", t):
            continue
        out.append(t)
    return out


def rrf_fuse(ranked_lists: list, top_k: int) -> list:
    """Reciprocal Rank Fusion：score(node)=Σ 1/(RRF_K+rank)，按 node id 去重排序截 top_k。

    ranked_lists：每个是已排序的 NodeWithScore 列表。返回新 NodeWithScore（score=RRF 分）。
    """
    scores: dict = {}
    keep: dict = {}
    for nodes in ranked_lists:
        for rank, nws in enumerate(nodes):
            nid = nws.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + rank)
            keep.setdefault(nid, nws)
    ordered = sorted(scores, key=lambda nid: -scores[nid])
    return [
        NodeWithScore(node=keep[nid].node, score=scores[nid])
        for nid in ordered[:top_k]
    ]


class VectorRetriever:
    """基线：当前向量检索（dense），等价改造前的 as_retriever 路径。"""

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        retriever = index_manager.get_index().as_retriever(
            similarity_top_k=top_k,
            filters=build_book_filters(book_titles),
        )
        return await retriever.aretrieve(query)


# 名字 → 构造器。新增策略在此登记一行。
_REGISTRY = {
    "vector": VectorRetriever,
}

# 名字 → 已构造实例缓存（一进程一次；HybridRetriever 的 BM25 索引只建一次）。
_INSTANCES: dict = {}


def make_retriever(name):
    """名字 → 实例（按名缓存）。None/"vector" → VectorRetriever；未知 → ValueError。"""
    key = name or "vector"
    if key not in _REGISTRY:
        raise ValueError(f"未知 retriever 名字：{name!r}，可选：{list(_REGISTRY)}")
    if key not in _INSTANCES:
        _INSTANCES[key] = _REGISTRY[key]()
    return _INSTANCES[key]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_retrieval_retrieve.py -v`
Expected: 全部 passed（9 个）

- [ ] **Step 5: 守卫 + 提交**

Run: `python scripts/check_layering.py` → 通过
```bash
git add core/retrieval/retrieve.py tests/test_retrieval_retrieve.py
git commit -m "feat(retrieval): Retriever 协议 + VectorRetriever + RRF/分词/过滤工具 + make_retriever"
```

---

## Task 2: HybridRetriever（dense + BM25 RRF 融合）

**Files:**
- Modify: `core/retrieval/retrieve.py`（加 `HybridRetriever`，注册 `"hybrid"`）
- Test: `tests/test_retrieval_retrieve.py`（追加）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_retrieval_retrieve.py`）**

```python
# ── HybridRetriever ───────────────────────────────────────────────────
from core.retrieval.retrieve import HybridRetriever


class _FakeChromaCollection:
    def __init__(self, ids, docs, metas):
        self._data = {"ids": ids, "documents": docs, "metadatas": metas}
        self.get_calls = 0

    def get(self, include=None):
        self.get_calls += 1
        return self._data


class _FakeIMWithCorpus:
    """dense 走 as_retriever；BM25 语料走 chroma_collection.get。"""

    def __init__(self, dense_nodes, ids, docs, metas):
        self._index = _FakeIndex(dense_nodes)
        self.chroma_collection = _FakeChromaCollection(ids, docs, metas)

    def get_index(self):
        return self._index


async def test_hybrid_builds_bm25_once_and_fuses_dense_and_sparse():
    dense = [_nws("d1"), _nws("d2")]
    im = _FakeIMWithCorpus(
        dense_nodes=dense,
        ids=["d2", "s1", "s2"],
        docs=["范围查询 扫描", "哈希 等值查询", "事务 隔离性"],
        metas=[{"book_title": "《A》"}, {"book_title": "《A》"}, {"book_title": "《A》"}],
    )
    hr = HybridRetriever()

    out = await hr.retrieve("等值查询", index_manager=im, book_titles=["《A》"], top_k=3)
    ids = [o.node.node_id for o in out]
    # dense 命中 d1/d2，BM25 命中 s1（含「等值查询」）→ 融合后应包含三者的子集
    assert "d1" in ids or "d2" in ids
    assert all(isinstance(o, NodeWithScore) for o in out)
    assert len(ids) <= 3

    # 再检索一次：BM25 只构造一次（chroma.get 不再被调）
    await hr.retrieve("隔离性", index_manager=im, book_titles=["《A》"], top_k=3)
    assert im.chroma_collection.get_calls == 1


async def test_hybrid_bm25_scope_post_filter():
    """BM25 对全库打分后，按 book_titles 后过滤；scope 外的 node 不应出现在 sparse 侧。"""
    im = _FakeIMWithCorpus(
        dense_nodes=[],                       # dense 空，结果只来自 BM25
        ids=["inA", "inB"],
        docs=["等值查询 命中", "等值查询 命中"],
        metas=[{"book_title": "《A》"}, {"book_title": "《B》"}],
    )
    hr = HybridRetriever()
    out = await hr.retrieve("等值查询", index_manager=im, book_titles=["《A》"], top_k=5)
    ids = [o.node.node_id for o in out]
    assert "inB" not in ids                   # 《B》被 scope 过滤掉
    assert ids == ["inA"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_retrieval_retrieve.py -k hybrid -v`
Expected: FAIL，`ImportError: cannot import name 'HybridRetriever'`

- [ ] **Step 3: 在 `core/retrieval/retrieve.py` 加 `HybridRetriever`**

在 `VectorRetriever` 之后、`_REGISTRY` 之前插入。顶部 import 区加：

```python
import asyncio
from llama_index.core.schema import TextNode
```
（`asyncio` 和 `TextNode` 加到现有 import；`NodeWithScore` 已在。）

```python
class HybridRetriever:
    """dense + BM25 的混合检索，RRF 融合。

    BM25 语料从 chroma 全量重建（懒构造 + 缓存 + 并发守卫，一进程只建一次）；
    dense 用 chroma 元数据过滤，BM25 对全库打分后按 scope 后过滤。
    """

    def __init__(self):
        self._bm25 = None
        self._nodes = None       # list[TextNode]，与 BM25 语料同序
        self._lock = asyncio.Lock()

    async def _ensure_bm25(self, index_manager):
        if self._bm25 is not None:
            return
        async with self._lock:
            if self._bm25 is not None:      # 双检：等锁期间别人已建好
                return
            data = index_manager.chroma_collection.get(
                include=["documents", "metadatas"])
            # 重建 + 分词 + 建索引是 CPU 活，卸到线程不堵事件循环
            self._nodes, self._bm25 = await asyncio.to_thread(self._build_bm25, data)

    @staticmethod
    def _build_bm25(data):
        from rank_bm25 import BM25Okapi

        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        nodes = [
            TextNode(text=docs[i], id_=ids[i], metadata=metas[i] or {})
            for i in range(len(ids))
        ]
        corpus = [bm25_tokenize(n.text) for n in nodes]
        return nodes, BM25Okapi(corpus)

    def _bm25_search(self, query, book_titles, top_k):
        scores = self._bm25.get_scores(bm25_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            node = self._nodes[i]
            if book_titles and (node.metadata or {}).get("book_title") not in book_titles:
                continue
            out.append(NodeWithScore(node=node, score=float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        await self._ensure_bm25(index_manager)
        dense_retriever = index_manager.get_index().as_retriever(
            similarity_top_k=top_k,
            filters=build_book_filters(book_titles),
        )
        dense = await dense_retriever.aretrieve(query)
        sparse = self._bm25_search(query, book_titles, top_k)
        return rrf_fuse([dense, sparse], top_k)
```

在 `_REGISTRY` 里登记：

```python
_REGISTRY = {
    "vector": VectorRetriever,
    "hybrid": HybridRetriever,
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_retrieval_retrieve.py -v`
Expected: 全部 passed（含 2 个 hybrid + 之前 9 个）

- [ ] **Step 5: make_retriever hybrid 分支 smoke（不下任何模型）**

Run:
```bash
python -c "from core.retrieval.retrieve import make_retriever, HybridRetriever; assert isinstance(make_retriever('hybrid'), HybridRetriever); print('hybrid OK')"
```
Expected: `hybrid OK`

- [ ] **Step 6: 守卫 + 提交**

Run: `python scripts/check_layering.py` → 通过
```bash
git add core/retrieval/retrieve.py tests/test_retrieval_retrieve.py
git commit -m "feat(retrieval): HybridRetriever（dense+BM25，jieba 分词，RRF 融合，scope 后过滤）"
```

---

## Task 3: 接入 QaCapability（委托策略 + 上移过滤器）

**Files:**
- Modify: `core/workflow/qa_capability.py`（imports、`__init__`、`_retrieve_nodes`，删 `_make_filters`）
- Test: `tests/test_qa_capability.py`（追加）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_qa_capability.py`）**

```python
# ── retriever 接入 ────────────────────────────────────────────────────
class _RecordingRetriever:
    def __init__(self, nodes):
        self._nodes = nodes
        self.calls = []

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        self.calls.append((query, book_titles, top_k))
        return self._nodes


async def test_retrieve_nodes_delegates_to_injected_retriever():
    rr = _RecordingRetriever(nodes=["x", "y"])
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM(),
                      similarity_top_k=4, retriever=rr)

    nodes = await qa._retrieve_nodes("B+树", ["《A》"])

    assert nodes == ["x", "y"]
    assert rr.calls == [("B+树", ["《A》"], 4)]   # 无 reranker → top_k=similarity_top_k


async def test_retrieve_nodes_default_retriever_is_vector():
    from core.retrieval.retrieve import VectorRetriever
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM())
    assert isinstance(qa.retriever, VectorRetriever)


async def test_retrieve_nodes_retriever_overfetches_when_reranker_set():
    rr_ret = _RecordingRetriever(nodes=["a", "b", "c", "d", "e"])
    qa = QaCapability(FakeIndexManager(nodes=[]), FakeLLM(),
                      similarity_top_k=2, retriever=rr_ret,
                      reranker=_RecordingReranker(), rerank_candidate_k=5)

    await qa._retrieve_nodes("q", None)
    # retriever 拿候选池大小 5（reranker 再截 2）
    assert rr_ret.calls[0][2] == 5
```

> `_RecordingReranker` 已在本文件（reranker 那批测试定义）。`FakeIndexManager`/`FakeLLM` 同。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_qa_capability.py -k "retriever or vector" -v`
Expected: FAIL（`TypeError: unexpected keyword argument 'retriever'`）

- [ ] **Step 3: 改 imports**

`core/workflow/qa_capability.py` 顶部：删除这段（逻辑上移到 retrieve.py）：

```python
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
```

在 reranker import 旁加：

```python
from core.retrieval.retrieve import Retriever, VectorRetriever
```

- [ ] **Step 4: 改 `__init__`**

在 `reranker` / `rerank_candidate_k` 参数旁加 `retriever`，并设默认实例：

```python
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        reranker: "Reranker | None" = None,
        rerank_candidate_k: int = 20,
        retriever: "Retriever | None" = None,
    ):
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.max_sub_queries = max_sub_queries
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k
        # 检索不可跳过，基线=具体 VectorRetriever（不传即基线）
        self.retriever = retriever or VectorRetriever()
        self.preprocessor = QueryPreprocessor(llm)
        self.decomposer = QueryDecomposer(llm)
        self.dimensioner = DimensionExtractor(llm)
```

- [ ] **Step 5: 改 `_retrieve_nodes` + 删 `_make_filters`**

把 `_retrieve_nodes` 改为委托策略：

```python
    async def _retrieve_nodes(self, query: str, book_titles: Optional[list[str]]):
        # 检索策略可插拔（默认 VectorRetriever=基线）；有 reranker 时过召回候选池再重排截断
        fetch_k = self.rerank_candidate_k if self.reranker else self.similarity_top_k
        nodes = await self.retriever.retrieve(
            query, index_manager=self.index_manager,
            book_titles=book_titles, top_k=fetch_k,
        )
        if self.reranker:
            nodes = await self.reranker.rerank(query, nodes, self.similarity_top_k)
        return nodes
```

删除整个 `_make_filters` 方法（约在 `_book_chapters` 之后、`_retrieve_nodes` 之前那段）。

- [ ] **Step 6: 确认无其它引用 `_make_filters`**

Run: `python -m pytest -q 2>&1 | tail -3; grep -rn "_make_filters" core eval tests`
Expected: grep 无输出（已无引用）；若有，把该处改用 `build_book_filters`（从 `core.retrieval.retrieve` 导入）。

- [ ] **Step 7: 跑测试确认通过 + 零回归**

Run: `python -m pytest tests/test_qa_capability.py -v`
Expected: 全部 passed（新 3 个 + 旧的；旧 reranker/baseline 测试因 VectorRetriever 走同一 `as_retriever` 路径而不变）

- [ ] **Step 8: 守卫 + 提交**

Run: `python scripts/check_layering.py` → 通过
```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(workflow): QaCapability 委托可插拔 Retriever，过滤器上移 build_book_filters"
```

---

## Task 4: DocQueryWorkflow 注入 retriever

**Files:**
- Modify: `core/workflow/doc_workflow.py`（imports、`__init__`）
- Test: `tests/test_doc_workflow.py`（追加）

- [ ] **Step 1: 写失败测试（追加到 `tests/test_doc_workflow.py`）**

```python
def test_no_retriever_defaults_to_vector():
    from core.retrieval.retrieve import VectorRetriever
    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert isinstance(wf.qa.retriever, VectorRetriever)


def test_retriever_name_resolved_and_injected(monkeypatch):
    sentinel = object()
    import core.workflow.doc_workflow as mod

    captured = {}

    def fake_make_retriever(name):
        captured["name"] = name
        return sentinel

    monkeypatch.setattr(mod, "make_retriever", fake_make_retriever)

    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM(), retriever="hybrid")

    assert captured["name"] == "hybrid"
    assert wf.qa.retriever is sentinel
```

> `_StubIndexManager` / `_StubLLM` 已在本文件（reranker 那批测试定义）。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_doc_workflow.py -k retriever -v`
Expected: FAIL（`TypeError: unexpected keyword argument 'retriever'`）

- [ ] **Step 3: 改 `core/workflow/doc_workflow.py`**

import 区加：

```python
from core.retrieval.retrieve import make_retriever
```

`__init__` 在 `reranker: str | None = None` 之后加 `retriever: str | None = None`：

```python
        max_sub_queries: int = 6,
        # 可插拔检索后处理组件（按名字注入；名字→对象在 core 解析，None=基线无重排）。
        # 与下面的布尔决策开关不同：那是二元开关，这是带多实现的具名组件。
        reranker: str | None = None,
        # 可插拔检索策略（None/"vector"=基线向量检索，"hybrid"=dense+BM25）。
        retriever: str | None = None,
        probe_then_classify: bool = True,
```

QaCapability 构造行加 `retriever=make_retriever(retriever)`：

```python
        self.qa = QaCapability(
            index_manager, llm, similarity_top_k, max_sub_queries,
            reranker=make_reranker(reranker),
            retriever=make_retriever(retriever),
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_doc_workflow.py -v`
Expected: 全部 passed

- [ ] **Step 5: 守卫 + 提交**

Run: `python scripts/check_layering.py` → 通过
```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): DocQueryWorkflow 按名字解析并注入 retriever 策略"
```

---

## Task 5: eval 变体 + 依赖 + 文档

**Files:**
- Modify: `eval/harness/compare.py`、`requirements.txt`、`docs/ARCHITECTURE.md`

- [ ] **Step 1: compare.py 加变体**

`eval/harness/compare.py` 的 `VARIANTS` 在 `"全开+rerank"` 之后追加：

```python
    "全开+hybrid": dict(probe_then_classify=True, split_enabled=True,
                        assume_enabled=True, other_agent_enabled=True,
                        retriever="hybrid"),
    "全开+hybrid+rerank": dict(probe_then_classify=True, split_enabled=True,
                               assume_enabled=True, other_agent_enabled=True,
                               retriever="hybrid", reranker="bge-reranker-v2-m3"),
```

- [ ] **Step 2: 校验 kwargs 兼容**

Run:
```bash
python -c "from eval.harness.compare import VARIANTS; import inspect; from core.workflow.doc_workflow import DocQueryWorkflow; sig=set(inspect.signature(DocQueryWorkflow.__init__).parameters); bad=[k for v in VARIANTS.values() for k in v if k not in sig]; print('未知kwarg:', bad); assert not bad"
```
Expected: `未知kwarg: []`（无报错）

- [ ] **Step 3: requirements 加依赖**

`requirements.txt` 在 `sentence-transformers` 那行附近加：

```
rank_bm25  # Hybrid 检索的 BM25 稀疏打分（core/retrieval/retrieve.py）
jieba      # 中文分词（BM25 tokenizer）
```

Run: `python -c "import rank_bm25, jieba; print('deps OK')"`
Expected: `deps OK`

- [ ] **Step 4: ARCHITECTURE 加一行**

`docs/ARCHITECTURE.md` 的「现状→目标」表，在 reranker 行之后加：

```
| `core/retrieval/retrieve.py` | 可插拔 Retriever 策略：`"vector"`(基线 dense) / `"hybrid"`(dense+BM25，jieba 分词，RRF 融合，scope 后过滤)；治召回。eval `VARIANTS` 按名选择 | Layer 0 检索数据源 |
```

- [ ] **Step 5: 全量测试 + 提交**

Run: `python -m pytest -q`
Expected: 全部 passed
```bash
git add eval/harness/compare.py requirements.txt docs/ARCHITECTURE.md
git commit -m "feat(eval): 加 hybrid 检索变体 + 记录 rank_bm25/jieba 依赖与架构说明"
```

- [ ] **Step 6: 真实库 smoke（手动，可选，需 chroma 书库 + API 额度）**

Run:
```bash
python -m eval.harness.compare --variants "全开" "全开+hybrid" "全开+hybrid+rerank" --limit 5
```
Expected: 跑通并打 delta 表；BM25 索引一进程内只建一次。重点看 `context_recall` 列 hybrid 相对基线的变化（设计预期正向）。网络/数据受限可跳过留待联网环境——前 4 个 Task 单测不依赖真实库。

---

## 收尾

- [ ] 跑全量测试：`python -m pytest -q`，确认零回归。
- [ ] 更新记忆 `project_pluggable_retrieval_pipeline.md`：补「第二增量 Hybrid Retriever 已落地」+ 下一步（smoke 量化 / HyDE 或纯 BM25 第二策略 / dedup-filter stage）。
