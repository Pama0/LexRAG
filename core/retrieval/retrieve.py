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
