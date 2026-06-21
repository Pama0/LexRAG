"""会话作用域收窄：全库模式下从会话历史锚定主体，把检索硬锁到主导书。

注入式协作单元（仿 QueryGate/Admitter）：注入依赖、对外只暴露 run、失败降级、独立可测。
本单元【不消指代、不改 query 文本】——只决定「本轮检索锁哪几本书」。
设计见 docs/superpowers/specs/2026-06-21-conversation-scoper-design.md。
"""
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from llama_index.core.base.llms.types import MessageRole

from core.retrieval.retrieve import Retriever

logger = logging.getLogger(__name__)


@dataclass
class ScopeDecision:
    """scoper 产出：effective_book_titles=None 表示不收窄（保持全库）。"""

    effective_book_titles: Optional[list[str]]
    note: str = ""


def _book_of(node) -> str:
    """从 NodeWithScore/TextNode 取 book_title（缺失返回空串）。"""
    meta = getattr(node, "metadata", None) or {}
    return meta.get("book_title") or ""


class ConversationScoper:
    """全库多轮的隐式作用域收窄。仅当用户未手选书时动作。"""

    def __init__(
        self,
        index_manager,
        probe_retriever: Retriever,
        probe_k: int = 8,
        n_history_turns: int = 2,
        dominant_share: float = 0.60,
        dominant_ratio: float = 2.0,
        cover_share: float = 0.80,
        max_books: int = 2,
        min_count: int = 2,
    ):
        self.index_manager = index_manager
        self.probe_retriever = probe_retriever
        self.probe_k = probe_k
        self.n_history_turns = n_history_turns
        self.dominant_share = dominant_share
        self.dominant_ratio = dominant_ratio
        self.cover_share = cover_share
        self.max_books = max_books
        self.min_count = min_count

    def _probe_text(self, clean_query: str, memory) -> str:
        """拼接：最近 n 轮【用户】问 + 本轮 clean_query。

        memory 此刻【不含本轮】（本轮在 finalize 才入记忆），故取到的是历史用户问。
        裸概念续问靠这步把上文主体带进 probe；自包含强 query 则自己立得住。
        """
        turns: list[str] = []
        if memory is not None:
            msgs = memory.get() or []
            users = [
                str(m.content)
                for m in msgs
                if m.role == MessageRole.USER and m.content
            ]
            turns = users[-self.n_history_turns:]
        return "\n".join(turns + [clean_query])

    def _decide(self, titles: list[str]) -> Optional[list[str]]:
        """命中书名列表 → 主导书集合 or None（不收窄）。"""
        titles = [t for t in titles if t]
        if not titles:
            return None
        counts = Counter(titles).most_common()      # [(book, n), ...] 降序
        total = sum(n for _b, n in counts)
        top_book, top_n = counts[0]
        second_n = counts[1][1] if len(counts) > 1 else 0
        # 单一主导：占比过线、≥第二名 dominant_ratio 倍、且自身命中足够（防单点噪声）
        if (
            top_n / total >= self.dominant_share
            and top_n >= self.dominant_ratio * second_n
            and top_n >= self.min_count
        ):
            return [top_book]
        # 少数主导：最小前缀累计 ≥ cover_share，前缀 ≤ max_books，
        # 前缀内每本 ≥ min_count，尾部每本 < min_count（确属长尾噪声）
        prefix: list[str] = []
        acc = 0
        for book, n in counts:
            if n < self.min_count:
                break
            prefix.append(book)
            acc += n
            if len(prefix) > self.max_books:
                return None
            if acc / total >= self.cover_share:
                tail = counts[len(prefix):]
                if all(tn < self.min_count for _tb, tn in tail):
                    return prefix
                return None
        return None

    async def run(
        self, clean_query: str, user_book_titles: Optional[list[str]], memory
    ) -> ScopeDecision:
        # 手选硬约束永远赢：非空直接 no-op，根本不 probe
        if user_book_titles:
            return ScopeDecision(user_book_titles, "")
        try:
            probe_text = self._probe_text(clean_query, memory)
            nodes = await self.probe_retriever.retrieve(
                probe_text,
                index_manager=self.index_manager,
                book_titles=None,
                top_k=self.probe_k,
            )
            books = self._decide([_book_of(n) for n in nodes])
        except Exception as exc:
            logger.warning("scoper probe/判定失败，保持全库：%s", exc)
            return ScopeDecision(None, "")
        if not books:
            return ScopeDecision(None, "")
        label = "》《".join(books)
        note = f"（我按《{label}》里的内容回答；想看全部书可以说\"在所有书里讲\"。）\n"
        logger.info("scoper: 收窄到 %s", books)
        return ScopeDecision(books, note)
