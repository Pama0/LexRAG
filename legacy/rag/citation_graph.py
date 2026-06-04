"""法条引用有向图 — 构建、查询、持久化"""
import json
import logging
import os
from collections import defaultdict, deque
from typing import Optional

from llama_index.core import VectorStoreIndex

from legacy.rag.citation_extractor import Citation, extract_citations

logger = logging.getLogger(__name__)


class CitationGraph:
    """法条引用有向图

    邻接表结构:
        {file_name: {source_article_int: [Citation, ...]}}

    支持持久化为 JSON，增量构建（只更新变更文件）。
    """

    def __init__(self):
        # {file_name: {source_article_int: [Citation]}}
        self._graph: dict[str, dict[int, list[Citation]]] = defaultdict(lambda: defaultdict(list))
        self._built = False

    # ==================== 构建 ====================

    def build_from_nodes(self, nodes) -> None:
        """从 LlamaIndex node 列表构建引用图

        Args:
            nodes: TextNode 列表，每个 node 应有 file_name 和 article_no_int 元数据
        """
        for node in nodes:
            file_name = node.metadata.get("file_name", "")
            article_no_int = node.metadata.get("article_no_int")
            article_no = node.metadata.get("article_no", "")

            if not file_name or article_no_int is None:
                continue

            citations = extract_citations(
                text=node.text,
                file_name=file_name,
                current_article=article_no,
            )

            for cite in citations:
                self._graph[file_name][article_no_int].append(cite)

        total = sum(len(arts) for arts in self._graph.values())
        logger.info(f"引用图构建完成: {len(self._graph)} 个法规, {total} 处引用关系")
        self._built = True

    def update_file(self, file_name: str, nodes: list) -> None:
        """增量更新：替换某个文件的引用关系"""
        # 删除旧数据
        if file_name in self._graph:
            del self._graph[file_name]

        for node in nodes:
            article_no_int = node.metadata.get("article_no_int")
            article_no = node.metadata.get("article_no", "")

            if article_no_int is None:
                continue

            citations = extract_citations(
                text=node.text,
                file_name=file_name,
                current_article=article_no,
            )
            for cite in citations:
                self._graph[file_name][article_no_int].append(cite)

    def remove_file(self, file_name: str) -> None:
        """删除某个文件的引用关系"""
        self._graph.pop(file_name, None)

    # ==================== 查询 ====================

    def get_citations(self, file_name: str, article_int: int) -> list[Citation]:
        """获取某条文的直接引用"""
        return self._graph.get(file_name, {}).get(article_int, [])

    def expand(
        self,
        file_name: str,
        article_ints: list[int],
        depth: int = 1,
    ) -> list[Citation]:
        """BFS 展开引用链

        Args:
            file_name: 法规文件名
            article_ints: 起始条文号列表
            depth: 展开深度，1=直接引用，2=引用的引用

        Returns:
            所有被引用的 Citation 列表（含中间节点），去重
        """
        visited: set[tuple[str, int]] = set()
        result: list[Citation] = []

        # BFS 队列: (file_name, article_int, current_depth)
        queue = deque()
        for art_int in article_ints:
            queue.append((file_name, art_int, 0))
            visited.add((file_name, art_int))

        while queue:
            cur_file, cur_art, cur_depth = queue.popleft()

            if cur_depth >= depth:
                continue

            for cite in self.get_citations(cur_file, cur_art):
                key = (file_name, cite.target_article_int)
                if key not in visited:
                    visited.add(key)
                    result.append(cite)
                    queue.append((file_name, cite.target_article_int, cur_depth + 1))

        return result

    def get_reverse_citations(self, file_name: str, article_int: int) -> list[Citation]:
        """反向查询：哪些条文引用了指定条文"""
        result = []
        for source_art, cites in self._graph.get(file_name, {}).items():
            for cite in cites:
                if cite.target_article_int == article_int:
                    result.append(cite)
        return result

    # ==================== 统计 ====================

    @property
    def stats(self) -> dict:
        total_citations = sum(
            len(cites)
            for arts in self._graph.values()
            for cites in arts.values()
        )
        internal = sum(
            1
            for arts in self._graph.values()
            for cites in arts.values()
            for c in cites
            if c.citation_type == "internal"
        )
        external = total_citations - internal
        return {
            "file_count": len(self._graph),
            "total_citations": total_citations,
            "internal": internal,
            "external": external,
        }

    # ==================== 持久化 ====================

    def save(self, path: str = "./citation_graph.json") -> None:
        """持久化到 JSON"""
        data = {}
        for file_name, arts in self._graph.items():
            data[file_name] = {}
            for art_int, cites in arts.items():
                data[file_name][str(art_int)] = [
                    {
                        "source_article": c.source_article,
                        "source_article_int": c.source_article_int,
                        "target_article": c.target_article,
                        "target_article_int": c.target_article_int,
                        "citation_type": c.citation_type,
                        "citation_verb": c.citation_verb,
                        "target_law": c.target_law,
                        "context": c.context,
                    }
                    for c in cites
                ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"引用图已保存到 {path}")

    def load(self, path: str = "./citation_graph.json") -> bool:
        """从 JSON 加载

        Returns:
            True 加载成功，False 文件不存在
        """
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._graph = defaultdict(lambda: defaultdict(list))
        for file_name, arts in data.items():
            for art_int_str, cites_data in arts.items():
                art_int = int(art_int_str)
                for cd in cites_data:
                    self._graph[file_name][art_int].append(Citation(
                        source_article=cd["source_article"],
                        source_article_int=cd["source_article_int"],
                        target_article=cd["target_article"],
                        target_article_int=cd["target_article_int"],
                        citation_type=cd["citation_type"],
                        citation_verb=cd["citation_verb"],
                        target_law=cd.get("target_law"),
                        context=cd.get("context", ""),
                    ))

        self._built = True
        logger.info(f"引用图已从 {path} 加载: {self.stats}")
        return True
