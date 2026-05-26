"""书籍知识库 Agent 工具

工具职责：
- book_search: 在书籍向量库中检索；可选按 book_title 过滤
- list_books: 返回当前知识库已入库书籍清单

工具内通过 api.source_context 把检索到的 source nodes 写入请求级容器，
请求结束时由 chat handler 统一取出回传给前端。
"""
from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from core.agent.source_context import add_sources, node_to_source_ref


def create_book_search_tool(
    index_manager,
    llm: LLM,
    similarity_top_k: int = 5,
) -> FunctionTool:
    """书籍内容检索工具

    持有 index_manager 引用而非 index 对象本身——这样书籍入库后能立刻
    被工具检索到（不需要重启 Agent）。
    """

    async def book_search(query: str, book_title: Optional[str] = None) -> str:
        """从书籍知识库中检索内容并合成答案。

        Args:
            query: 用户问题，必须是字符串。
            book_title: 可选，限定在某本书中检索，如 "深入理解MySQL"。
                        留空则在所有书中检索。

        Returns:
            基于检索片段合成的答案文本。
        """
        # 防御：LLM 偶尔返回 dict
        if not isinstance(query, str):
            query = (query.get("title") or query.get("text") or str(query)) if isinstance(query, dict) else str(query)
        query = query.strip()
        if not query:
            return "请提供要查询的问题"

        index = index_manager.get_index()
        if index is None:
            return "知识库为空，请先在「文档管理」上传 PDF。"

        filters = None
        if book_title:
            filters = MetadataFilters(filters=[
                MetadataFilter(key="book_title", value=book_title),
            ])

        retriever = index.as_retriever(
            similarity_top_k=similarity_top_k,
            filters=filters,
        )
        nodes = await retriever.aretrieve(query)

        if not nodes:
            scope = f"《{book_title}》中" if book_title else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。"

        # 把 source 写入请求级容器
        add_sources([node_to_source_ref(n) for n in nodes])

        synthesizer = get_response_synthesizer(llm=llm)
        response = await synthesizer.asynthesize(query=query, nodes=nodes)
        return str(response)

    return FunctionTool.from_defaults(
        fn=book_search,
        name="book_search",
        description=(
            "书籍知识库检索：根据用户问题在已入库的技术书籍中查找相关内容。"
            "可选指定 book_title 限定单本书，留空则跨书检索。"
            "回答用户对书籍内容的具体技术问题时调用此工具。"
        ),
    )


def create_list_books_tool(index_manager) -> FunctionTool:
    """列出已入库书籍工具"""

    def list_books() -> str:
        """列出当前知识库中已入库的所有书籍名称。

        Returns:
            书名清单字符串（每本一行，附块数）。
        """
        all_data = index_manager.chroma_collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in all_data.get("metadatas", []) or []:
            title = (meta or {}).get("book_title")
            if not title:
                continue
            counts[title] = counts.get(title, 0) + 1

        if not counts:
            return "知识库当前为空，请先在「文档管理」上传 PDF。"

        lines = [f"- 《{t}》（{c} 个向量块）" for t, c in sorted(counts.items())]
        return "已入库书籍：\n" + "\n".join(lines)

    return FunctionTool.from_defaults(
        fn=list_books,
        name="list_books",
        description=(
            "查询当前知识库中已入库的书籍清单。"
            "用户问'你有哪些书'、'知识库里有什么'，或问题模糊需要先了解可用书籍时调用。"
        ),
    )
