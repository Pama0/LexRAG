"""库藏元数据聚合：从 chroma 的 book_title 元数据产人读文本。

front_door 的 converse 路径（元查询"库里有什么/有 X 吗/多少本"）与
ListBooksTool（QaAgent 工具）共用此函数，避免 core/workflow → core/agent.tools
的模块循环（core/agent/qa_agent.py 与 auto_agent.py 已依赖 core/workflow/qa_capability.py）。
"""


def _collect_titles(index_manager) -> list[str]:
    """从 chroma 元数据抽出非空 book_title 列表（保留重复，供计数）。"""
    data = index_manager.chroma_collection.get(include=["metadatas"])
    metas = (data or {}).get("metadatas") or []
    titles: list[str] = []
    for meta in metas:
        title = (meta or {}).get("book_title")
        if title:
            titles.append(title)
    return titles


def list_books_text(
    index_manager,
    title_filter: str = "",
    count_only: bool = False,
) -> str:
    """聚合库藏书单为人读文本。

    - title_filter：大小写不敏感子串匹配 book_title；空串 = 全量。
    - count_only：True → 只回计数；False → 列出每本书 + 块数。
    """
    titles = _collect_titles(index_manager)
    if title_filter:
        fl = title_filter.lower()
        titles = [t for t in titles if fl in t.lower()]

    if count_only:
        if not titles and title_filter:
            return f"没有匹配「{title_filter}」的书。"
        if not titles:
            return "知识库当前为空。"
        if title_filter:
            # 去重计数：同一书名只算 1 本
            n = len(set(titles))
            return f"匹配「{title_filter}」的书有 {n} 本。"
        n = len(set(titles))
        return f"已入库 {n} 本。"

    # 列表形式
    if not titles and title_filter:
        return f"没有匹配「{title_filter}」的书籍。"
    if not titles:
        return "知识库当前为空。"

    # 按书名排序、计数
    counts: dict[str, int] = {}
    for t in titles:
        counts[t] = counts.get(t, 0) + 1
    head = f"匹配「{title_filter}」的书籍：" if title_filter else "已入库书籍："
    lines = [f"- 《{t}》（{c} 块）" for t, c in sorted(counts.items())]
    return head + "\n" + "\n".join(lines)
