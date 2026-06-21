"""list_books_text 单测：从 chroma 元数据聚合书单，验全量/filter/count_only/大小写。"""
from core.rag.inventory import list_books_text


class _FakeCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class _FakeIndexManager:
    def __init__(self, metas):
        self.chroma_collection = _FakeCollection(metas)


def test_empty_library_returns_empty_message():
    assert list_books_text(_FakeIndexManager([])) == "知识库当前为空。"


def test_full_list_counts_and_sorts_titles():
    metas = [{"book_title": "乙"}, {"book_title": "甲"}, {"book_title": "甲"}]
    out = list_books_text(_FakeIndexManager(metas))
    assert "已入库书籍：" in out
    assert "《甲》（2 块）" in out
    assert "《乙》（1 块）" in out
    # 按书名 Unicode 码点排序：乙(U+4E59) 在 甲(U+7532) 前（与现有 ListBooksTool 的 sorted() 行为一致）
    assert out.index("《乙》") < out.index("《甲》")


def test_filter_case_insensitive_substring_match():
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}, {"book_title": "MySQL实战"}]
    out = list_books_text(_FakeIndexManager(metas), title_filter="mysql")
    assert "匹配「mysql」的书籍：" in out
    assert "《高性能MySQL》" in out
    assert "《MySQL实战》" in out
    assert "Redis" not in out


def test_filter_no_match_returns_no_match_message():
    metas = [{"book_title": "MySQL"}]
    out = list_books_text(_FakeIndexManager(metas), title_filter="oracle")
    assert out == "没有匹配「oracle」的书籍。"


def test_count_only_full_returns_count():
    metas = [{"book_title": "甲"}, {"book_title": "乙"}, {"book_title": "丙"}]
    assert list_books_text(_FakeIndexManager(metas), count_only=True) == "已入库 3 本。"


def test_count_only_empty_returns_empty_message():
    assert list_books_text(_FakeIndexManager([]), count_only=True) == "知识库当前为空。"


def test_count_only_with_filter_returns_match_count():
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}, {"book_title": "MySQL实战"}]
    out = list_books_text(_FakeIndexManager(metas), title_filter="mysql", count_only=True)
    assert out == "匹配「mysql」的书有 2 本。"


def test_count_only_with_filter_no_match():
    metas = [{"book_title": "MySQL"}]
    out = list_books_text(_FakeIndexManager(metas), title_filter="oracle", count_only=True)
    assert out == "没有匹配「oracle」的书。"


def test_filter_empty_string_is_full_list_not_match():
    # title_filter="" 应等同全量，不是"匹配空串"
    metas = [{"book_title": "甲"}]
    out = list_books_text(_FakeIndexManager(metas), title_filter="")
    assert "已入库书籍：" in out
    assert "匹配" not in out


def test_metas_without_book_title_skipped():
    metas = [{"book_title": "甲"}, {"other": "x"}, None, {"book_title": ""}]
    out = list_books_text(_FakeIndexManager(metas))
    assert "《甲》（1 块）" in out
    assert "已入库 1 本" not in out  # 只 1 本有效，但默认是列表形式不是计数
    # 确认只计了 1 本
    assert out.count("《") == 1
