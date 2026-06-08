from eval.generate_testset import chunks_to_langchain


def test_chunks_to_langchain_wraps_text_and_metadata():
    docs = ["正文1", "正文2"]
    metas = [{"book_title": "MySQL", "chapter": "3"}, {"book_title": "MySQL", "chapter": "4"}]
    out = chunks_to_langchain(docs, metas)
    assert len(out) == 2
    assert out[0].page_content == "正文1"
    assert out[0].metadata["book_title"] == "MySQL"
    assert out[1].metadata["chapter"] == "4"


def test_chunks_to_langchain_skips_empty_text():
    docs = ["正文", "", "  "]
    metas = [{"book_title": "X"}, {"book_title": "X"}, {"book_title": "X"}]
    out = chunks_to_langchain(docs, metas)
    assert len(out) == 1
    assert out[0].page_content == "正文"
