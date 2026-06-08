from eval.generate_testset import chunks_to_langchain, sample_chunks


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


def test_sample_chunks_caps_to_max():
    chunks = list(range(100))
    out = sample_chunks(chunks, max_chunks=10, seed=42)
    assert len(out) == 10
    assert set(out).issubset(set(chunks))  # 子集，未引入新元素


def test_sample_chunks_deterministic_with_seed():
    chunks = list(range(100))
    assert sample_chunks(chunks, max_chunks=10, seed=42) == sample_chunks(chunks, max_chunks=10, seed=42)


def test_sample_chunks_returns_all_when_below_cap():
    chunks = list(range(5))
    assert sample_chunks(chunks, max_chunks=10, seed=42) == chunks


def test_sample_chunks_none_returns_all():
    chunks = list(range(50))
    assert sample_chunks(chunks, max_chunks=None, seed=42) == chunks
