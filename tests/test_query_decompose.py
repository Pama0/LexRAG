"""QueryDecomposer 单测：把"章节标题 + 召回正文"拆成 ≤N 个子查询。

mock LLM 控返回，验证：解析 / 上限裁剪 / 去空 / 失败降级为空 / prompt 带素材。
拆解质量本身依赖真 LLM，不在单测范围。
"""
from core.workflow.query_decompose import QueryDecomposer


class _Resp:
    def __init__(self, text):
        self._t = text

    def __str__(self):
        return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


def _dec(llm):
    return QueryDecomposer(llm)


async def test_run_parses_sub_queries():
    llm = FakeLLM(['{"sub_queries": ["工具A 是什么", "工具B 怎么用"]}'])
    subs = await _dec(llm).run("openclaw 的工具系统", ["3.2.1 工具A", "3.2.2 工具B"], ["正文片段"])
    assert subs == ["工具A 是什么", "工具B 怎么用"]


async def test_run_caps_at_max_items():
    payload = '{"sub_queries": ["a", "b", "c", "d", "e", "f", "g"]}'
    subs = await _dec(FakeLLM([payload])).run("q", [], ["p"], max_items=3)
    assert subs == ["a", "b", "c"]


async def test_run_drops_blank_sub_queries():
    llm = FakeLLM(['{"sub_queries": ["有效", "  ", ""]}'])
    subs = await _dec(llm).run("q", [], ["p"])
    assert subs == ["有效"]


async def test_run_returns_empty_on_parse_failure(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        subs = await _dec(FakeLLM(["这不是JSON"])).run("q", [], ["p"])
    assert subs == []
    assert any("decompose 失败" in r.getMessage() for r in caplog.records)  # 降级显形


async def test_run_returns_empty_on_empty_content():
    subs = await _dec(FakeLLM([""])).run("q", [], ["p"])
    assert subs == []


async def test_run_prompt_includes_headings_and_passages():
    llm = FakeLLM(['{"sub_queries": ["x"]}'])
    await _dec(llm).run("openclaw 工具系统", ["3.2.1 工具A"], ["这是召回正文ZZZ"])
    assert "3.2.1 工具A" in llm.prompts[0]
    assert "这是召回正文ZZZ" in llm.prompts[0]
