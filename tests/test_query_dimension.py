"""DimensionExtractor 单测：把"问题 + 召回正文"归纳成 ≤N 个评判维度。

mock LLM 控返回，验证：解析 / 上限裁剪 / 去空 / 失败降级为空 / prompt 带素材。
维度质量本身依赖真 LLM，不在单测范围。
"""
from core.workflow.query_dimension import DimensionExtractor


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


def _ext(llm):
    return DimensionExtractor(llm)


async def test_run_parses_dimensions():
    payload = '{"dimensions": [{"label": "读写性能", "query": "Redis 缓存读写性能"}, {"label": "一致性", "query": "Redis 缓存数据一致性"}]}'
    dims = await _ext(FakeLLM([payload])).run("Redis 做缓存好吗", ["正文片段"])
    assert [(d.label, d.query) for d in dims] == [
        ("读写性能", "Redis 缓存读写性能"),
        ("一致性", "Redis 缓存数据一致性"),
    ]


async def test_run_caps_at_max_items():
    items = ", ".join(
        '{"label": "L%d", "query": "Q%d"}' % (i, i) for i in range(7)
    )
    payload = '{"dimensions": [%s]}' % items
    dims = await _ext(FakeLLM([payload])).run("q", ["p"], max_items=3)
    assert [d.label for d in dims] == ["L0", "L1", "L2"]


async def test_run_drops_items_with_blank_label_or_query():
    payload = '{"dimensions": [{"label": "有效", "query": "有效检索"}, {"label": "", "query": "缺label"}, {"label": "缺query", "query": "  "}]}'
    dims = await _ext(FakeLLM([payload])).run("q", ["p"])
    assert [(d.label, d.query) for d in dims] == [("有效", "有效检索")]


async def test_run_returns_empty_on_parse_failure():
    dims = await _ext(FakeLLM(["这不是JSON"])).run("q", ["p"])
    assert dims == []


async def test_run_returns_empty_on_empty_content():
    dims = await _ext(FakeLLM([""])).run("q", ["p"])
    assert dims == []


async def test_run_prompt_includes_query_and_passages():
    payload = '{"dimensions": [{"label": "x", "query": "xq"}]}'
    llm = FakeLLM([payload])
    await _ext(llm).run("Redis 做缓存好吗", ["这是召回正文ZZZ"])
    assert "Redis 做缓存好吗" in llm.prompts[0]
    assert "这是召回正文ZZZ" in llm.prompts[0]
