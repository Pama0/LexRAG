"""QueryClassifier 单测：mock LLM 控返回，验证 4 类解析 / 证据进 prompt / 降级。

分类质量依赖真 LLM，不在单测范围。
设计见 docs/superpowers/specs/2026-06-21-multi-subject-split-pipeline-design.md。
"""
import pytest

from core.workflow.query_classifier import QueryClassifier, ClassifyResult


class _Resp:
    def __init__(self, text): self._t = text
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


@pytest.mark.parametrize("cat", ["explain", "compare", "simple", "complex"])
async def test_parses_each_category(cat):
    llm = FakeLLM([f'{{"category":"{cat}","reason":"r"}}'])
    res = await QueryClassifier(llm).run("讲讲MySQL索引", "召回片段")
    assert isinstance(res, ClassifyResult)
    assert res.category == cat


async def test_evidence_in_prompt():
    llm = FakeLLM(['{"category":"simple","reason":""}'])
    await QueryClassifier(llm).run("MySQL有哪些锁", "命中3段：第8章 锁")
    assert "命中3段：第8章 锁" in llm.prompts[0]
    assert "MySQL有哪些锁" in llm.prompts[0]


async def test_illegal_category_degrades_simple():
    llm = FakeLLM(['{"category":"banana","reason":"x"}'])
    res = await QueryClassifier(llm).run("MySQL锁", "")
    assert res.category == "simple"


async def test_parse_failure_degrades_simple(caplog):
    import logging
    llm = FakeLLM(["不是JSON"])
    with caplog.at_level(logging.WARNING):
        res = await QueryClassifier(llm).run("MySQL锁", "")
    assert res.category == "simple"
    assert any("classifier" in r.getMessage().lower() for r in caplog.records)
