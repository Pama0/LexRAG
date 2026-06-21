"""AnswerOutliner（教学维度化列骨架）单测。mock LLM 控返回，验解析/词表过滤/TOC/降级。"""
from core.workflow.answer_outliner import AnswerOutliner
from core.workflow.query_dimension import Dimension


class _Resp:
    def __init__(self, t): self._t = t
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


async def test_outline_returns_dimensions_from_vocab():
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"什么是MySQL"},'
                   '{"label":"组成","query":"MySQL由哪些部分组成"}]}'])
    dims = await AnswerOutliner(llm).run("MySQL基础知识", ["片段1", "片段2"])
    assert dims == [Dimension(label="是什么", query="什么是MySQL"),
                    Dimension(label="组成", query="MySQL由哪些部分组成")]


async def test_outline_drops_labels_not_in_vocab():
    # FIL_PAGE_UNDO_LOG 这种碎细节被模型当 label → 必须被词表过滤丢掉
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"什么是X"},'
                   '{"label":"FIL_PAGE_UNDO_LOG","query":"FIL_PAGE_UNDO_LOG细节"}]}'])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert dims == [Dimension(label="是什么", query="什么是X")]


async def test_outline_drops_dimension_with_empty_query():
    llm = FakeLLM(['{"dimensions":[{"label":"作用","query":""},'
                   '{"label":"原理","query":"X的原理"}]}'])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert dims == [Dimension(label="原理", query="X的原理")]


async def test_outline_atomic_single_dimension():
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"脏读的定义"}]}'])
    dims = await AnswerOutliner(llm).run("什么是脏读", ["片段"])
    assert dims == [Dimension(label="是什么", query="脏读的定义")]


async def test_outline_passages_passed_to_prompt():
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"x"}]}'])
    await AnswerOutliner(llm).run("讲讲X", ["关键片段ABC"])
    assert "关键片段ABC" in llm.prompts[0]


async def test_outline_toc_hint_passed_to_prompt():
    llm = FakeLLM(['{"dimensions":[{"label":"组成","query":"x"}]}'])
    await AnswerOutliner(llm).run("讲讲X", ["片段"], toc_hint=["第1章 索引", "第2章 事务"])
    assert "第1章 索引" in llm.prompts[0] and "第2章 事务" in llm.prompts[0]


async def test_outline_respects_max_items():
    llm = FakeLLM(['{"dimensions":[{"label":"是什么","query":"a"},{"label":"作用","query":"b"},'
                   '{"label":"组成","query":"c"},{"label":"原理","query":"d"}]}'])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"], max_items=2)
    assert [d.label for d in dims] == ["是什么", "作用"]


async def test_outline_empty_on_parse_failure():
    llm = FakeLLM(["这不是JSON"])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert dims == []          # 空 → explain 将落 agent 兜底


async def test_outline_empty_on_empty_list():
    llm = FakeLLM(['{"dimensions":[]}'])
    dims = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert dims == []
