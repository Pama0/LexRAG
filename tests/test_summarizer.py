"""会话历史增量摘要单测：plan_overflow（纯函数，无 LLM）+ fold_summary（mock LLM）。"""
from core.workflow.summarizer import (
    fold_summary,
    plan_overflow,
)


class _Msg:
    def __init__(self, id, role, content):
        self.id = id
        self.role = role
        self.content = content


class _Resp:
    def __init__(self, t):
        self._t = t

    def __str__(self):
        return self._t


class FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._r.pop(0))


def _msgs(n, start=1):
    return [_Msg(i, "user" if i % 2 else "assistant", f"m{i}")
            for i in range(start, start + n)]


def test_plan_overflow_noop_below_trigger():
    overflow, upto = plan_overflow(_msgs(5), 0, trigger=10, keep_last=3)
    assert overflow is None
    assert upto == 0


def test_plan_overflow_folds_oldest_keeps_last():
    # ids 1..12，trigger=10 触发；保留最近 3 条（10,11,12），折叠 1..9
    overflow, upto = plan_overflow(_msgs(12), 0, trigger=10, keep_last=3)
    assert [m.id for m in overflow] == list(range(1, 10))
    assert upto == 9


def test_plan_overflow_only_counts_unsummarized():
    # upto=6 → 未摘要 7..12 共 6 条 ≤ trigger 10 → 不压缩
    overflow, upto = plan_overflow(_msgs(12), 6, trigger=10, keep_last=3)
    assert overflow is None
    assert upto == 6


def test_plan_overflow_advances_from_watermark():
    # upto=5 → 未摘要 6..30 共 25 条 > 10；keep_last 3 → 折叠 6..27，水位推到 27
    overflow, upto = plan_overflow(_msgs(30), 5, trigger=10, keep_last=3)
    assert overflow[0].id == 6
    assert upto == 27


def test_plan_overflow_guards_keep_last_ge_unsummarized():
    # 配置异常：keep_last 比未摘要还多 → 不产生空折叠
    overflow, upto = plan_overflow(_msgs(12), 0, trigger=5, keep_last=20)
    assert overflow is None


async def test_fold_summary_merges_prev_and_overflow():
    llm = FakeLLM(["新摘要"])
    out = await fold_summary(llm, "旧摘要", _msgs(3))
    assert out == "新摘要"
    p = llm.prompts[0]
    assert "旧摘要" in p and "m1" in p and "m3" in p
    # 稳定指令应在最前（缓存友好），变量在后
    assert p.index("对话历史压缩器") < p.index("旧摘要")


async def test_fold_summary_handles_no_prev():
    llm = FakeLLM(["S"])
    out = await fold_summary(llm, None, _msgs(2))
    assert out == "S"
    assert "（无）" in llm.prompts[0]
