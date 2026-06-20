"""QueryGate（Call A：检索降噪 + 意图二判）单测。mock LLM 控返回，验解析/降级。"""
from core.workflow.query_gate import QueryGate


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


async def test_gate_explain_intent():
    llm = FakeLLM(['{"denoised_query":"MySQL索引","intent":"explain"}'])
    denoised, intent = await QueryGate(llm).run("给我讲讲MySQL索引啊")
    assert denoised == "MySQL索引"
    assert intent == "explain"


async def test_gate_other_intent():
    llm = FakeLLM(['{"denoised_query":"redo日志的LSN是什么","intent":"other"}'])
    denoised, intent = await QueryGate(llm).run("redo日志的LSN是什么")
    assert intent == "other"
    assert denoised == "redo日志的LSN是什么"


async def test_gate_parse_failure_degrades_to_other_original():
    llm = FakeLLM(["这不是JSON"])
    denoised, intent = await QueryGate(llm).run("讲讲数据库")
    assert intent == "other"            # 降级默认 other（落已验证的难度分类路径）
    assert denoised == "讲讲数据库"      # 用原 query


async def test_gate_empty_content_degrades():
    llm = FakeLLM([""])
    denoised, intent = await QueryGate(llm).run("讲讲数据库")
    assert intent == "other"
    assert denoised == "讲讲数据库"


async def test_gate_invalid_intent_rejected():
    llm = FakeLLM(['{"denoised_query":"x","intent":"compare"}'])  # 枚举外
    denoised, intent = await QueryGate(llm).run("讲讲数据库")
    assert intent == "other"            # Pydantic 拒 → 降级
    assert denoised == "讲讲数据库"


async def test_gate_empty_denoised_uses_original():
    llm = FakeLLM(['{"denoised_query":"","intent":"explain"}'])
    denoised, intent = await QueryGate(llm).run("讲讲B+树")
    assert denoised == "讲讲B+树"
    assert intent == "explain"
