"""IntentRouter（Layer 1 门口）单测：净化（指代+规范化）+ 意图分类。

mock LLM 控返回，验证：解析 / Pydantic 校验 / 历史拼接 / 失败降级。
净化质量本身依赖真 LLM，不在单测范围。
"""
from core.workflow.intent_router import IntentRouter, RouterResult


class _Resp:
    def __init__(self, text: str):
        self._t = text

    def __str__(self) -> str:
        return self._t


class FakeLLM:
    """按队列依次返回预设文本，并记录收到的 prompt（用于断言历史拼接）。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


class _Msg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeMemory:
    def __init__(self, msgs=None):
        self._msgs = list(msgs or [])

    def get(self):
        return self._msgs


def _router(llm):
    return IntentRouter(llm)


async def test_run_classifies_qa():
    llm = FakeLLM(['{"intent": "qa", "clean_query": "MySQL有哪些锁"}'])
    result = await _router(llm).run("MySQL有哪些锁啊")
    assert isinstance(result, RouterResult)
    assert result.intent == "qa"
    assert result.clean_query == "MySQL有哪些锁"


async def test_run_classifies_study_plan():
    llm = FakeLLM(['{"intent": "study_plan", "clean_query": "为《Redis设计与实现》制定学习计划"}'])
    result = await _router(llm).run("给我做份学Redis的计划")
    assert result.intent == "study_plan"
    assert result.clean_query == "为《Redis设计与实现》制定学习计划"


async def test_run_resolves_coreference_into_clean_query():
    # 指代消解：clean_query 应是 LLM 基于历史补全后的自包含句
    llm = FakeLLM(['{"intent": "qa", "clean_query": "MySQL索引的应用场景"}'])
    result = await _router(llm).run("它的应用场景是什么")
    assert result.clean_query == "MySQL索引的应用场景"


async def test_run_passes_history_to_llm():
    # 历史必须拼进 prompt，门口才能消指代
    llm = FakeLLM(['{"intent": "qa", "clean_query": "MySQL索引的应用场景"}'])
    memory = FakeMemory([_Msg("user", "MySQL索引有哪些"), _Msg("assistant", "B+树索引……")])
    await _router(llm).run("它的应用场景是什么", memory)
    assert "MySQL索引有哪些" in llm.prompts[0]


async def test_run_injects_selected_books_into_prompt():
    # 用户选中的书要进 prompt，门口才能把"这本书"补全成具体书名
    llm = FakeLLM(['{"intent": "qa", "clean_query": "《openclaw》讲了什么"}'])
    await _router(llm).run("这本书讲了什么", None, book_titles=["openclaw"])
    assert "openclaw" in llm.prompts[0]


async def test_run_resolves_book_reference_returns_clean_query():
    llm = FakeLLM(['{"intent": "qa", "clean_query": "《openclaw》讲了什么"}'])
    result = await _router(llm).run("这本书讲了什么", None, book_titles=["openclaw"])
    assert result.intent == "qa"
    assert result.clean_query == "《openclaw》讲了什么"


async def test_run_falls_back_to_qa_on_parse_failure(caplog):
    import logging
    llm = FakeLLM(["这不是JSON"])
    with caplog.at_level(logging.WARNING):
        result = await _router(llm).run("讲讲数据库")
    assert result.intent == "qa"        # 解析失败 → 默认 qa，不阻塞
    assert result.clean_query == "讲讲数据库"   # 用原 query
    assert any("router 解析失败" in r.getMessage() for r in caplog.records)  # 降级显形


async def test_run_falls_back_on_empty_content():
    llm = FakeLLM([""])                  # DeepSeek json 模式偶发空 content
    result = await _router(llm).run("讲讲数据库")
    assert result.intent == "qa"
    assert result.clean_query == "讲讲数据库"


async def test_run_rejects_invalid_intent():
    # taxonomy 外的 intent（如 life_plan）应被 Pydantic 拒，降级为 qa + 原 query
    llm = FakeLLM(['{"intent": "life_plan", "clean_query": "帮我规划人生"}'])
    result = await _router(llm).run("帮我规划人生")
    assert result.intent == "qa"
    assert result.clean_query == "帮我规划人生"
