"""DocQueryService 单测：会话锁 / 记忆构造 / 驱动 workflow handler。

不跑真 LLM：run_handler 用 stub workflow 验证参数透传。
"""
from llama_index.core.base.llms.types import MessageRole

from core.workflow.doc_query_service import DocQueryService


class _DBMsg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


def _svc():
    return DocQueryService(index_manager=object(), llm=object(), similarity_top_k=3)


def test_get_lock_same_session_returns_same_lock():
    svc = _svc()
    a = svc.get_lock("s1")
    b = svc.get_lock("s1")
    assert a is b


def test_get_lock_none_session_returns_fresh_each_time():
    svc = _svc()
    assert svc.get_lock(None) is not svc.get_lock(None)


def test_reset_removes_lock():
    svc = _svc()
    svc.get_lock("s1")
    assert svc.reset("s1") is True
    assert svc.reset("s1") is False


def test_build_memory_maps_roles_and_skips_empty():
    svc = _svc()
    mem = svc.build_memory([
        _DBMsg("user", "问题1"),
        _DBMsg("assistant", "答案1"),
        _DBMsg("assistant", ""),          # 空内容跳过
        _DBMsg("system", "忽略"),          # 非 user/assistant 跳过
    ])
    msgs = mem.get()
    assert [m.role for m in msgs] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert [m.content for m in msgs] == ["问题1", "答案1"]


def test_run_handler_passes_args_to_workflow(monkeypatch):
    import core.workflow.doc_query_service as svc_mod

    class StubWF:
        def __init__(self, **kw):
            StubWF.init_kw = kw

        def run(self, **kw):
            StubWF.run_kw = kw
            return "HANDLER"

    monkeypatch.setattr(svc_mod, "DocQueryWorkflow", StubWF)

    svc = DocQueryService(index_manager="IM", llm="LLM", similarity_top_k=7)
    handler = svc.run_handler(query="q", memory="MEM", book_titles=["b1"])

    assert handler == "HANDLER"
    assert StubWF.run_kw["query"] == "q"
    assert StubWF.run_kw["memory"] == "MEM"
    assert StubWF.run_kw["book_titles"] == ["b1"]
    assert StubWF.init_kw["index_manager"] == "IM"
    assert StubWF.init_kw["llm"] == "LLM"
    assert StubWF.init_kw["similarity_top_k"] == 7
