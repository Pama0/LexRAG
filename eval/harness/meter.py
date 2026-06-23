"""按行测被测系统（SUT）的 LLM token 消耗。

基于 LlamaIndex **instrumentation dispatcher**（与 configs/usage_logging 同机制）：
监听 LLMChatEndEvent / LLMCompletionEndEvent，优先读响应 raw.usage 的 token；
流式调用（DeepSeek 流式不回 usage）则退回客户端 tokenizer 数 prompt/completion 文本。

为什么不用 TokenCountingHandler（老 callback_manager）：它挂在 llm 实例的
callback_manager 上，一旦下游组件（嵌套 Workflow / FunctionAgent）替换了
`llm.callback_manager`，handler 即被甩脱，之后所有读数恒为 0（实测 workflow 跑到
触发内置 agent 的那条后，本变体剩余行 + 后续 naive/agent 变体全部归零）。dispatcher
是全局的、不会被实例级替换甩脱，稳。

judge（ragas 用另一套 openai 客户端，非 LlamaIndex）不发这些事件 → 天然不计入。
只统计 LLM token，不含 embedding。
"""
from typing import Any

import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
)

# 当前活动 meter（serial 评测下单例；handler 通过它累加，便于逐行 reset/read）。
_ACTIVE: "RunMeter | None" = None
_registered = False
_tokenizer = None


class RunMeter:
    """按行测 token：reset 清零、read 取 {prompt,completion,total}_tokens。"""

    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0

    def reset(self) -> None:
        self.prompt = 0
        self.completion = 0

    def read(self) -> dict:
        return {
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "total_tokens": self.prompt + self.completion,
        }

    def add(self, prompt: int, completion: int) -> None:
        self.prompt += prompt
        self.completion += completion


def _count(text: str) -> int:
    """客户端 tokenizer 计数（流式无 usage 时兜底）。"""
    global _tokenizer
    if not text:
        return 0
    if _tokenizer is None:
        from llama_index.core.utils import get_tokenizer
        _tokenizer = get_tokenizer()
    return len(_tokenizer(text))


def _usage(resp: Any) -> Any | None:
    """从 ChatResponse.raw 取 usage（raw 可能是对象或 dict）。"""
    raw = getattr(resp, "raw", None)
    if raw is None:
        return None
    return raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)


def _messages_text(event: Any) -> str:
    """事件里的输入文本（chat 的 messages / completion 的 prompt）。"""
    parts = []
    for m in (getattr(event, "messages", None) or []):
        c = getattr(m, "content", None)
        if c:
            parts.append(str(c))
    prompt = getattr(event, "prompt", None)
    if prompt:
        parts.append(str(prompt))
    return "\n".join(parts)


def _response_text(resp: Any) -> str:
    if resp is None:
        return ""
    msg = getattr(resp, "message", None)
    if msg is not None and getattr(msg, "content", None):
        return str(msg.content)
    return str(getattr(resp, "text", "") or "")


class _MeterEventHandler(BaseEventHandler):
    """dispatcher 事件 → 累加到当前活动 RunMeter（usage 优先、tokenizer 兜底）。"""

    @classmethod
    def class_name(cls) -> str:
        return "EvalMeterEventHandler"

    def handle(self, event: Any, **kwargs: Any) -> Any:
        if _ACTIVE is None:
            return
        if not isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)):
            return
        resp = getattr(event, "response", None)
        usage = _usage(resp)
        p = getattr(usage, "prompt_tokens", None) if usage is not None else None
        c = getattr(usage, "completion_tokens", None) if usage is not None else None
        if p is None and c is None:
            # 流式无 usage → 客户端 tokenizer 兜底
            p = _count(_messages_text(event))
            c = _count(_response_text(resp))
        _ACTIVE.add(p or 0, c or 0)


def attach_token_meter(llm=None) -> RunMeter:
    """注册全局 dispatcher 上的计量 handler（幂等），返回新的当前 RunMeter。

    llm 参数仅为兼容旧调用签名保留——dispatcher 是全局的，无需挂到具体实例，
    因而也不会被下游替换 llm.callback_manager 甩脱。
    """
    global _ACTIVE, _registered
    _ACTIVE = RunMeter()
    if not _registered:
        instrument.get_dispatcher().add_event_handler(_MeterEventHandler())
        _registered = True
    return _ACTIVE
