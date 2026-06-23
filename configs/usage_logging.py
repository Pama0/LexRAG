"""LLM token usage / 缓存命中观测（基于 LlamaIndex instrumentation）。

放在 configs（最低层）：只依赖 llama_index，不碰 core/api，符合分层方向
api → core → configs。在 configure_llm() 里一次性注册，覆盖所有经 LlamaIndex
发出的 LLM chat 调用（router / classify / 合成 …）。

DeepSeek 的缓存命中字段 prompt_cache_hit_tokens / prompt_cache_miss_tokens 只在
【原始 API 响应的 usage】里，客户端 token 计数器拿不到——这里从 ChatResponse.raw.usage
直接抠。流式调用（合成）的 usage 常缺失（DeepSeek 需 stream_options.include_usage），
缺失即跳过，不报错。
"""
import logging
from typing import Any

import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
)

logger = logging.getLogger("llm.usage")

# 同时覆盖 chat（achat：router/合成）与 completion（acomplete：judge）两类结束事件——
# 二者都带 .response.raw.usage，但走不同事件类，漏一个就有调用记不到。
_USAGE_EVENTS = (LLMChatEndEvent, LLMCompletionEndEvent)


class CacheUsageEventHandler(BaseEventHandler):
    """每次 LLM chat 结束时记录 token usage + DeepSeek 缓存命中率。"""

    @classmethod
    def class_name(cls) -> str:
        return "CacheUsageEventHandler"

    def handle(self, event: Any, **kwargs: Any) -> Any:
        if not isinstance(event, _USAGE_EVENTS):
            return
        usage = self._extract_usage(getattr(event, "response", None))
        if usage is None:
            return  # 流式无 usage / 非预期响应：静默跳过

        hit = getattr(usage, "prompt_cache_hit_tokens", None)
        miss = getattr(usage, "prompt_cache_miss_tokens", None)
        completion = getattr(usage, "completion_tokens", None)

        if hit is None and miss is None:
            # 非 DeepSeek 或无缓存字段：退而记总量
            logger.info(
                "LLM usage: prompt=%s completion=%s (无缓存字段)",
                getattr(usage, "prompt_tokens", None), completion,
            )
            return

        total_prompt = (hit or 0) + (miss or 0)
        rate = (hit / total_prompt * 100) if total_prompt else 0.0
        logger.info(
            "LLM cache: hit=%s miss=%s 命中率=%.0f%%（prompt 共 %s tok）completion=%s",
            hit, miss, rate, total_prompt, completion,
        )

    @staticmethod
    def _extract_usage(resp: Any) -> Any | None:
        """从 ChatResponse.raw 取 usage。raw 可能是对象（ChatCompletion）或 dict。"""
        if resp is None:
            return None
        raw = getattr(resp, "raw", None)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw.get("usage")
        return getattr(raw, "usage", None)


_registered = False


def install_usage_logging() -> None:
    """注册一次（幂等）：重复调用（如多次 configure_llm / 测试）不重复挂处理器。"""
    global _registered
    if _registered:
        return
    instrument.get_dispatcher().add_event_handler(CacheUsageEventHandler())
    _registered = True
