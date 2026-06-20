"""Call A：检索降噪 + 答案意图二判（explain / other）。

把原 QueryPreprocessor 的"降噪"步抽出来，并加意图判定。意图判"答案形状"（要不要讲透），
不需检索；intent=explain 走 explain 精修工作流，other 滑入难度分类。
单次 LLM 结构化决策，注入 LLM、json_object + Pydantic 校验、失败降级。
设计见 docs/superpowers/specs/2026-06-20-explain-intent-workflow-design.md。
"""
import logging
from typing import Literal

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# 用 .replace 注入，避免 JSON 示例花括号被 str.format 误当占位符。
_GATE_PROMPT = """你是检索 query 处理器。下面的 query 已净化（指代已消解、错别字已纠正）。做两件事：

第一步 降噪：去掉口语化/礼貌/请求词，保留关键词、实体、技术名词、限定词。已干净则不动，不要强行改写。

第二步 判意图（二选一，判【用户想要什么形状的答案】，不是判话题）：
- explain：用户想【理解 / 讲清楚 / 讲透】一个概念或主题（如"什么是X""讲讲X""讲懂X""X的原理是什么""X是怎么回事"）。
- other：其余一切——查具体事实、对比、设计方案、操作步骤、罗列等，交给下游难度分类处理。
拿不准 → other。

只返回 JSON，不要其它任何内容：
{"denoised_query":"降噪后的检索 query","intent":"explain / other"}

query：{query}"""


class GateDecision(BaseModel):
    """LLM 判定目标 schema（代码侧 Pydantic 校验）。intent 用 Literal 锁枚举，非法值被拒。"""

    denoised_query: str = Field(default="")
    intent: Literal["explain", "other"] = "other"


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class QueryGate:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self, clean_query: str) -> tuple[str, str]:
        prompt = _GATE_PROMPT.replace("{query}", clean_query)
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            d = GateDecision.model_validate_json(text)
            denoised = (d.denoised_query or clean_query).strip() or clean_query
            logger.info("gate: intent=%s denoised=%r", d.intent, denoised[:80])
            return denoised, d.intent
        except Exception as exc:
            # 任何失败 → other + 原 query（落已验证的难度分类路径，最安全）
            logger.warning("gate 解析失败，降级 other + 原 query：%s", exc)
            return clean_query, "other"
