"""统一流式合成 agent：所有路线（dispatch_qa / converse / clarify）的唯一出口嘴巴。

拿【用户原始问题 + format_history(memory) + 各路线材料】→ 一次 astream_complete 流式合成
最终回复（逐 token 发 AnswerDeltaEvent）。不带工具、不检索（不做补洞 agent）。
"""
import logging

from llama_index.core.llms import LLM

from core.prompts.template import load_prompt
from core.workflow.qa_capability import AnswerDeltaEvent, Material, REFUSAL_TEXT

logger = logging.getLogger(__name__)

COMBINE_PROMPT = load_prompt("combine_prompt")


class Combiner:
    """统一合成 agent：材料 → 流式产出唯一用户可见回复。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def combine(
        self, ctx, original: str, history: str, materials: list[Material]
    ) -> tuple[str, list]:
        """流式合成最终回复。逐 token 发 AnswerDeltaEvent；失败/空 → REFUSAL_TEXT 兜底。"""
        block = self._format_materials(materials)
        prompt = (
            COMBINE_PROMPT.replace("{question}", original or "")
            .replace("{history}", history or "")
            .replace("{materials}", block)
        )
        parts: list[str] = []
        try:
            handle = await self.llm.astream_complete(prompt)
            async for chunk in handle:
                delta = chunk.delta or ""
                if delta:
                    ctx.write_event_to_stream(AnswerDeltaEvent(delta=delta))
                    parts.append(delta)
        except Exception as exc:
            logger.warning("combine 合成失败，兜底拒答：%s", exc)
        text = "".join(parts).strip()
        if not text:
            text = REFUSAL_TEXT
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=text))
        nodes = [n for m in materials for n in m.nodes]
        return text, nodes

    @staticmethod
    def _format_materials(materials: list[Material]) -> str:
        """把材料拼成 combine_prompt 约定的文本块（与 prompt 的「各类材料怎么处理」对齐）。"""
        lines: list[str] = []
        for m in materials:
            if m.verdict == "ok":
                lines.append(f"【子问题】{m.query}\n【已检索分答案】{m.answer}")
            elif m.verdict == "converse":
                lines.append(f"【交流/元问题】{m.query}")
            elif m.verdict in ("missing_info", "clarify"):
                lines.append(f"【需澄清】{m.query}：{m.reason or '信息不足，需向用户追问'}")
            elif m.verdict == "out_of_scope":
                lines.append(f"【库外】{m.query}：知识库未收录相关内容")
        return "\n\n".join(lines) if lines else "（无材料）"
