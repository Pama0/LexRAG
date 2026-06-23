import logging
from dataclasses import dataclass

from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from pydantic import BaseModel, Field

from core.prompts.template import load_prompt
from core.workflow.summarizer import SUMMARY_MARKER

logger = logging.getLogger(__name__)

CLEAN_PROMPT = load_prompt("clean_prompt")
SPLIT_QUERY_PROMPT = load_prompt("split_query_prompt")

# 门口消指代只取最近几轮历史，别灌全量（省 token，也避免远古上下文误导）
MAX_HISTORY_MSGS = 6

@dataclass
class RoutedSubQuery:
    """拆分后的一个子问题及其路由出口。route → qa_capability 的活契约。"""
    query: str
    action: str = "dispatch_qa"      # dispatch_qa | converse
    reply: str = ""                  # converse 婉拒文案（dispatch_qa 时空）


class _CleanResultModel(BaseModel):
    is_missing_info: bool = Field(default=False,description="是否缺失信息")
    clean_query: str = Field(default="", description="净化后的自包含 query")
    missing_reason: str = Field(default="", description="信息缺失的原因")


class _SplitResultModel(BaseModel):
    """拆分产物：子问题列表（不拆时为单元素）。"""
    sub_queries: list[str] = Field(
        default_factory=list, description="拆分后的子问题，不拆则只含原句一项"
    )



def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()

def format_history(
    memory: ChatMemoryBuffer | None, max_msgs: int = MAX_HISTORY_MSGS
) -> str:
    """取最近几轮历史拼成文本，喂给门口做指代消解 + 对话判断。

    若首条是摘要消息（SUMMARY_MARKER 前缀），【永远保留】它再接最近 max_msgs 条——
    摘要承载被压缩掉的远期上下文，落窗口外被截断则压缩白做。
    """
    if memory is None:
        return ""
    msgs = memory.get()
    if not msgs:
        return ""
    head: list = []
    rest = msgs
    first = msgs[0]
    if first.content and str(first.content).startswith(SUMMARY_MARKER):
        head = [first]
        rest = msgs[1:]
    rest = rest[-max_msgs:]
    return "\n".join(f"{m.role}: {m.content}" for m in (head + rest))


class FrontDoor:
    def __init__(self, llm: LLM, index_manager=None):
        self.llm = llm
        self.index_manager = index_manager

    async def _complete_json(self, prompt: str) -> str:
        """单次 json_object LLM 调用，去围栏，空返回抛错（交由各步降级）。"""
        resp = await self.llm.acomplete(prompt, response_format={"type": "json_object"})
        text = _strip_fences(str(resp)).strip()
        if not text:
            raise ValueError("empty content")
        return text

    async def clean(self, original: str, memory: ChatMemoryBuffer | None) -> tuple[str, bool, str]:
        """original + history → clean_query。失败/空 → 原 query。"""
        history = format_history(memory)
        prompt = (
            CLEAN_PROMPT.replace("{query}", original)
            .replace("{history}", history)
        )
        is_missing_info = False
        missing_reason = ""
        try:
            text = await self._complete_json(prompt)
            c = _CleanResultModel.model_validate_json(text)
            clean_q = (c.clean_query or original).strip() or original
            is_missing_info = c.is_missing_info
            missing_reason = c.missing_reason
        except Exception as exc:
            logger.warning("front_door 净化失败，用原 query：%s", exc)
            clean_q = original
        logger.info("front_door clean: %r", clean_q[:80])
        return clean_q,is_missing_info,missing_reason

    async def split_query(self, clean_query: str) -> list[str]:
        """clean_query → 子问题列表。单次 LLM 调用（同 clean/route），凭模型自身知识判拆分。

        query 在 prompt 模板里被包成【数据】（反注入），失败/空/输出漂移都降级不拆。
        拆错代价低（多检一次或并到一起），故全程从宽，异常一律退回 [clean_query]。
        """
        fallback = [clean_query]
        prompt = SPLIT_QUERY_PROMPT.replace("{query}", clean_query)
        try:
            text = await self._complete_json(prompt)
            r = _SplitResultModel.model_validate_json(text)
            subs = [s.strip() for s in r.sub_queries if s and s.strip()]
            if not subs:
                raise ValueError("empty sub_queries")
            # 锚定校验（纵深防御）：子问题应是原句的子串/降噪改写，而非模型自由发挥。
            # 注入得逞时模型多半会「展开作答」，长度远超原句——据此判异常并降级。
            if any(len(s) > len(clean_query) * 2 for s in subs):
                raise ValueError("split output drifted (suspected injection)")
            logger.info("front_door split: %d 子问题 | %s", len(subs), " || ".join(subs))
            return subs
        except Exception as exc:
            logger.warning("front_door 拆分失败，降级不拆：%s", exc)
            return fallback

