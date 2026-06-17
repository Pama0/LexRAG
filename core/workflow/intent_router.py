"""Intent Router（Layer 1 门口）：通用净化 + 意图分类。

职责：把【用户原始 query + 会话历史】→【clean_query + intent】。
- 指代消解（它/这个/上面说的 → 自包含问句，读会话历史）
- 规范化（纠错别字、统一全半角、展无歧义缩写）
- 意图分类 → qa / study_plan / ...

这是横切的「干净、自包含、规范 query」产出层，所有 capability 共享其结果。
检索专属的【降噪 + 难度分类】不在此，留在 QA capability 内部（query_preprocess）。

不持有 memory，不碰 ctx，不做 dispatch——dispatch 是 workflow 编排层的事。
解析失败一律降级为 intent=qa + clean_query=原 query，绝不阻塞。
"""
import logging
from dataclasses import dataclass
from typing import Literal, Optional

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer

from core.workflow.summarizer import SUMMARY_MARKER

logger = logging.getLogger(__name__)

# 门口消指代只取最近几轮历史，别灌全量（省 token，也避免远古上下文误导）
MAX_HISTORY_MSGS = 6

# 意图 taxonomy v1：qa（RAG 问答）+ study_plan（占位，仅验证 dispatch 这条缝）
Intent = Literal["qa", "study_plan", "chitchat"]

# 用 {history} 把指代补全 + 规范化，再分类意图。用 .replace 注入，避免 prompt 内
# JSON 示例的花括号被 str.format 误当占位符。
# 【prompt 顺序约定】稳定指令在前、每轮变化的输入（history/scope/query）在末尾，
# 让前缀命中 DeepSeek 上下文缓存（context caching）。改动顺序前先想清楚这点。
_ROUTER_PROMPT = """你是请求入口处理器，对下面的 query 依次做两件事：先净化，再分类意图。

第一步 净化（产出 clean_query，自包含、规范）：
1) 指代消解：用对话历史 +「当前选中的书」把 query 中的指代词补全为不依赖上文、能独立成立的句子。
   - 指代会话内容的（它/这个/上面说的/前面提到的）→ 用对话历史补全。
   - 指代选中书籍的（这本书/本书/该书/这本/这部书）→ 用「当前选中的书」补全（如选中《openclaw》，"这本书讲了什么"→"《openclaw》讲了什么"）。若选中多本而指代不明，保留原词交由下游判定。
   无指代则不动。
2) 规范化（只改形式不改意图）：纠正错别字、明显的同音/形近字错误（如"装饰起"→"装饰器"），统一全半角/大小写，仅展开毫无歧义的常见技术缩写（如 K8s→Kubernetes）。不确定时保留原词。
净化只补全指代 + 修形式，严禁改变用户意图或新增用户没提到的话题。若 query 已自包含且规范，clean_query 原样返回。

第二步 意图分类（基于 clean_query）：
- qa：针对已入库书籍/文档内容的具体问答（如"X是什么""X和Y的区别""第3章讲了什么"）。
- study_plan：要求基于某本书/文档生成结构化学习计划或学习路线（如"给我做份学Redis的计划""按这本书排个学习路线"）。
- chitchat：寒暄、问候、致谢、闲聊，或与知识库内容无关的元问题（如"你好""谢谢""你是谁""你能做什么"）。这类不需要检索知识库。
拿不准是否要检索时默认 qa；明显是寒暄/无关闲聊才判 chitchat。

intent 仅取 [qa|study_plan|chitchat]，clean_query 始终返回净化后的 query。
结果只返回 JSON，不要其他任何内容：
{"intent": "qa / study_plan / chitchat", "clean_query": "净化后的 query"}

对话历史：
{history}

当前选中的书：{scope}

query：{query}"""


@dataclass
class RouterResult:
    """门口产出：intent 决定 dispatch，clean_query 供所有 capability 共享。"""

    intent: str
    clean_query: str


class RouterDecision(BaseModel):
    """LLM 判定的目标 schema。

    DeepSeek 稳定端点只有 json_object（保语法不保 schema），故本模型不发给模型做
    约束，而用于【代码侧】对返回 JSON 做 Pydantic 校验。intent 用 Literal 锁枚举，
    非法值（如 life_plan）会在 model_validate 阶段被拒、走降级。
    """

    intent: Literal["qa", "study_plan", "chitchat"]
    clean_query: str = Field(..., min_length=1, description="净化后的自包含 query")


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def format_history(
    memory: Optional[ChatMemoryBuffer], max_msgs: int = MAX_HISTORY_MSGS
) -> str:
    """取最近几轮历史拼成文本，喂给门口做指代消解。

    若首条是摘要消息（SUMMARY_MARKER 前缀），【永远保留】它再接最近 max_msgs 条——
    摘要承载被压缩掉的远期上下文，落在窗口外会被截断则压缩白做。
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


def format_scope(book_titles: Optional[list[str]]) -> str:
    """把用户选中的书拼成文本，喂给门口消解"这本书"类指代。"""
    if not book_titles:
        return "（用户未选择特定书籍，范围为全部已入库书籍）"
    return "".join(f"《{t}》" for t in book_titles)


class IntentRouter:
    """注入 LLM，对外只暴露一个 run。便于单测（mock LLM 控分类输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self,
        original: str,
        memory: Optional[ChatMemoryBuffer] = None,
        book_titles: Optional[list[str]] = None,
    ) -> RouterResult:
        history = format_history(memory)
        scope = format_scope(book_titles)
        prompt = (
            _ROUTER_PROMPT.replace("{query}", original)
            .replace("{history}", history)
            .replace("{scope}", scope)
        )
        try:
            # json_object 模式保 JSON 语法合法（DeepSeek 稳定端点能力）；
            # 只能按调用传，别塞进全局 llm，否则 agent/synthesizer 也被迫 json 模式。
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                # DeepSeek json 模式偶发空 content，专门兜底
                raise ValueError("empty content")
            # schema 校验交给 Pydantic（json_object 不保 schema，这步才是约束）
            decision = RouterDecision.model_validate_json(text)
            clean = (decision.clean_query or original).strip() or original
            logger.info("router: intent=%s clean_query=%r", decision.intent, clean[:80])
            return RouterResult(decision.intent, clean)
        except Exception as exc:
            # 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）都降级为 qa + 原 query，绝不阻塞
            logger.warning("router 解析失败，降级 intent=qa + 原 query：%s", exc)
            return RouterResult("qa", original)
