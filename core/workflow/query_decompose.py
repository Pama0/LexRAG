"""QueryDecomposer（QA capability 内部）：把宽问题拆成 ≤N 个可检索子查询。

结构主 + 内容辅：输入「章节子树标题（结构，保完整）+ 召回正文（内容，补正文级
实体、去噪）」，由 LLM 在【给定素材】上产出并列子查询——禁止编造文档里没有的
实体。LLM 在此是归纳器，不是知识源，故对训练时未见的概念同样有效。

解析失败 / 空 -> 返回空列表，由调用方（split_branch）降级为单轮检索，绝不阻塞。
"""
from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

# 用 .replace 注入，避免 prompt 内 JSON 示例花括号被 str.format 误当占位符。
_DECOMPOSE_PROMPT = """你是检索 query 拆解器。下面给出一个较宽的问题，以及与它相关的
【章节标题】和【召回正文片段】。请【只依据给定素材】把问题拆成若干并列的子查询，
每个子查询聚焦一个具体子项/小节/对比维度，便于逐个检索。

铁律：
- 子查询只能来自给定的章节标题或召回正文里真实出现的内容，严禁编造素材里没有的实体。
- 若问题是"对比/区别"，子查询应是各对比维度（如"X 与 Y 在适用场景上的区别"）。
- 子查询数量不超过 {max} 个；素材子项更多时，归并或取最重要的若干个。
- 每个子查询是能独立检索的完整短句。

问题：{query}

章节标题：
{headings}

召回正文片段：
{passages}

只返回 JSON，不要其他任何内容：
{"sub_queries": ["子查询1", "子查询2", ...]}"""


class Decomposition(BaseModel):
    """LLM 拆解结果的目标 schema（代码侧 Pydantic 校验）。"""

    sub_queries: List[str] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class QueryDecomposer:
    """注入 LLM，对外只暴露一个 run。便于单测（mock LLM 控拆解输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self,
        clean_query: str,
        headings: List[str],
        passages: List[str],
        max_items: int = 6,
    ) -> List[str]:
        prompt = (
            _DECOMPOSE_PROMPT.replace("{query}", clean_query)
            .replace("{headings}", "\n".join(f"- {h}" for h in headings) or "（无）")
            .replace("{passages}", "\n---\n".join(passages) or "（无）")
            .replace("{max}", str(max_items))
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            data = Decomposition.model_validate_json(text)
            subs = [s.strip() for s in data.sub_queries if s and s.strip()]
            return subs[:max_items]
        except Exception:
            # 任何失败都返回空，交由 split_branch 降级为单轮检索，绝不阻塞
            return []
