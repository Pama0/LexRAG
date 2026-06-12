"""DimensionExtractor（QA capability 内部）：把"角度不定"的问题归纳成 ≤N 个评判维度。

输入「问题 + 召回正文」，由 LLM 在【给定素材】上产出并列的评判维度（如 性能 / 一致性 /
成本），每个维度含 label（维度名，进分节标题与角度声明）+ query（该维度的检索子查询）。
铁律：只依据召回正文，严禁编造素材里没有的维度。LLM 在此是归纳器，不是知识源。

解析失败 / 空 -> 返回空列表，由调用方（assume）降级为单轮检索，绝不阻塞。
"""
from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

# 用 .replace 注入，避免 prompt 内 JSON 示例花括号被 str.format 误当占位符。
_DIMENSION_PROMPT = """你是检索角度归纳器。下面给出一个"主题已具体、但回答角度/评判维度不定"的问题，以及与它相关的【召回正文片段】。请【只依据给定素材】归纳出若干个并列的评判维度，便于分角度回答。

铁律：
- 维度只能来自召回正文里真实出现的内容，严禁编造素材里没有的维度。
- 每个维度给 label（简短维度名，如"读写性能""数据一致性""成本"）和 query（该维度下能独立检索的完整子查询）。
- 维度数量不超过 {max} 个；取最重要、区分度最高的若干个。

问题：{query}

召回正文片段：
{passages}

只返回 JSON，不要其他任何内容：
{"dimensions": [{"label": "维度名1", "query": "子查询1"}, {"label": "维度名2", "query": "子查询2"}]}"""


class Dimension(BaseModel):
    """单个评判维度：label 进分节标题/角度声明，query 进检索。"""

    label: str = ""
    query: str = ""


class DimensionSet(BaseModel):
    """LLM 归纳结果的目标 schema（代码侧 Pydantic 校验）。"""

    dimensions: List[Dimension] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class DimensionExtractor:
    """注入 LLM，对外只暴露一个 run。便于单测（mock LLM 控归纳输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self,
        clean_query: str,
        passages: List[str],
        max_items: int = 6,
    ) -> List[Dimension]:
        prompt = (
            _DIMENSION_PROMPT.replace("{query}", clean_query)
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
            data = DimensionSet.model_validate_json(text)
            dims = [
                Dimension(label=d.label.strip(), query=d.query.strip())
                for d in data.dimensions
                if d.label and d.label.strip() and d.query and d.query.strip()
            ]
            return dims[:max_items]
        except Exception:
            # 任何失败都返回空，交由 assume 降级为单轮检索，绝不阻塞
            return []
