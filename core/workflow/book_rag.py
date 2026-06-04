"""book 知识库 RAG workflow：judge_query → retrieve → synthesize。

judge_query 步骤判定 query 是否够明确：宽泛则自动改写，最多 MAX_ROUNDS 轮，
再进入检索。指代/缺上下文类问题由 Agent 层 system_prompt 解决，不在此处理。
"""
import json
from typing import Optional

from llama_index.core import get_response_synthesizer
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

MAX_ROUNDS = 2

_JUDGE_PROMPT = """你是检索 query 质量判定器。判断下面的 query 作为技术书籍知识库的检索词是否足够明确具体。

判定标准：
- 明确：指向具体的技术概念/章节/问题，能检索到精准内容。
- 不明确：过于宽泛或模糊（如"讲讲数据库"、"介绍一下"），检索会命中很杂。

如果不明确，把它改写得更具体——但只能在原 query 的语义范围内收窄，严禁新增用户没提到的约束或话题。

只返回 JSON，不要其他任何内容：
{{"clear": true 或 false, "rewritten_query": "改写后的 query（若已明确则原样返回）"}}

query：{query}"""


class JudgeEvent(Event):
    query: str
    book_title: Optional[str] = None
    round: int = 0


class RetrieveEvent(Event):
    query: str
    book_title: Optional[str] = None


class SynthesizeEvent(Event):
    query: str
    nodes: list


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 代码块围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class BookRagWorkflow(Workflow):
    def __init__(self, index_manager, llm: LLM, similarity_top_k: int = 5, **kw):
        super().__init__(**kw)
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k

    async def _judge_query(self, query: str) -> tuple[bool, str]:
        """判定 query 是否明确。返回 (clear, query_or_rewrite)。

        解析失败一律当作 clear=True 并用原 query，绝不阻塞检索。
        """
        resp = await self.llm.acomplete(_JUDGE_PROMPT.format(query=query))
        try:
            data = json.loads(_strip_fences(str(resp)))
            clear = bool(data["clear"])
            rewritten = str(data.get("rewritten_query") or query).strip() or query
            return clear, rewritten
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return True, query

    async def _decide(self, query: str, round: int) -> tuple[str, str]:
        """决定下一步。返回 (action, query)，action ∈ {'retrieve', 'rewrite'}。

        达到 MAX_ROUNDS 直接检索，不再调用 LLM。
        """
        if round >= MAX_ROUNDS:
            return "retrieve", query
        clear, rewritten = await self._judge_query(query)
        if clear:
            return "retrieve", query
        return "rewrite", rewritten

    def _make_filters(self, book_title: Optional[str]):
        if not book_title:
            return None
        return MetadataFilters(filters=[
            MetadataFilter(key="book_title", value=book_title),
        ])

    async def _retrieve_nodes(self, query: str, book_title: Optional[str]):
        index = self.index_manager.get_index()
        retriever = index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=self._make_filters(book_title),
        )
        return await retriever.aretrieve(query)

    @step
    async def start(self, ev: StartEvent) -> JudgeEvent:
        return JudgeEvent(
            query=ev.query,
            book_title=getattr(ev, "book_title", None),
            round=0,
        )

    @step
    async def judge(self, ev: JudgeEvent) -> "JudgeEvent | RetrieveEvent":
        action, q = await self._decide(ev.query, ev.round)
        if action == "retrieve":
            return RetrieveEvent(query=q, book_title=ev.book_title)
        return JudgeEvent(query=q, book_title=ev.book_title, round=ev.round + 1)

    @step
    async def retrieve(self, ev: RetrieveEvent) -> "SynthesizeEvent | StopEvent":
        nodes = await self._retrieve_nodes(ev.query, ev.book_title)
        if not nodes:
            return StopEvent(result=Response(response="", source_nodes=[]))
        return SynthesizeEvent(query=ev.query, nodes=nodes)

    @step
    async def synthesize(self, ev: SynthesizeEvent) -> StopEvent:
        synthesizer = get_response_synthesizer(llm=self.llm)
        response = await synthesizer.asynthesize(query=ev.query, nodes=ev.nodes)
        return StopEvent(result=response)
