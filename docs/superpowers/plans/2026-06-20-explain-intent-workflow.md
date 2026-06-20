# explain 意图轴 + 精修工作流 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把"讲清楚"做成一等公民——QA 预处理抽出答案意图轴（v1 二元闸 explain/other），`explain` 走独立的"宽召回→列骨架→每节点检索→教学体合成"工作流，非 explain 原样滑入难度分类。

**Architecture:** 新增三决策单元——`QueryGate`（降噪+意图二判，Call A）、`AnswerOutliner`（据宽 hybrid 召回列概念骨架）、`QueryPreprocessor` 瘦身为难度分类（Call B）。`DocQueryWorkflow.preprocess` 先闸后分：explain→新 `ExplainEvent`→`explain_branch`（委托 `qa.explain`，空骨架落 `qa_agent` 再落单轮）；非 explain→难度分类→现有分支。

**Tech Stack:** Python 3.12 async，LlamaIndex Workflow（step 图）+ get_response_synthesizer 流式合成，DeepSeek（`acomplete` + `json_object`），可插拔 Retriever（hybrid=dense+BM25），Pydantic 校验，pytest（`pytest-asyncio` 已配，async 测试函数直接写）。

## Global Constraints

- **从项目根目录运行**，绝对导入（`from core.workflow.x import Y`），子模块内相对导入。
- 决策单元统一模式：注入 LLM、对外只暴露 `run`、`response_format={"type":"json_object"}`、Pydantic 校验、失败降级、`_strip_fences` 按模块自带副本。
- **所有 I/O `async/await`**；函数签名带类型注解；中文注释可接受。
- **grounding 红线**：合成只从检索 chunk 出事实；开场/收束是串场框架，不引入 chunk 外内容。
- **降级绝不阻塞**：gate 失败→`(原query,"other")`；outliner 空→explain 落 agent→再失败落单轮。
- 每次 commit 用**显式文件路径** `git add <file> ...`，禁止 `git add -A/.`。
- 设计依据：`docs/superpowers/specs/2026-06-20-explain-intent-workflow-design.md`。

**执行前**：已在分支 `feat/explain-intent` 上（spec 已提交于此）。

---

### Task 1: QueryGate 决策单元（降噪 + 意图二判）

新模块自包含，不碰 workflow。

**Files:**
- Create: `core/workflow/query_gate.py`
- Test: `tests/test_query_gate.py`

**Interfaces:**
- Produces: `QueryGate.run(self, clean_query: str) -> tuple[str, str]`，返回 `(denoised_query, intent)`，`intent ∈ {"explain","other"}`。

- [ ] **Step 1: 写失败测试** — 创建 `tests/test_query_gate.py`

```python
"""QueryGate（Call A：检索降噪 + 意图二判）单测。mock LLM 控返回，验解析/降级。"""
from core.workflow.query_gate import QueryGate


class _Resp:
    def __init__(self, t): self._t = t
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


async def test_gate_explain_intent():
    llm = FakeLLM(['{"denoised_query":"MySQL索引","intent":"explain"}'])
    denoised, intent = await QueryGate(llm).run("给我讲讲MySQL索引啊")
    assert denoised == "MySQL索引"
    assert intent == "explain"


async def test_gate_other_intent():
    llm = FakeLLM(['{"denoised_query":"redo日志的LSN是什么","intent":"other"}'])
    denoised, intent = await QueryGate(llm).run("redo日志的LSN是什么")
    assert intent == "other"
    assert denoised == "redo日志的LSN是什么"


async def test_gate_parse_failure_degrades_to_other_original():
    llm = FakeLLM(["这不是JSON"])
    denoised, intent = await QueryGate(llm).run("讲讲数据库")
    assert intent == "other"            # 降级默认 other（落已验证的难度分类路径）
    assert denoised == "讲讲数据库"      # 用原 query


async def test_gate_empty_content_degrades():
    llm = FakeLLM([""])
    denoised, intent = await QueryGate(llm).run("讲讲数据库")
    assert intent == "other"
    assert denoised == "讲讲数据库"


async def test_gate_invalid_intent_rejected():
    llm = FakeLLM(['{"denoised_query":"x","intent":"compare"}'])  # 枚举外
    denoised, intent = await QueryGate(llm).run("讲讲数据库")
    assert intent == "other"            # Pydantic 拒 → 降级
    assert denoised == "讲讲数据库"


async def test_gate_empty_denoised_uses_original():
    llm = FakeLLM(['{"denoised_query":"","intent":"explain"}'])
    denoised, intent = await QueryGate(llm).run("讲讲B+树")
    assert denoised == "讲讲B+树"
    assert intent == "explain"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_query_gate.py -q`
Expected: FAIL（`ModuleNotFoundError: core.workflow.query_gate`）

- [ ] **Step 3: 写实现** — 创建 `core/workflow/query_gate.py`

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_query_gate.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add core/workflow/query_gate.py tests/test_query_gate.py
git commit -m "feat(workflow): 新增 QueryGate 降噪+意图二判决策单元

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: AnswerOutliner 决策单元（据宽召回列概念骨架）

新模块自包含。

**Files:**
- Create: `core/workflow/answer_outliner.py`
- Test: `tests/test_answer_outliner.py`

**Interfaces:**
- Produces: `AnswerOutliner.run(self, query: str, passages: list[str], max_items: int = 8) -> list[str]`，返回子主题列表；空/失败 → `[]`。

- [ ] **Step 1: 写失败测试** — 创建 `tests/test_answer_outliner.py`

```python
"""AnswerOutliner（据宽召回列概念骨架）单测。mock LLM 控返回，验解析/降级。"""
from core.workflow.answer_outliner import AnswerOutliner


class _Resp:
    def __init__(self, t): self._t = t
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


async def test_outline_multi_node():
    llm = FakeLLM(['{"sub_queries":["MySQL索引基础","MySQL事务基础","MySQL锁基础"]}'])
    subs = await AnswerOutliner(llm).run("MySQL基础知识", ["片段1", "片段2"])
    assert subs == ["MySQL索引基础", "MySQL事务基础", "MySQL锁基础"]


async def test_outline_atomic_single_node():
    llm = FakeLLM(['{"sub_queries":["脏读的定义与例子"]}'])  # 原子概念 1 节
    subs = await AnswerOutliner(llm).run("什么是脏读", ["片段"])
    assert subs == ["脏读的定义与例子"]


async def test_outline_passages_passed_to_prompt():
    llm = FakeLLM(['{"sub_queries":["x"]}'])
    await AnswerOutliner(llm).run("讲讲X", ["关键片段ABC"])
    assert "关键片段ABC" in llm.prompts[0]


async def test_outline_respects_max_items():
    llm = FakeLLM(['{"sub_queries":["a","b","c","d"]}'])
    subs = await AnswerOutliner(llm).run("讲讲X", ["片段"], max_items=2)
    assert subs == ["a", "b"]


async def test_outline_empty_on_parse_failure():
    llm = FakeLLM(["这不是JSON"])
    subs = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert subs == []          # 空 → explain 将落 agent 兜底


async def test_outline_empty_on_empty_list():
    llm = FakeLLM(['{"sub_queries":[]}'])
    subs = await AnswerOutliner(llm).run("讲讲X", ["片段"])
    assert subs == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_answer_outliner.py -q`
Expected: FAIL（`ModuleNotFoundError: core.workflow.answer_outliner`）

- [ ] **Step 3: 写实现** — 创建 `core/workflow/answer_outliner.py`

```python
"""AnswerOutliner：据【宽 hybrid 覆盖召回的片段】把"答案"拆成并列概念子主题（骨架）。

explain 专用。骨架对齐【库里实际覆盖】（喂宽召回片段），不绑章节树、不靠模型世界知识。
尺寸自适应：原子概念 1~2 节、宽主题多节，下限 1。空/失败 → []，由 qa.explain 落 agent 兜底。
设计见 docs/superpowers/specs/2026-06-20-explain-intent-workflow-design.md。
"""
import logging
from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

_OUTLINE_PROMPT = """你是答案大纲规划器。下面是一个用户想"讲清楚"的问题，以及在知识库里宽召回到的相关片段。请【只依据召回片段覆盖到的内容】，把"这个问题的答案"拆成若干并列子主题，每个子主题是答案的一个方面/一节，便于逐个检索后分节讲解。

铁律：
- 子主题只能来自召回片段真实覆盖的内容，严禁凭世界知识编库里没有的子主题。
- 数量按概念复杂度【自适应】：原子概念 1~2 个即可，宽主题可多个（最多 {max} 个）。下限 1，不强凑。
- 每个子主题写成一个能独立检索的【完整短句】，含主体技术实体（别只写"应用场景"这种裸限定）。

只返回 JSON，不要其它任何内容：
{"sub_queries":["子主题1","子主题2", ...]}

问题：{query}

召回片段：
{passages}"""


class Outline(BaseModel):
    """LLM 列骨架的目标 schema（代码侧 Pydantic 校验）。"""

    sub_queries: List[str] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class AnswerOutliner:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(
        self, query: str, passages: List[str], max_items: int = 8
    ) -> List[str]:
        prompt = (
            _OUTLINE_PROMPT.replace("{query}", query)
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
            data = Outline.model_validate_json(text)
            subs = [s.strip() for s in data.sub_queries if s and s.strip()][:max_items]
            logger.info("outline: 列出 %d 个子主题：%s", len(subs), " | ".join(subs))
            return subs
        except Exception as exc:
            logger.warning("outline 解析失败，返回空（explain 将落 agent 兜底）：%s", exc)
            return []
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_answer_outliner.py -q`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add core/workflow/answer_outliner.py tests/test_answer_outliner.py
git commit -m "feat(workflow): 新增 AnswerOutliner 据宽召回列概念骨架

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: QaCapability.explain() 精修管线

在 `QaCapability` 加 `gate`/`explain` 方法 + 宽召回 + 教学体合成 + `EmptySkeleton`，全是新增，不碰 `classify`/`split`。

**Files:**
- Modify: `core/workflow/qa_capability.py`
- Test: `tests/test_qa_capability.py`

**Interfaces:**
- Consumes: `core.workflow.query_gate.QueryGate`（Task 1）、`core.workflow.answer_outliner.AnswerOutliner`（Task 2）。
- Produces:
  - `QaCapability.gate(self, clean_query: str) -> tuple[str, str]`
  - `QaCapability.explain(self, ctx, query: str, book_titles) -> tuple[str, list]`（空骨架 raise `EmptySkeleton`）
  - 模块级 `class EmptySkeleton(Exception)`
  - `QaCapability.__init__` 新增参数 `explain_retriever=None, explain_recall_k: int = 12`

- [ ] **Step 1: 写失败测试** — 在 `tests/test_qa_capability.py` 末尾追加

```python
# ── explain：宽召回 → 列骨架 → 每节点检索 → 教学体合成 ──────────────────
import pytest
from core.workflow.qa_capability import EmptySkeleton


async def test_explain_builds_sections_from_skeleton():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    async def fake_recall(query, book_titles):
        return ["w1", "w2"]                      # 宽召回片段

    async def fake_outline(query, passages, max_items=8):
        return ["索引基础", "事务基础"]          # 骨架两节

    async def fake_retrieve_all(sub_queries, book_titles):
        return [["a1"], ["b1"]]                  # 每节点各自命中

    async def fake_synth(ctx, query, nodes):
        return f"[{query}]"

    qa._explain_recall = fake_recall
    qa.outliner.run = fake_outline
    qa._retrieve_all = fake_retrieve_all
    qa._synthesize_stream = fake_synth

    answer, nodes = await qa.explain(ctx, "MySQL基础知识", None)
    assert "## 索引基础" in answer and "## 事务基础" in answer   # 逐节标题
    assert "[索引基础]" in answer                                # 逐节正文来自该节合成
    assert nodes == ["a1", "b1"]                                # 去重合并池


async def test_explain_empty_skeleton_raises():
    qa = _qa(FakeIndexManager(nodes=[]))
    ctx = FakeCtx()

    async def fake_recall(query, book_titles):
        return ["w1"]

    async def fake_outline(query, passages, max_items=8):
        return []                                # 列不出骨架

    qa._explain_recall = fake_recall
    qa.outliner.run = fake_outline

    with pytest.raises(EmptySkeleton):
        await qa.explain(ctx, "讲讲X", None)


async def test_gate_delegates_to_query_gate():
    qa = _qa()

    async def fake_run(clean_query):
        return "降噪后", "explain"

    qa._gate.run = fake_run
    denoised, intent = await qa.gate("原始 query")
    assert (denoised, intent) == ("降噪后", "explain")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_qa_capability.py -k "explain or gate_delegates" -q`
Expected: FAIL（`ImportError: EmptySkeleton` / `AttributeError: explain`）

- [ ] **Step 3: 写实现** — 改 `core/workflow/qa_capability.py`

3a. 顶部 import 加（与既有 `from core.workflow.query_decompose import QueryDecomposer` 同处）：
```python
from core.workflow.query_gate import QueryGate
from core.workflow.answer_outliner import AnswerOutliner
```

3b. 模块级（紧接 `logger = logging.getLogger(__name__)` 之后）加异常：
```python
class EmptySkeleton(Exception):
    """AnswerOutliner 列不出骨架 → 由 explain_branch 落 agent 兜底。"""
```

3c. `__init__` 签名加参数（在 `probe_reranker` 之后）：
```python
        probe_reranker: "Reranker | None" = None,
        explain_retriever: "Retriever | None" = None,
        explain_recall_k: int = 12,
```
并在 `self._retrieve_concurrency = 4` 之前加实例：
```python
        # explain 专用：宽覆盖召回（hybrid + 大 top_k + 不重排，求覆盖不求精）
        self.explain_retriever = explain_retriever or VectorRetriever()
        self.explain_recall_k = explain_recall_k
        self._gate = QueryGate(llm)
        self.outliner = AnswerOutliner(llm)
```

3d. 在 `classify` 方法之前加 `gate` 方法：
```python
    async def gate(self, clean_query: str) -> tuple[str, str]:
        """Call A：检索降噪 + 意图二判（explain / other）。委托 QueryGate。"""
        return await self._gate.run(clean_query)
```

3e. 在 `_probe_retrieve` 之后加 `_explain_recall`：
```python
    async def _explain_recall(self, query: str, book_titles: Optional[list[str]]):
        """explain 宽覆盖召回：hybrid + 大 top_k + 不重排（求"有哪几块"，不求精）。"""
        return await self.explain_retriever.retrieve(
            query, index_manager=self.index_manager,
            book_titles=book_titles, top_k=self.explain_recall_k,
        )
```

3f. 在 `assume` 方法之后加 `explain` 方法：
```python
    async def explain(
        self, ctx: Context, query: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """讲清楚：宽覆盖召回 → 列概念骨架 → 每节点检索 → 教学体分节合成。

        空骨架 → raise EmptySkeleton（由 explain_branch 落 agent 兜底）。
        广度从骨架节点数自然涌现（1 节=单轮、N 节=扇出），不预分类。
        """
        # 1. 宽覆盖召回（内部，不发流事件——空骨架时要静默落 agent，别先污染 UI）
        located = await self._explain_recall(query, book_titles)
        passages = [
            (n.get_content() if hasattr(n, "get_content") else n.text)[:500]
            for n in located
        ]

        # 2. 列骨架
        sub_queries = await self.outliner.run(query, passages)
        if not sub_queries:
            raise EmptySkeleton(query)

        # 3. 每节点检索（此时才发 RetrievalStart）
        ctx.write_event_to_stream(RetrievalStartEvent(query=query))
        retrieved = await self._retrieve_all(sub_queries, book_titles)
        pool = self._merge_pool(retrieved)
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(pool)))

        # 4. 教学体合成：开场全景 → 逐节接地 → 收束（每段只从对应 chunk 出事实）
        parts: list[str] = []
        if pool:
            intro = await self._synthesize_stream(
                ctx, f"请用一段话总览，引出下面要分述的几个方面：{query}", pool
            )
            parts.append(intro)
        for sub_q, ns in zip(sub_queries, retrieved):
            h = f"\n\n## {sub_q}\n"
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=h))
            body = (
                await self._synthesize_stream(ctx, sub_q, ns)
                if ns
                else "（未检索到相关内容）"
            )
            parts.append(h + body)
        if pool:
            outro = await self._synthesize_stream(
                ctx, f"请用一两句话小结上面关于「{query}」的内容", pool
            )
            parts.append("\n\n" + outro)
        return "".join(parts).strip(), pool
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_qa_capability.py -q`
Expected: PASS（原有 + 3 新，全绿）

注：`test_explain_builds_sections_from_skeleton` stub 了 `_synthesize_stream`，故 intro/outro 也走 stub（返回 `[query]` 串），不影响 `## 标题` 与节点池断言。

- [ ] **Step 5: 提交**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(qa): explain 精修管线（宽召回→列骨架→每节点检索→教学体合成）

QaCapability 加 gate()/explain()/EmptySkeleton + 宽 hybrid 覆盖召回。
暂未接入 workflow（Task 4）。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 接入 DocQueryWorkflow（preprocess 先闸后分 + explain_branch）

`preprocess` step 改"先 gate 后分"，新增 `ExplainEvent` + `explain_branch`（空骨架落 agent 再落单轮），`QaCapability` 构造传 hybrid explain_retriever，更新 `test_doc_workflow`。

**Files:**
- Modify: `core/workflow/doc_workflow.py`
- Modify: `tests/test_doc_workflow.py`

**Interfaces:**
- Consumes: `core.workflow.qa_capability.EmptySkeleton`（Task 3）、`qa.gate`/`qa.explain`（Task 3）。
- Produces: `preprocess` 现返回多一个 `ExplainEvent`；`explain_branch` step。

- [ ] **Step 1: 改实现 — import EmptySkeleton + 构造传 explain_retriever**

`doc_workflow.py` 顶部 import 块里，把
```python
from core.workflow.qa_capability import (  # noqa: F401  (事件类 re-export 供 api 层 import)
    AnswerDeltaEvent,
    QaCapability,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)
```
改为（加 `EmptySkeleton`）：
```python
from core.workflow.qa_capability import (  # noqa: F401  (事件类 re-export 供 api 层 import)
    AnswerDeltaEvent,
    EmptySkeleton,
    QaCapability,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)
```

`__init__` 里构造 `QaCapability` 处，把
```python
            probe_retriever=make_retriever(probe_retriever),  # None → VectorRetriever
            probe_reranker=make_reranker(probe_reranker),     # None → None（不重排）
        )
```
改为（加 explain 宽召回默认 hybrid）：
```python
            probe_retriever=make_retriever(probe_retriever),  # None → VectorRetriever
            probe_reranker=make_reranker(probe_reranker),     # None → None（不重排）
            explain_retriever=make_retriever("hybrid"),       # explain 宽覆盖召回默认 hybrid
        )
```

- [ ] **Step 2: 改实现 — 新增 ExplainEvent**

在 `class PreprocessEvent(Event):` 定义之后加：
```python
class ExplainEvent(Event):
    """intent=explain → 讲清楚精修工作流（宽召回→列骨架→每节点检索→教学体合成）。"""
```

- [ ] **Step 3: 改实现 — preprocess step 先闸后分**

把整个 `preprocess` 方法的签名与开头替换。将
```python
    @step
    async def preprocess(
        self, ctx: Context, ev: PreprocessEvent
    ) -> "RetrieveAgentEvent | SplitEvent | AssumeEvent | ClarifyEvent | OtherEvent | OutOfScopeEvent":
        clean_query = await ctx.store.get("clean_query")
        book_titles = await ctx.store.get("book_titles")

        result = await self.qa.classify(clean_query, book_titles, probe=self._probe)

        await ctx.store.set("rewritten_query", result.rewritten_query)
        await ctx.store.set("category", result.category)

        rewritten = result.rewritten_query
```
替换为：
```python
    @step
    async def preprocess(
        self, ctx: Context, ev: PreprocessEvent
    ) -> "RetrieveAgentEvent | SplitEvent | AssumeEvent | ClarifyEvent | OtherEvent | OutOfScopeEvent | ExplainEvent":
        clean_query = await ctx.store.get("clean_query")
        book_titles = await ctx.store.get("book_titles")

        # Call A：检索降噪 + 意图二判。降噪后的检索 query 来源唯一在此（存 ctx 供下游分支）。
        denoised_query, intent = await self.qa.gate(clean_query)
        await ctx.store.set("rewritten_query", denoised_query)
        await ctx.store.set("intent", intent)
        if intent == "explain":
            return ExplainEvent()

        # Call B：难度六分类（非 explain）。rewritten_query 已由 gate 提供，不取 classify 的。
        result = await self.qa.classify(denoised_query, book_titles, probe=self._probe)
        await ctx.store.set("category", result.category)

        rewritten = denoised_query
```
（`match result.category:` 及其后所有 case 分支不动——它们用局部变量 `rewritten`。）

- [ ] **Step 4: 改实现 — 新增 explain_branch（空骨架落 agent 再落单轮）**

在 `out_of_scope_branch` 方法之后加：
```python
    @step
    async def explain_branch(self, ctx: Context, ev: ExplainEvent) -> FinalizeEvent:
        # explain：讲清楚精修工作流。空骨架 → 落有界 agent 多轮探索 → agent 再失败 → 单轮兜底。
        rewritten = await ctx.store.get("rewritten_query")
        book_titles = await ctx.store.get("book_titles")
        try:
            answer, nodes = await self.qa.explain(ctx, rewritten, book_titles)
        except EmptySkeleton:
            logger.info("explain: 空骨架，落 agent 兜底")
            try:
                answer, nodes = await self.qa_agent.run(ctx, rewritten, book_titles)
            except Exception as exc:
                logger.warning("explain agent 兜底失败，降级单轮：%s", exc)
                answer, nodes = await self.qa.retrieve(ctx, rewritten, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

- [ ] **Step 5: 改测试 — `_wf` 默认 stub gate 为 echo+other**

`tests/test_doc_workflow.py` 里把
```python
def _wf(llm, index_manager=None):
    return DocQueryWorkflow(index_manager=index_manager, llm=llm, similarity_top_k=3, timeout=10)
```
替换为（默认 stub gate：回显 clean_query + intent=other，使存量 FakeLLM 队列不变、走难度分类路径）：
```python
def _wf(llm, index_manager=None):
    wf = DocQueryWorkflow(index_manager=index_manager, llm=llm, similarity_top_k=3, timeout=10)

    async def _echo_other_gate(clean_query):
        return clean_query, "other"            # 默认非 explain；explain 测试自行覆盖

    wf.qa.gate = _echo_other_gate
    return wf
```

- [ ] **Step 6: 改测试 — 新增 explain 接线用例**

在 `test_out_of_scope_responds_without_retrieval_or_clarify` 之后加：
```python
async def test_explain_intent_routes_to_explain_branch():
    # front_door dispatch_qa → preprocess → gate intent=explain → explain_branch（不进难度分类）
    llm = FakeLLM(['{"action": "dispatch_qa", "clean_query": "讲讲MVCC"}'])  # 仅 front_door 调 LLM
    wf = _wf(llm)
    called = {"explain": False, "classify": False}

    async def fake_gate(clean_query):
        return clean_query, "explain"

    async def fake_explain(ctx, query, book_titles):
        called["explain"] = True
        assert query == "讲讲MVCC"               # rewritten_query 来自 gate
        return "教学体答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        called["classify"] = True
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable", clean_query)

    wf.qa.gate = fake_gate
    wf.qa.explain = fake_explain
    wf.qa.classify = fake_classify

    result = await wf.run(query="讲讲MVCC", memory=FakeMemory())
    assert called["explain"] is True
    assert called["classify"] is False          # explain 跳过难度分类
    assert str(result.response) == "教学体答案"
    assert result.source_nodes == ["n1"]
    assert llm.calls == 1                        # 只 front_door（gate/explain 都 stub）
```

- [ ] **Step 7: 跑测试确认通过**

Run: `python -m pytest tests/test_doc_workflow.py -q`
Expected: PASS（全绿；存量用例因 gate 默认 stub 为 other、FakeLLM 队列不变而保持，新增 explain 用例通过）

若个别存量用例失败：检查它是否断言 captured query 等于 classify 响应里的 `rewritten_query` 且该值与 clean_query 不同——echo gate 下 rewritten_query=clean_query，把该断言改为对 clean_query 即可。

- [ ] **Step 8: 提交**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): preprocess 先闸后分，explain 走独立 explain_branch

gate(降噪+意图) → explain 走 ExplainEvent/explain_branch（空骨架落 agent 再落单轮），
非 explain 滑入难度分类；rewritten_query 来源上移到 gate。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: QueryPreprocessor 瘦身（去掉 rewritten_query）

降噪已移到 Call A（Task 4 起 doc_workflow 用 gate 的 denoised 作 rewritten_query）。把 `PreprocessResult` 的 `rewritten_query` 字段去掉，`QueryPreprocessor` 不再产出它，难度六分类逻辑不动。

**Files:**
- Modify: `core/workflow/query_preprocess.py`
- Modify: `tests/test_query_preprocess.py`

**Interfaces:**
- Produces: `PreprocessResult(category, reason="", clarify_question="")`（去掉 `rewritten_query`）。

- [ ] **Step 1: 改测试 — 去掉 rewritten_query 的断言与构造位置**

定位全部受影响处：
Run: `grep -rn "rewritten_query\|PreprocessResult(" tests/test_query_preprocess.py tests/test_doc_workflow.py`

两类改动：
- `tests/test_query_preprocess.py`：每处 `assert result.rewritten_query == ...` 删除该行（其同测试通常另有 `assert result.category == ...` 仍覆盖核心行为）；若某测试**仅**断言 rewritten_query，改为断言 `result.category`。
- `tests/test_doc_workflow.py`：`fake_classify` 里的 `PreprocessResult("retrievable", clean_query)`（含 Task 4 新增的 explain 用例）去掉第二个位置参数，改为 `PreprocessResult("retrievable")`——否则 `clean_query` 会被绑到 `reason`（虽不致测试失败，但语义错）。`PreprocessResult("missing_info", ..., reason=..., clarify_question=...)` 这类带关键字的构造若用了位置 `rewritten_query` 也一并去掉。

- [ ] **Step 2: 改实现 — PreprocessResult 去字段 + run 不再回 rewritten_query**

`core/workflow/query_preprocess.py` 把
```python
@dataclass
class PreprocessResult:
    """QA 内部 step1 产出：category 决定 workflow 路由，rewritten_query 进检索。"""

    category: str
    rewritten_query: str
    reason: str = ""
    clarify_question: str = ""
```
改为：
```python
@dataclass
class PreprocessResult:
    """QA 内部 step1 产出：只判 category（降噪/rewritten_query 已上移到 QueryGate/Call A）。"""

    category: str
    reason: str = ""
    clarify_question: str = ""
```

`run` 方法里把
```python
            judgment = QueryJudgment.model_validate_json(text)
            rewritten = (judgment.rewritten_query or clean_query).strip() or clean_query
            result = PreprocessResult(
                judgment.category, rewritten, judgment.reason, judgment.clarify_question
            )
            logger.info("preprocess: category=%s reason=%s", result.category, result.reason)
            return result
        except Exception as exc:
            # 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）都降级为可检索，绝不阻塞
            logger.warning("preprocess 解析失败，降级 retrievable：%s", exc)
            return PreprocessResult("retrievable", clean_query, "")
```
改为：
```python
            judgment = QueryJudgment.model_validate_json(text)
            result = PreprocessResult(
                judgment.category, judgment.reason, judgment.clarify_question
            )
            logger.info("preprocess: category=%s reason=%s", result.category, result.reason)
            return result
        except Exception as exc:
            # 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）都降级为可检索，绝不阻塞
            logger.warning("preprocess 解析失败，降级 retrievable：%s", exc)
            return PreprocessResult("retrievable")
```

注：`QueryJudgment` 仍保留 `rewritten_query` 字段（prompt 的 JSON 示例仍含它，模型照常回；这里只是不再向 `PreprocessResult` 传递）。`_JUDGE_PROMPT` 的降噪步因 query 已上游降噪而成近似 noop，本刀不重写其大段示例，留作后续清理。

- [ ] **Step 3: 跑测试确认通过**

Run: `python -m pytest tests/test_query_preprocess.py tests/test_doc_workflow.py tests/test_qa_capability.py -q`
Expected: PASS（全绿）

- [ ] **Step 4: 提交**

```bash
git add core/workflow/query_preprocess.py tests/test_query_preprocess.py
git commit -m "refactor(preprocess): PreprocessResult 去掉 rewritten_query（来源上移 QueryGate）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 验证（end-to-end，可选手动冒烟）

需 `.env` 的 `DEEPSEEK_API_KEY` + 已建索引。`python -m uvicorn api.main:app --port 8000` 后：
- 「什么是聚簇索引」→ explain 单节骨架 → 一段结构化讲解（不注水）。
- 「讲懂MySQL基础知识」→ explain 多节骨架（索引/事务/锁…）→ 开场全景 + 逐节 + 收束，不再是零散噪声。
- 「redo日志的LSN是什么」→ other → 难度分类→现有分支（行为不变）。
- 看日志 `gate: intent=...`、`outline: 列出 N 个子主题：...`，确认链路与骨架质量。

## 全量回归

Run: `python -m pytest tests/ -q`
Expected: PASS（新增 test_query_gate / test_answer_outliner / qa explain 用例 + 改动的 doc_workflow / query_preprocess 全绿）。

## 自查（spec 覆盖）

- QueryGate 降噪+意图二判 → Task 1。
- AnswerOutliner 据宽召回列骨架 → Task 2。
- qa.explain 宽 hybrid 召回→骨架→每节点检索去重→教学体三段；空骨架 EmptySkeleton → Task 3。
- preprocess 先闸后分、ExplainEvent/explain_branch、空骨架落 agent 再落单轮、rewritten_query 来源上移 → Task 4。
- QueryPreprocessor 瘦身去 rewritten_query → Task 5。
- v1 非目标（lookup/compare/design、多跳专门处理、显式剪枝、教学 prompt 精修、评测度量）→ 未触碰，留以后。
