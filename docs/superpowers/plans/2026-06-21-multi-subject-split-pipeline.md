# 多主体拆分管线（Plan 1：核心）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 QA 管线从"原子可答性判定 + 分解在后"改成"先拆多主体 → 每个子问题独立判可答性/类型 → 执行 → 合并"，修掉复合问题被一个判决整体拒答/放行的根因。

**Architecture:** 入口新增 `QuerySplitter`（降噪 + 多主体拆分）；新增 `QueryClassifier`（4 类 `explain/compare/simple/complex`，替代 gate.intent + 旧 4 类 judge）；`QaCapability.answer` 编排逐子问题 `probe→admit→classify→执行` 并合并；`DocQueryWorkflow` step 图收敛为 `route→split_answer→finalize`；删 `QueryGate`/`QueryPreprocessor` 及旧分支。strangler 顺序：先建新单元（不接线）→ 建编排 → 切换 → 删旧。

**Tech Stack:** Python 3.12 async、LlamaIndex Workflow、DeepSeek（`OpenAILike`，`json_object` 模式）、Pydantic v2 校验、pytest（`asyncio_mode=auto`）。

## Global Constraints

- 所有 I/O 用 `async/await`；函数签名加类型注解；中文注释可接受。（CLAUDE.md）
- 各判定单元沿用约定：注入 LLM、对外只暴露 `run`、`acomplete(prompt, response_format={"type":"json_object"})`、Pydantic `model_validate_json` 校验、**失败优雅降级、绝不阻塞**、自带 `_strip_fences` 副本。
- prompt 用 `.replace("{x}", ...)` 注入，**不用 `str.format`**（避免 JSON 示例花括号被误当占位符）。
- prompt 顺序：稳定指令在前、每轮变化输入（evidence/query）在末尾，命中 DeepSeek 缓存。
- 降级方向：splitter 失败→不拆（原问题单元素）；admit 失败→`ok`；classifier 失败→`simple`；complex agent 异常→单轮 retrieve。
- 依赖方向单向 `api → core → configs`；`core` 不依赖 `api`。守卫 `python scripts/check_layering.py`。
- 从项目根目录运行；测试用 `.venv\Scripts\python.exe -m pytest`。
- 类别枚举锁定 `explain / compare / simple / complex`（Pydantic `Literal`）。
- 跨轮 missing_info 本计划用 MVP：当轮把反问追加进答案，**不持久化**（`PendingClarification` 在 Plan 2）。

---

### Task 1: QuerySplitter（多主体拆分器，新单元，不接线）

**Files:**
- Create: `core/workflow/query_splitter.py`
- Test: `tests/test_query_splitter.py`

**Interfaces:**
- Consumes: 注入的 `LLM`（`acomplete`）。
- Produces: `QuerySplitter(llm).run(clean_query: str) -> list[str]`，返回 ≥1 个降噪后的自包含子问题；不可拆 → 单元素列表（降噪后的原问题）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_splitter.py
"""QuerySplitter 单测：mock LLM 控返回，验证拆分解析 / 单问题透传 / 降级。

拆分质量（多主体 vs 比较 vs 多跳的边界判断）依赖真 LLM，不在单测范围。
设计见 docs/superpowers/specs/2026-06-21-multi-subject-split-pipeline-design.md。
"""
from core.workflow.query_splitter import QuerySplitter


class _Resp:
    def __init__(self, text): self._t = text
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


async def test_splits_independent_multi_subject():
    llm = FakeLLM(['{"sub_queries":["MySQL的锁有哪些","Redis的持久化机制"]}'])
    subs = await QuerySplitter(llm).run("讲讲MySQL的锁和Redis的持久化")
    assert subs == ["MySQL的锁有哪些", "Redis的持久化机制"]


async def test_single_question_returns_one_element():
    llm = FakeLLM(['{"sub_queries":["什么是聚簇索引"]}'])
    subs = await QuerySplitter(llm).run("什么是聚簇索引啊")
    assert subs == ["什么是聚簇索引"]


async def test_empty_sub_queries_degrades_to_original():
    llm = FakeLLM(['{"sub_queries":[]}'])
    subs = await QuerySplitter(llm).run("讲讲MySQL")
    assert subs == ["讲讲MySQL"]


async def test_parse_failure_degrades_to_original(caplog):
    import logging
    llm = FakeLLM(["这不是JSON"])
    with caplog.at_level(logging.WARNING):
        subs = await QuerySplitter(llm).run("讲讲MySQL")
    assert subs == ["讲讲MySQL"]
    assert any("splitter" in r.getMessage().lower() for r in caplog.records)


async def test_query_in_prompt():
    llm = FakeLLM(['{"sub_queries":["x"]}'])
    await QuerySplitter(llm).run("讲讲MySQL和Redis的区别")
    assert "讲讲MySQL和Redis的区别" in llm.prompts[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_query_splitter.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.workflow.query_splitter'`

- [ ] **Step 3: Write minimal implementation**

```python
# core/workflow/query_splitter.py
"""QuerySplitter：QA 入口的多主体拆分器（降噪 + 拆分二合一）。

职责：把【已净化的 clean_query】→【≥1 个降噪后的自包含子问题】。
- 降噪：去口语/礼貌/请求词，留实体/技术名词/限定词（原 QueryGate 的降噪职责并入）。
- 拆分：仅拆"显式并列、话题独立、无比较词、无依赖"的多主体问题；比较/多跳/广度
  发散/话题共享的居中句式一律【不拆】，返回单元素，交下游 classifier 判类型。

只看问题文本，不检索。解析失败/空 → 单元素（原 query），绝不阻塞。
设计见 docs/superpowers/specs/2026-06-21-multi-subject-split-pipeline-design.md。
"""
import logging
from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# 用 .replace 注入，避免 JSON 示例花括号被 str.format 误当占位符。
_SPLIT_PROMPT = """你是检索 query 预处理器。下面的 query 已净化（指代已消解、错别字已纠正）。做两件事：先降噪，再判断是否需要拆成多个独立子问题。

第一步 降噪：去掉口语化/礼貌/请求词，保留关键词、实体、技术名词、限定词。已干净则不动，不要强行改写。

第二步 拆分（只以"多主体"为判据，宁可不拆）：
【拆】同时满足：① 显式并列（A和B、A与B、A、B分别…）；② 两侧话题不同（"A的x和B的y"）或带"分别/各自"标记；③ 无比较/对比/区别词；④ 无依赖（后半不靠前半的答案）。把每个子问题写成降噪后、能独立检索的自包含短句。
【不拆】（任一即整体作为单元素返回）：
  · 比较/评价："A和B的区别""A和B哪个好"——不拆。
  · 多跳依赖：后半要先知道前半的答案——不拆。
  · 单主题广度发散："怎么优化X""讲懂X的核心概念"——不拆。
  · 话题共享且无"分别"标记的居中句式："讲讲A和B的缓存机制"——默认不拆。
铁律：拆是不可逆的（拆开就回不到跨主体对照），拿不准一律不拆，返回单元素。

无论拆不拆，sub_queries 都是降噪后的自包含短句；不拆时只含 1 个元素。

只返回 JSON，不要其它任何内容：
{"sub_queries": ["子问题1", "子问题2", ...]}

query：{query}"""


class SplitResult(BaseModel):
    """LLM 拆分结果的目标 schema（代码侧 Pydantic 校验）。"""

    sub_queries: List[str] = Field(default_factory=list)


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class QuerySplitter:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self, clean_query: str) -> list[str]:
        prompt = _SPLIT_PROMPT.replace("{query}", clean_query)
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            result = SplitResult.model_validate_json(text)
            subs = [s.strip() for s in result.sub_queries if s and s.strip()]
            if not subs:
                raise ValueError("empty sub_queries")
            logger.info("splitter: %d 个子问题 %r", len(subs), subs)
            return subs
        except Exception as exc:
            logger.warning("splitter 解析失败，降级不拆（原 query）：%s", exc)
            return [clean_query]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_query_splitter.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add core/workflow/query_splitter.py tests/test_query_splitter.py
git commit -m "feat: add QuerySplitter (多主体拆分 + 降噪入口单元)"
```

---

### Task 2: QueryClassifier（4 类类型分类器，新单元，不接线）

**Files:**
- Create: `core/workflow/query_classifier.py`
- Test: `tests/test_query_classifier.py`

**Interfaces:**
- Consumes: 注入的 `LLM`；调用方喂 `evidence`（probe 召回格式化文本）。
- Produces: `QueryClassifier(llm).run(query: str, evidence: str = "") -> ClassifyResult`，`ClassifyResult.category ∈ {explain, compare, simple, complex}`，`ClassifyResult.reason: str`。失败 → `ClassifyResult("simple")`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_query_classifier.py
"""QueryClassifier 单测：mock LLM 控返回，验证 4 类解析 / 证据进 prompt / 降级。

分类质量依赖真 LLM，不在单测范围。
设计见 docs/superpowers/specs/2026-06-21-multi-subject-split-pipeline-design.md。
"""
import pytest

from core.workflow.query_classifier import QueryClassifier, ClassifyResult


class _Resp:
    def __init__(self, text): self._t = text
    def __str__(self): return self._t


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
    async def acomplete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(self._responses.pop(0))


@pytest.mark.parametrize("cat", ["explain", "compare", "simple", "complex"])
async def test_parses_each_category(cat):
    llm = FakeLLM([f'{{"category":"{cat}","reason":"r"}}'])
    res = await QueryClassifier(llm).run("讲讲MySQL索引", "召回片段")
    assert isinstance(res, ClassifyResult)
    assert res.category == cat


async def test_evidence_in_prompt():
    llm = FakeLLM(['{"category":"simple","reason":""}'])
    await QueryClassifier(llm).run("MySQL有哪些锁", "命中3段：第8章 锁")
    assert "命中3段：第8章 锁" in llm.prompts[0]
    assert "MySQL有哪些锁" in llm.prompts[0]


async def test_illegal_category_degrades_simple():
    llm = FakeLLM(['{"category":"banana","reason":"x"}'])
    res = await QueryClassifier(llm).run("MySQL锁", "")
    assert res.category == "simple"


async def test_parse_failure_degrades_simple(caplog):
    import logging
    llm = FakeLLM(["不是JSON"])
    with caplog.at_level(logging.WARNING):
        res = await QueryClassifier(llm).run("MySQL锁", "")
    assert res.category == "simple"
    assert any("classifier" in r.getMessage().lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_query_classifier.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.workflow.query_classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# core/workflow/query_classifier.py
"""QueryClassifier：可答性闸放行后的类型分类器（替代 gate.intent + 旧 4 类 judge）。

吃 query + probe 召回证据，判 4 类【答案形状/难度】：
- explain：想理解/讲透一个概念（"什么是X""讲讲X""X的原理"）。
- compare：比较/评价（"A和B的区别""A和B哪个好""X做缓存好吗"）——继承原 ambiguous 路线。
- simple：单一信息需求，一条检索能集中命中（原 retrievable）。
- complex：多跳依赖 / 单主题广度发散 / 开放综合权衡（原 other + pending_split），交有界 agent。

可答性（out_of_scope/missing_info）已由前置 Admitter 判完、非 ok 不会走到这里。
解析失败/空/非法 → simple（最便宜确定路径），绝不阻塞。
设计见 docs/superpowers/specs/2026-06-21-multi-subject-split-pipeline-design.md。
"""
import logging
from dataclasses import dataclass
from typing import Literal

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# 用 .replace 注入；稳定指令在前、证据/query 在末尾命中缓存。
_CLASSIFY_PROMPT = """你是检索 query 类型分类器。下面的 query 已确认可答（库内有相关内容）。请判定它属于哪一类【答案形状/难度】。

【铁律·必读】判定以末尾【知识库探测召回】为准，绝不以"你是否认识问题中的词"为准。库里全是你训练时没见过的专名（书名/工具名/项目名），其含义由检索决定。绝不要因为不认识某个词就判 complex。

四类（先判 explain/compare，再在其余里分 simple/complex）：

- explain：用户想【理解 / 讲清楚 / 讲透】一个概念或主题（"什么是X""讲讲X""讲懂X""X的原理是什么""X是怎么回事"）。
  返回 {"category":"explain","reason":"理由"}

- compare：【比较 / 评价 / 选型】两个或多个对象，或对一个对象求某种立场/角度的评价。
  如「Vue和React的区别」「Vue和React哪个好」「Redis做缓存好吗」「MySQL大表查询慢怎么优化」（有多个角度可选）。
  返回 {"category":"compare","reason":"理由"}

- simple：单一信息需求，**一条检索 query 就能集中命中**——哪怕答案要枚举若干项，只要集中在同一片区域。
  如「MySQL有哪些锁」（锁列在同一节，一次命中）。旁证：末尾召回命中集中在 1 个章节、有明显主导章 → 倾向 simple。
  返回 {"category":"simple","reason":"理由"}

- complex：需要【多跳依赖检索、单一大主题铺成多个子领域、或开放设计/权衡】，单轮答不全，须多轮检索+推理。
  · 多跳依赖：后一步查什么要看前一步检索回的答案（如「MySQL默认隔离级别会有哪些并发问题」——先查默认级别是RR，再查RR的并发问题）。
  · 广度发散：单一大主题散在多个互不重叠子领域（如「怎么优化MySQL」索引/查询/配置/架构散在多章）；旁证：末尾召回跨多个章节、无明显主导章。
  返回 {"category":"complex","reason":"理由"}

category 仅为 [explain|compare|simple|complex]，结果只返回 JSON，不要其它任何内容。

系统已用该 query 在知识库做了一次探测检索：
【知识库探测召回】
{evidence}

query：{query}"""


@dataclass
class ClassifyResult:
    """类型分类产出。"""

    category: str
    reason: str = ""


class ClassifyJudgment(BaseModel):
    """LLM 判定目标 schema（代码侧 Pydantic 校验）。category 用 Literal 锁枚举。"""

    category: Literal["explain", "compare", "simple", "complex"]
    reason: str = Field(default="")


def _strip_fences(text: str) -> str:
    """去掉 LLM 偶尔包裹的 ```json ... ``` 围栏。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


class QueryClassifier:
    """注入 LLM，对外只暴露 run。便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm

    async def run(self, query: str, evidence: str = "") -> ClassifyResult:
        prompt = (
            _CLASSIFY_PROMPT.replace("{query}", query)
            .replace("{evidence}", evidence or "（系统未能探测知识库，请仅依据问题文本判定）")
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            j = ClassifyJudgment.model_validate_json(text)
            logger.info("classifier: category=%s reason=%s", j.category, j.reason)
            return ClassifyResult(j.category, j.reason)
        except Exception as exc:
            logger.warning("classifier 解析失败，降级 simple：%s", exc)
            return ClassifyResult("simple")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_query_classifier.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add core/workflow/query_classifier.py tests/test_query_classifier.py
git commit -m "feat: add QueryClassifier (explain/compare/simple/complex 类型分类)"
```

---

### Task 3: qa._decide_subq —— 逐子问题判定（probe→admit→classify）

**Files:**
- Modify: `core/workflow/qa_capability.py`（`__init__` 注入 splitter/classifier；新增 `_SubDecision`、`_decide_subq`）
- Test: `tests/test_qa_capability.py`（追加）

**Interfaces:**
- Consumes: `QuerySplitter.run`、`QueryClassifier.run`、现有 `Admitter.run`、`_probe_retrieve`、`_format_probe`。
- Produces: `qa._decide_subq(q: str, book_titles, probe: bool=True) -> _SubDecision`；`_SubDecision(query, verdict, category, reason, clarify_question)`，`verdict ∈ {ok, missing_info, out_of_scope}`，`category` 仅 ok 时非空。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qa_capability.py 追加（文件顶部已有 _qa / FakeCtx 等替身）
from core.workflow.admitter import AdmitVerdict
from core.workflow.query_classifier import ClassifyResult


def _qa_for_decide():
    qa = _qa()
    async def fake_probe(q, bt): return []
    qa._probe_retrieve = fake_probe
    qa._format_probe = lambda nodes, bt: "EVIDENCE"
    return qa


async def test_decide_subq_out_of_scope_short_circuits_classify():
    qa = _qa_for_decide()
    async def fake_admit(q, passages): return AdmitVerdict(verdict="out_of_scope", reason="库外")
    qa.admitter.run = fake_admit
    async def boom(q, e): raise AssertionError("classify 不该被调用")
    qa.classifier.run = boom
    d = await qa._decide_subq("PostgreSQL的MVCC", None)
    assert d.verdict == "out_of_scope"
    assert d.category == ""


async def test_decide_subq_missing_info_carries_clarify():
    qa = _qa_for_decide()
    async def fake_admit(q, passages):
        return AdmitVerdict(verdict="missing_info", reason="指代不明", clarify_question="你说的索引指哪个？")
    qa.admitter.run = fake_admit
    d = await qa._decide_subq("这个索引的应用场景", None)
    assert d.verdict == "missing_info"
    assert d.clarify_question == "你说的索引指哪个？"


async def test_decide_subq_ok_runs_classifier():
    qa = _qa_for_decide()
    async def fake_admit(q, passages): return AdmitVerdict(verdict="ok")
    qa.admitter.run = fake_admit
    async def fake_classify(q, e): return ClassifyResult("complex", "多跳")
    qa.classifier.run = fake_classify
    d = await qa._decide_subq("MySQL默认隔离级别有哪些并发问题", None)
    assert d.verdict == "ok"
    assert d.category == "complex"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k decide_subq -q`
Expected: FAIL — `AttributeError: 'QaCapability' object has no attribute 'classifier'`（及 `_decide_subq`）

- [ ] **Step 3: Write minimal implementation**

在 `core/workflow/qa_capability.py` 顶部 import 区追加：

```python
from core.workflow.query_splitter import QuerySplitter
from core.workflow.query_classifier import QueryClassifier, ClassifyResult
```

在 `QaCapability.__init__` 末尾（`self.admitter = Admitter(llm)` 附近）追加注入与 agent 占位：

```python
        self.splitter = QuerySplitter(llm)
        self.classifier = QueryClassifier(llm)
        # 有界 agent 由 doc_workflow 构造后注入（complex / simple 升级用）；None → 降级单轮
        self.qa_agent = None
```

在 `dataclass` import 处确保 `from dataclasses import dataclass`（文件已用 dataclass 风格则复用），并在模块内（类外，靠近顶部异常类附近）新增：

```python
from dataclasses import dataclass, field


@dataclass
class _SubDecision:
    """单个子问题的判定结果：可答性 verdict + （ok 时）类型 category。"""

    query: str
    verdict: str = "ok"          # ok / missing_info / out_of_scope
    category: str = ""           # explain/compare/simple/complex（仅 ok）
    reason: str = ""
    clarify_question: str = ""
```

在 `QaCapability` 内新增方法（紧跟现有 `classify` 之后即可）：

```python
    async def _decide_subq(
        self, q: str, book_titles: Optional[list[str]], probe: bool = True
    ) -> "_SubDecision":
        """单子问题判定：probe → admit（非 ok 短路）→ classify。失败一律放行/降级。"""
        evidence = ""
        if probe:
            try:
                located = await self._probe_retrieve(q, book_titles)
                evidence = self._format_probe(located, book_titles)
            except Exception as exc:
                logger.warning("_decide_subq probe 失败，纯文本判定：%s", exc)
        try:
            verdict = await self.admitter.run(q, [evidence])
        except Exception as exc:
            logger.warning("_decide_subq admit 抛错，降级 ok：%s", exc)
            verdict = None
        if verdict is not None and verdict.verdict == "out_of_scope":
            return _SubDecision(q, "out_of_scope", reason=verdict.reason)
        if verdict is not None and verdict.verdict == "missing_info":
            return _SubDecision(
                q, "missing_info", reason=verdict.reason,
                clarify_question=verdict.clarify_question,
            )
        result = await self.classifier.run(q, evidence)
        return _SubDecision(q, "ok", category=result.category, reason=result.reason)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k decide_subq -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat: qa._decide_subq 逐子问题 probe→admit→classify 判定"
```

---

### Task 4: qa._execute_subq —— 类型分派执行 + simple 安全网

**Files:**
- Modify: `core/workflow/qa_capability.py`（新增 `_evidence_weak`、`_execute_subq`）
- Test: `tests/test_qa_capability.py`（追加）

**Interfaces:**
- Consumes: 现有 `explain`/`assume`/`retrieve`/`_retrieve_nodes`/`_synthesize_stream`、`self.qa_agent`、Task 3 的 category。
- Produces: `qa._execute_subq(ctx, q: str, category: str, book_titles) -> tuple[str, list]`。simple 证据不足 → 升级 agent；complex → agent，异常降级单轮。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qa_capability.py 追加


class _FakeAgent:
    def __init__(self, answer="AGENT答案", nodes=None):
        self._a = answer
        self._n = nodes or ["an"]
        self.called_with = None
    async def run(self, ctx, q, bt):
        self.called_with = q
        return self._a, self._n


async def test_execute_simple_with_enough_evidence_uses_retrieve():
    qa = _qa()
    async def fake_retrieve(ctx, q, bt, preamble=""):
        return "单轮答案", ["n1", "n2"]
    qa.retrieve = fake_retrieve
    qa.qa_agent = _FakeAgent()
    ans, nodes = await qa._execute_subq(FakeCtx(), "MySQL有哪些锁", "simple", None)
    assert ans == "单轮答案"
    assert qa.qa_agent.called_with is None  # 没升级


async def test_execute_simple_weak_evidence_escalates_to_agent():
    qa = _qa()
    async def fake_nodes(q, bt): return []   # 召回空 = 证据不足
    qa._retrieve_nodes = fake_nodes
    qa.qa_agent = _FakeAgent(answer="AGENT答案")
    ans, nodes = await qa._execute_subq(FakeCtx(), "冷门问题", "simple", None)
    assert ans == "AGENT答案"
    assert qa.qa_agent.called_with == "冷门问题"


async def test_execute_complex_uses_agent():
    qa = _qa()
    qa.qa_agent = _FakeAgent(answer="AGENT答案")
    ans, nodes = await qa._execute_subq(FakeCtx(), "怎么优化MySQL", "complex", None)
    assert ans == "AGENT答案"


async def test_execute_complex_agent_none_degrades_single_retrieve():
    qa = _qa()
    qa.qa_agent = None
    async def fake_retrieve(ctx, q, bt, preamble=""): return "降级单轮", ["n"]
    qa.retrieve = fake_retrieve
    ans, nodes = await qa._execute_subq(FakeCtx(), "怎么优化MySQL", "complex", None)
    assert ans == "降级单轮"


async def test_execute_explain_and_compare_route():
    qa = _qa()
    async def fake_explain(ctx, q, bt): return "讲解答案", ["e"]
    async def fake_assume(ctx, q, bt): return "比较答案", ["c"]
    qa.explain = fake_explain
    qa.assume = fake_assume
    a1, _ = await qa._execute_subq(FakeCtx(), "讲讲索引", "explain", None)
    a2, _ = await qa._execute_subq(FakeCtx(), "A和B区别", "compare", None)
    assert a1 == "讲解答案" and a2 == "比较答案"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k execute -q`
Expected: FAIL — `AttributeError: 'QaCapability' object has no attribute '_execute_subq'`

- [ ] **Step 3: Write minimal implementation**

在 `QaCapability` 内新增（紧跟 `_decide_subq`）：

```python
    def _evidence_weak(self, nodes: list) -> bool:
        """simple 安全网触发判据：召回空 / top-1 分数低于阈值（complex 误判成 simple 时升级）。"""
        if not nodes:
            return True
        top = max((getattr(n, "score", 0) or 0) for n in nodes)
        return top < self.simple_escalate_min_score

    async def _execute_subq(
        self, ctx: Context, q: str, category: str, book_titles: Optional[list[str]]
    ) -> tuple[str, list]:
        """按 category 分派执行；simple 证据不足升级 agent，complex agent 异常降级单轮。"""
        if category == "explain":
            return await self.explain(ctx, q, book_titles)
        if category == "compare":
            return await self.assume(ctx, q, book_titles)
        if category == "complex":
            if self.qa_agent is None:
                return await self.retrieve(ctx, q, book_titles)
            try:
                return await self.qa_agent.run(ctx, q, book_titles)
            except Exception as exc:
                logger.warning("complex agent 失败，降级单轮：%s", exc)
                return await self.retrieve(ctx, q, book_titles)
        # simple（含分类降级）：先探召回，证据不足且有 agent → 升级
        nodes = await self._retrieve_nodes(q, book_titles)
        if self._evidence_weak(nodes) and self.qa_agent is not None:
            logger.info("simple 证据不足，升级 agent：%r", q[:60])
            try:
                return await self.qa_agent.run(ctx, q, book_titles)
            except Exception as exc:
                logger.warning("simple 升级 agent 失败，回落单轮：%s", exc)
        return await self.retrieve(ctx, q, book_titles)
```

在 `QaCapability.__init__` 参数表新增（紧跟 `explain_recall_k` 之后），并在体内赋值：

```python
        simple_escalate_min_score: float = 0.0,
```
```python
        self.simple_escalate_min_score = simple_escalate_min_score
```

> 默认 `0.0`：只在召回**为空**时升级（最保守，不依赖未标定的分数阈值）。后续冷烟标定后再调高。

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k execute -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat: qa._execute_subq 类型分派 + simple 证据不足升级 agent"
```

---

### Task 5: qa.answer —— 顶层编排（拆分 → 并行判定 → 按序执行 → 合并装饰）

**Files:**
- Modify: `core/workflow/qa_capability.py`（新增 `answer`）
- Test: `tests/test_qa_capability.py`（追加）

**Interfaces:**
- Consumes: `splitter.run`、`_decide_subq`、`_execute_subq`、`_synthesize_stream`/`AnswerDeltaEvent`、`REFUSAL_TEXT`/`REFUSAL_FALLBACK`。
- Produces: `qa.answer(ctx, clean_query: str, book_titles, probe: bool=True) -> tuple[str, list, dict]`，meta 含 `categories: list[str]`、`sub_count: int`、`category: str`（单问题=该类；多问题="multi"）。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qa_capability.py 追加


def _qa_answer_stub(decisions, exec_map):
    """decisions: list[_SubDecision]; exec_map: {query: (answer, nodes)}。"""
    from core.workflow.qa_capability import _SubDecision  # noqa
    qa = _qa()
    async def fake_split(cq): return [d.query for d in decisions]
    qa.split_query = fake_split
    di = {d.query: d for d in decisions}
    async def fake_decide(q, bt, probe=True): return di[q]
    qa._decide_subq = fake_decide
    async def fake_exec(ctx, q, cat, bt): return exec_map[q]
    qa._execute_subq = fake_exec
    return qa


async def test_answer_single_ok_no_decoration():
    from core.workflow.qa_capability import _SubDecision
    d = _SubDecision("什么是B+树", "ok", category="explain")
    qa = _qa_answer_stub([d], {"什么是B+树": ("B+树是…", ["n"])})
    ans, nodes, meta = await qa.answer(FakeCtx(), "什么是B+树", None)
    assert ans == "B+树是…"
    assert "##" not in ans            # 单问题不加分节标题
    assert meta["category"] == "explain"
    assert meta["sub_count"] == 1


async def test_answer_multi_ok_sections_joined():
    from core.workflow.qa_capability import _SubDecision
    ds = [_SubDecision("MySQL锁", "ok", category="simple"),
          _SubDecision("Redis持久化", "ok", category="explain")]
    qa = _qa_answer_stub(ds, {"MySQL锁": ("锁有X", ["a"]), "Redis持久化": ("RDB/AOF", ["b"])})
    ans, nodes, meta = await qa.answer(FakeCtx(), "讲讲MySQL锁和Redis持久化", None)
    assert "锁有X" in ans and "RDB/AOF" in ans
    assert nodes == ["a", "b"]
    assert meta["category"] == "multi"
    assert meta["categories"] == ["simple", "explain"]


async def test_answer_partial_out_of_scope_appends_hint():
    from core.workflow.qa_capability import _SubDecision
    ds = [_SubDecision("MySQL锁", "ok", category="simple"),
          _SubDecision("OpenCL的session", "out_of_scope", reason="库外")]
    qa = _qa_answer_stub(ds, {"MySQL锁": ("锁有X", ["a"])})
    ans, nodes, meta = await qa.answer(FakeCtx(), "MySQL锁和OpenCL的session", None)
    assert "锁有X" in ans
    assert "OpenCL的session" in ans       # 末尾提示该子问题不在库
    assert nodes == ["a"]


async def test_answer_partial_missing_info_appends_clarify():
    from core.workflow.qa_capability import _SubDecision
    ds = [_SubDecision("MySQL锁", "ok", category="simple"),
          _SubDecision("这个索引的场景", "missing_info", clarify_question="指哪个索引？")]
    qa = _qa_answer_stub(ds, {"MySQL锁": ("锁有X", ["a"])})
    ans, _, _ = await qa.answer(FakeCtx(), "MySQL锁和这个索引的场景", None)
    assert "锁有X" in ans and "指哪个索引？" in ans


async def test_answer_all_out_of_scope_pure_refusal():
    from core.workflow.qa_capability import _SubDecision, REFUSAL_TEXT
    ds = [_SubDecision("PG的MVCC", "out_of_scope", reason="库外")]
    qa = _qa_answer_stub(ds, {})
    ans, nodes, meta = await qa.answer(FakeCtx(), "PG的MVCC", None)
    assert ans == REFUSAL_TEXT
    assert nodes == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k answer -q`
Expected: FAIL — `AttributeError: 'QaCapability' object has no attribute 'answer'`（及 `split_query`）

- [ ] **Step 3: Write minimal implementation**

在 `QaCapability` 内新增：

```python
    async def split_query(self, clean_query: str) -> list[str]:
        """委托 QuerySplitter：clean_query → ≥1 个降噪自包含子问题。"""
        return await self.splitter.run(clean_query)

    async def answer(
        self,
        ctx: Context,
        clean_query: str,
        book_titles: Optional[list[str]],
        probe: bool = True,
    ) -> tuple[str, list, dict]:
        """顶层编排：拆分 → 并行逐子问题判定 → 按序执行 ok 子问题 → 合并装饰。

        - 并行只用于判定阶段（无用户可见输出）；执行/合成按子问题顺序串行（保流式顺序）。
        - 单问题：无分节标题、无合并装饰，等价旧单路径。
        - 部分非 ok：先答 ok 的，末尾追加 missing_info 反问 / out_of_scope "不在库" 提示。
        - 全非 ok：纯拒答（out_of_scope→REFUSAL_TEXT）/反问（missing_info）。
        """
        sub_qs = await self.split_query(clean_query)
        decisions = await asyncio.gather(
            *(self._decide_subq(q, book_titles, probe=probe) for q in sub_qs)
        )
        oks = [d for d in decisions if d.verdict == "ok"]
        missing = [d for d in decisions if d.verdict == "missing_info"]
        oos = [d for d in decisions if d.verdict == "out_of_scope"]
        multi = len(sub_qs) > 1
        meta = {
            "categories": [d.category for d in oks],
            "sub_count": len(sub_qs),
            "category": (oks[0].category if oks else "out_of_scope")
            if len(sub_qs) == 1 else "multi",
        }

        # 全非 ok：退化纯拒答/反问（单条复用原话术）
        if not oks:
            if missing:
                q = missing[0].clarify_question or REFUSAL_FALLBACK
                return q, [], meta
            return REFUSAL_TEXT, [], meta

        # 执行 ok 子问题（按序流式）。多问题加分节标题；单问题裸答。
        parts: list[str] = []
        all_nodes: list = []
        for d in oks:
            if multi:
                heading = f"\n## {d.query}\n"
                ctx.write_event_to_stream(AnswerDeltaEvent(delta=heading))
                parts.append(heading)
            ans, nodes = await self._execute_subq(ctx, d.query, d.category, book_titles)
            parts.append(ans)
            all_nodes.extend(nodes)

        # 末尾装饰：out_of_scope / missing_info 子问题（仅多问题且存在时）
        tail = self._compose_tail(oos, missing)
        if tail:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=tail))
            parts.append(tail)

        return "".join(parts).strip(), all_nodes, meta

    @staticmethod
    def _compose_tail(oos: list, missing: list) -> str:
        """合并末尾提示：库外子问题如实告知 + 信息不足子问题反问。"""
        lines: list[str] = []
        if oos:
            names = "、".join(f"「{d.query}」" for d in oos)
            lines.append(f"另外，{names} 知识库里暂未收录相关内容，无法作答。")
        for d in missing:
            lines.append(d.clarify_question or f"关于「{d.query}」，能再说具体一点吗？")
        return ("\n\n" + "\n".join(lines)) if lines else ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -k answer -q`
Expected: PASS (5 passed)，并跑全文件回归 `.venv\Scripts\python.exe -m pytest tests/test_qa_capability.py -q`（全绿）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat: qa.answer 多子问题编排（并行判定+按序执行+合并装饰）"
```

---

### Task 6: DocQueryWorkflow 切换 —— route → split_answer → finalize

**Files:**
- Modify: `core/workflow/doc_workflow.py`（删 `preprocess` 及 6 个 category 分支/事件，新增 `split_answer` step；注入 `qa.qa_agent`）
- Test: `tests/test_doc_workflow.py`（改写为新 step 图）

**Interfaces:**
- Consumes: `qa.answer(ctx, clean_query, book_titles, probe) -> (answer, nodes, meta)`；`front_door.run`（不变）。
- Produces: workflow 结果 `Response(response, source_nodes, metadata=meta)`，meta 由 `qa.answer` 提供 + `action`。

- [ ] **Step 1: Write the failing test**

```python
# tests/test_doc_workflow.py —— 替换原 preprocess/分支相关用例，保留 route/finalize 骨架风格
# 关键新增/改写用例（其余 converse/clarify/study_plan 用例保持，断言不变）：


async def test_dispatch_qa_goes_through_split_answer(monkeypatch):
    wf = _make_workflow()  # 复用文件既有的 workflow 构造替身（front_door/qa stub）
    # front_door → dispatch_qa
    async def fake_front(original, memory, bt):
        from core.workflow.front_door import FrontDoorDecision
        return FrontDoorDecision("dispatch_qa", clean_query="讲讲MySQL锁和Redis持久化")
    wf.front_door.run = fake_front
    # qa.answer 返回多段答案 + meta
    async def fake_answer(ctx, cq, bt, probe=True):
        return "合并答案", ["n1"], {"category": "multi", "categories": ["simple", "explain"], "sub_count": 2}
    wf.qa.answer = fake_answer

    handler = wf.run(query="...", memory=None, book_titles=None)
    result = await handler
    assert result.response == "合并答案"
    assert result.metadata["category"] == "multi"
    assert result.source_nodes == ["n1"]
```

> 实施者注意：`_make_workflow` 按本文件既有替身风格构造（mock LLM、stub index_manager）。删除原断言 `category in (out_of_scope/pending_split/...)` 的用例，它们对应的分支已不存在。

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_doc_workflow.py -k split_answer -q`
Expected: FAIL（`qa.answer` 未被 workflow 调用 / 旧 `preprocess` 仍在）

- [ ] **Step 3: Write minimal implementation**

`core/workflow/doc_workflow.py` 改动：

1) 事件区：删 `PreprocessEvent / ExplainEvent / RetrieveAgentEvent / SplitEvent / AssumeEvent / ClarifyEvent / OtherEvent / OutOfScopeEvent`，新增：

```python
class SplitAnswerEvent(Event):
    """dispatch_qa → 多子问题拆分 + 编排作答。纯信号；clean_query 从 ctx 取。"""
```

2) `route` step 返回类型与分支：`dispatch_qa` 改为返回 `SplitAnswerEvent()`（study_plan/converse/clarify 不变）：

```python
    async def route(
        self, ctx: Context, ev: RouteEvent
    ) -> "SplitAnswerEvent | StudyPlanEvent | DirectReplyEvent":
        ...
        await ctx.store.set("clean_query", decision.clean_query)
        return SplitAnswerEvent()
```

3) 删除整个 `preprocess` step 与全部 6 个 category 分支 step（`retrieve_branch/other_branch/split_branch/assume_branch/clarify_branch/out_of_scope_branch/explain_branch`），新增单一编排 step：

```python
    @step
    async def split_answer(self, ctx: Context, ev: SplitAnswerEvent) -> FinalizeEvent:
        clean_query = await ctx.store.get("clean_query")
        book_titles = await ctx.store.get("book_titles")
        answer, nodes, meta = await self.qa.answer(
            ctx, clean_query, book_titles, probe=self._probe
        )
        await ctx.store.set("qa_meta", meta)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

4) `__init__`：构造后把 agent 注入 qa（让 complex/simple 升级可用）：

```python
        self.qa_agent = QaAgent(index_manager, llm, similarity_top_k, max_iterations=6)
        self.qa.qa_agent = self.qa_agent
```

5) `finalize`：meta 改从 `qa_meta` 取并叠加 `action`：

```python
        qa_meta = await ctx.store.get("qa_meta", {}) or {}
        meta = {**qa_meta, "action": await ctx.store.get("action", None)}
```

> `_split_enabled/_assume_enabled/_other_agent_enabled` 这些 ablation 开关本计划不再驱动分支；保留构造参数以免破坏 eval 调用方签名，但标注 TODO（Plan 2 或 eval 适配时再清）。`probe_then_classify` 仍经 `self._probe` 传给 `qa.answer`。

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_doc_workflow.py -q`
Expected: PASS（新用例通过；converse/clarify/study_plan 回归绿；已删用例移除）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "refactor: doc_workflow 切换为 route→split_answer→finalize"
```

---

### Task 7: 清理 —— 删 QueryGate / QueryPreprocessor / 旧 classify·gate 接线

**Files:**
- Delete: `core/workflow/query_gate.py`、`core/workflow/query_preprocess.py`、`tests/test_query_gate.py`、`tests/test_query_preprocess.py`
- Modify: `core/workflow/qa_capability.py`（删 `gate`/`classify`/`_gate`/`preprocessor` 及其 import）
- Grep 验证无残留引用。

**Interfaces:**
- Produces: 无新接口；移除死代码后全套件仍绿。

- [ ] **Step 1: 找出所有引用（先测后删的"红"= 引用存在）**

Run（用 Grep 工具或）：`.venv\Scripts\python.exe -m pytest -q` 当前应全绿；然后定位引用：

```bash
grep -rn "query_gate\|QueryGate\|query_preprocess\|QueryPreprocessor\|PreprocessResult\|\.gate(\|\.classify(" core api eval tests
```
Expected: 命中 `qa_capability.py`（import + `self._gate`/`self.preprocessor`/`gate`/`classify`）及两个待删测试文件；**eval/api 不应再有**（若有，记录待改）。

- [ ] **Step 2: 删除死代码与文件**

`core/workflow/qa_capability.py`：
- 删 import：`from core.workflow.query_gate import QueryGate`、`from core.workflow.query_preprocess import QueryPreprocessor, PreprocessResult`。
- 删 `__init__` 里 `self.preprocessor = QueryPreprocessor(llm)`、`self._gate = QueryGate(llm)`。
- 删方法 `gate`、`classify`（已被 `split_query`/`_decide_subq`/`answer` 取代）。

删文件：

```bash
git rm core/workflow/query_gate.py core/workflow/query_preprocess.py tests/test_query_gate.py tests/test_query_preprocess.py
```

- [ ] **Step 3: 验证无残留引用**

```bash
grep -rn "query_gate\|QueryGate\|query_preprocess\|QueryPreprocessor\|PreprocessResult\|\.gate(\|\.classify(" core api eval tests
```
Expected: 无输出（全部清除）。若 `eval/` 仍引用 `classify`，在此步改为调用 `qa.answer` 或 `_decide_subq`（按 eval 实际用法最小适配）。

- [ ] **Step 4: 全套件 + 分层守卫**

Run:
```bash
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/check_layering.py
```
Expected: 全绿；分层守卫通过。

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: 删 QueryGate/QueryPreprocessor 及旧 gate/classify 接线"
```

---

## Self-Review

**Spec coverage：**
- 多主体拆分器（splitter）→ Task 1 ✓
- 类型分类器 4 类（explain/compare/simple/complex）→ Task 2 ✓
- 逐子问题 probe→admit→classify → Task 3 ✓
- 类型分派 + simple 安全网（空/低分升级 agent）→ Task 4 ✓
- 顶层编排：并行判定 + 按序执行 + 合并装饰 + 退化 + 单问题透传 → Task 5 ✓
- workflow 切换 route→split_answer→finalize；agent 注入 qa；meta → Task 6 ✓
- gate 溶解 + pending_split 并入 complex + 删旧分类器 → Task 6/7 ✓
- 流式（判定并行、合成按序分节）→ Task 5 实现 ✓
- 降级矩阵（splitter/admit/classifier/agent）→ Task 1/3/4 ✓
- **missing_info 跨轮持久化 → 不在本计划**（Plan 2，spec 决策点 4 的 MVP：当轮反问入答案、不持久化）。已在 Global Constraints 标注。
- **compare 双主体特别优化 / agent prompt 增强 → 已知缺口，不在本计划**（spec 已列）。

**Placeholder scan：** 无 TBD/TODO 占位（Task 6 的 ablation 开关 TODO 是"保留参数不删"的明确说明，非代码占位）。

**Type consistency：** `_SubDecision(query, verdict, category, reason, clarify_question)` 在 Task 3 定义、Task 4/5 消费一致；`ClassifyResult(category, reason)` Task 2 定义、Task 3 消费一致；`qa.answer(...) -> (answer, nodes, meta)` Task 5 定义、Task 6 消费一致；`_execute_subq(ctx, q, category, book_titles)` Task 4 定义、Task 5 调用签名一致。

## Execution Handoff

见下条消息的执行方式选择。
