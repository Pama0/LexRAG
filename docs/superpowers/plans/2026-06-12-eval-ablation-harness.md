# 评测 ablation 框架 Implementation Plan（Phase 1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让评测接上当前 `DocQueryWorkflow`，把每个工作流决策做成可开关的变体，跑同一金标准测试集，输出 baseline vs 变体的 delta 对比表（含 ragas 5 指标 + 分类准确率 + 分支分布）。

**Architecture:** `DocQueryWorkflow` 加 4 个决策 flag（off→分支降级单轮），并在 `finalize` 把 `category`/`intent` 附到 `Response.metadata`；`eval/sut.py` 新 `DocQueryWorkflowSystem` 按 flag 构造、映射出 `RagOutput`（含 category）；新增分类准确率聚合 + 金标准带 category 标注的测试集；`eval/compare.py` 跑变体列表出 Markdown delta 表。

**Tech Stack:** Python 3.12，LlamaIndex Workflow，ragas，DeepSeek 评测侧，pytest。

参考 spec：`docs/superpowers/specs/2026-06-12-eval-ablation-harness-design.md`

---

## File Structure

- **Modify** `core/workflow/doc_workflow.py` — `__init__` 加决策 flag；分支 step 按 flag 降级；`finalize` 把 category/intent 附 `Response.metadata`；`classify` 调用受 probe flag 控制。
- **Modify** `core/workflow/qa_capability.py` — `classify` 加 `probe: bool = True` 开关。
- **Modify** `tests/test_doc_workflow.py` — flag 降级 + category 暴露测试。
- **Modify** `eval/sut.py` — 新增 `DocQueryWorkflowSystem` + `map_doc_result`；`RagOutput` 加 `category`。
- **Modify** `eval/run_eval.py` — `aggregate` 加分类准确率 + category 分布；`_run` 用新 SUT。
- **Create** `eval/compare.py` — 变体列表 runner + Markdown delta 表。
- **Create** `eval/dataset/golden.seed.jsonl` — 金标准种子（含 category 标注，每类样本 + openclaw 易错 case）。
- **Create** `eval/dataset/README.md` — category 标注准则。
- **Create** `tests/test_eval_compare.py` / 扩 `tests/test_eval_*.py` — 纯逻辑单测（delta 表、分类准确率）。

---

### Task 1: DocQueryWorkflow 决策 flag + category 暴露

**Files:** Modify `core/workflow/doc_workflow.py`, `core/workflow/qa_capability.py`; Test `tests/test_doc_workflow.py`

- [ ] **Step 1: 追加失败测试**（`tests/test_doc_workflow.py` 末尾）

```python
async def test_flags_off_degrade_branches_to_single_retrieve():
    # split/assume/other flag 关 → 对应 category 走单轮 retrieve（baseline 对比用）
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "讲讲MySQL"}',
        '{"category": "pending_split", "rewritten_query": "讲讲MySQL", "reason": "需罗列"}',
    ])
    wf = DocQueryWorkflow(
        index_manager=None, llm=llm, similarity_top_k=3, timeout=10,
        split_enabled=False, assume_enabled=False, other_agent_enabled=False,
        probe_then_classify=False,
    )
    used = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        used["retrieve"] = True
        return "单轮答案", ["n1"]

    async def boom_split(ctx, query, book_titles):
        raise AssertionError("split 不应被调用（flag off）")

    wf.qa.retrieve = fake_retrieve
    wf.qa.split = boom_split
    result = await wf.run(query="讲讲MySQL", memory=FakeMemory())
    assert used.get("retrieve") is True
    assert str(result.response) == "单轮答案"


async def test_finalize_exposes_category_in_metadata():
    llm = FakeLLM([
        '{"intent": "qa", "clean_query": "MySQL锁"}',
        '{"category": "retrievable", "rewritten_query": "MySQL锁"}',
    ])
    wf = _wf(llm)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", ["n1"]

    wf.qa.retrieve = fake_retrieve
    result = await wf.run(query="MySQL锁", memory=FakeMemory())
    assert result.metadata.get("category") == "retrievable"
    assert result.metadata.get("intent") == "qa"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_doc_workflow.py -k "flags_off or category_in_metadata" -q`
Expected: FAIL（`DocQueryWorkflow` 不接受这些 flag / metadata 无 category）

- [ ] **Step 3: 改 qa_capability.py 的 classify（probe 开关）**

把 `classify` 签名与 probe 调用改为受 `probe` 控制：

```python
    async def classify(
        self,
        clean_query: str,
        book_titles: Optional[list[str]] = None,
        probe: bool = True,
    ):
        """先用 clean_query 探测召回（probe=True），把召回信号喂给 judge，堵住「盲判」。

        probe=False（ablation baseline）→ 不探测，纯文本判定；probe 失败亦容错为空。
        """
        retrieval_context = ""
        if probe:
            try:
                located = await self._retrieve_nodes(clean_query, book_titles)
                retrieval_context = self._format_probe(located, book_titles)
            except Exception as exc:
                logger.warning("classify probe 探测失败，退回纯文本判定：%s", exc)
        return await self.preprocessor.run(clean_query, retrieval_context)
```

- [ ] **Step 4: 改 doc_workflow.py**

(a) `__init__` 加 flag（在 `max_sub_queries` 后）:

```python
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        probe_then_classify: bool = True,
        split_enabled: bool = True,
        assume_enabled: bool = True,
        other_agent_enabled: bool = True,
        **kw,
    ):
        super().__init__(**kw)
        self.router = IntentRouter(llm)
        self.qa = QaCapability(index_manager, llm, similarity_top_k, max_sub_queries)
        self.qa_agent = QaAgent(index_manager, llm, similarity_top_k, max_iterations=6)
        self._probe = probe_then_classify
        self._split_enabled = split_enabled
        self._assume_enabled = assume_enabled
        self._other_agent_enabled = other_agent_enabled
```

(b) `preprocess` step 的 classify 调用传 probe flag:

```python
        result = await self.qa.classify(clean_query, book_titles, probe=self._probe)
```

(c) `split_branch` / `assume_branch` / `other_branch` 按 flag 降级。例如 `split_branch`:

```python
    @step
    async def split_branch(self, ctx: Context, ev: SplitEvent) -> FinalizeEvent:
        book_titles = await ctx.store.get("book_titles")
        if self._split_enabled:
            answer, nodes = await self.qa.split(ctx, ev.rewritten_query, book_titles)
        else:
            answer, nodes = await self.qa.retrieve(ctx, ev.rewritten_query, book_titles)
        return FinalizeEvent(answer=answer, source_nodes=nodes)
```

`assume_branch` 同理（`self._assume_enabled` 否则 `qa.retrieve`）；`other_branch` 用 `self._other_agent_enabled`（否则 `qa.retrieve`，且不进 try/except agent）。

(d) `finalize` 把 category/intent 附 `Response.metadata`:

```python
    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> StopEvent:
        memory: Optional[ChatMemoryBuffer] = await ctx.store.get("memory")
        original = await ctx.store.get("original_query")
        if memory is not None:
            memory.put(ChatMessage(role=MessageRole.USER, content=original))
            memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=ev.answer))
        logger.info(
            "finalize: answer_len=%d source_nodes=%d",
            len(ev.answer or ""), len(ev.source_nodes or []),
        )
        meta = {
            "category": await ctx.store.get("category"),
            "intent": await ctx.store.get("intent"),
        }
        return StopEvent(
            result=Response(
                response=ev.answer, source_nodes=ev.source_nodes, metadata=meta
            )
        )
```

- [ ] **Step 5: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_doc_workflow.py -q`
Expected: 全部 passed（含新增 2 + 原有不破——flag 默认全开，现有行为不变）

- [ ] **Step 6: Commit**

```bash
git add core/workflow/doc_workflow.py core/workflow/qa_capability.py tests/test_doc_workflow.py
git commit -m "feat(workflow): 决策 flag（probe/split/assume/other）+ category 暴露到 metadata（评测用）"
```

---

### Task 2: DocQueryWorkflowSystem（SUT 接当前系统）

**Files:** Modify `eval/sut.py`; Test `tests/test_eval_sut.py`（若无则 Create）

- [ ] **Step 1: 失败测试**（`tests/test_eval_sut.py`）

```python
"""map_doc_result 纯逻辑单测：DocQueryWorkflow 结果 → RagOutput（含 category）。"""
from eval.sut import RagOutput, map_doc_result


class _Resp:
    def __init__(self, response, source_nodes, metadata):
        self.response = response
        self.source_nodes = source_nodes
        self.metadata = metadata


class _N:
    def __init__(self, text):
        self._t = text

    class _Node:
        def __init__(self, t):
            self._t = t

        def get_content(self):
            return self._t

    @property
    def node(self):
        return self._Node(self._t)


def test_map_answered_with_category():
    r = _Resp("答案", [_N("片段")], {"category": "retrievable", "intent": "qa"})
    out = map_doc_result(r, response_cls=_Resp)
    assert out.outcome == "answered"
    assert out.category == "retrievable"
    assert out.retrieved_contexts == ["片段"]


def test_map_empty_when_no_nodes():
    r = _Resp("反问句", [], {"category": "missing_info", "intent": "qa"})
    out = map_doc_result(r, response_cls=_Resp)
    assert out.outcome == "empty"
    assert out.category == "missing_info"
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_sut.py -q`
Expected: FAIL（无 `map_doc_result` / `RagOutput.category`）

- [ ] **Step 3: 改 eval/sut.py**

(a) `RagOutput` 加 `category`:

```python
@dataclass
class RagOutput:
    response: str
    retrieved_contexts: list[str]
    outcome: str  # answered | clarify | split | empty | error
    category: str = ""   # judge 判的 category（评测分类准确率用）
```

(b) 新增 `map_doc_result`（DocQueryWorkflow 的 Response → RagOutput）:

```python
def map_doc_result(result, response_cls=None) -> RagOutput:
    """DocQueryWorkflow.run() 的 Response → RagOutput（读 metadata.category）。"""
    if response_cls is None:
        from llama_index.core.base.response.schema import Response as response_cls  # noqa: N813
    meta = getattr(result, "metadata", None) or {}
    category = meta.get("category", "") or ""
    if isinstance(result, response_cls):
        text = (getattr(result, "response", None) or "").strip()
        nodes = getattr(result, "source_nodes", None) or []
        if not text or not nodes:
            return RagOutput(text, [], "empty", category)
        contexts = [n.node.get_content() for n in nodes]
        return RagOutput(text, contexts, "answered", category)
    return RagOutput(str(result), [], "empty", category)
```

(c) 新增 `DocQueryWorkflowSystem`:

```python
class DocQueryWorkflowSystem:
    """包装当前 DocQueryWorkflow，按决策 flag 构造，实现 RagSystem。"""

    def __init__(self, index_manager, llm, flags: dict | None = None,
                 similarity_top_k: int = 5, timeout: float = 120.0):
        self._index_manager = index_manager
        self._llm = llm
        self._flags = flags or {}
        self._similarity_top_k = similarity_top_k
        self._timeout = timeout

    async def answer(self, query: str, book_titles=None) -> RagOutput:
        from core.workflow.doc_workflow import DocQueryWorkflow
        wf = DocQueryWorkflow(
            index_manager=self._index_manager, llm=self._llm,
            similarity_top_k=self._similarity_top_k, timeout=self._timeout,
            **self._flags,
        )
        try:
            result = await wf.run(query=query, book_titles=book_titles)
        except Exception as e:  # noqa: BLE001 — 单条异常记 error 不中断
            return RagOutput(f"{type(e).__name__}: {e}", [], "error", "")
        return map_doc_result(result)
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_sut.py -q`
Expected: passed

- [ ] **Step 5: Commit**

```bash
git add eval/sut.py tests/test_eval_sut.py
git commit -m "feat(eval): DocQueryWorkflowSystem 接当前系统 + map_doc_result（带 category）"
```

---

### Task 3: 分类准确率聚合 + 金标准种子集

**Files:** Modify `eval/run_eval.py`; Create `eval/dataset/golden.seed.jsonl`, `eval/dataset/README.md`; Test `tests/test_eval_run.py`（扩 aggregate 测试）

- [ ] **Step 1: aggregate 加分类准确率 + category 分布（失败测试）**

在 `tests/test_eval_run.py`（若无则建）加纯逻辑测试：

```python
from eval.run_eval import aggregate


def test_aggregate_classification_accuracy_and_distribution():
    rows = [
        {"outcome": "answered", "category": "retrievable", "expected_category": "retrievable"},
        {"outcome": "answered", "category": "other", "expected_category": "retrievable"},  # 误判
        {"outcome": "empty", "category": "missing_info", "expected_category": "missing_info"},
    ]
    rep = aggregate(rows)
    assert rep["classification"]["accuracy"] == 2 / 3   # 2/3 判对
    assert rep["category_distribution"]["other"] == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_run.py -k classification -q`
Expected: FAIL（aggregate 无 classification）

- [ ] **Step 3: 改 aggregate（`eval/run_eval.py`）**

在 `aggregate` 返回 dict 里追加分类统计（`score_row` 需把 SUT 的 `category` 和测试集标注 `expected_category` 都带进 row——见 Step 4）:

```python
def aggregate(rows: list[dict]) -> dict:
    outcomes: dict[str, int] = {}
    cat_dist: dict[str, int] = {}
    cls_total = cls_correct = 0
    for r in rows:
        oc = r.get("outcome", "error")
        outcomes[oc] = outcomes.get(oc, 0) + 1
        cat = r.get("category") or ""
        if cat:
            cat_dist[cat] = cat_dist.get(cat, 0) + 1
        exp = r.get("expected_category")
        if exp:
            cls_total += 1
            cls_correct += int(cat == exp)
    answered = [r for r in rows if r.get("outcome") == "answered"]
    metric_means: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        vals = [r[name] for r in answered if r.get(name) is not None]
        metric_means[name] = (sum(vals) / len(vals)) if vals else None
    return {
        "total": len(rows),
        "answered": len(answered),
        "outcome_distribution": outcomes,
        "category_distribution": cat_dist,
        "classification": {
            "total": cls_total,
            "correct": cls_correct,
            "accuracy": (cls_correct / cls_total) if cls_total else None,
        },
        "metric_means": metric_means,
    }
```

并在 `score_row` 的 `base` 里带上 `category` 与 `expected_category`:

```python
    base = {
        "user_input": row["user_input"],
        "reference": row.get("reference", ""),
        "response": out.response,
        "outcome": out.outcome,
        "category": out.category,
        "expected_category": row.get("category", ""),   # 测试集金标准标注
        "num_contexts": len(out.retrieved_contexts),
    }
```

- [ ] **Step 4: 运行确认通过 + 建金标准种子集与准则**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_run.py -q` → passed

创建 `eval/dataset/golden.seed.jsonl`（**种子，仅示范格式；完整 30~50 条需按你书库内容人工补全/校验**），每行:

```json
{"user_input": "给我讲明白openclaw", "category": "retrievable", "scope": null, "reference": "<openclaw 的核心定义，按书填>", "reference_contexts": ["<相关原文片段>"]}
{"user_input": "讲讲MySQL", "category": "pending_split", "scope": null, "reference": "<MySQL 主题概览>", "reference_contexts": ["<片段>"]}
{"user_input": "这个索引的应用场景是什么", "category": "missing_info", "scope": null, "reference": "", "reference_contexts": []}
{"user_input": "Redis做缓存好吗", "category": "ambiguous", "scope": null, "reference": "<多角度评判>", "reference_contexts": ["<片段>"]}
{"user_input": "综合对比 openclaw 与传统方案的架构取舍", "category": "other", "scope": null, "reference": "<跨主题综合答案>", "reference_contexts": ["<片段>"]}
```

创建 `eval/dataset/README.md`，写 category 标注准则（边界约定）:
- retrievable：单一概念、单轮检索可集中命中（含"X是什么/讲明白X"当 X 是单一概念）。
- pending_split：X 是大主题、答案需罗列并列子项/多章节。
- ambiguous：主题具体但缺评判维度/角度。
- missing_info：缺检索必需限定且**库里确实没有**。
- other：召回到但需跨主题综合/多步推理/开放权衡。
- 难度分层：每类至少 covers 简单/中/难；务必含"库里有但易误判"case（openclaw）。

- [ ] **Step 5: Commit**

```bash
git add eval/run_eval.py eval/dataset/golden.seed.jsonl eval/dataset/README.md tests/test_eval_run.py
git commit -m "feat(eval): 分类准确率/分支分布聚合 + 金标准种子集与标注准则"
```

---

### Task 4: 对比 runner + delta 表

**Files:** Create `eval/compare.py`; Test `tests/test_eval_compare.py`

- [ ] **Step 1: 失败测试**（纯逻辑：delta 表渲染）

```python
"""对比表渲染纯逻辑单测。"""
from eval.compare import render_delta_table


def test_render_delta_table_marks_improvement():
    variants = [
        {"name": "baseline", "report": {"classification": {"accuracy": 0.6},
            "metric_means": {"context_recall": 0.62}}},
        {"name": "+probe", "report": {"classification": {"accuracy": 0.9},
            "metric_means": {"context_recall": 0.78}}},
    ]
    md = render_delta_table(variants, baseline="baseline")
    assert "| baseline |" in md
    assert "+0.30" in md or "+0.3" in md   # 分类准确率 delta
    assert "0.78" in md
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_compare.py -q`
Expected: FAIL（无 `eval.compare`）

- [ ] **Step 3: 写 eval/compare.py**

```python
"""决策对比 runner：对变体列表跑同一测试集，渲染 baseline vs 变体的 delta 表。"""
import argparse
import asyncio

# 对比表展示的列（确定性指标优先）
_COLS = [
    ("分类准确率", lambda rep: rep.get("classification", {}).get("accuracy")),
    ("context_recall", lambda rep: rep.get("metric_means", {}).get("context_recall")),
    ("factual_correctness", lambda rep: rep.get("metric_means", {}).get("factual_correctness")),
    ("faithfulness", lambda rep: rep.get("metric_means", {}).get("faithfulness")),
    ("answer_relevancy", lambda rep: rep.get("metric_means", {}).get("answer_relevancy")),
]


def _fmt(val, base):
    if val is None:
        return "—"
    if base is None or val == base:
        return f"{val:.2f}"
    return f"{val:.2f} ({val - base:+.2f})"


def render_delta_table(variants: list[dict], baseline: str) -> str:
    base_rep = next(v["report"] for v in variants if v["name"] == baseline)
    header = "| 配置 | " + " | ".join(c[0] for c in _COLS) + " |"
    sep = "|" + "---|" * (len(_COLS) + 1)
    lines = [header, sep]
    for v in variants:
        cells = []
        for _, getter in _COLS:
            cells.append(_fmt(getter(v["report"]), getter(base_rep)))
        lines.append(f"| {v['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 变体矩阵：baseline 全单轮，逐个打开决策
VARIANTS = {
    "baseline(全单轮)": dict(probe_then_classify=False, split_enabled=False,
                             assume_enabled=False, other_agent_enabled=False),
    "+probe": dict(probe_then_classify=True, split_enabled=False,
                   assume_enabled=False, other_agent_enabled=False),
    "+probe+split": dict(probe_then_classify=True, split_enabled=True,
                         assume_enabled=False, other_agent_enabled=False),
    "全开": dict(probe_then_classify=True, split_enabled=True,
                 assume_enabled=True, other_agent_enabled=True),
}


async def _run_variants(testset_path, limit, names):
    from eval.config import CHROMA_DIR, make_eval_embeddings, make_eval_llm
    from eval.metrics import build_metric_specs
    from eval.run_eval import load_testset, score_row, aggregate
    from eval.sut import DocQueryWorkflowSystem
    from configs.embedding import configure_embedding
    from configs.llm import configure_llm
    from core.rag.data_loader import RAGIndexManager

    rows = load_testset(testset_path)
    if limit:
        rows = rows[:limit]
    eval_llm, eval_emb = make_eval_llm(), make_eval_embeddings()
    metric_specs = build_metric_specs(eval_llm, eval_emb)
    sut_llm = configure_llm()
    configure_embedding()
    index_manager = RAGIndexManager(persist_dir=CHROMA_DIR)

    variants = []
    for name in names:
        sut = DocQueryWorkflowSystem(index_manager, sut_llm, flags=VARIANTS[name])
        scored = [await score_row(r, sut, metric_specs) for r in rows]
        variants.append({"name": name, "report": aggregate(scored)})
    return variants


def main():
    p = argparse.ArgumentParser(description="决策对比评测")
    p.add_argument("--testset", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    p.add_argument("--baseline", default="baseline(全单轮)")
    args = p.parse_args()
    variants = asyncio.run(_run_variants(args.testset, args.limit, args.variants))
    print(render_delta_table(variants, baseline=args.baseline))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_eval_compare.py -q`
Expected: passed

- [ ] **Step 5: Commit**

```bash
git add eval/compare.py tests/test_eval_compare.py
git commit -m "feat(eval): 决策对比 runner + Markdown delta 表（baseline vs 变体）"
```

---

### Task 5: 真实 ablation 跑通（验证 + 首张对比表）

**Files:** 无（验证 only；需真实 chroma + DeepSeek + 已标注金标准集）

- [ ] **Step 1: 单测全绿 + 编译**

Run: `.venv/Scripts/python.exe -m py_compile core/workflow/doc_workflow.py core/workflow/qa_capability.py eval/sut.py eval/run_eval.py eval/compare.py`
Run: `.venv/Scripts/python.exe -m pytest tests/test_doc_workflow.py tests/test_eval_sut.py tests/test_eval_run.py tests/test_eval_compare.py -q`
Expected: 全 passed。

- [ ] **Step 2: 跑首张对比表**（需先把 `golden.seed.jsonl` 补成完整金标准集）

Run: `.venv/Scripts/python.exe -m eval.compare --testset eval/dataset/golden.jsonl --variants "baseline(全单轮)" "+probe" --baseline "baseline(全单轮)"`
Expected: 输出 Markdown delta 表；`+probe` 的**分类准确率**与 `context_recall` 应高于 baseline，`other` 误判（在 category_distribution）应下降——量化 openclaw 修复。

- [ ] **Step 3: 存档**（把对比表写进 docs，作简历/项目证据）

```bash
git add docs/eval-results-*.md
git commit -m "docs(eval): 首张决策对比表（probe-then-classify 提升量化）"
```

---

## Self-Review Notes

- **Spec coverage:** 换 SUT 接当前系统 → Task 2；决策 flag（4 个 off→降级单轮）→ Task 1；category 暴露 metadata → Task 1 finalize；分类准确率 + 分支分布 → Task 3；金标准带 category 标注 → Task 3（种子+准则，完整标注需领域知识）；对比 runner + delta 表 → Task 4；真实验证 → Task 5。ragas 5 指标复用未动。
- **Type consistency:** `classify(clean_query, book_titles=None, probe=True)`；`DocQueryWorkflow(..., probe_then_classify, split_enabled, assume_enabled, other_agent_enabled)`；`RagOutput(response, retrieved_contexts, outcome, category="")`；`map_doc_result(result, response_cls=None)`；`aggregate` 返回加 `classification`/`category_distribution`；`render_delta_table(variants, baseline)`。各 Task 一致。
- **No placeholders:** 代码完整。例外：`golden.seed.jsonl` 的 `reference`/`reference_contexts` 是 `<按书填>` 占位——这是**领域数据**（依赖你书库内容），非代码占位；Task 3 给了格式 + 准则 + 5 类种子，完整 30~50 条需人工标注/校验。
- **风险点:** ① flag 默认全开 → 现有 doc_workflow 测试不破（Task 1 Step 5 验证）；② `finalize` 加 metadata 不影响 api（chat.py 读 response/source_nodes，metadata 是附加）；③ `score_row`/`sut.answer` 的 scope（book_titles）透传——金标准集 `scope` 字段，Phase 1 默认 None（全库），若需按书评测，后续让 `score_row` 传 `row.get("scope")`；④ Task 5 真实跑依赖完整金标准集 + 环境，单测（Task 1-4）不依赖。
```
