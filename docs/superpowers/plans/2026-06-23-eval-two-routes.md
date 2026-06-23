# eval 收敛为 workflow vs agent 两路线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 eval harness 收敛为「workflow 默认配置」与「纯 agent」两条 SUT 路线，删掉失效的 flag-ablation 矩阵、分类准确率指标与单系统 runner（`run_eval.py`）。

**Architecture:** `compare.py` 成为唯一 runner，吸收原 `run_eval.py` 的纯打分函数（`load_testset` / `_row_to_dict` / `score_row` / `aggregate`）；`VARIANTS` 退化为两条命名路线；`sut.py` 去掉 category 维度；`report.py` 去掉分类准确率列与 match 列。保留 5 ragas 质量指标 + 成本（时延/tokens）。

**Tech Stack:** Python 3.12 / pytest（async）/ ragas（judge）。

## Global Constraints

- 分层：`api/`(Web) → `core/`(领域) → `configs/`，core 不依赖 api（守卫 `python scripts/check_layering.py`）。eval 仅依赖 core/configs。
- 所有 I/O 用 `async/await`；函数签名带类型注解；中文注释可接受。
- 从项目根目录运行；模块用绝对导入（`from eval.harness.compare import ...`）。
- 测试命令统一：`.venv\Scripts\python.exe -m pytest <path> -v`（Windows PowerShell）。
- ragas judge 配置（`config.py` / `metrics.py` / `meter.py`）与 `golden.jsonl` 标签**不动**。

---

## File Structure

- `eval/harness/compare.py` — 修改：吸收打分函数 + 两路线 VARIANTS（唯一 runner）。
- `eval/harness/run_eval.py` — **删除**。
- `eval/harness/sut.py` — 修改：`RagOutput` 去 category；mapper 去 category 逻辑。
- `eval/harness/report.py` — 修改：去分类准确率列、去 detail 的 category/match 列。
- `tests/test_eval_run.py` — **删除**（相关测试迁入 `test_eval_compare.py`）。
- `tests/test_eval_compare.py` — 修改：新增打分函数测试段 + 改 build_sut/VARIANTS/render 测试。
- `tests/test_eval_sut.py` — 修改：去 category 断言。
- `tests/test_eval_report.py` — 修改：去分类准确率列 / match 列断言。
- `eval/datagen/merge_golden.py`、`eval/datagen/generate_testset.py` — 修改：注释措辞。
- `docs/EVAL_OVERVIEW.md` — 修改：删 run_eval / 分类准确率，改两路线口径。

---

## Task 1: compare.py 吸收打分函数 + 删除 run_eval.py

把 `run_eval.py` 的纯打分逻辑搬进 `compare.py`（同时去掉 classification 与 SUT category），删除 `run_eval.py` 与 `tests/test_eval_run.py`，把仍有效的打分测试迁入 `tests/test_eval_compare.py`。

**Files:**
- Modify: `eval/harness/compare.py`
- Delete: `eval/harness/run_eval.py`
- Delete: `tests/test_eval_run.py`
- Modify (新增打分测试段): `tests/test_eval_compare.py`

**Interfaces:**
- Produces（供 Task 3/4 与测试使用，均在 `eval/harness/compare.py`）：
  - `REFUSE_CATEGORIES: frozenset[str]` = `{"missing_info", "out_of_scope"}`
  - `def load_testset(path: str) -> list[dict]`
  - `def _row_to_dict(row) -> dict`
  - `async def score_row(row: dict, sut, metric_specs: list, meter=None) -> dict`
  - `def aggregate(rows: list[dict]) -> dict`（返回 dict 含 `total` / `answered` / `outcome_distribution` / `metric_means` / `cost`，**不再有** `classification` / `category_distribution`）
- Consumes：`eval.harness.metrics.METRIC_NAMES` / `MetricSpec`、`eval.harness.sut.RagOutput`。

- [ ] **Step 1: 在 test_eval_compare.py 末尾追加「打分函数」测试段（先红）**

在 `tests/test_eval_compare.py` 文件末尾追加：

```python
# ── 打分函数（原 run_eval，已搬入 compare.py）────────────────────────
from dataclasses import dataclass

from eval.harness.compare import aggregate, score_row, _row_to_dict, load_testset
from eval.harness.metrics import MetricSpec
from eval.harness.sut import RagOutput


class _AttrRow:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_row_to_dict_from_dict():
    assert _row_to_dict({"user_input": "Q", "reference": "R"})["user_input"] == "Q"


def test_row_to_dict_from_attr_object():
    d = _row_to_dict(_AttrRow(user_input="Q", reference="R"))
    assert d["user_input"] == "Q" and d["reference"] == "R"


def test_aggregate_means_only_over_answered():
    rows = [
        {"outcome": "answered", "faithfulness": 1.0, "answer_relevancy": 0.8,
         "context_precision": 1.0, "context_recall": 0.5, "factual_correctness": 0.6},
        {"outcome": "answered", "faithfulness": 0.0, "answer_relevancy": 0.6,
         "context_precision": 0.0, "context_recall": 0.5, "factual_correctness": 0.4},
        {"outcome": "empty"},
    ]
    rep = aggregate(rows)
    assert rep["total"] == 3
    assert rep["answered"] == 2
    assert rep["outcome_distribution"] == {"answered": 2, "empty": 1}
    assert rep["metric_means"]["faithfulness"] == 0.5
    assert rep["metric_means"]["answer_relevancy"] == 0.7
    assert "classification" not in rep          # 分类准确率已删
    assert "category_distribution" not in rep


def test_aggregate_ignores_none_scores():
    rows = [{"outcome": "answered", "faithfulness": None, "answer_relevancy": 0.4,
             "context_precision": None, "context_recall": None, "factual_correctness": None}]
    rep = aggregate(rows)
    assert rep["metric_means"]["faithfulness"] is None
    assert rep["metric_means"]["answer_relevancy"] == 0.4


def test_aggregate_cost_block_means_and_total():
    rows = [
        {"outcome": "answered", "latency_s": 1.0, "total_tokens": 100},
        {"outcome": "answered", "latency_s": 3.0, "total_tokens": 300},
    ]
    rep = aggregate(rows)
    assert rep["cost"]["mean_latency_s"] == 2.0
    assert rep["cost"]["mean_total_tokens"] == 200
    assert rep["cost"]["total_tokens"] == 400


def test_aggregate_cost_no_tokens_gives_none():
    rep = aggregate([{"outcome": "answered", "latency_s": 1.5}])
    assert rep["cost"]["mean_latency_s"] == 1.5
    assert rep["cost"]["mean_total_tokens"] is None
    assert rep["cost"]["total_tokens"] is None


@dataclass
class _MetricResult:
    value: float


class _FakeMetric:
    def __init__(self, value):
        self._v = value

    async def ascore(self, **kw):
        return _MetricResult(self._v)


class _FakeSUT:
    def __init__(self, out):
        self._out = out

    async def answer(self, query):
        return self._out


class _FakeMeter:
    def __init__(self, tokens):
        self._tokens = tokens
        self.reset_called = 0

    def reset(self):
        self.reset_called += 1

    def read(self):
        return self._tokens


async def test_score_row_answered_scores_all_metrics():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert res["outcome"] == "answered"
    assert res["faithfulness"] == 0.9
    assert res["response"] == "A"
    assert "category" not in res                # SUT category 已删


async def test_score_row_non_answered_skips_metrics():
    out = RagOutput(response="", retrieved_contexts=[], outcome="empty")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert res["outcome"] == "empty"
    assert "faithfulness" not in res


async def test_score_row_refuse_category_skips_metrics_even_if_answered():
    # golden 拒答行（out_of_scope）即便被硬答也不打分：指标归 null，避免污染质量均值
    out = RagOutput(response="编造的答案", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.0), lambda r, o: {})]
    res = await score_row(
        {"user_input": "Q", "reference": "", "category": "out_of_scope"}, _FakeSUT(out), specs
    )
    assert res["outcome"] == "answered"
    assert res["expected_category"] == "out_of_scope"
    assert "faithfulness" not in res
    assert res["latency_s"] >= 0


async def test_score_row_missing_info_category_skips_metrics():
    out = RagOutput(response="瞎猜的答案", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("answer_relevancy", _FakeMetric(0.4), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "category": "missing_info"}, _FakeSUT(out), specs)
    assert "answer_relevancy" not in res


async def test_score_row_explain_row_scores_and_is_not_refuse_skipped():
    # explain 金标准行：无 category、无 reference、answered → 照常打分（不被 REFUSE 短路）
    out = RagOutput(response="教学体答案", retrieved_contexts=["c"], outcome="answered")
    specs = [
        MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {}),
        MetricSpec("answer_relevancy", _FakeMetric(0.8), lambda r, o: {}),
    ]
    res = await score_row({"user_input": "讲讲MVCC", "reference": ""}, _FakeSUT(out), specs)
    assert res["expected_category"] == ""
    assert res["faithfulness"] == 0.9
    assert res["answer_relevancy"] == 0.8


async def test_score_row_records_latency_without_meter():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered")
    specs = [MetricSpec("faithfulness", _FakeMetric(0.9), lambda r, o: {})]
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), specs)
    assert isinstance(res["latency_s"], float) and res["latency_s"] >= 0
    assert "total_tokens" not in res


async def test_score_row_with_meter_records_tokens_and_resets():
    out = RagOutput(response="A", retrieved_contexts=["c"], outcome="answered")
    meter = _FakeMeter({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    res = await score_row({"user_input": "Q", "reference": "R"}, _FakeSUT(out), [], meter=meter)
    assert res["prompt_tokens"] == 100
    assert res["completion_tokens"] == 20
    assert res["total_tokens"] == 120
    assert meter.reset_called == 1


async def test_score_row_non_answered_still_records_latency_and_tokens():
    out = RagOutput(response="", retrieved_contexts=[], outcome="error")
    meter = _FakeMeter({"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5})
    res = await score_row({"user_input": "Q"}, _FakeSUT(out), [], meter=meter)
    assert res["latency_s"] >= 0
    assert res["total_tokens"] == 5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_compare.py -v`
Expected: FAIL —— `ImportError: cannot import name 'aggregate' from 'eval.harness.compare'`（compare.py 尚未有这些函数）。

- [ ] **Step 3: 在 compare.py 顶部追加 imports**

把 `eval/harness/compare.py` 现有 import 段（`from eval.harness.report import (...)` 之后）追加：

```python
import json
from time import perf_counter

from eval.harness.metrics import METRIC_NAMES, MetricSpec  # noqa: F401  (MetricSpec 供类型/测试)
from eval.harness.sut import RagOutput
```

- [ ] **Step 4: 在 compare.py 插入打分函数（VARIANTS 定义之前）**

在 `eval/harness/compare.py` 的 `VARIANTS = {` 之前插入：

```python
# 「拒答类」金标准：正确行为是反问澄清 / 告知库外，而非给出可被 ragas 打分的答案。
# 按金标准 expected_category 把这两类的指标归 null（对所有被测系统一致），避免污染质量均值。
REFUSE_CATEGORIES = frozenset({"missing_info", "out_of_scope"})


def load_testset(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_dict(row) -> dict:
    """把测试集行（dict / pydantic / 带属性对象）归一成 dict。"""
    if isinstance(row, dict):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    keys = ["user_input", "reference", "reference_contexts"]
    return {k: getattr(row, k, None) for k in keys}


async def score_row(
    row: dict, sut, metric_specs: list[MetricSpec], meter=None
) -> dict:
    # meter（可选）：测本条 SUT 的 token 消耗；reset 在 answer 前、read 在 answer 后。
    if meter is not None:
        meter.reset()
    t0 = perf_counter()
    out: RagOutput = await sut.answer(row["user_input"])
    latency_s = perf_counter() - t0
    base = {
        "user_input": row["user_input"],
        "reference": row.get("reference", ""),
        "response": out.response,
        "outcome": out.outcome,
        "expected_category": row.get("category", ""),   # golden 标注（仅供 REFUSE 闸门 + 人工查阅）
        "num_contexts": len(out.retrieved_contexts),
        "latency_s": round(latency_s, 3),
    }
    if meter is not None:
        base.update(meter.read())                        # prompt/completion/total_tokens
    # 非 answered（empty/error）不打分；拒答类金标准即便被硬答也不打分（指标归 null）。
    if out.outcome != "answered" or base["expected_category"] in REFUSE_CATEGORIES:
        return base
    for spec in metric_specs:
        try:
            result = await spec.metric.ascore(**spec.kwargs(row, out))
            base[spec.name] = result.value
        except Exception as e:  # noqa: BLE001 — 单指标失败不影响其他指标
            base[spec.name] = None
            base[f"{spec.name}_error"] = f"{type(e).__name__}: {e}"
    return base


def aggregate(rows: list[dict]) -> dict:
    """指标均值（仅 answered 行、忽略 None）+ outcome 分布 + 成本。"""
    outcomes: dict[str, int] = {}
    for r in rows:
        oc = r.get("outcome", "error")
        outcomes[oc] = outcomes.get(oc, 0) + 1
    answered = [r for r in rows if r.get("outcome") == "answered"]
    metric_means: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        vals = [r[name] for r in answered if r.get(name) is not None]
        metric_means[name] = (sum(vals) / len(vals)) if vals else None
    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    token_vals = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
    cost = {
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "mean_total_tokens": (sum(token_vals) / len(token_vals)) if token_vals else None,
        "total_tokens": sum(token_vals) if token_vals else None,
    }
    return {
        "total": len(rows),
        "answered": len(answered),
        "outcome_distribution": outcomes,
        "metric_means": metric_means,
        "cost": cost,
    }
```

- [ ] **Step 5: 把 compare.py 内对 run_eval 的 3 处 import 删掉**

在 `eval/harness/compare.py` 中删除这三行（函数已是本模块局部）：
- `_run_variants` 内：`from eval.harness.run_eval import load_testset, score_row, aggregate`
- `_score_rows_serial` 内：`from eval.harness.run_eval import score_row`
- `_score_rows_parallel` 内：`from eval.harness.run_eval import score_row`

- [ ] **Step 6: 删除 run_eval.py 与 test_eval_run.py**

```bash
git rm eval/harness/run_eval.py tests/test_eval_run.py
```

- [ ] **Step 7: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_compare.py -v`
Expected: PASS（新增打分测试段全绿；原有 render/build_sut 测试仍绿——本任务未改 VARIANTS/render）。

- [ ] **Step 8: 确认无残留 run_eval 引用**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q -k eval`
Expected: 全绿，无 `ModuleNotFoundError: eval.harness.run_eval`。

- [ ] **Step 9: Commit**

```bash
git add eval/harness/compare.py tests/test_eval_compare.py
git commit -m "refactor(eval): compare 吸收打分函数，删除 run_eval 单系统 runner

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: sut.py 去掉 category 维度

`RagOutput` 删 `category` 字段；`map_doc_result` 不再读 `metadata.category`、`map_agent_result` 去掉 category 入参；outcome 映射（answered/empty/error）保持不变。

**Files:**
- Modify: `eval/harness/sut.py`
- Modify: `tests/test_eval_sut.py`

**Interfaces:**
- Produces：`RagOutput(response: str, retrieved_contexts: list[str], outcome: str)`（**无** category 字段）；`map_doc_result(result, response_cls=None) -> RagOutput`；`map_agent_result(answer: str, sources: list) -> RagOutput`。

- [ ] **Step 1: 改 test_eval_sut.py，去掉所有 category 断言（先红）**

把 `tests/test_eval_sut.py` 中 `map_doc_result` / `map_agent_result` / `AgentSystem` 三组测试里**所有 `out.category == ...` 断言行删除**，并把首个 doc 测试改名（去掉 metadata.category 语义）。改后这两个 doc 测试为：

```python
def test_doc_answered():
    r = _RespMeta("答案", [_NodeWithScore("片段")], {"intent": "qa"})
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "answered"
    assert out.retrieved_contexts == ["片段"]


def test_doc_empty_when_no_nodes():
    r = _RespMeta("反问句", [], {"intent": "qa"})
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "empty"


def test_doc_handles_missing_metadata():
    r = _RespMeta("答案", [_NodeWithScore("片段")], None)
    out = map_doc_result(r, response_cls=_RespMeta)
    assert out.outcome == "answered"
```

（`test_agent_answered_with_sources` / `test_agent_empty_when_no_sources` / `test_agent_system_answered` / `test_agent_system_error_is_caught` 等：仅删去其中的 `assert out.category == ""` 行，其余断言保留。）

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_sut.py -v`
Expected: 仍 PASS（旧字段尚在，断言只是变少）—— 本步实为「准备」，真正先红在 Step 3 删字段后由旧断言引发。若想严格先红，可跳到 Step 3 先删字段再跑：Expected FAIL `AttributeError: 'RagOutput' object has no attribute 'category'`。

- [ ] **Step 3: 改 sut.py —— RagOutput 去 category**

`eval/harness/sut.py` 中 `RagOutput` 改为：

```python
@dataclass
class RagOutput:
    response: str
    retrieved_contexts: list[str]
    outcome: str  # answered | empty | error
```

- [ ] **Step 4: 改 sut.py —— map_doc_result 去 category**

替换 `map_doc_result` 为：

```python
def map_doc_result(result, response_cls=None) -> RagOutput:
    """DocQueryWorkflow.run() 的 Response → RagOutput（有 nodes→answered，无→empty）。"""
    if response_cls is None:
        from llama_index.core.base.response.schema import Response as response_cls  # noqa: N813
    if isinstance(result, response_cls):
        text = (getattr(result, "response", None) or "").strip()
        nodes = getattr(result, "source_nodes", None) or []
        if not text or not nodes:
            return RagOutput(text, [], "empty")
        contexts = [n.node.get_content() for n in nodes]
        return RagOutput(text, contexts, "answered")
    return RagOutput(str(result), [], "empty")
```

- [ ] **Step 5: 改 sut.py —— map_agent_result 去 category**

替换 `map_agent_result` 为：

```python
def map_agent_result(answer: str, sources: list) -> RagOutput:
    """AutoAgent.run() 的 (answer, source_nodes) → RagOutput。"""
    text = (answer or "").strip()
    if not text or not sources:
        return RagOutput(text, [], "empty")
    contexts = [_node_text(n) for n in sources]
    return RagOutput(text, contexts, "answered")
```

（`DocQueryWorkflowSystem.answer` / `AgentSystem.answer` 里的 `RagOutput(..., "error", "")` 第四参数也要删掉，改为 `RagOutput(f"{type(e).__name__}: {e}", [], "error")`。共两处 error 兜底。）

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_sut.py tests/test_eval_compare.py -v`
Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add eval/harness/sut.py tests/test_eval_sut.py
git commit -m "refactor(eval): RagOutput 去掉 category 维度（分类指标已删）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: compare.py VARIANTS 收敛为两路线

把 ablation flag 矩阵替换为两条命名路线（`workflow` / `agent`），`build_sut` 与 CLI 默认相应调整。

**Files:**
- Modify: `eval/harness/compare.py`
- Modify: `tests/test_eval_compare.py`

**Interfaces:**
- Produces：`WORKFLOW_VARIANT = "workflow"`、`AGENT_VARIANT = "agent"`、`VARIANTS = {"workflow": {}, "agent": None}`、`build_sut(name, index_manager, llm)`、`resolve_baseline(baseline, variant_names)`（不变）。

- [ ] **Step 1: 改 test_eval_compare.py 的 build_sut / VARIANTS / resolve 测试（先红）**

把 `tests/test_eval_compare.py` 中「build_sut 工厂与 agent 哨兵变体」与「baseline 回退」两段（`test_agent_variant_registered_as_sentinel` 到 `test_resolve_baseline_absent_falls_back_to_first`）整体替换为：

```python
# ── build_sut 工厂与两路线 VARIANTS ──────────────────────────────
from eval.harness.compare import build_sut, AGENT_VARIANT, WORKFLOW_VARIANT, VARIANTS
from eval.harness.sut import AgentSystem, DocQueryWorkflowSystem


def test_variants_are_exactly_two_routes():
    assert set(VARIANTS) == {WORKFLOW_VARIANT, AGENT_VARIANT}
    assert VARIANTS[WORKFLOW_VARIANT] == {}     # 默认 flags = 生产配置
    assert VARIANTS[AGENT_VARIANT] is None      # 哨兵 → AgentSystem


def test_build_sut_agent_returns_agent_system():
    sut = build_sut(AGENT_VARIANT, index_manager=object(), llm=object())
    assert isinstance(sut, AgentSystem)


def test_build_sut_workflow_returns_workflow_system():
    sut = build_sut(WORKFLOW_VARIANT, index_manager=object(), llm=object())
    assert isinstance(sut, DocQueryWorkflowSystem)


def test_build_sut_unknown_name_raises():
    import pytest
    with pytest.raises(KeyError):
        build_sut("不存在的变体", index_manager=object(), llm=object())


# ── baseline 回退（默认 baseline 不在所选 --variants 子集里时回退首个）──
from eval.harness.compare import resolve_baseline


def test_resolve_baseline_present_returns_it():
    assert resolve_baseline("workflow", ["workflow", "agent"]) == "workflow"


def test_resolve_baseline_absent_falls_back_to_first():
    assert resolve_baseline("不存在", ["agent", "workflow"]) == "agent"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_compare.py -v`
Expected: FAIL —— `ImportError: cannot import name 'WORKFLOW_VARIANT'`。

- [ ] **Step 3: 改 compare.py —— 替换 VARIANTS 与 AGENT_VARIANT 段**

把 `eval/harness/compare.py` 中从 `# 变体矩阵：...` 注释到 `VARIANTS[AGENT_VARIANT] = None` 的整段（含 `VARIANTS = {...}` 与 AGENT_VARIANT 定义）替换为：

```python
# 两条 SUT 路线：workflow（默认 flags = DocQueryService 生产配置）vs agent（自主规划）。
# workflow 用空 flags dict（= 默认）；agent 用 None 作哨兵，build_sut 据此分流到 AgentSystem。
WORKFLOW_VARIANT = "workflow"
AGENT_VARIANT = "agent"
VARIANTS = {
    WORKFLOW_VARIANT: {},
    AGENT_VARIANT: None,
}
```

- [ ] **Step 4: 改 compare.py —— build_sut docstring 对齐（逻辑不变）**

`build_sut` 内逻辑保持（`flags is None` → AgentSystem，否则 `DocQueryWorkflowSystem(index_manager, llm, flags=flags)`）。仅把 docstring 改为：

```python
    """按路线名构造被测系统：哨兵(None) → AgentSystem，否则 DocQueryWorkflowSystem(默认 flags)。"""
```

- [ ] **Step 5: 改 compare.py main —— CLI 默认两路线全跑、baseline=workflow**

把 `main()` 里 `--variants` 与 `--baseline` 两个 `add_argument` 改为：

```python
    p.add_argument("--variants", nargs="+",
                   default=list(VARIANTS),
                   choices=list(VARIANTS.keys()),
                   help=f"路线子集，可选：{list(VARIANTS.keys())}（默认两条都跑）")
    p.add_argument("--baseline", default=WORKFLOW_VARIANT,
                   help="作为 delta 基准的路线名（默认 workflow，delta 列即 agent 相对 workflow）")
```

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_compare.py -v`
Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add eval/harness/compare.py tests/test_eval_compare.py
git commit -m "refactor(eval): VARIANTS 收敛为 workflow vs agent 两路线

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: report.py 去分类准确率列与 match 列

对比表删「分类准确率」列；明细 CSV 删 `category` / `match` 列，保留 `expected_category`。

**Files:**
- Modify: `eval/harness/report.py`
- Modify: `tests/test_eval_report.py`
- Modify: `tests/test_eval_compare.py`（render 测试段去 classification）

**Interfaces:**
- Produces：`render_delta_table(variants, baseline)`（列 = 5 ragas + 2 成本，无分类准确率）；`write_detail_csv(detail, path)`（列见下，无 category/match）；`default_result_paths(prefix="compare", now=None)`。

- [ ] **Step 1: 改 test_eval_report.py（先红）**

把 `tests/test_eval_report.py` 整体替换为（去 classification 列、去 match 列、prefix 默认 compare）：

```python
"""eval/harness/report.py 纯展示+落盘单测。"""
import csv
import os
from datetime import datetime

from eval.harness.report import (
    default_result_paths,
    render_delta_table,
    write_detail_csv,
)


def test_render_delta_table_single_row_no_delta():
    variants = [{"name": "workflow", "report": {"metric_means": {"faithfulness": 0.9}}}]
    md = render_delta_table(variants, baseline="workflow")
    assert "| workflow |" in md
    assert "0.90" in md
    assert "(+0" not in md and "(-0" not in md


def test_render_delta_table_no_classification_column():
    variants = [{"name": "workflow", "report": {"metric_means": {"faithfulness": 0.9}}}]
    md = render_delta_table(variants, baseline="workflow")
    assert "分类准确率" not in md           # 分类列已删
    assert "faithfulness" in md


def test_render_delta_table_shows_cost_columns():
    variants = [{"name": "S", "report": {
        "metric_means": {"faithfulness": 0.9},
        "cost": {"mean_latency_s": 2.35, "mean_total_tokens": 1200.0, "total_tokens": 2400},
    }}]
    md = render_delta_table(variants, baseline="S")
    assert "时延(s/条)" in md and "tokens/条" in md
    assert "2.35" in md and "1200.00" in md


def test_render_delta_table_cost_missing_shows_dash():
    variants = [{"name": "S", "report": {"metric_means": {}}}]
    md = render_delta_table(variants, baseline="S")
    assert "时延(s/条)" in md and "tokens/条" in md
    assert "—" in md


def test_render_delta_table_raises_on_missing_baseline():
    import pytest
    variants = [{"name": "workflow", "report": {"metric_means": {}}}]
    with pytest.raises(ValueError):
        render_delta_table(variants, baseline="不存在")


def test_default_result_paths_defaults_to_compare_prefix():
    now = datetime(2026, 6, 18, 13, 0, 0)
    md, detail = default_result_paths(now=now)
    assert md.endswith(os.path.join("20260618_130000", "compare.md"))
    assert detail.endswith(os.path.join("20260618_130000", "compare_detail.csv"))


def test_write_detail_csv_includes_cost_columns(tmp_path):
    detail = [{"variant": "S", "user_input": "Q", "expected_category": "retrievable",
               "outcome": "answered", "response": "A", "num_contexts": 2,
               "latency_s": 1.2, "prompt_tokens": 100, "completion_tokens": 20,
               "total_tokens": 120}]
    path = tmp_path / "cost_detail.csv"
    write_detail_csv(detail, str(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["latency_s"] == "1.2"
    assert rows[0]["total_tokens"] == "120"
    assert rows[0]["expected_category"] == "retrievable"


def test_write_detail_csv_has_no_category_or_match_columns(tmp_path):
    detail = [{"variant": "S", "user_input": "Q", "expected_category": "x",
               "outcome": "answered", "response": "A", "num_contexts": 1}]
    path = tmp_path / "no_match.csv"
    write_detail_csv(detail, str(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert "match" not in header
    assert "category" not in header
    assert "expected_category" in header
```

- [ ] **Step 2: 改 test_eval_compare.py render 段去 classification**

把 `tests/test_eval_compare.py` 顶部三个 render 测试（`test_render_delta_table_marks_improvement` / `test_render_delta_table_baseline_row_has_no_delta` / `test_render_delta_table_none_metric_shows_dash`）替换为：

```python
"""对比表渲染纯逻辑单测（render_delta_table）。"""
from eval.harness.compare import render_delta_table


def test_render_delta_table_marks_improvement():
    variants = [
        {"name": "workflow", "report": {"metric_means": {"context_recall": 0.62}}},
        {"name": "agent", "report": {"metric_means": {"context_recall": 0.78}}},
    ]
    md = render_delta_table(variants, baseline="workflow")
    assert "| workflow |" in md
    assert "| agent |" in md
    assert "0.78" in md
    assert "+0.16" in md                       # context_recall delta（0.78-0.62）


def test_render_delta_table_baseline_row_has_no_delta():
    variants = [{"name": "base", "report": {"metric_means": {"faithfulness": 0.5}}}]
    md = render_delta_table(variants, baseline="base")
    assert "0.50" in md
    assert "(+0" not in md and "(-0" not in md


def test_render_delta_table_none_metric_shows_dash():
    variants = [{"name": "base", "report": {"metric_means": {}}}]
    md = render_delta_table(variants, baseline="base")
    assert "—" in md
```

（同时删掉文件后段那个 `test_render_delta_table_raises_on_missing_baseline`——它已迁到 test_eval_report.py；若仍想留，把其 report dict 里的 `"classification": {...}` 去掉。为避免重复，删除 test_eval_compare.py 内的该函数。）

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_report.py tests/test_eval_compare.py -v`
Expected: FAIL（report.py 仍渲染「分类准确率」列、detail 仍写 match）。

- [ ] **Step 4: 改 report.py —— _COLS 去分类列**

把 `eval/harness/report.py` 的 `_COLS` 改为（删除首个「分类准确率」元组）：

```python
_COLS = [
    ("context_precision", lambda rep: rep.get("metric_means", {}).get("context_precision")),
    ("context_recall", lambda rep: rep.get("metric_means", {}).get("context_recall")),
    ("factual_correctness", lambda rep: rep.get("metric_means", {}).get("factual_correctness")),
    ("faithfulness", lambda rep: rep.get("metric_means", {}).get("faithfulness")),
    ("answer_relevancy", lambda rep: rep.get("metric_means", {}).get("answer_relevancy")),
    # 成本列：越低越好——delta 为正＝更贵（与上面质量列符号相反）
    ("时延(s/条)", lambda rep: rep.get("cost", {}).get("mean_latency_s")),
    ("tokens/条", lambda rep: rep.get("cost", {}).get("mean_total_tokens")),
]
```

- [ ] **Step 5: 改 report.py —— _DETAIL_COLS 去 category/match，write_detail_csv 去 match 计算**

`_DETAIL_COLS` 改为：

```python
_DETAIL_COLS = [
    "variant", "user_input", "expected_category", "outcome",
    "reference", "response", "num_contexts",
    "faithfulness", "answer_relevancy", "context_precision",
    "context_recall", "factual_correctness",
    "latency_s", "prompt_tokens", "completion_tokens", "total_tokens",
]
```

`write_detail_csv` 改为（去掉 `row["match"] = ...`）：

```python
def write_detail_csv(detail: list[dict], path: str) -> None:
    """每条明细写 CSV（utf-8-sig，Excel 直开）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_DETAIL_COLS, extrasaction="ignore")
        w.writeheader()
        for d in detail:
            w.writerow(d)
```

- [ ] **Step 6: 改 report.py 顶部 docstring**

把 `report.py` 模块 docstring 里「分类准确率 + 5 ragas 列」「compare 与 run_eval 共用」措辞改为「5 ragas + 成本列」「compare 用」。具体：第 4 行 `render_delta_table：分类准确率 + 5 ragas 列...` → `render_delta_table：5 ragas + 成本列的 Markdown 表（单行时无 delta）。`；第 5 行去掉 match 提法。

- [ ] **Step 7: 跑测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_eval_report.py tests/test_eval_compare.py -v`
Expected: PASS。

- [ ] **Step 8: Commit**

```bash
git add eval/harness/report.py tests/test_eval_report.py tests/test_eval_compare.py
git commit -m "refactor(eval): 对比表去分类准确率列，明细去 category/match 列

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: 文档/注释收尾 + 全量 smoke

更新 datagen 注释与 `docs/EVAL_OVERVIEW.md`，跑分层守卫与真实集成 smoke 验证端到端可用。

**Files:**
- Modify: `eval/datagen/merge_golden.py`、`eval/datagen/generate_testset.py`
- Modify: `docs/EVAL_OVERVIEW.md`

- [ ] **Step 1: 改 datagen 注释**

- `eval/datagen/merge_golden.py` 第 7 行注释 `category=构造意图金标准，run_eval 读作...` → 把 `run_eval` 改为 `compare`。
- `eval/datagen/generate_testset.py` 第 4 行与第 177 行 `供 run_eval 使用` / `再跑 run_eval` → 改为 `供 compare 使用` / `再跑 compare`。

- [ ] **Step 2: 改 docs/EVAL_OVERVIEW.md**

- 架构图（第 19 行）删 `run_eval.py 单系统跑分...` 这一行；第 24 行 `+ run_eval.aggregate: 分类准确率` 改为 `（agent vs workflow 两路线）`。
- 删「3.2 分类准确率」整节（第 80 行附近的小节）；如该节是质量指标里唯一确定性项，改为一句说明：分类准确率指标已随门口路由重构移除，现以 ragas 质量 + 成本对比两路线。
- 目录树（第 144 行）删 `run_eval.py ...` 行；第 157 行 `run_eval / compare 落盘` → `compare 落盘`。
- 第 200 行历史注记可保留或补一句「run_eval 已并入 compare」。

- [ ] **Step 3: 分层守卫**

Run: `.venv\Scripts\python.exe scripts/check_layering.py`
Expected: 通过（无 core→api 依赖）。

- [ ] **Step 4: 全量 eval 单测**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q -k eval`
Expected: 全绿，无 run_eval 残留导入。

- [ ] **Step 5: 真实集成 smoke（两路线各跑 2 条）**

需 `.env` 有 `DEEPSEEK_API_KEY` 且 `chroma_db` 已建索引。

Run: `.venv\Scripts\python.exe -m eval.harness.compare --testset eval/dataset/golden.jsonl --limit 2`
Expected: stdout 打印含 `workflow` / `agent` 两行的 Markdown 对比表（5 ragas + 时延 + tokens 列，无分类准确率列）；`eval/results/<时间戳>/compare.md` 与 `compare_detail.csv` 落盘，detail CSV 表头无 `category`/`match`、有 `expected_category`。

- [ ] **Step 6: Commit**

```bash
git add eval/datagen/merge_golden.py eval/datagen/generate_testset.py docs/EVAL_OVERVIEW.md
git commit -m "docs(eval): 文案更新为 workflow vs agent 两路线，移除 run_eval/分类准确率

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review 结果

- **Spec 覆盖**：run_eval 删除 + 函数搬迁（T1）、sut category（T2）、VARIANTS 两路线（T3）、report 分类列/match（T4）、datagen+EVAL_OVERVIEW 文案+smoke+分层（T5）—— 全覆盖。
- **占位符**：无；每个改动给出完整代码块或精确行级指令。
- **类型一致**：`RagOutput(response, retrieved_contexts, outcome)` 三参贯穿 T1/T2 测试与 mapper；`aggregate` 返回不含 classification 在 T1 定义、T4 渲染依赖一致；`VARIANTS`/`WORKFLOW_VARIANT`/`AGENT_VARIANT` 在 T3 定义、T3 测试消费一致。
- **顺序安全**：T1 后旧 `RagOutput.category` 仍在（score_row 不再读它）→ 套件绿；T2 删字段时 score_row 已不依赖；T3 改 VARIANTS 同步改其测试；T4 改 render 同步改两处 render 测试。每个 Task 收尾套件均绿。
