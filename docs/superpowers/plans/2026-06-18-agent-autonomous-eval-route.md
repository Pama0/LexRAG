# 评测「agent 自主规划」对照路线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给评测 ablation 加第二种被测系统——绕过 `DocQueryWorkflow` 决策路由、复用 `QaAgent` 让 agent 自主规划检索——并作为一行进同一张对比表。

**Architecture:** 新增 SUT 适配器 `AgentSystem`（实现既有 `RagSystem` 协议）内部跑 `QaAgent`，用 no-op `_NullCtx` 复用其 `run()`；`aggregate` 一行语义微调让无 category 系统分类准确率显示 N/A；`compare.py` 抽 `build_sut` 工厂按变体名分流到两种 SUT。

**Tech Stack:** Python 3.12 / asyncio / pytest（pytest-asyncio，测试用 `async def` 直接写）/ LlamaIndex FunctionAgent。

## Global Constraints

- 所有 I/O 用 `async/await`；函数签名加类型注解（CLAUDE.md Code Style）。
- 必须从项目根目录运行；子模块内用相对导入，跨包用绝对导入。
- core 层不依赖 api；评测代码在 `eval/`，复用 `core/` 组件。
- 对 `DocQueryWorkflowSystem`、现有 compare 变体、golden 数据集**零行为回归**。
- 被测 LLM = DeepSeek（`configs/llm.py`）；评测 judge = `eval/config.py`。本计划不碰 LLM 配置。
- 哨兵变体名固定为 `"agent(自主规划)"`（全角括号），workflow 变体名沿用 `VARIANTS` 现有键。

---

### Task 1: `map_agent_result` 纯映射函数

把 `QaAgent.run()` 的 `(answer, sources)` 归一成 `RagOutput`，纯函数便于单测（镜像现有 `map_doc_result`）。agent 不产分类 → `category` 恒空。

**Files:**
- Modify: `eval/harness/sut.py`（在 `map_doc_result` 之后新增 `map_agent_result`）
- Test: `tests/test_eval_sut.py`（追加）

**Interfaces:**
- Consumes: 既有 `RagOutput` dataclass（`response, retrieved_contexts, outcome, category`）。
- Produces: `def map_agent_result(answer: str, sources: list) -> RagOutput`，`outcome` ∈ `{"answered","empty"}`，`category` 恒 `""`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_eval_sut.py` 末尾追加（复用文件顶部已有的 `_Node`：有 `get_content()`）：

```python
from eval.harness.sut import map_agent_result


def test_agent_answered_with_sources():
    out = map_agent_result("综合答案", [_Node("片段A"), _Node("片段B")])
    assert out.outcome == "answered"
    assert out.response == "综合答案"
    assert out.retrieved_contexts == ["片段A", "片段B"]
    assert out.category == ""          # agent 不产分类


def test_agent_empty_when_no_sources():
    out = map_agent_result("答案", [])
    assert out.outcome == "empty"
    assert out.category == ""


def test_agent_empty_when_blank_answer():
    out = map_agent_result("   ", [_Node("片段A")])
    assert out.outcome == "empty"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_eval_sut.py -k agent -v`
Expected: FAIL（`ImportError: cannot import name 'map_agent_result'`）

- [ ] **Step 3: 写最小实现**

在 `eval/harness/sut.py` 中 `map_doc_result` 函数之后插入：

```python
def _node_text(n) -> str:
    """从 NodeWithScore / Node 取正文（镜像 QaAgent._search 的提取逻辑）。"""
    return n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")


def map_agent_result(answer: str, sources: list) -> RagOutput:
    """QaAgent.run() 的 (answer, source_nodes) → RagOutput；agent 不产分类，category 恒空。"""
    text = (answer or "").strip()
    if not text or not sources:
        return RagOutput(text, [], "empty", "")
    contexts = [_node_text(n) for n in sources]
    return RagOutput(text, contexts, "answered", "")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_eval_sut.py -k agent -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add eval/harness/sut.py tests/test_eval_sut.py
git commit -m "feat(eval): map_agent_result 把 agent (answer,sources) 归一成 RagOutput"
```

---

### Task 2: `_NullCtx` + `AgentSystem` 适配器

`AgentSystem` 实现 `RagSystem` 协议，内部每条新建 `QaAgent`，用 no-op ctx 复用其 `run()`，映射经 Task 1 的 `map_agent_result`；异常兜底 `outcome="error"`（同 `DocQueryWorkflowSystem`）。

**Files:**
- Modify: `eval/harness/sut.py`（文件末尾新增 `_NullCtx`、`AgentSystem`）
- Test: `tests/test_eval_sut.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `map_agent_result`；`core.agent.qa_agent.QaAgent`（签名 `QaAgent(index_manager, llm, similarity_top_k=5, max_iterations=6)`，`async run(self, ctx, query, book_titles) -> tuple[str, list]`）。
- Produces: `class AgentSystem`，`async def answer(self, query: str, book_titles=None) -> RagOutput`，满足 `RagSystem` 协议。`class _NullCtx`，`write_event_to_stream(self, event) -> None`（no-op）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_eval_sut.py` 末尾追加。用 monkeypatch 把 `QaAgent` 换成假替身（`AgentSystem.answer` 内部 `from core.agent.qa_agent import QaAgent`，故 patch 模块属性即可注入）：

```python
import pytest
from eval.harness.sut import AgentSystem, _NullCtx


class _FakeQaAgent:
    """记录构造与 run 入参，返回预置 (answer, sources)；可设为抛异常。"""
    last_instance = None

    def __init__(self, index_manager, llm, similarity_top_k=5, max_iterations=6):
        self.kw = dict(similarity_top_k=similarity_top_k, max_iterations=max_iterations)
        self.run_args = None
        type(self).last_instance = self

    async def run(self, ctx, query, book_titles):
        self.run_args = dict(ctx=ctx, query=query, book_titles=book_titles)
        if query == "boom":
            raise RuntimeError("agent 崩了")
        if query == "empty":
            return ("", [])
        return ("综合答案", [_Node("片段A"), _Node("片段B")])


def test_nullctx_write_is_noop():
    assert _NullCtx().write_event_to_stream("任意事件") is None


async def test_agent_system_answered(monkeypatch):
    monkeypatch.setattr("core.agent.qa_agent.QaAgent", _FakeQaAgent)
    sut = AgentSystem(index_manager=object(), llm=object())
    out = await sut.answer("openclaw 架构与权衡")
    assert out.outcome == "answered"
    assert out.response == "综合答案"
    assert out.retrieved_contexts == ["片段A", "片段B"]
    assert out.category == ""
    # 复用 QaAgent.run 时传入的是 _NullCtx
    assert isinstance(_FakeQaAgent.last_instance.run_args["ctx"], _NullCtx)


async def test_agent_system_empty(monkeypatch):
    monkeypatch.setattr("core.agent.qa_agent.QaAgent", _FakeQaAgent)
    sut = AgentSystem(index_manager=object(), llm=object())
    out = await sut.answer("empty")
    assert out.outcome == "empty"


async def test_agent_system_error_is_caught(monkeypatch):
    monkeypatch.setattr("core.agent.qa_agent.QaAgent", _FakeQaAgent)
    sut = AgentSystem(index_manager=object(), llm=object())
    out = await sut.answer("boom")
    assert out.outcome == "error"
    assert "RuntimeError" in out.response
    assert out.category == ""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_eval_sut.py -k "agent_system or nullctx" -v`
Expected: FAIL（`ImportError: cannot import name 'AgentSystem'`）

- [ ] **Step 3: 写最小实现**

在 `eval/harness/sut.py` 文件末尾（`DocQueryWorkflowSystem` 之后）追加：

```python
# ── agent 自主规划路线（QaAgent，绕过 DocQueryWorkflow 决策路由）──────
class _NullCtx:
    """QaAgent.run 需要带 write_event_to_stream 的 ctx 推前端流式事件；
    评测无 workflow ctx，用 no-op 替身。最终答案来自 await handler，与 ctx 无关。"""

    def write_event_to_stream(self, event) -> None:  # noqa: D401 — no-op
        pass


class AgentSystem:
    """被测系统：每条 query 直接喂有界 QaAgent 自主规划检索，实现 RagSystem。"""

    def __init__(self, index_manager, llm,
                 similarity_top_k: int = 5, max_iterations: int = 6):
        self._index_manager = index_manager
        self._llm = llm
        self._similarity_top_k = similarity_top_k
        self._max_iterations = max_iterations

    async def answer(self, query: str, book_titles=None) -> RagOutput:
        from core.agent.qa_agent import QaAgent

        qa = QaAgent(
            self._index_manager, self._llm,
            similarity_top_k=self._similarity_top_k,
            max_iterations=self._max_iterations,
        )
        try:
            answer, sources = await qa.run(_NullCtx(), query, book_titles)
        except Exception as e:  # noqa: BLE001 — 单条异常记 error 不中断
            return RagOutput(f"{type(e).__name__}: {e}", [], "error", "")
        return map_agent_result(answer, sources)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_eval_sut.py -k "agent_system or nullctx" -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add eval/harness/sut.py tests/test_eval_sut.py
git commit -m "feat(eval): AgentSystem 适配器——QaAgent 自主规划路线 + NullCtx"
```

---

### Task 3: `aggregate` 分类准确率 N/A

把分类计分门控从 `if exp:` 改为 `if exp and cat:`，让不产 category 的系统（agent）分类准确率 `accuracy=None`（表里「—」），workflow 行为不变。

**Files:**
- Modify: `eval/harness/run_eval.py:65-75`（`aggregate` 内分类计分块）
- Test: `tests/test_eval_run.py`（追加）

**Interfaces:**
- Consumes: 既有 `aggregate(rows: list[dict]) -> dict`，`rows` 含 `category` / `expected_category`。
- Produces: 行为变化——当某行 `category` 为空时不计入 `classification.total`；全空时 `accuracy is None`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_eval_run.py` 末尾追加（已有 `test_aggregate_classification_accuracy_and_distribution` 充当「有 category 不回归」基准，本步只加 N/A 新例）：

```python
def test_aggregate_no_category_system_gives_na_accuracy():
    # agent 路线：每行 category 空、但金标准 expected_category 非空
    rows = [
        {"outcome": "answered", "category": "", "expected_category": "retrievable"},
        {"outcome": "answered", "category": "", "expected_category": "other"},
    ]
    rep = aggregate(rows)
    assert rep["classification"]["accuracy"] is None
    assert rep["classification"]["total"] == 0


def test_aggregate_skips_blank_category_but_keeps_others():
    # 混合：一行有 category（计分），一行空（跳过，不算误判）
    rows = [
        {"outcome": "answered", "category": "retrievable", "expected_category": "retrievable"},
        {"outcome": "error", "category": "", "expected_category": "other"},
    ]
    rep = aggregate(rows)
    assert rep["classification"]["total"] == 1
    assert rep["classification"]["accuracy"] == 1.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_eval_run.py -k "na_accuracy or skips_blank" -v`
Expected: FAIL（旧逻辑 `if exp:` 把空 category 行计入 → `total` 非 0、`accuracy` 为 0.0/0.5）

- [ ] **Step 3: 写最小实现**

在 `eval/harness/run_eval.py` 的 `aggregate` 中，把分类计分判据从：

```python
        exp = r.get("expected_category")
        if exp:
            cls_total += 1
            cls_correct += int(cat == exp)
```

改为（加 `and cat`，并补一行注释）：

```python
        exp = r.get("expected_category")
        if exp and cat:  # 仅在系统确实产出类别时计分；agent(无 category) → N/A，error 行不算误判
            cls_total += 1
            cls_correct += int(cat == exp)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_eval_run.py -v`
Expected: PASS（含既有 `test_aggregate_classification_accuracy_and_distribution` 不回归）

- [ ] **Step 5: 提交**

```bash
git add eval/harness/run_eval.py tests/test_eval_run.py
git commit -m "fix(eval): aggregate 分类计分加 and cat——无 category 系统显示 N/A"
```

---

### Task 4: `compare.py` 接入——`build_sut` 工厂 + 哨兵变体

抽 `build_sut(name, ...)` 按变体名分流到 `AgentSystem` / `DocQueryWorkflowSystem`；新增哨兵 `AGENT_VARIANT`（在 `VARIANTS` 里值为 `None` 以进默认全集与 CLI 可选名，但不当作 flags）。

**Files:**
- Modify: `eval/harness/compare.py`（新增 `AGENT_VARIANT` 常量 + `VARIANTS` 加键 + `build_sut` 函数 + `_run_variants` 改用 `build_sut`）
- Test: `tests/test_eval_compare.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 `AgentSystem`；既有 `DocQueryWorkflowSystem`、`VARIANTS`。
- Produces: `AGENT_VARIANT: str = "agent(自主规划)"`；`def build_sut(name: str, index_manager, llm)` → `AgentSystem`（当 `VARIANTS[name] is None`）或 `DocQueryWorkflowSystem`（否则，`flags=VARIANTS[name]`）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_eval_compare.py` 末尾追加（`build_sut` 只构造、存字段、不触发重导入，可用 dummy 入参）：

```python
from eval.harness.compare import build_sut, AGENT_VARIANT, VARIANTS
from eval.harness.sut import AgentSystem, DocQueryWorkflowSystem


def test_agent_variant_registered_as_sentinel():
    assert AGENT_VARIANT in VARIANTS
    assert VARIANTS[AGENT_VARIANT] is None   # 哨兵：非 flags dict


def test_build_sut_agent_variant_returns_agent_system():
    sut = build_sut(AGENT_VARIANT, index_manager=object(), llm=object())
    assert isinstance(sut, AgentSystem)


def test_build_sut_workflow_variant_returns_workflow_system():
    sut = build_sut("baseline(全单轮)", index_manager=object(), llm=object())
    assert isinstance(sut, DocQueryWorkflowSystem)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_eval_compare.py -k "agent_variant or build_sut" -v`
Expected: FAIL（`ImportError: cannot import name 'build_sut'`）

- [ ] **Step 3: 写最小实现**

3a. 在 `eval/harness/compare.py` 的 `VARIANTS` 字典定义**之后**追加哨兵注册与工厂：

```python
# agent 自主规划路线：另一个 SUT 类（非 flags 组合）。值置 None 作哨兵，
# 既进默认全集 / CLI 可选名，又能在 build_sut 里按 None 分流。
AGENT_VARIANT = "agent(自主规划)"
VARIANTS[AGENT_VARIANT] = None


def build_sut(name: str, index_manager, llm):
    """按变体名构造被测系统：哨兵(None) → AgentSystem，否则 DocQueryWorkflowSystem(flags)。"""
    from eval.harness.sut import AgentSystem, DocQueryWorkflowSystem
    if VARIANTS.get(name) is None:
        return AgentSystem(index_manager, llm)
    return DocQueryWorkflowSystem(index_manager, llm, flags=VARIANTS[name])
```

3b. 在 `_run_variants` 内，把：

```python
        sut = DocQueryWorkflowSystem(index_manager, sut_llm, flags=VARIANTS[name])
```

改为：

```python
        sut = build_sut(name, index_manager, sut_llm)
```

并删除 `_run_variants` 顶部 import 块里的 `from eval.harness.sut import DocQueryWorkflowSystem`（现由 `build_sut` 内部导入；若该行还导入了其它名则只删 `DocQueryWorkflowSystem`）。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_eval_compare.py -v`
Expected: PASS（含既有 `render_delta_table` 三测不回归）

- [ ] **Step 5: 全量回归 + 提交**

Run: `python -m pytest tests/test_eval_sut.py tests/test_eval_run.py tests/test_eval_compare.py -v`
Expected: PASS（全绿）

```bash
git add eval/harness/compare.py tests/test_eval_compare.py
git commit -m "feat(eval): compare 接入 agent(自主规划) 变体——build_sut 工厂分流"
```

---

### Task 5: 文档更新（EVAL_OVERVIEW.md）

把第二被测系统写进总览，让读者知道 ablation 里多了「agent 自主规划」对照行及其跑法。

**Files:**
- Modify: `docs/EVAL_OVERVIEW.md`（§2 被测系统、§7 怎么跑）

**Interfaces:**
- Consumes: 无（纯文档）。
- Produces: 无代码接口。

- [ ] **Step 1: §2 末尾补「第二被测系统」小节**

在 `docs/EVAL_OVERVIEW.md` 第 2 节（被测系统 SUT）的 `**probe（探测检索）** 是横切开关 ...` 那段之后，新增：

```markdown
### Layer 0 替代 · agent 自主规划路线（对照系）
除上面的 `DocQueryWorkflow`，评测另有第二被测系统 `eval/harness/sut.py` 的
`AgentSystem`：**绕过 IntentRouter + category 分类**，每条 query 直接喂给有界
`QaAgent`（FunctionAgent + book_search/list_books，自主多轮规划检索）。它**不产
category**，故分类准确率列显示「—」，只在 5 个 ragas 答案质量指标上与 workflow 同台对比。
用途：回答「显式决策路由 vs 让 agent 自己规划」到底谁强。
```

- [ ] **Step 2: §7 跑法补一条**

在 `docs/EVAL_OVERVIEW.md` 第 7 节「逐步加决策」那条命令之后，新增：

````markdown
# 对照：workflow 全开 vs agent 自主规划（同表，agent 行分类准确率列为 —）
python -m eval.harness.compare --testset eval/dataset/golden.jsonl --variants "全开" "agent(自主规划)"
````

- [ ] **Step 3: 提交**

```bash
git add docs/EVAL_OVERVIEW.md
git commit -m "docs(eval): EVAL_OVERVIEW 记第二被测系统 agent 自主规划路线"
```

---

## Self-Review

**Spec coverage：**
- 组件 1 `AgentSystem` + `_NullCtx` → Task 1（映射）+ Task 2（适配器/NullCtx）。✓
- 组件 2 `aggregate` `if exp and cat:` → Task 3。✓
- 组件 3 `compare.py` `AGENT_VARIANT` 分流 → Task 4。✓
- spec §3.1 retrieved_contexts 取 node 文本 → Task 1 `_node_text`（镜像 QaAgent）。✓
- spec §5 错误处理（error / empty）→ Task 2 try/except、Task 1 empty 分支。✓
- spec §6 测试三项（映射 / NullCtx / aggregate N/A）→ Task 1、2、3 测试步。✓
- spec §7 文档补 EVAL_OVERVIEW → Task 5。✓

**Placeholder scan：** 无 TBD/TODO；每个代码步含完整代码与确切命令/预期。✓

**Type consistency：**
- `map_agent_result(answer: str, sources: list) -> RagOutput`：Task 1 定义、Task 2 调用，签名一致。✓
- `AgentSystem(index_manager, llm, similarity_top_k=5, max_iterations=6)` + `answer(query, book_titles=None)`：Task 2 定义、Task 4 `build_sut` 构造（只传 `index_manager, llm`，其余默认）一致。✓
- `QaAgent(index_manager, llm, similarity_top_k, max_iterations)` 与 `run(ctx, query, book_titles)`：与 `core/agent/qa_agent.py` 实际签名一致。✓
- `AGENT_VARIANT = "agent(自主规划)"`：Task 4 定义、Task 5 文档命令、Global Constraints 三处字面一致。✓
- `_NullCtx.write_event_to_stream(event)`：Task 2 定义，与 `QaAgent.run` 内 `ctx.write_event_to_stream(...)` 调用契合。✓
