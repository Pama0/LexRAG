# ConversationScoper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 全库模式下，从会话历史锚定本轮主体，把检索硬锁到主导书，消除「讲一下gateway」这类裸概念续问被其它书同名概念污染的问题。

**Architecture:** 新增注入式单元 `ConversationScoper`（仿 `QueryGate`/`Admitter`），在 `doc_workflow.route` 的 `dispatch_qa` 分支后调用一次：用「最近 N 轮用户问 ⊕ clean_query」拼接文本跑一次轻量 vector probe，按命中 `book_title` 集中度判主导书，写回 ctx 的 `book_titles`（下游检索零改动地硬过滤）+ `scope_note`（透明声明，由各答案分支当前缀流式输出）。门口保持保守不动；纠偏用 `disable_scope`（用户说"在所有书里"时跳过收窄）。

**Tech Stack:** Python 3.12 async、llama-index Workflow、pytest（async 自动模式）、Chroma 元数据过滤。

## Global Constraints

- 所有 I/O 用 `async/await`；函数签名加类型注解；中文注释可接受。
- 依赖方向单向 `api/ → core/ → configs/`；`core` 不依赖 `api`。
- 根目录绝对导入（`from core.workflow.x import ...`），子模块内相对导入。
- 降级铁律：scoper 任意失败（probe 异常 / 空命中 / 统计异常）→ 不收窄、保持全库、绝不阻塞。
- 用户手选的 `book_titles`（前端选定）是硬约束，永远优先；scoper 仅在其为空（全库）时动作。
- 设计依据：`docs/superpowers/specs/2026-06-21-conversation-scoper-design.md`。

---

## File Structure

- **Create** `core/workflow/conversation_scoper.py` — `ConversationScoper` 单元 + `ScopeDecision`。唯一职责：决定本轮检索锁哪几本书（不消指代、不改 query 文本）。
- **Create** `tests/test_conversation_scoper.py` — scoper 单测。
- **Modify** `core/workflow/doc_workflow.py` — `__init__` 构造 scoper；`route` 接线；新增 `_scope_prefix` helper；各答案分支前缀输出 scope_note。
- **Modify** `core/workflow/front_door.py` — 新增 `disable_scope` 字段（纠偏）。
- **Modify** `tests/test_doc_workflow.py` — 装配/接线/前缀/纠偏测试。
- **Modify** `tests/test_front_door.py` — `disable_scope` 解析测试。

---

## Task 1: ConversationScoper 单元

**Files:**
- Create: `core/workflow/conversation_scoper.py`
- Test: `tests/test_conversation_scoper.py`

**Interfaces:**
- Consumes: `core.retrieval.retrieve.Retriever`（probe 数据源，`async retrieve(query, *, index_manager, book_titles, top_k)`）。
- Produces:
  - `ScopeDecision(effective_book_titles: Optional[list[str]], note: str = "")`
  - `ConversationScoper(index_manager, probe_retriever, probe_k=8, n_history_turns=2, dominant_share=0.60, dominant_ratio=2.0, cover_share=0.80, max_books=2, min_count=2)`
  - `async ConversationScoper.run(clean_query: str, user_book_titles: Optional[list[str]], memory) -> ScopeDecision`
  - `ConversationScoper._probe_text(clean_query, memory) -> str`（内部，测试可见）

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_conversation_scoper.py
"""ConversationScoper 单测：拼接 probe 文本 + 主导书判据 + 降级。"""
from llama_index.core.schema import NodeWithScore, TextNode

from core.workflow.conversation_scoper import ConversationScoper, ScopeDecision


class _Msg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeMemory:
    def __init__(self, msgs=None):
        self._msgs = list(msgs or [])

    def get(self):
        return self._msgs


class _FakeProbe:
    """按给定 book_title 序列造命中 node；记录最后一次 probe 文本。"""
    def __init__(self, books):
        self._books = books
        self.last_query = None

    async def retrieve(self, query, *, index_manager, book_titles, top_k):
        self.last_query = query
        return [
            NodeWithScore(node=TextNode(text="x", id_=str(i), metadata={"book_title": b}))
            for i, b in enumerate(self._books)
        ]


class _BoomProbe:
    async def retrieve(self, *a, **k):
        raise RuntimeError("probe down")


def test_probe_text_appends_recent_user_turns_only():
    sc = ConversationScoper(index_manager=None, probe_retriever=_FakeProbe([]), n_history_turns=2)
    mem = FakeMemory([
        _Msg("user", "讲讲openclaw"),
        _Msg("assistant", "……其中 gateway 是……"),
        _Msg("user", "讲一下A"),
    ])
    text = sc._probe_text("讲一下gateway", mem)
    assert "讲讲openclaw" in text                 # 历史用户问被拼进
    assert text.endswith("讲一下gateway")          # 本轮 clean_query 在末尾
    assert "其中 gateway 是" not in text           # assistant 内容被过滤


async def test_run_locks_single_dominant_book_and_uses_augmented_probe():
    probe = _FakeProbe(["openclaw"] * 6 + ["X"] * 2)
    sc = ConversationScoper(index_manager=None, probe_retriever=probe)
    d = await sc.run("讲一下gateway", None, FakeMemory([_Msg("user", "讲讲openclaw")]))
    assert d.effective_book_titles == ["openclaw"]
    assert "openclaw" in d.note
    assert "讲讲openclaw" in probe.last_query       # probe 文本带上了历史主体


async def test_run_locks_two_books_when_concept_spans():
    probe = _FakeProbe(["A"] * 4 + ["B"] * 3 + ["C"] * 1)
    sc = ConversationScoper(index_manager=None, probe_retriever=probe)
    d = await sc.run("q", None, FakeMemory())
    assert d.effective_book_titles == ["A", "B"]


async def test_run_no_narrow_when_diffuse():
    probe = _FakeProbe(["A"] * 3 + ["B"] * 3 + ["C"] * 2)
    sc = ConversationScoper(index_manager=None, probe_retriever=probe)
    d = await sc.run("q", None, FakeMemory())
    assert d.effective_book_titles is None
    assert d.note == ""


async def test_run_no_narrow_when_probe_empty():
    sc = ConversationScoper(index_manager=None, probe_retriever=_FakeProbe([]))
    d = await sc.run("q", None, FakeMemory())
    assert d.effective_book_titles is None


async def test_run_degrades_to_full_library_on_probe_error():
    sc = ConversationScoper(index_manager=None, probe_retriever=_BoomProbe())
    d = await sc.run("q", None, FakeMemory())
    assert d.effective_book_titles is None
    assert d.note == ""


async def test_run_noop_when_user_selected_books():
    probe = _FakeProbe(["A"] * 8)              # 若被咨询会收窄，但手选时不应 probe
    sc = ConversationScoper(index_manager=None, probe_retriever=probe)
    d = await sc.run("q", ["高性能MySQL"], FakeMemory())
    assert d.effective_book_titles == ["高性能MySQL"]
    assert d.note == ""
    assert probe.last_query is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_conversation_scoper.py -v`
Expected: FAIL（`ModuleNotFoundError: core.workflow.conversation_scoper`）

- [ ] **Step 3: Write the implementation**

```python
# core/workflow/conversation_scoper.py
"""会话作用域收窄：全库模式下从会话历史锚定主体，把检索硬锁到主导书。

注入式协作单元（仿 QueryGate/Admitter）：注入依赖、对外只暴露 run、失败降级、独立可测。
本单元【不消指代、不改 query 文本】——只决定「本轮检索锁哪几本书」。
设计见 docs/superpowers/specs/2026-06-21-conversation-scoper-design.md。
"""
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from llama_index.core.base.llms.types import MessageRole

from core.retrieval.retrieve import Retriever

logger = logging.getLogger(__name__)


@dataclass
class ScopeDecision:
    """scoper 产出：effective_book_titles=None 表示不收窄（保持全库）。"""

    effective_book_titles: Optional[list[str]]
    note: str = ""


def _book_of(node) -> str:
    """从 NodeWithScore/TextNode 取 book_title（缺失返回空串）。"""
    meta = getattr(node, "metadata", None) or {}
    return meta.get("book_title") or ""


class ConversationScoper:
    """全库多轮的隐式作用域收窄。仅当用户未手选书时动作。"""

    def __init__(
        self,
        index_manager,
        probe_retriever: Retriever,
        probe_k: int = 8,
        n_history_turns: int = 2,
        dominant_share: float = 0.60,
        dominant_ratio: float = 2.0,
        cover_share: float = 0.80,
        max_books: int = 2,
        min_count: int = 2,
    ):
        self.index_manager = index_manager
        self.probe_retriever = probe_retriever
        self.probe_k = probe_k
        self.n_history_turns = n_history_turns
        self.dominant_share = dominant_share
        self.dominant_ratio = dominant_ratio
        self.cover_share = cover_share
        self.max_books = max_books
        self.min_count = min_count

    def _probe_text(self, clean_query: str, memory) -> str:
        """拼接：最近 n 轮【用户】问 + 本轮 clean_query。

        memory 此刻【不含本轮】（本轮在 finalize 才入记忆），故取到的是历史用户问。
        裸概念续问靠这步把上文主体带进 probe；自包含强 query 则自己立得住。
        """
        turns: list[str] = []
        if memory is not None:
            msgs = memory.get() or []
            users = [
                str(m.content)
                for m in msgs
                if m.role == MessageRole.USER and m.content
            ]
            turns = users[-self.n_history_turns:]
        return "\n".join(turns + [clean_query])

    def _decide(self, titles: list[str]) -> Optional[list[str]]:
        """命中书名列表 → 主导书集合 or None（不收窄）。"""
        titles = [t for t in titles if t]
        if not titles:
            return None
        counts = Counter(titles).most_common()      # [(book, n), ...] 降序
        total = sum(n for _b, n in counts)
        top_book, top_n = counts[0]
        second_n = counts[1][1] if len(counts) > 1 else 0
        # 单一主导：占比过线、≥第二名 dominant_ratio 倍、且自身命中足够（防单点噪声）
        if (
            top_n / total >= self.dominant_share
            and top_n >= self.dominant_ratio * second_n
            and top_n >= self.min_count
        ):
            return [top_book]
        # 少数主导：最小前缀累计 ≥ cover_share，前缀 ≤ max_books，
        # 前缀内每本 ≥ min_count，尾部每本 < min_count（确属长尾噪声）
        prefix: list[str] = []
        acc = 0
        for book, n in counts:
            if n < self.min_count:
                break
            prefix.append(book)
            acc += n
            if len(prefix) > self.max_books:
                return None
            if acc / total >= self.cover_share:
                tail = counts[len(prefix):]
                if all(tn < self.min_count for _tb, tn in tail):
                    return prefix
                return None
        return None

    async def run(
        self, clean_query: str, user_book_titles: Optional[list[str]], memory
    ) -> ScopeDecision:
        # 手选硬约束永远赢：非空直接 no-op，根本不 probe
        if user_book_titles:
            return ScopeDecision(user_book_titles, "")
        try:
            probe_text = self._probe_text(clean_query, memory)
            nodes = await self.probe_retriever.retrieve(
                probe_text,
                index_manager=self.index_manager,
                book_titles=None,
                top_k=self.probe_k,
            )
            books = self._decide([_book_of(n) for n in nodes])
        except Exception as exc:
            logger.warning("scoper probe/判定失败，保持全库：%s", exc)
            return ScopeDecision(None, "")
        if not books:
            return ScopeDecision(None, "")
        label = "》《".join(books)
        note = f"（我按《{label}》里的内容回答；想看全部书可以说\"在所有书里讲\"。）\n"
        logger.info("scoper: 收窄到 %s", books)
        return ScopeDecision(books, note)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_conversation_scoper.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/conversation_scoper.py tests/test_conversation_scoper.py
git commit -m "feat(workflow): ConversationScoper 单元（全库多轮作用域收窄）"
```

---

## Task 2: 接线进 DocQueryWorkflow

**Files:**
- Modify: `core/workflow/doc_workflow.py`（`__init__` 构造；`route` 的 dispatch_qa 分支）
- Test: `tests/test_doc_workflow.py`

**Interfaces:**
- Consumes: Task 1 的 `ConversationScoper`、`ScopeDecision`。
- Produces: ctx 键 `book_titles`（收窄后）、`scope_note`（透明声明文本，可空）。

- [ ] **Step 1: Write the failing tests**（追加到 `tests/test_doc_workflow.py` 末尾）

```python
# ── ConversationScoper 接线（Task 2）──────────────────────────────────
def test_scoper_constructed_with_probe_vector_retriever():
    from core.workflow.conversation_scoper import ConversationScoper
    from core.retrieval.retrieve import VectorRetriever
    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert isinstance(wf.scoper, ConversationScoper)
    assert isinstance(wf.scoper.probe_retriever, VectorRetriever)


async def test_scoper_narrows_book_titles_in_full_library():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    wf = _wf(llm)

    async def fake_scope(clean_query, user_book_titles, memory):
        from core.workflow.conversation_scoper import ScopeDecision
        return ScopeDecision(["openclaw"], "（我按《openclaw》回答…）\n")
    wf.scoper.run = fake_scope

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["book_titles"] = book_titles
        return "答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        captured["classify_books"] = book_titles
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    await wf.run(query="讲一下gateway", memory=FakeMemory([_Msg("user", "讲讲openclaw")]))
    assert captured["book_titles"] == ["openclaw"]      # 收窄透传到检索
    assert captured["classify_books"] == ["openclaw"]   # 也透传到 classify probe


async def test_scoper_called_with_user_books_and_result_flows():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    wf = _wf(llm)

    seen = {}

    async def fake_scope(clean_query, user_book_titles, memory):
        from core.workflow.conversation_scoper import ScopeDecision
        seen["args"] = (clean_query, user_book_titles)
        return ScopeDecision(user_book_titles, "")       # 模拟手选 no-op
    wf.scoper.run = fake_scope

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["book_titles"] = book_titles
        return "答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    await wf.run(query="讲一下gateway", memory=FakeMemory(), book_titles=["高性能MySQL"])
    assert seen["args"] == ("讲一下gateway", ["高性能MySQL"])
    assert captured["book_titles"] == ["高性能MySQL"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_doc_workflow.py -k scoper -v`
Expected: FAIL（`AttributeError: 'DocQueryWorkflow' object has no attribute 'scoper'`）

- [ ] **Step 3: Implement — 构造 + 接线**

在 `core/workflow/doc_workflow.py` 顶部 import 区加：

```python
from core.workflow.conversation_scoper import ConversationScoper
```

`__init__` 中 `self.qa_agent = QaAgent(...)` 之后加：

```python
        # 全库多轮作用域收窄：probe 复用 workflow 的 probe_retriever 名字（None→vector）
        self.scoper = ConversationScoper(
            index_manager, probe_retriever=make_retriever(probe_retriever)
        )
```

`route` 的 dispatch_qa 分支，把：

```python
        # dispatch_qa（含降级）
        await ctx.store.set("clean_query", decision.clean_query)
        return PreprocessEvent()
```

改成：

```python
        # dispatch_qa（含降级）—— memory/book_titles 在 route 顶部已取
        await ctx.store.set("clean_query", decision.clean_query)
        scope = await self.scoper.run(decision.clean_query, book_titles, memory)
        await ctx.store.set("book_titles", scope.effective_book_titles or book_titles)
        await ctx.store.set("scope_note", scope.note)
        return PreprocessEvent()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_doc_workflow.py -v`
Expected: PASS（含新 3 个 + 原有全部回归通过）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): route 接入 ConversationScoper（全库收窄写回 book_titles/scope_note）"
```

---

## Task 3: 透明声明（scope_note 当答案前缀）

**Files:**
- Modify: `core/workflow/doc_workflow.py`（新增 `_scope_prefix` helper；retrieve/other/split/assume/explain 成功路径前缀输出）
- Test: `tests/test_doc_workflow.py`

> **机制说明（对 spec §4.3 的实现细化）**：spec 提"复用 preamble 把 scope_note 输出"。实现上不逐个穿透 qa_capability 的 split/explain/assume 内部多条降级路径（易漏），改为在 doc_workflow 各答案分支统一：调用前先推一个 `AnswerDeltaEvent(scope_note)`、再把 scope_note 拼到最终答案前缀。用户可见行为一致（声明流式在最前），但改动集中、不碰 qa_capability。

**Interfaces:**
- Consumes: ctx 键 `scope_note`（Task 2 产）。
- Produces: `DocQueryWorkflow._scope_prefix(ctx) -> str`（推 delta 并返回前缀；空则无副作用）。

- [ ] **Step 1: Write the failing tests**（追加到 `tests/test_doc_workflow.py`）

```python
# ── 透明声明前缀（Task 3）─────────────────────────────────────────────
async def test_scope_note_prepended_to_answer():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    wf = _wf(llm)

    async def fake_scope(clean_query, user_book_titles, memory):
        from core.workflow.conversation_scoper import ScopeDecision
        return ScopeDecision(["openclaw"], "（我按《openclaw》回答…）\n")
    wf.scoper.run = fake_scope

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "正文答案", ["n1"]

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    result = await wf.run(query="讲一下gateway", memory=FakeMemory([_Msg("user", "讲讲openclaw")]))
    resp = str(result.response)
    assert resp.startswith("（我按《openclaw》回答")     # 声明在最前
    assert "正文答案" in resp


async def test_no_scope_note_when_not_narrowed():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    wf = _wf(llm)

    async def fake_scope(clean_query, user_book_titles, memory):
        from core.workflow.conversation_scoper import ScopeDecision
        return ScopeDecision(None, "")                    # 不收窄
    wf.scoper.run = fake_scope

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "正文答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    result = await wf.run(query="讲一下gateway", memory=FakeMemory())
    assert str(result.response) == "正文答案"             # 无前缀
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_doc_workflow.py -k scope_note -v`
Expected: FAIL（`test_scope_note_prepended_to_answer`：response 不含前缀）

- [ ] **Step 3: Implement — helper + 各答案分支前缀**

`AnswerDeltaEvent` 已在 doc_workflow import（顶部 from qa_capability）。在 `route` step 之后、`study_plan_branch` 之前加 helper：

```python
    async def _scope_prefix(self, ctx: Context) -> str:
        """全库收窄的透明声明：流式先推一个 delta，并返回前缀供拼进最终答案。空则无副作用。"""
        note = await ctx.store.get("scope_note", "")
        if note:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=note))
        return note
```

在以下 5 个答案分支，于调用 `self.qa.*`/`self.qa_agent.*` 之前取前缀、返回时拼接：

`retrieve_branch`：

```python
    @step
    async def retrieve_branch(self, ctx: Context, ev: RetrieveAgentEvent) -> FinalizeEvent:
        book_titles = await ctx.store.get("book_titles")
        prefix = await self._scope_prefix(ctx)
        answer, nodes = await self.qa.retrieve(
            ctx, ev.rewritten_query, book_titles, ev.assumption_note
        )
        return FinalizeEvent(answer=prefix + answer, source_nodes=nodes)
```

`other_branch`：在 `book_titles = ...` 后加 `prefix = await self._scope_prefix(ctx)`；两个 `return FinalizeEvent(answer=answer, ...)` 都改成 `answer=prefix + answer`。

`split_branch`：同理加 `prefix`，`return` 改 `answer=prefix + answer`。

`assume_branch`：同理加 `prefix`，`return` 改 `answer=prefix + answer`。

`explain_branch`：在 `book_titles = ...` 后加 `prefix = await self._scope_prefix(ctx)`。**仅成功路径**拼前缀——`try` 体的 `return FinalizeEvent(answer=answer, ...)`、`except EmptySkeleton` 后 agent/单轮兜底的 `return` 改成 `answer=prefix + answer`；`except OutOfScope`（REFUSAL_TEXT）、`except MissingInfo`（反问）**不拼前缀**（拒答/反问不是按某书的作答）。

> 注意：`_scope_prefix` 在 explain_branch 顶部调用会先推 delta；若随后落 OutOfScope/MissingInfo，则该 delta 已推但答案不拼前缀——为避免拒答前先冒出"我按《X》回答"，把 `prefix = await self._scope_prefix(ctx)` 移到**确认走成功路径后**再调。具体：explain_branch 不在顶部调用 helper，而在 `try` 内 `explain` 返回成功后、及 `except EmptySkeleton` 兜底成功后，分别 `prefix = await self._scope_prefix(ctx)` 再拼。OutOfScope/MissingInfo 分支不调用 helper。

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_doc_workflow.py -v`
Expected: PASS（含新 2 个 + 原有回归，尤其 explain 的 OutOfScope/MissingInfo 用例仍等于纯 REFUSAL_TEXT/反问句，无前缀）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): scope_note 透明声明作为答案前缀流式输出"
```

---

## Task 4: 纠偏 disable_scope（门口识别"在所有书里"）

**Files:**
- Modify: `core/workflow/front_door.py`（prompt + schema + decision 字段 + run 透传）
- Modify: `core/workflow/doc_workflow.py`（route 按 `decision.disable_scope` 跳过 scoper）
- Test: `tests/test_front_door.py`、`tests/test_doc_workflow.py`

**Interfaces:**
- Produces: `FrontDoorDecision.disable_scope: bool`（默认 False，仅 dispatch_qa 有意义）。

- [ ] **Step 1: Write the failing tests**

`tests/test_front_door.py`（追加；沿用该文件已有的 FakeLLM 与 `FrontDoorAgent` 构造方式）：

```python
async def test_front_door_sets_disable_scope_on_all_books_request():
    from core.workflow.front_door import FrontDoorAgent
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway","disable_scope":true}'])
    d = await FrontDoorAgent(llm).run("在所有书里讲一下gateway")
    assert d.action == "dispatch_qa"
    assert d.disable_scope is True


async def test_front_door_disable_scope_defaults_false():
    from core.workflow.front_door import FrontDoorAgent
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway"}'])
    d = await FrontDoorAgent(llm).run("讲一下gateway")
    assert d.disable_scope is False
```

`tests/test_doc_workflow.py`（追加）：

```python
async def test_disable_scope_skips_scoper():
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"讲一下gateway","disable_scope":true}'])
    wf = _wf(llm)

    async def boom_scope(*a, **k):
        raise AssertionError("disable_scope=true 时不应调用 scoper")
    wf.scoper.run = boom_scope

    captured = {}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        captured["book_titles"] = book_titles
        return "答案", []

    async def fake_classify(clean_query, book_titles=None, probe=True):
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable")

    wf.qa.retrieve = fake_retrieve
    wf.qa.classify = fake_classify

    await wf.run(query="在所有书里讲一下gateway", memory=FakeMemory())
    assert captured["book_titles"] is None        # 跳过收窄，保持全库
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_front_door.py -k disable_scope tests/test_doc_workflow.py -k disable_scope -v`
Expected: FAIL（`FrontDoorDecision` 无 `disable_scope` / scoper 仍被调用）

- [ ] **Step 3: Implement**

`core/workflow/front_door.py`：

`_FRONT_DOOR_PROMPT` 的 dispatch_qa 铁律后补一句：

```
  纠偏：若用户明确要求【在所有书/全部书里】或【不要限定范围】回答，置 disable_scope=true（仅 dispatch_qa 有意义；默认 false，不要随意置 true）。
```

并把 JSON 模板那一行的尾部加 `disable_scope` 字段：

```
{"action":"dispatch_qa / dispatch_study_plan / converse / clarify","clean_query":"...","reply":"...","reason":"...","tool":"list_books 或空串（仅 converse 元查询时填 list_books）","tool_filter":"...","tool_count_only":false,"disable_scope":false}
```

`FrontDoorDecision` dataclass 加字段：

```python
    disable_scope: bool = False
```

`FrontDoorDecisionModel` 加字段：

```python
    disable_scope: bool = Field(default=False, description="用户要求全库/不限定时 true（仅 dispatch_qa 有意义）")
```

`run` 中 dispatch_qa/dispatch_study_plan 的返回，带上 `disable_scope`：

```python
            if d.action in ("dispatch_qa", "dispatch_study_plan"):
                clean = (d.clean_query or original).strip() or original
                logger.info(
                    "front_door: action=%s clean_query=%r", d.action, clean[:80]
                )
                return FrontDoorDecision(
                    d.action, clean_query=clean, reason=d.reason,
                    disable_scope=d.disable_scope,
                )
```

`core/workflow/doc_workflow.py` route 的 dispatch_qa 分支（Task 2 写的那段）改为按 `disable_scope` 跳过：

```python
        # dispatch_qa（含降级）—— memory/book_titles 在 route 顶部已取
        await ctx.store.set("clean_query", decision.clean_query)
        if not decision.disable_scope:
            scope = await self.scoper.run(decision.clean_query, book_titles, memory)
            await ctx.store.set("book_titles", scope.effective_book_titles or book_titles)
            await ctx.store.set("scope_note", scope.note)
        else:
            await ctx.store.set("scope_note", "")
        return PreprocessEvent()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_front_door.py tests/test_doc_workflow.py -v`
Expected: PASS（含新用例 + 全部回归）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/front_door.py core/workflow/doc_workflow.py tests/test_front_door.py tests/test_doc_workflow.py
git commit -m "feat(workflow): disable_scope 纠偏——用户要求全库时跳过 ConversationScoper 收窄"
```

---

## End-to-End Verification

- [ ] 全量回归：`python -m pytest tests/ -q` → 全绿。
- [ ] 分层守卫：`python scripts/check_layering.py` → 通过（scoper 在 core，无 api 依赖）。
- [ ] 真实链路冒烟（需 `.env` 的 DEEPSEEK_API_KEY + 已入库 ≥2 本含同名概念的书）：
  - 启动 `python -m uvicorn api.main:app --port 8000`；前端**不选书**（全库）。
  - 第一轮问「讲讲 openclaw」，第二轮问「讲一下 gateway」。
  - 看后端日志应出现 `scoper: 收窄到 ['openclaw']`；答案开头有「（我按《openclaw》里的内容回答…）」；内容来自 openclaw，不串别的书。
  - 第三轮问「在所有书里讲一下 gateway」→ 日志 `front_door` 决策带 `disable_scope`，无收窄、无前缀声明。

---

## Self-Review

**Spec coverage：**
- §3/§4.1 ConversationScoper 单元 + 拼接 probe + 主导书判据 → Task 1。
- §4.2 route 接线（写回 book_titles/scope_note，手选 no-op）→ Task 2。
- §4.3 透明声明 → Task 3（机制细化为分支级前缀，行为等价，已注明）。
- §4.4 disable_scope 纠偏 → Task 4。
- §6 错误处理（probe 失败/空/分散/手选/空 memory）→ Task 1 单测覆盖；降级保持全库。
- §7 测试（单测 + 装配测，explain 与非 explain 两路继承 book_titles）→ Task 1/2/3/4。
- §8 非目标（probe 复用、阈值评测、sticky 持久化）→ 未实现，符合预期。

**Type consistency：** `ScopeDecision(effective_book_titles, note)`、`ConversationScoper.run(clean_query, user_book_titles, memory)`、`FrontDoorDecision.disable_scope`、ctx 键 `book_titles`/`scope_note` 在各 Task 间一致。

**Placeholder scan：** 无 TODO/占位；每个改动步骤含完整代码与命令。
