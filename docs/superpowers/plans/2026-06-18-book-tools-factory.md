# Book Tools 工厂解耦 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `QaAgent` 内联的 `book_search`/`list_books` 工具抽到 `core/agent/tools/`，用 ToolContext + 注册表 + 工厂组装，供任意 agent 复用，运行行为不变。

**Architecture:** 新增 `core/agent/tools/book_tools.py`：一个 `ToolContext` dataclass 收口共享依赖（index_manager / similarity_top_k）与 per-run 状态（scope / sources）；两个独立工具类 `BookSearchTool`/`ListBooksTool`，各自 `__call__` 执行 + `to_function_tool()` 包装；`@register_tool` 装饰器入 `_TOOL_REGISTRY`，`build_book_tools(ctx)` 工厂按名实例化并组装为 FunctionTool 列表。`QaAgent` 改持 `self.ctx` 并委托工厂。

**Tech Stack:** Python 3.12, LlamaIndex（FunctionTool/FunctionAgent）, pytest（asyncio_mode=auto）。

## Global Constraints

- 只搬不改：`book_search` 的 `[:500]` 截断、空 query 提示「请提供要检索的问题。」、空库提示「知识库为空，请先上传 PDF。」、无命中「（未检索到相关内容）」、`sources.extend` 不去重；`list_books` 不带 scope filter、空库「知识库当前为空。」——全部逐字保留。
- 分层：`core/agent/tools` 只能依赖 `core/retrieval`（`build_book_filters`）等 core 内模块，**禁止依赖 `api/`**。守卫 `python scripts/check_layering.py` 必须通过。
- 所有 I/O 用 `async/await`；函数签名加类型注解；中文注释可接受。
- 从项目根目录运行；子模块内用相对导入，根脚本用绝对导入。
- `QaAgent.run()` 对外契约不变：签名 `run(ctx, query, book_titles)`、返回 `(answer: str, sources: list)`。

---

### Task 1: 新建 `core/agent/tools` 工具模块（ToolContext + 两工具 + 注册表 + 工厂）

**Files:**
- Create: `core/agent/tools/__init__.py`
- Create: `core/agent/tools/book_tools.py`
- Test: `tests/test_book_tools.py`

**Interfaces:**
- Produces:
  - `ToolContext(index_manager, similarity_top_k=5, scope=None, sources=<list>)` — dataclass。
  - `BookSearchTool(ctx)` 有 `name="book_search"`、`async __call__(query: str) -> str`、`to_function_tool() -> FunctionTool`。
  - `ListBooksTool(ctx)` 有 `name="list_books"`、`__call__() -> str`、`to_function_tool() -> FunctionTool`。
  - `register_tool(cls) -> cls` 装饰器；`_TOOL_REGISTRY: dict[str, type]`。
  - `build_book_tools(ctx, names=None) -> list[FunctionTool]` 工厂。

- [ ] **Step 1: 写失败测试 `tests/test_book_tools.py`**

```python
"""book_tools 单测：两个工具类的方法 + 注册表工厂 + ctx 状态收集。"""
import pytest

from core.agent.tools.book_tools import (
    BookSearchTool,
    ListBooksTool,
    ToolContext,
    build_book_tools,
)


class FakeRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    async def aretrieve(self, query):
        return self._nodes


class FakeIndex:
    def __init__(self, nodes):
        self._nodes = nodes
        self.last_kw = None

    def as_retriever(self, **kw):
        self.last_kw = kw
        return FakeRetriever(self._nodes)


class _FakeCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class FakeIndexManager:
    def __init__(self, nodes, metas=None):
        self._index = FakeIndex(nodes)
        self.chroma_collection = _FakeCollection(metas or [])

    def get_index(self):
        return self._index


class _Node:
    def __init__(self, content):
        self._c = content

    def get_content(self):
        return self._c


def _ctx(nodes=(), metas=None, top_k=3):
    im = FakeIndexManager(nodes=list(nodes), metas=metas)
    return ToolContext(index_manager=im, similarity_top_k=top_k)


async def test_book_search_joins_passages_and_collects_sources():
    ctx = _ctx(nodes=[_Node("片段A"), _Node("片段B")])
    out = await BookSearchTool(ctx)("分布式事务")
    assert "片段A" in out and "片段B" in out
    assert len(ctx.sources) == 2


async def test_book_search_empty_returns_placeholder_no_collect():
    ctx = _ctx(nodes=[])
    out = await BookSearchTool(ctx)("不存在")
    assert out == "（未检索到相关内容）"
    assert ctx.sources == []


async def test_book_search_blank_query_prompts():
    ctx = _ctx(nodes=[_Node("x")])
    assert await BookSearchTool(ctx)("   ") == "请提供要检索的问题。"


async def test_book_search_passes_scope_and_top_k_to_retriever():
    ctx = _ctx(nodes=[_Node("x")], top_k=3)
    ctx.scope = ["书A"]
    await BookSearchTool(ctx)("q")
    kw = ctx.index_manager.get_index().last_kw
    assert kw["similarity_top_k"] == 3
    assert kw["filters"] is not None  # build_book_filters(["书A"]) 非空


def test_list_books_counts_titles():
    ctx = _ctx(metas=[{"book_title": "甲"}, {"book_title": "甲"}, {"book_title": "乙"}])
    out = ListBooksTool(ctx)()
    assert "《甲》（2 块）" in out
    assert "《乙》（1 块）" in out


def test_list_books_empty():
    ctx = _ctx(metas=[])
    assert ListBooksTool(ctx)() == "知识库当前为空。"


def test_build_book_tools_default_returns_both():
    tools = build_book_tools(_ctx())
    names = sorted(t.metadata.name for t in tools)
    assert names == ["book_search", "list_books"]


def test_build_book_tools_unknown_name_raises():
    with pytest.raises(ValueError):
        build_book_tools(_ctx(), names=["nope"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_book_tools.py -q`
Expected: FAIL（`ModuleNotFoundError: core.agent.tools.book_tools`）

- [ ] **Step 3: 写 `core/agent/tools/book_tools.py`**

```python
"""书籍知识库检索工具集：工厂 + 注册表组装，供任意 agent 复用。

设计同 core/retrieval/retrieve.py 的注册表风格：每个工具一个类（自带 name/
description + 执行方法），@register_tool 入表，build_book_tools 工厂按名实例化并
包成 LlamaIndex FunctionTool。共享依赖与 per-run 状态收口到 ToolContext。
"""
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core.tools import FunctionTool

from core.retrieval.retrieve import build_book_filters


@dataclass
class ToolContext:
    """工具共享依赖 + 可重置的 per-run 状态。

    所有工具只接此一个 ctx 构造，故注册表能统一实例化。scope/sources 由 agent 在
    每次 run 前设置/重置：scope 是本轮检索范围（None=全库），sources 收集本轮命中。
    """
    index_manager: object
    similarity_top_k: int = 5
    scope: Optional[list[str]] = None
    sources: list = field(default_factory=list)


_TOOL_REGISTRY: dict[str, type] = {}  # name → 工具类


def register_tool(cls):
    """装饰器：按 cls.name 登记工具类。新增工具加一行 @register_tool 即可。"""
    _TOOL_REGISTRY[cls.name] = cls
    return cls


@register_tool
class BookSearchTool:
    """书籍知识库检索：按 query 取 top-k 原文片段并把命中 nodes 收进 ctx.sources。"""

    name = "book_search"
    description = "书籍知识库检索：按 query 返回相关原文片段，范围由用户选定。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    async def __call__(self, query: str) -> str:
        if not isinstance(query, str):
            query = str(query)
        query = query.strip()
        if not query:
            return "请提供要检索的问题。"
        index = self.ctx.index_manager.get_index()
        if index is None:
            return "知识库为空，请先上传 PDF。"
        retriever = index.as_retriever(
            similarity_top_k=self.ctx.similarity_top_k,
            filters=build_book_filters(self.ctx.scope),
        )
        nodes = await retriever.aretrieve(query)
        if not nodes:
            return "（未检索到相关内容）"
        self.ctx.sources.extend(nodes)
        return "\n---\n".join(
            (n.get_content() if hasattr(n, "get_content") else getattr(n, "text", ""))[:500]
            for n in nodes
        )

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )


@register_tool
class ListBooksTool:
    """列出当前已入库书籍清单（按 book_title 计数）。"""

    name = "list_books"
    description = "列出当前已入库书籍清单。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def __call__(self) -> str:
        data = self.ctx.index_manager.chroma_collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in data.get("metadatas", []) or []:
            title = (meta or {}).get("book_title")
            if not title:
                continue
            counts[title] = counts.get(title, 0) + 1
        if not counts:
            return "知识库当前为空。"
        return "已入库书籍：\n" + "\n".join(
            f"- 《{t}》（{c} 块）" for t, c in sorted(counts.items())
        )

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )


def build_book_tools(ctx: ToolContext, names: Optional[list[str]] = None) -> list:
    """工厂：按名从注册表实例化工具并包成 FunctionTool 列表。

    names=None → 注册表全部（登记顺序：book_search, list_books）。未知名 → ValueError。
    """
    names = names or list(_TOOL_REGISTRY)
    tools = []
    for n in names:
        if n not in _TOOL_REGISTRY:
            raise ValueError(f"未知工具名字：{n!r}，可选：{list(_TOOL_REGISTRY)}")
        tools.append(_TOOL_REGISTRY[n](ctx).to_function_tool())
    return tools
```

- [ ] **Step 4: 写 `core/agent/tools/__init__.py`**

```python
"""agent 可复用工具包：检索工具的工厂 + 注册表。"""
from core.agent.tools.book_tools import (
    BookSearchTool,
    ListBooksTool,
    ToolContext,
    build_book_tools,
    register_tool,
)

__all__ = [
    "ToolContext",
    "BookSearchTool",
    "ListBooksTool",
    "build_book_tools",
    "register_tool",
]
```

- [ ] **Step 5: 跑测试确认通过 + 分层守卫**

Run: `python -m pytest tests/test_book_tools.py -q && python scripts/check_layering.py`
Expected: 8 passed；分层检查通过（无输出或 OK）。

- [ ] **Step 6: 提交**

```bash
git add core/agent/tools/__init__.py core/agent/tools/book_tools.py tests/test_book_tools.py
git commit -m "feat(tools): book_search/list_books 抽为 ToolContext+工厂+注册表

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `QaAgent` 改用工厂 + 迁移既有测试 + 更新 CLAUDE.md

**Files:**
- Modify: `core/agent/qa_agent.py`
- Modify: `tests/test_qa_agent.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `ToolContext`, `build_book_tools`（Task 1）。
- Produces: `QaAgent.ctx: ToolContext`（替代旧 `_run_scope`/`_run_sources`）；`run()` 契约不变。

- [ ] **Step 1: 改 `tests/test_qa_agent.py`——删 `_search` 用例、`run` 用例改读 `qa.ctx.sources`**

删除 `test_search_returns_joined_passages_and_collects_nodes` 与
`test_search_empty_returns_placeholder_and_collects_nothing` 两个函数（已迁至
`tests/test_book_tools.py`）。

把 `test_run_resets_sources_each_call_and_passes_max_iterations` 中
```python
    qa._run_sources = ["stale"]
```
改为
```python
    qa.ctx.sources = ["stale"]
```

其余用例（`test_run_bridges_tool_events_and_emits_final_delta`、Fake* 辅助类）保持不变。

- [ ] **Step 2: 跑改后的 qa 测试确认失败**

Run: `python -m pytest tests/test_qa_agent.py -q`
Expected: FAIL（`AttributeError: 'QaAgent' object has no attribute 'ctx'` 等），证明测试已对准新接口。

- [ ] **Step 3: 改 `core/agent/qa_agent.py`**

把顶部 import 中的
```python
from core.retrieval.retrieve import build_book_filters
```
替换为
```python
from core.agent.tools.book_tools import ToolContext, build_book_tools
```

`__init__` 方法体替换为（保留 docstring 之外的赋值逻辑）：
```python
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_iterations: int = 6,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.ctx = ToolContext(
            index_manager=index_manager, similarity_top_k=similarity_top_k
        )
        # 懒构造：FunctionAgent 需合法 LLM 且较重，只在真走 other 分支时才建。
        # 这样 DocQueryWorkflow 每请求构造（含单测替身 LLM）不被 FunctionAgent 校验拖累，
        # 多数不走 other 的请求也省下构造开销。
        self.agent = None
```

删除整个 `_search` 方法和整个 `_make_tools` 方法。

`_ensure_agent` 中
```python
                tools=self._make_tools(),
```
改为
```python
                tools=build_book_tools(self.ctx),
```

`run` 方法体中：
```python
        self._run_scope = book_titles
        self._run_sources = []
```
改为
```python
        self.ctx.scope = book_titles
        self.ctx.sources = []
```
`RetrievalDoneEvent(count=len(self._run_sources))` 改为
`RetrievalDoneEvent(count=len(self.ctx.sources))`；
`logger.info("qa_agent 完成: %d sources", len(self._run_sources))` 改为
`len(self.ctx.sources)`；
`return answer, list(self._run_sources)` 改为 `return answer, list(self.ctx.sources)`。

- [ ] **Step 4: 跑全量测试确认通过**

Run: `python -m pytest tests/test_qa_agent.py tests/test_book_tools.py -q`
Expected: 全 passed。

- [ ] **Step 5: 更新 `CLAUDE.md`「工具在组装层创建，注入 Agent」段**

把该段中过时的
```
`core.tools.book_tools.create_book_search_tool / create_list_books_tool`
```
改为描述新事实：检索工具现位于 `core/agent/tools/`，用
`core.agent.tools.build_book_tools(ToolContext(index_manager))` 经注册表 + 工厂组装，
`QaAgent` 已接入；其它 agent（如 `BookAgent`）可同样注入复用。

- [ ] **Step 6: 全量回归 + 分层守卫**

Run: `python -m pytest tests/ -q && python scripts/check_layering.py`
Expected: 全 passed；分层检查通过。

- [ ] **Step 7: 提交**

```bash
git add core/agent/qa_agent.py tests/test_qa_agent.py CLAUDE.md
git commit -m "refactor(agent): QaAgent 改用 book_tools 工厂，状态收口 ToolContext

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage**：ToolContext/两工具类/注册表/工厂（Task 1）；QaAgent 改造（Task 2 Step 3）；测试新增与迁移（Task 1 Step 1、Task 2 Step 1）；CLAUDE.md 更新（Task 2 Step 5）；分层守卫（两 Task 末步）；非目标（旧瑕疵不修、BookAgent 不接线）由 Global Constraints「只搬不改」+ 计划不含装配层改动保证。全覆盖。
- **Placeholder scan**：无 TODO/TBD，所有代码步给出完整代码。
- **Type consistency**：`ToolContext(index_manager, similarity_top_k, scope, sources)`、`build_book_tools(ctx, names=None)`、`BookSearchTool/ListBooksTool(ctx).__call__/.to_function_tool()`、`QaAgent.ctx` 跨 Task 命名一致；`run()` 返回 `(answer, list(ctx.sources))` 与既有 `test_run_*` 断言一致。
