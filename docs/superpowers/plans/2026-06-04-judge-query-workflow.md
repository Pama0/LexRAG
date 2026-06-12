# judge_query Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `book_search` 的内联检索改造为一个多步 LlamaIndex Workflow,在检索前加 `judge_query` 步骤——宽泛 query 自动改写最多 2 轮直到明确;并在 Agent 层 system_prompt 补强指代改写。

**Architecture:** 新建 `core/workflow/book_rag.py` 定义 `BookRagWorkflow`(start → judge → retrieve → synthesize,judge 自循环最多 2 轮)。`core/tools/book_tools.py` 的 `book_search` 保持工具名/签名不变,内部委托该 workflow。`core/agent/agent.py` 的 `BOOK_SYSTEM_PROMPT` 追加指代改写规则。装配层(`api/main.py`、`main.py`)不动。

**Tech Stack:** Python 3.12,LlamaIndex Workflow(`llama-index-workflows`),pytest + pytest-asyncio。

参考 spec:`docs/superpowers/specs/2026-06-04-judge-query-workflow-design.md`

---

## File Structure

- **Create** `core/workflow/book_rag.py` — `BookRagWorkflow` 与事件定义、judge/retrieve 逻辑。
- **Create** `tests/test_book_rag_workflow.py` — workflow 单元测试。
- **Create** `tests/test_book_search_tool.py` — 工具委托 workflow 的接线测试。
- **Create** `tests/test_book_system_prompt.py` — system_prompt 含指代规则的断言。
- **Create** `pytest.ini` — pytest-asyncio 配置。
- **Modify** `requirements.txt` — 加 pytest、pytest-asyncio。
- **Modify** `core/tools/book_tools.py` — `book_search` 内部改为委托 workflow。
- **Modify** `core/agent/agent.py` — `BOOK_SYSTEM_PROMPT` 追加指代规则。

---

### Task 1: 测试基建

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`

- [ ] **Step 1: requirements.txt 追加测试依赖**

在 `requirements.txt` 末尾追加两行:

```
pytest
pytest-asyncio
```

- [ ] **Step 2: 创建 pytest.ini**

`pytest.ini` 内容:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 3: 安装依赖**

Run: `.venv\Scripts\python.exe -m pip install pytest pytest-asyncio`
Expected: 安装成功,`pytest --version` 可用。

- [ ] **Step 4: 验证 pytest 能跑空集**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: `no tests ran`(或 collected 0 items),无报错。

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini
git commit -m "chore: add pytest + pytest-asyncio test infra"
```

---

### Task 2: 事件定义 + judge 解析逻辑(`_judge_query`)

**Files:**
- Create: `core/workflow/book_rag.py`
- Test: `tests/test_book_rag_workflow.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_book_rag_workflow.py`:

```python
import pytest

from core.workflow.book_rag import BookRagWorkflow


class _Resp:
    """模拟 LLM 返回对象,str(resp) 即文本。"""
    def __init__(self, text: str):
        self._t = text
    def __str__(self) -> str:
        return self._t


class FakeLLM:
    """按队列依次返回预设文本;acomplete 是 workflow judge 唯一用到的方法。"""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
    async def acomplete(self, prompt, **kw):
        self.calls += 1
        return _Resp(self._responses.pop(0))


def _make_wf(llm, index_manager=None):
    return BookRagWorkflow(index_manager=index_manager, llm=llm, similarity_top_k=3)


async def test_judge_query_clear_returns_original():
    llm = FakeLLM(['{"clear": true, "rewritten_query": "B+树的索引结构"}'])
    wf = _make_wf(llm)
    clear, q = await wf._judge_query("B+树的索引结构")
    assert clear is True
    assert q == "B+树的索引结构"


async def test_judge_query_unclear_returns_rewrite():
    llm = FakeLLM(['{"clear": false, "rewritten_query": "数据库索引的实现原理"}'])
    wf = _make_wf(llm)
    clear, q = await wf._judge_query("讲讲数据库")
    assert clear is False
    assert q == "数据库索引的实现原理"


async def test_judge_query_malformed_falls_back_to_clear():
    llm = FakeLLM(["这不是JSON"])
    wf = _make_wf(llm)
    clear, q = await wf._judge_query("讲讲数据库")
    assert clear is True          # 解析失败 → 当作明确,不阻塞
    assert q == "讲讲数据库"       # 用原 query
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_rag_workflow.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'core.workflow.book_rag'`

- [ ] **Step 3: 写最小实现**

创建 `core/workflow/book_rag.py`:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_rag_workflow.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/workflow/book_rag.py tests/test_book_rag_workflow.py
git commit -m "feat(workflow): add BookRagWorkflow events + judge_query parsing"
```

---

### Task 3: 路由 + 轮数封顶(`_decide`)

**Files:**
- Modify: `core/workflow/book_rag.py`
- Test: `tests/test_book_rag_workflow.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_book_rag_workflow.py` 末尾追加:

```python
async def test_decide_clear_routes_to_retrieve():
    llm = FakeLLM(['{"clear": true, "rewritten_query": "B+树"}'])
    wf = _make_wf(llm)
    action, q = await wf._decide("B+树", round=0)
    assert action == "retrieve"
    assert q == "B+树"


async def test_decide_unclear_routes_to_rewrite():
    llm = FakeLLM(['{"clear": false, "rewritten_query": "数据库索引原理"}'])
    wf = _make_wf(llm)
    action, q = await wf._decide("讲讲数据库", round=0)
    assert action == "rewrite"
    assert q == "数据库索引原理"


async def test_decide_caps_at_max_rounds_without_calling_llm():
    llm = FakeLLM([])  # 队列为空：若被调用会 IndexError
    wf = _make_wf(llm)
    action, q = await wf._decide("还是很泛", round=2)
    assert action == "retrieve"   # 达上限直接检索
    assert q == "还是很泛"
    assert llm.calls == 0         # 未再调用 LLM
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_rag_workflow.py -k decide -v`
Expected: FAIL，`AttributeError: ... has no attribute '_decide'`

- [ ] **Step 3: 实现 `_decide`**

在 `core/workflow/book_rag.py` 的 `BookRagWorkflow` 类内、`_judge_query` 方法之后追加:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_rag_workflow.py -k decide -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add core/workflow/book_rag.py tests/test_book_rag_workflow.py
git commit -m "feat(workflow): add _decide routing with MAX_ROUNDS cap"
```

---

### Task 4: 检索逻辑 + 串联 steps(`_retrieve_nodes` + start/judge/retrieve/synthesize)

**Files:**
- Modify: `core/workflow/book_rag.py`
- Test: `tests/test_book_rag_workflow.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_book_rag_workflow.py` 顶部 import 区下方追加 fake 检索件,并在文件末尾追加测试:

```python
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


class FakeIndexManager:
    def __init__(self, nodes):
        self._index = FakeIndex(nodes)
    def get_index(self):
        return self._index


async def test_retrieve_nodes_passes_top_k_and_returns_nodes():
    llm = FakeLLM([])
    im = FakeIndexManager(nodes=["n1", "n2"])
    wf = _make_wf(llm, index_manager=im)
    nodes = await wf._retrieve_nodes("B+树", book_title=None)
    assert nodes == ["n1", "n2"]
    assert im._index.last_kw["similarity_top_k"] == 3
    assert im._index.last_kw["filters"] is None


async def test_retrieve_nodes_builds_book_title_filter():
    llm = FakeLLM([])
    im = FakeIndexManager(nodes=["n1"])
    wf = _make_wf(llm, index_manager=im)
    await wf._retrieve_nodes("B+树", book_title="MySQL是怎样运行的")
    assert im._index.last_kw["filters"] is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_rag_workflow.py -k retrieve_nodes -v`
Expected: FAIL，`AttributeError: ... has no attribute '_retrieve_nodes'`

- [ ] **Step 3: 实现 `_retrieve_nodes` 与四个 step**

在 `core/workflow/book_rag.py` 的 `BookRagWorkflow` 类内、`_decide` 之后追加:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_rag_workflow.py -v`
Expected: 全部 passed（含此前 6 个 + 新增 2 个）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/book_rag.py tests/test_book_rag_workflow.py
git commit -m "feat(workflow): add retrieve/synthesize steps + empty-result short-circuit"
```

---

### Task 5: 工具委托 workflow + source 接线

**Files:**
- Modify: `core/tools/book_tools.py:31-74`(`book_search` 函数体)
- Test: `tests/test_book_search_tool.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_book_search_tool.py`:

```python
import pytest

import core.tools.book_tools as book_tools
from core.tools.book_tools import create_book_search_tool


class _FakeNode:
    pass


class FakeResponse:
    def __init__(self, text, source_nodes):
        self._t = text
        self.source_nodes = source_nodes
    def __str__(self):
        return self._t


class FakeIndexManager:
    """get_index 非 None 表示库非空。"""
    def __init__(self, index=object()):
        self._index = index
    def get_index(self):
        return self._index


async def _call_tool(tool):
    # FunctionTool 的底层 async fn 可通过 tool.async_fn 访问
    return tool


async def test_book_search_propagates_sources(monkeypatch):
    captured = {}
    monkeypatch.setattr(book_tools, "add_sources", lambda refs: captured.setdefault("refs", refs))
    monkeypatch.setattr(book_tools, "node_to_source_ref", lambda n: f"ref:{id(n)}")

    nodes = [_FakeNode(), _FakeNode()]

    class StubWorkflow:
        def __init__(self, **kw):
            pass
        async def run(self, **kw):
            return FakeResponse("合成答案", source_nodes=nodes)

    monkeypatch.setattr(book_tools, "BookRagWorkflow", StubWorkflow)

    tool = create_book_search_tool(FakeIndexManager(), llm=object())
    result = await tool.async_fn(query="B+树的索引结构")

    assert result == "合成答案"
    assert len(captured["refs"]) == 2


async def test_book_search_empty_index_returns_hint():
    tool = create_book_search_tool(
        FakeIndexManager(index=None), llm=object()
    )
    result = await tool.async_fn(query="B+树")
    assert "知识库为空" in result


async def test_book_search_no_nodes_returns_scope_hint(monkeypatch):
    monkeypatch.setattr(book_tools, "add_sources", lambda refs: None)

    class StubWorkflow:
        def __init__(self, **kw):
            pass
        async def run(self, **kw):
            return FakeResponse("", source_nodes=[])

    monkeypatch.setattr(book_tools, "BookRagWorkflow", StubWorkflow)

    tool = create_book_search_tool(FakeIndexManager(), llm=object())
    result = await tool.async_fn(query="不存在的内容", book_title="某本书")
    assert "没有检索到" in result
    assert "某本书" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_search_tool.py -v`
Expected: FAIL，`AttributeError: module 'core.tools.book_tools' has no attribute 'BookRagWorkflow'`

- [ ] **Step 3: 改 book_search 委托 workflow**

编辑 `core/tools/book_tools.py`。在文件顶部 import 区(第 17 行 `from core.agent.source_context ...` 之后)追加:

```python
from core.workflow.book_rag import BookRagWorkflow
```

然后把 `book_search` 函数体(原第 49-74 行,从 `index = index_manager.get_index()` 到 `return str(response)`)整体替换为:

```python
        index = index_manager.get_index()
        if index is None:
            return "知识库为空，请先在「文档管理」上传 PDF。"

        workflow = BookRagWorkflow(
            index_manager=index_manager,
            llm=llm,
            similarity_top_k=similarity_top_k,
        )
        result = await workflow.run(query=query, book_title=book_title)

        if not result.source_nodes:
            scope = f"《{book_title}》中" if book_title else "知识库中"
            return f"在{scope}没有检索到与「{query}」相关的内容。"

        add_sources([node_to_source_ref(n) for n in result.source_nodes])
        return str(result)
```

注意:删除原先内联的 `filters` 构造、`as_retriever`、`retriever.aretrieve`、`get_response_synthesizer`、`synthesizer.asynthesize` 代码,以及现在用不到的 import(`get_response_synthesizer`、`MetadataFilter`、`MetadataFilters`)。保留顶部对 `query` 的字符串防御与空 query 检查(原第 42-47 行)不变。

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_search_tool.py -v`
Expected: 3 passed

- [ ] **Step 5: 跑分层守卫 + 全量测试**

Run: `.venv\Scripts\python.exe scripts/check_layering.py`
Expected: 通过(core 仍不依赖 api;新增的是 core→core 依赖)

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全部 passed

- [ ] **Step 6: Commit**

```bash
git add core/tools/book_tools.py tests/test_book_search_tool.py
git commit -m "refactor(tools): book_search delegates to BookRagWorkflow"
```

---

### Task 6: Agent 层 system_prompt 指代改写补强

**Files:**
- Modify: `core/agent/agent.py:18-32`(`BOOK_SYSTEM_PROMPT`)
- Test: `tests/test_book_system_prompt.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_book_system_prompt.py`:

```python
from core.agent.agent import BOOK_SYSTEM_PROMPT


def test_system_prompt_has_coreference_rewrite_rule():
    # 必须提示 LLM 在调用工具前，把含指代词的问题改写为自包含 query
    assert "指代" in BOOK_SYSTEM_PROMPT
    assert "book_search" in BOOK_SYSTEM_PROMPT
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_system_prompt.py -v`
Expected: FAIL，`AssertionError: assert '指代' in BOOK_SYSTEM_PROMPT`

- [ ] **Step 3: 在 BOOK_SYSTEM_PROMPT 追加规则**

编辑 `core/agent/agent.py`,在 `BOOK_SYSTEM_PROMPT` 的「回答规则」区块(原第 29-32 行)末尾、闭合的 `"""` 之前追加一行:

```
- 调用 book_search 前：若用户问题含指代词（它/这个/上面说的/前面提到的），先根据会话历史把 query 改写为不依赖上文、能独立成立的句子，再传入 query 参数。
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv\Scripts\python.exe -m pytest tests/test_book_system_prompt.py -v`
Expected: 1 passed

- [ ] **Step 5: 全量测试 + 冒烟启动**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 全部 passed

Run: `.venv\Scripts\python.exe -c "import api.main"`
Expected: 无 import 错误(装配层未改,workflow 能被正常 import)

- [ ] **Step 6: Commit**

```bash
git add core/agent/agent.py tests/test_book_system_prompt.py
git commit -m "feat(agent): system_prompt 指代改写规则补强"
```

---

## Self-Review Notes

- **Spec coverage:** A(system_prompt 指代)→ Task 6;B(judge_query workflow)→ Task 2/3/4;工具委托不改签名 → Task 5;错误处理(非法 JSON fallback / 轮数封顶 / 空检索 / 空库)→ Task 2 Step1 第 3 测、Task 3、Task 5;测试 1-6 → 分布于 Task 2-6。覆盖完整。
- **Type consistency:** `_judge_query → (bool, str)`、`_decide → (str, str)`、`_retrieve_nodes → list`、事件 `JudgeEvent(query, book_title, round)` / `RetrieveEvent(query, book_title)` / `SynthesizeEvent(query, nodes)`、`MAX_ROUNDS=2` 在各 task 一致引用。
- **No placeholders:** 所有 step 含完整代码与确切命令。
- **风险点提示:** `tool.async_fn` 是 LlamaIndex `FunctionTool` 暴露异步底层函数的属性;若该版本属性名不同(如 `acall`/`fn`),Task 5 测试需相应调整——执行时以实际属性为准。
