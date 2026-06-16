# 可插拔 Reranker 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 RAG 检索加一个装配时注入的 `Reranker` 组件——不传=基线（直召 top_k），传入 bge 交叉编码器=过召回(20)→重排→截(5)，并让 eval ablation 量化其 `context_precision` 增益。

**Architecture:** 在唯一咽喉点 `QaCapability._retrieve_nodes` 接入重排。组件协议 + bge 实现 + 名字→对象工厂全落 `core/retrieval/`；`DocQueryWorkflow` 用工厂把名字解析成对象注入 `QaCapability`；eval 只在 `VARIANTS` 配置里传名字字符串，经现有 `**flags` 透传链自动流通。默认 `reranker=None` → 现有行为与测试零变化。

**Tech Stack:** Python 3.12 async、LlamaIndex（`SentenceTransformerRerank` 已在 `llama_index.core.postprocessor`，依赖 sentence-transformers + torch 均已安装）、pytest。

---

## 文件结构

| 文件 | 职责 | 动作 |
|------|------|------|
| `core/retrieval/__init__.py` | 新包标记 | 创建（空） |
| `core/retrieval/rerank.py` | `Reranker` 协议 + `BgeReranker` 实现 + `make_reranker` 工厂 | 创建 |
| `core/workflow/qa_capability.py` | `QaCapability` 接收 reranker，`_retrieve_nodes` 过召回→重排→截断 | 修改 |
| `core/workflow/doc_workflow.py` | `DocQueryWorkflow` 接收 `reranker: str\|None`，工厂解析后注入 | 修改 |
| `eval/harness/compare.py` | `VARIANTS` 加一条带 `reranker=...` 的变体 | 修改 |
| `tests/test_retrieval_rerank.py` | 工厂 + 协议单测 | 创建 |
| `tests/test_qa_capability.py` | rerank 接入单测（过召回/截断/None 跳过） | 修改 |
| `tests/test_doc_workflow.py` | 注入路径单测（名字→对象） | 修改或创建 |
| `requirements.txt` | 固定 sentence-transformers（已装，显式记录） | 修改 |
| `docs/ARCHITECTURE.md` | 记一句可插拔检索组件方向 | 修改 |

---

## Task 1: Reranker 协议 + 工厂（core/retrieval/rerank.py）

**Files:**
- Create: `core/retrieval/__init__.py`
- Create: `core/retrieval/rerank.py`
- Test: `tests/test_retrieval_rerank.py`

- [ ] **Step 1: 建空包标记**

创建 `core/retrieval/__init__.py`，内容为空（与 `core/workflow/__init__.py` 惯例一致）。

- [ ] **Step 2: 写失败测试**

创建 `tests/test_retrieval_rerank.py`：

```python
"""core/retrieval/rerank.py 单测：工厂映射 + 协议一致性。

真实 bge 模型需下载 ~600MB，不在单测范围（见 plan 的手动 smoke 步骤）；
这里只验证名字→对象的解析与边界，构造真实 BgeReranker 不触发。
"""
import pytest

from core.retrieval.rerank import Reranker, make_reranker


def test_make_reranker_none_returns_none():
    assert make_reranker(None) is None
    assert make_reranker("") is None


def test_make_reranker_unknown_name_raises():
    with pytest.raises(ValueError):
        make_reranker("no-such-reranker")


def test_make_reranker_bge_name_is_registered():
    # 已知名字在注册表里（不构造真实模型，只查表）
    from core.retrieval.rerank import _REGISTRY
    assert "bge-reranker-v2-m3" in _REGISTRY


class _FakeReranker:
    async def rerank(self, query, nodes, top_n):
        return nodes[:top_n]


def test_protocol_runtime_check_accepts_conforming_object():
    assert isinstance(_FakeReranker(), Reranker)
```

- [ ] **Step 3: 跑测试确认失败**

Run: `python -m pytest tests/test_retrieval_rerank.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'core.retrieval'`

- [ ] **Step 4: 写实现**

创建 `core/retrieval/rerank.py`：

```python
"""可插拔 Reranker：装配时注入的检索后处理组件。

不传（None）= 没有重排步骤（基线）；传入实现 = 过召回后重新打分截断。
名字→对象的解析住在本模块（core），eval 只传名字字符串，评测概念不漏进 core。
"""
import asyncio
from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """对召回候选重新打分排序，返回前 top_n 个。"""

    async def rerank(self, query: str, nodes: list, top_n: int) -> list: ...


class BgeReranker:
    """本地交叉编码器，包 LlamaIndex SentenceTransformerRerank（默认 bge-reranker-v2-m3）。

    模型同步推理，用 asyncio.to_thread 卸到线程，不堵事件循环。首次使用下载模型。
    """

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3"):
        from llama_index.core.postprocessor import SentenceTransformerRerank

        # top_n 占位，真实值每次调用前按 top_n 覆盖
        self._pp = SentenceTransformerRerank(model=model, top_n=5)

    async def rerank(self, query: str, nodes: list, top_n: int) -> list:
        if not nodes:
            return nodes
        return await asyncio.to_thread(self._postprocess, query, nodes, top_n)

    def _postprocess(self, query: str, nodes: list, top_n: int) -> list:
        from llama_index.core import QueryBundle

        self._pp.top_n = top_n
        return self._pp.postprocess_nodes(nodes, query_bundle=QueryBundle(query))


# 名字 → 构造器。新增实现在此登记一行即可。
_REGISTRY = {
    "bge-reranker-v2-m3": lambda: BgeReranker("BAAI/bge-reranker-v2-m3"),
}


def make_reranker(name: str | None) -> "Reranker | None":
    """名字 → 实例。None/"" → None（跳过这步）；未知名字 → ValueError（配置错误尽早暴露）。"""
    if not name:
        return None
    if name not in _REGISTRY:
        raise ValueError(
            f"未知 reranker 名字：{name!r}，可选：{list(_REGISTRY)}"
        )
    return _REGISTRY[name]()
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_retrieval_rerank.py -v`
Expected: 4 passed

- [ ] **Step 6: 守卫分层不破坏**

Run: `python scripts/check_layering.py`
Expected: 通过（core/retrieval 只依赖 llama_index，不碰 api/eval）

- [ ] **Step 7: 提交**

```bash
git add core/retrieval/__init__.py core/retrieval/rerank.py tests/test_retrieval_rerank.py
git commit -m "feat(retrieval): Reranker 协议 + BgeReranker + make_reranker 工厂"
```

---

## Task 2: QaCapability 接入重排（过召回→重排→截断）

**Files:**
- Modify: `core/workflow/qa_capability.py:63-73`（`__init__`）、`:283-289`（`_retrieve_nodes`）
- Test: `tests/test_qa_capability.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_qa_capability.py` 末尾追加（替身区已有 `FakeIndexManager` 等）：

```python
# ── rerank 接入 ───────────────────────────────────────────────────────
from core.workflow.qa_capability import QaCapability as _QaCap


class _RecordingReranker:
    """记录入参；把候选倒序后截 top_n（验证顺序确实被改 + 截断）。"""

    def __init__(self):
        self.calls = []

    async def rerank(self, query, nodes, top_n):
        self.calls.append((query, list(nodes), top_n))
        return list(reversed(nodes))[:top_n]


async def test_retrieve_nodes_without_reranker_keeps_top_k():
    im = FakeIndexManager(nodes=["a", "b", "c"])
    qa = _QaCap(im, FakeLLM(), similarity_top_k=3)

    nodes = await qa._retrieve_nodes("q", None)

    assert nodes == ["a", "b", "c"]
    # 基线：用 similarity_top_k 召回，不过召回
    assert im._index.last_kw["similarity_top_k"] == 3


async def test_retrieve_nodes_with_reranker_overfetches_then_truncates():
    im = FakeIndexManager(nodes=["a", "b", "c", "d", "e"])
    rr = _RecordingReranker()
    qa = _QaCap(im, FakeLLM(), similarity_top_k=2,
                reranker=rr, rerank_candidate_k=5)

    nodes = await qa._retrieve_nodes("B+树", None)

    # 召回用候选池大小，不是 top_k
    assert im._index.last_kw["similarity_top_k"] == 5
    # reranker 收到候选并按 top_n 截断
    assert rr.calls == [("B+树", ["a", "b", "c", "d", "e"], 2)]
    assert nodes == ["e", "d"]  # 倒序后截 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_qa_capability.py -k "reranker or rerank" -v`
Expected: FAIL，`TypeError: __init__() got an unexpected keyword argument 'reranker'`

- [ ] **Step 3: 改 `__init__` 接收 reranker**

`core/workflow/qa_capability.py` 顶部 import 区加：

```python
from core.retrieval.rerank import Reranker
```

把 `__init__`（当前 63-76 行）改为：

```python
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        reranker: "Reranker | None" = None,
        rerank_candidate_k: int = 20,
    ):
        self.index_manager = index_manager
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.max_sub_queries = max_sub_queries
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k
        self.preprocessor = QueryPreprocessor(llm)
        self.decomposer = QueryDecomposer(llm)
        self.dimensioner = DimensionExtractor(llm)
```

- [ ] **Step 4: 改 `_retrieve_nodes` 过召回→重排→截断**

把 `_retrieve_nodes`（当前 283-289 行）改为：

```python
    async def _retrieve_nodes(self, query: str, book_titles: Optional[list[str]]):
        # 无 reranker（基线）→ 直接召回 top_k；有 → 过召回到候选池再重排截回 top_k
        fetch_k = self.rerank_candidate_k if self.reranker else self.similarity_top_k
        index = self.index_manager.get_index()
        retriever = index.as_retriever(
            similarity_top_k=fetch_k,
            filters=self._make_filters(book_titles),
        )
        nodes = await retriever.aretrieve(query)
        if self.reranker:
            nodes = await self.reranker.rerank(query, nodes, self.similarity_top_k)
        return nodes
```

- [ ] **Step 5: 跑新测试确认通过**

Run: `python -m pytest tests/test_qa_capability.py -k "reranker or rerank" -v`
Expected: 2 passed

- [ ] **Step 6: 跑全量 QaCapability 测试确认零回归**

Run: `python -m pytest tests/test_qa_capability.py -v`
Expected: 全部 passed（默认 reranker=None，旧行为不变）

- [ ] **Step 7: 提交**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(workflow): QaCapability 接入可注入 reranker（过召回→重排→截断）"
```

---

## Task 3: DocQueryWorkflow 解析名字并注入

**Files:**
- Modify: `core/workflow/doc_workflow.py:119-135`（`__init__`）
- Test: `tests/test_doc_workflow.py`

- [ ] **Step 1: 写失败测试**

创建（若不存在）`tests/test_doc_workflow.py`，否则追加：

```python
"""DocQueryWorkflow 装配单测：reranker 名字 → 对象注入 QaCapability。"""
from core.workflow.doc_workflow import DocQueryWorkflow


class _StubIndexManager:
    pass


class _StubLLM:
    pass


def test_no_reranker_by_default():
    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM())
    assert wf.qa.reranker is None


def test_reranker_name_resolved_and_injected(monkeypatch):
    sentinel = object()
    import core.workflow.doc_workflow as mod

    captured = {}

    def fake_make(name):
        captured["name"] = name
        return sentinel

    monkeypatch.setattr(mod, "make_reranker", fake_make)

    wf = DocQueryWorkflow(_StubIndexManager(), _StubLLM(),
                          reranker="bge-reranker-v2-m3")

    assert captured["name"] == "bge-reranker-v2-m3"
    assert wf.qa.reranker is sentinel
```

> 注：`QaCapability.__init__` / `IntentRouter` / `QaAgent` 构造不触碰 index/llm 的真实方法，stub 即可；若构造期有真实调用导致失败，按报错把 stub 补成对应最小替身。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_doc_workflow.py -v`
Expected: FAIL，`TypeError: __init__() got an unexpected keyword argument 'reranker'`

- [ ] **Step 3: 改 DocQueryWorkflow**

`core/workflow/doc_workflow.py` import 区加：

```python
from core.retrieval.rerank import make_reranker
```

`__init__` 签名（当前 119-130 行）加参数 `reranker: str | None = None`，放在 `max_sub_queries` 之后、`probe_then_classify` 之前：

```python
    def __init__(
        self,
        index_manager,
        llm: LLM,
        similarity_top_k: int = 5,
        max_sub_queries: int = 6,
        reranker: str | None = None,
        probe_then_classify: bool = True,
        split_enabled: bool = True,
        assume_enabled: bool = True,
        other_agent_enabled: bool = True,
        **kw,
    ):
```

把 QaCapability 构造行（当前 135 行）改为：

```python
        self.qa = QaCapability(
            index_manager, llm, similarity_top_k, max_sub_queries,
            reranker=make_reranker(reranker),
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_doc_workflow.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat(workflow): DocQueryWorkflow 按名字解析并注入 reranker"
```

---

## Task 4: eval VARIANTS 增加带 reranker 的变体

**Files:**
- Modify: `eval/harness/compare.py:42-51`（`VARIANTS`）

- [ ] **Step 1: 加变体**

`eval/harness/compare.py` 的 `VARIANTS` 字典在 `"全开"` 之后追加一条：

```python
    "全开+rerank": dict(probe_then_classify=True, split_enabled=True,
                        assume_enabled=True, other_agent_enabled=True,
                        reranker="bge-reranker-v2-m3"),
```

> `sut.py` 已 `**self._flags` 透传，`DocQueryWorkflow` 已接受 `reranker` kwarg，无需改 sut。

- [ ] **Step 2: 校验配置可被消费（不下模型，只查 kwargs 兼容）**

Run:
```bash
python -c "from eval.harness.compare import VARIANTS; import inspect; from core.workflow.doc_workflow import DocQueryWorkflow; sig=set(inspect.signature(DocQueryWorkflow.__init__).parameters); bad=[k for v in VARIANTS.values() for k in v if k not in sig]; print('未知kwarg:', bad); assert not bad"
```
Expected: `未知kwarg: []`（无报错）

- [ ] **Step 3: 提交**

```bash
git add eval/harness/compare.py
git commit -m "feat(eval): compare 增加「全开+rerank」变体，量化 reranker 增益"
```

---

## Task 5: 依赖记录、文档、真实模型 smoke 验证

**Files:**
- Modify: `requirements.txt`
- Modify: `docs/ARCHITECTURE.md`

- [ ] **Step 1: 记录依赖**

确认 `requirements.txt` 已含 `sentence-transformers`（torch 由其传递）。若缺，加一行：

```
sentence-transformers
```

Run: `python -c "import sentence_transformers, torch; print('deps OK')"`
Expected: `deps OK`

- [ ] **Step 2: 文档加一句方向说明**

`docs/ARCHITECTURE.md` 检索相关章节加一句：

> 检索后处理走可插拔组件（`core/retrieval/rerank.py` 的 `Reranker` 协议）：装配时注入，
> 不传=基线（直召 top_k），传入实现=过召回后重排截断。eval `VARIANTS` 以名字选择，
> ablation 量化增益。

- [ ] **Step 3: 真实模型 smoke（手动，会下模型 ~600MB）**

Run（小样本，含基线与 rerank 两变体）:
```bash
python -m eval.harness.compare --variants "全开" "全开+rerank" --limit 5
```
Expected: 跑通并打出 delta 表；首次运行下载 bge 模型。重点看 `context_precision` 列 `全开+rerank` 相对 `全开` 的变化（设计预期为正向）。

> 若模型下载受网络限制或耗时过长，可跳过本步留待联网环境；前 4 个 Task 的单测不依赖真实模型。

- [ ] **Step 4: 提交**

```bash
git add requirements.txt docs/ARCHITECTURE.md
git commit -m "docs(retrieval): 记录可插拔 reranker 依赖与架构说明"
```

---

## 收尾

- [ ] 跑全量测试：`python -m pytest -q`，确认零回归。
- [ ] 更新记忆 `project_pluggable_retrieval_pipeline.md`：状态从「待 brainstorm+plan」改为「第一版 reranker 已落地（bge 注入式），下一步可加第二实现 / 其余组件类别」。
