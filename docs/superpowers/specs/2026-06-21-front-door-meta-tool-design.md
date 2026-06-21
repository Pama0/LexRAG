# 设计 + 实现 Plan：front_door converse 路径加 list_books 元工具

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Context

线上观察：用户问"现在库里都有什么"——`FrontDoorAgent` 正确判 `converse`（它是元问题，不是书里的内容知识，不该 `dispatch_qa` 走检索），但 converse 出口走 LLM 直接 `reply`，而"库里有哪些书"是**运行时数据**（chroma `book_title` 元数据），不在 LLM 训练知识里 → 它要么编、要么泛泛答。

系统里已有 `core/agent/tools/func/list_book_tool.py` 的 `ListBooksTool`（从 `chroma_collection.get(include=["metadatas"])` 聚合 `book_title` 计数），但它只挂在 QaAgent/AutoAgent（有界工具 agent）上；`FrontDoorAgent` 是单次结构化决策、无工具、无 `index_manager` 注入。**断层**：converse 出口里混了两类——(a) 纯寒暄/对上轮反馈（LLM 能答）、(b) 可事实回答的元查询（需要运行时数据）。被一刀切走 LLM reply，(b) 类答不准。

## Goals / Non-goals

**Goals**
- `FrontDoorAgent` 注入 `index_manager`，converse 路径条件性调 `list_books` 元工具答库藏元查询（"有什么""有 MySQL 的书吗""多少本"）。
- 抽共享函数 `core/rag/inventory.py::list_books_text(im, filter, count_only)`，`ListBooksTool` 与 `FrontDoorAgent` 共用，避免 `workflow ↔ agent.tools` 模块循环。
- `list_books_text` 带参数：`filter`（大小写不敏感子串匹配 `book_title`）、`count_only`（只回计数）。
- converse+tool 路径走 2nd LLM 调用组自然回复（1st 决策 → 执行工具 → 2nd 用工具结果 + 原 query 组回复）。
- 4 出口不变（`dispatch_qa / dispatch_study_plan / converse / clarify`）；`tool` 是 converse 内分支，不是新出口。
- clarify 不加 tool。

**Non-goals（明确不做）**
- **不给 front_door 加 `book_search`**（内容检索）。routing-note 红线 1+2：front_door 绝不自己答内容问题、不自己做检索。内容问题一律 `dispatch_qa` 下沉。`tool` 只能是 `list_books`。
- **不给 `ListBooksTool` 加参数**（QaAgent 侧契约不变，本刀只修 front_door）。
- **不扩 clarify 出口**（YAGNI，反问不涉及库藏数据）。
- **不切 feature 分支**（用户明确指示，在 master 上做）。

## 架构：组件（新增 + 改造）

| 组件 | 变化 | 职责 |
|---|---|---|
| `core/rag/inventory.py`（新） | 全新共享函数 | `list_books_text(im, filter="", count_only=False) -> str`：从 chroma 元数据聚合书单 |
| `core/agent/tools/func/list_book_tool.py`（改） | thin wrapper | `__call__` 改成 `return list_books_text(self.ctx.index_manager)`（无参，契约不变） |
| `core/workflow/front_door.py`（改） | schema +3 字段、prompt 加工具定义+红线、`__init__` 注入 im、`run()` 加 2nd 路径、`_compose_tool_reply` 新方法 | converse+tool 时 1st 决策 → 执行工具 → 2nd LLM 组回复 |
| `core/workflow/doc_workflow.py`（改） | 接线一行 | `FrontDoorAgent(llm, index_manager)` |
| `tests/test_inventory.py`（新） | 共享函数单测 | 空库/全量/filter 命中不命中/count_only/大小写不敏感 |
| `tests/test_front_door.py`（改） | 加 tool 路径测试 | converse+tool 各形、2nd 组回复、降级 |
| `tests/test_doc_workflow.py`（改） | 端到端 | "库里都有什么" → reply 含书名、不检索、metadata.action=converse |

### `list_books_text` 接口与输出格式

```
list_books_text(index_manager, filter: str = "", count_only: bool = False) -> str
```

- `filter="" + count_only=False`（默认）：全量列表
  - 非空：`"已入库书籍：\n- 《A》（2 块）\n- 《B》（1 块）"`（按书名排序）
  - 空：`"知识库当前为空。"`
- `filter="mysql" + count_only=False`：匹配列表
  - 有匹配：`"匹配「mysql」的书籍：\n- 《高性能MySQL》（5 块）"`
  - 无匹配：`"没有匹配「mysql」的书籍。"`
- `filter="" + count_only=True`：全量计数
  - 非空：`"已入库 3 本。"`
  - 空：`"知识库当前为空。"`
- `filter="mysql" + count_only=True`：匹配计数
  - 有匹配：`"匹配「mysql」的书有 1 本。"`
  - 无匹配：`"没有匹配「mysql」的书。"`

`filter` 大小写不敏感子串匹配 `book_title`。

### FrontDoorAgent schema 加 3 字段（flat，不嵌套）

```
class FrontDoorDecisionModel(BaseModel):
    action: Literal["dispatch_qa", "dispatch_study_plan", "converse", "clarify"]
    clean_query: str = Field(default="", ...)
    reply: str = Field(default="", ...)
    reason: str = Field(default="", ...)
    # 新增：converse 路径的元工具调用（仅 list_books；绝不可加 book_search）
    tool: Literal["list_books", ""] = ""
    tool_filter: str = Field(default="", description="书名子串过滤，大小写不敏感")
    tool_count_only: bool = False
```

`FrontDoorDecision` dataclass 同步加这 3 字段（默认 `""`/`""`/`False`）。

## 数据流

### converse + tool 路径（新增）

```
FrontDoorAgent.run(original, memory, book_titles):
  history = format_history(memory); scope = format_scope(book_titles)
  # 1st LLM：决策（prompt 含工具定义 + 红线）
  d = await self.llm.acomplete(决策 prompt, json_object)
  if d.action in ("dispatch_qa", "dispatch_study_plan"):
      return FrontDoorDecision(d.action, clean_query=..., reason=d.reason)   # 现状
  if d.action == "clarify":
      return FrontDoorDecision("clarify", reply=..., reason=d.reason)        # 现状
  # converse
  if d.tool == "list_books":
      try:
          tool_result = list_books_text(self.index_manager, d.tool_filter, d.tool_count_only)
      except Exception as exc:
          logger.warning("front_door list_books 查询失败：%s", exc)
          tool_result = "（未能读取库藏清单）"
      reply = await self._compose_tool_reply(original, tool_result)
      return FrontDoorDecision("converse", reply=reply, reason=d.reason)
  # converse 无 tool：reply 直接用（现状，寒暄/反馈/元问题无需数据）
  reply = (d.reply or "").strip() or _FALLBACK_REPLY
  return FrontDoorDecision("converse", reply=reply, reason=d.reason)
```

### `_compose_tool_reply`（新方法，2nd LLM）

```
async def _compose_tool_reply(self, original: str, tool_result: str) -> str:
    """2nd LLM：用工具结果 + 原 query 组自然回复。失败降级裸 tool_result。"""
    prompt = _COMPOSE_PROMPT.replace("{query}", original).replace("{data}", tool_result)
    try:
        resp = await self.llm.acomplete(prompt)   # 非 json_object，自然文本
        text = str(resp).strip()
        if text:
            return text
    except Exception as exc:
        logger.warning("front_door compose reply 失败，用裸 tool_result：%s", exc)
    return tool_result   # 裸文本本就是人读的，比 fallback 信息多
```

`_COMPOSE_PROMPT`（单独常量）：

```
_COMPOSE_PROMPT = """用户问了关于知识库藏书的问题。下面是系统从知识库元数据查到的真实结果。请据此用一句自然、面向用户的话回复，不要机械复述数据。

铁律：
- 只能基于下面的【库藏数据】答，不得编造未列出的书。
- 简短自然，别寒暄一堆。

用户问题：{query}

库藏数据：
{data}"""
```

### 1st 决策 prompt 加工具定义 + 红线（在"第二步 选出口"的 converse 描述后追加）

```
- converse：寒暄/...以及对上一轮的反馈/...。
  【元工具（仅 converse 路径可用）】若本轮是关于知识库藏书的元查询（"库里有什么""有 MySQL 的书吗""多少本"等），设 tool="list_books" + tool_filter（书名子串，大小写不敏感，如"mysql"）+ tool_count_only（只要计数时 true），reply 留空（系统查库后另行组织）。纯寒暄/反馈/无需库藏数据时 tool="" 照常填 reply。
  【红线】tool 只能是 list_books。绝不可用于答书里的内容问题——内容问题一律 dispatch_qa 下沉检索。
```

JSON 示例行同步加 tool/tool_filter/tool_count_only 字段。

## 降级（绝不阻塞）

| 触发 | 落点 |
|---|---|
| 1st LLM 解析失败/非法 action | 降级 `dispatch_qa + 原 query`（现状） |
| `list_books_text` 抛错 | `tool_result = "（未能读取库藏清单）"`，仍走 2nd LLM |
| 2nd LLM 失败/空 | 用裸 `tool_result` 当 reply（人读文本，比 `_FALLBACK_REPLY` 信息多） |
| converse 无 tool + 空 reply | `_FALLBACK_REPLY`（现状） |

## 测试（mock LLM，验解析/接线/降级，不验真 LLM 判断质量）

- `list_books_text`：空库、全量列表、filter 命中、filter 不命中、count_only 全量、count_only + filter、大小写不敏感。
- `FrontDoorAgent.run` 1st：converse+tool="list_books"（"库里都有什么"）、converse+tool="list_books"+filter（"有 MySQL 的书吗"）、converse+tool="list_books"+count_only（"多少本"）、converse+tool=""（"你好"现状）、dispatch_qa 不变、clarify 不变、1st 失败降级 dispatch_qa。
- `FrontDoorAgent.run` 2nd：`_compose_tool_reply` 拿 tool_result 组自然回复（mock 2nd LLM）；2nd 失败 → 裸 tool_result 当 reply。
- `FrontDoorAgent.run` 工具失败：`list_books_text` 抛错 → 占位文本进 2nd LLM。
- `DocQueryWorkflow` 端到端："库里都有什么" → reply 含书名、不进检索、metadata.action=converse。
- 回归：`test_book_tools.py` 3 个 ListBooksTool 测试仍绿（wrapper 行为不变）。

## 决策锁定（评审依据）

1. **工具方案**（非上下文注入）：front_door 注入 im，converse+tool 时 2nd LLM 组回复。可扩展、贴用户"调工具"心智。
2. **list_books 带参数**（filter + count_only）：一工具覆盖"有什么/有 X 吗/多少本"。
3. **归 converse**（4 出口不变）：tool 是 converse 内分支，不是新出口。
4. **2nd LLM 组回复**（非裸 tool_result）：回复自然。
5. **clarify 不加 tool**（YAGNI）。
6. **共享函数放 `core/rag/inventory.py`**（中性，避免 `workflow↔agent.tools` 循环）。
7. **`ListBooksTool` 不加参数**（QaAgent 契约不变）。
8. **红线：tool 只能是 list_books，book_search 绝不可加**。

---

## Global Constraints

- **工作目录**：项目根 = `C:\Users\11394\PycharmProjects\llmaLearn`。所有命令从项目根运行；子模块内用相对导入、根脚本用绝对导入。
- **不切分支**：用户明确指示，在 master 上做。
- **提交粒度**：用显式 `git add <文件路径>`，**绝不** `git add -A` / `git add .`。不跳 hooks、不跳签名。提交信息结尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。
- **测试运行**：PowerShell 用 `$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest ...`——**必须用 `.venv\Scripts\python.exe`**，裸 `python` 指向另一个项目（gpt-researcher）的 venv、缺 `llama_index`，会 ImportError。**不要修 venv 或装包。**
- **所有 I/O 用 `async/await`**；函数签名加类型注解（中文注释可接受）。
- **core 不依赖 api**（守卫 `.venv\Scripts\python.exe scripts\check_layering.py`）。
- **决策单元约定（沿用，不是缺陷）**：注入 LLM、对外只暴露 `run`、`response_format={"type":"json_object"}` 按调用传、Pydantic 校验、失败优雅降级、`_strip_fences` 每模块各自带一份副本——照抄，不要"消重"。
- **红线**：`tool` 只能是 `list_books`；`book_search` 绝不可加进 front_door；内容问题一律 `dispatch_qa` 下沉。
- **命名**：共享函数 `list_books_text`；schema 字段 `tool` / `tool_filter` / `tool_count_only`；2nd 方法 `_compose_tool_reply`；2nd prompt 常量 `_COMPOSE_PROMPT`。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `core/rag/inventory.py` | 新建 | `list_books_text` 共享函数 |
| `core/agent/tools/func/list_book_tool.py` | 修改 | `__call__` 改 thin wrapper |
| `core/workflow/front_door.py` | 修改 | schema +3 字段、prompt 加工具定义+红线、`__init__` 注入 im、`run()` 加 2nd 路径、`_compose_tool_reply` + `_COMPOSE_PROMPT` |
| `core/workflow/doc_workflow.py` | 修改 | `FrontDoorAgent(llm, index_manager)` 一行 |
| `tests/test_inventory.py` | 新建 | `list_books_text` 单测 |
| `tests/test_front_door.py` | 修改 | 加 tool 路径 + 2nd + 降级测试 |
| `tests/test_doc_workflow.py` | 修改 | 端到端"库里都有什么"测试 |

`core/workflow/doc_workflow.py` 的 `preprocess`/`explain_branch`/`out_of_scope_branch`/finalize 等**不动**——本刀只改门口 + 接线一行。

---

## Task 1: 抽共享函数 `core/rag/inventory.py`

纯加法：新建 `inventory.py` + 单测，不接进任何调用方。任务结束后测试全绿、运行时行为不变。

**Files:**
- Create: `core/rag/inventory.py`
- Test: `tests/test_inventory.py`

**Interfaces:**
- Consumes：`index_manager`（只要有 `.chroma_collection.get(include=["metadatas"])` 返回 `{"metadatas": list[dict]}`）。
- Produces（Task 2/3 依赖）：`list_books_text(index_manager, filter: str = "", count_only: bool = False) -> str`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_inventory.py`：

```python
"""list_books_text 单测：从 chroma 元数据聚合书单，验全量/filter/count_only/大小写。"""
from core.rag.inventory import list_books_text


class _FakeCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class _FakeIndexManager:
    def __init__(self, metas):
        self.chroma_collection = _FakeCollection(metas)


def test_empty_library_returns_empty_message():
    assert list_books_text(_FakeIndexManager([])) == "知识库当前为空。"


def test_full_list_counts_and_sorts_titles():
    metas = [{"book_title": "乙"}, {"book_title": "甲"}, {"book_title": "甲"}]
    out = list_books_text(_FakeIndexManager(metas))
    assert "已入库书籍：" in out
    assert "《甲》（2 块）" in out
    assert "《乙》（1 块）" in out
    # 按书名排序：甲 在 乙 前
    assert out.index("《甲》") < out.index("《乙》")


def test_filter_case_insensitive_substring_match():
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}, {"book_title": "MySQL实战"}]
    out = list_books_text(_FakeIndexManager(metas), filter="mysql")
    assert "匹配「mysql」的书籍：" in out
    assert "《高性能MySQL》" in out
    assert "《MySQL实战》" in out
    assert "Redis" not in out


def test_filter_no_match_returns_no_match_message():
    metas = [{"book_title": "MySQL"}]
    out = list_books_text(_FakeIndexManager(metas), filter="oracle")
    assert out == "没有匹配「oracle」的书籍。"


def test_count_only_full_returns_count():
    metas = [{"book_title": "甲"}, {"book_title": "乙"}, {"book_title": "丙"}]
    assert list_books_text(_FakeIndexManager(metas), count_only=True) == "已入库 3 本。"


def test_count_only_empty_returns_empty_message():
    assert list_books_text(_FakeIndexManager([]), count_only=True) == "知识库当前为空。"


def test_count_only_with_filter_returns_match_count():
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}, {"book_title": "MySQL实战"}]
    out = list_books_text(_FakeIndexManager(metas), filter="mysql", count_only=True)
    assert out == "匹配「mysql」的书有 2 本。"


def test_count_only_with_filter_no_match():
    metas = [{"book_title": "MySQL"}]
    out = list_books_text(_FakeIndexManager(metas), filter="oracle", count_only=True)
    assert out == "没有匹配「oracle」的书。"


def test_filter_empty_string_is_full_list_not_match():
    # filter="" 应等同全量，不是"匹配空串"
    metas = [{"book_title": "甲"}]
    out = list_books_text(_FakeIndexManager(metas), filter="")
    assert "已入库书籍：" in out
    assert "匹配" not in out


def test_metas_without_book_title_skipped():
    metas = [{"book_title": "甲"}, {"other": "x"}, None, {"book_title": ""}]
    out = list_books_text(_FakeIndexManager(metas))
    assert "《甲》（1 块）" in out
    assert "已入库 1 本" not in out  # 只 1 本有效，但默认是列表形式不是计数
    # 确认只计了 1 本
    assert out.count("《") == 1
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_inventory.py -v
```
Expected: 全部 FAIL / ERROR —— `ModuleNotFoundError: No module named 'core.rag.inventory'`。

- [ ] **Step 3: 实现 `list_books_text`**

新建 `core/rag/inventory.py`：

```python
"""库藏元数据聚合：从 chroma 的 book_title 元数据产人读文本。

front_door 的 converse 路径（元查询"库里有什么/有 X 吗/多少本"）与
ListBooksTool（QaAgent 工具）共用此函数，避免 core/workflow → core/agent.tools
的模块循环（agent 已依赖 workflow.qa_capability）。
"""
from typing import Optional


def _collect_titles(index_manager) -> list[str]:
    """从 chroma 元数据抽出非空 book_title 列表（保留重复，供计数）。"""
    data = index_manager.chroma_collection.get(include=["metadatas"])
    metas = (data or {}).get("metadatas") or []
    titles: list[str] = []
    for meta in metas:
        title = (meta or {}).get("book_title")
        if title:
            titles.append(title)
    return titles


def list_books_text(
    index_manager,
    filter: str = "",
    count_only: bool = False,
) -> str:
    """聚合库藏书单为人读文本。

    - filter：大小写不敏感子串匹配 book_title；空串 = 全量。
    - count_only：True → 只回计数；False → 列出每本书 + 块数。
    """
    titles = _collect_titles(index_manager)
    if filter:
        fl = filter.lower()
        titles = [t for t in titles if fl in t.lower()]

    if count_only:
        if not titles and filter:
            return f"没有匹配「{filter}」的书。"
        if not titles:
            return "知识库当前为空。"
        if filter:
            # 去重计数：同一书名只算 1 本
            n = len(set(titles))
            return f"匹配「{filter}」的书有 {n} 本。"
        n = len(set(titles))
        return f"已入库 {n} 本。"

    # 列表形式
    if not titles and filter:
        return f"没有匹配「{filter}」的书籍。"
    if not titles:
        return "知识库当前为空。"

    # 按书名排序、计数
    counts: dict[str, int] = {}
    for t in titles:
        counts[t] = counts.get(t, 0) + 1
    head = f"匹配「{filter}」的书籍：" if filter else "已入库书籍："
    lines = [f"- 《{t}》（{c} 块）" for t, c in sorted(counts.items())]
    return head + "\n" + "\n".join(lines)
```

- [ ] **Step 4: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_inventory.py -v
```
Expected: 全部 PASS。

- [ ] **Step 5: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿（纯加法，不应触碰任何现有测试）。

- [ ] **Step 6: 提交**

```powershell
git add core/rag/inventory.py tests/test_inventory.py
git commit -m "feat: add list_books_text shared helper for library inventory

从 chroma book_title 元数据聚合书单为人读文本，支持 filter（大小写
不敏感子串）和 count_only。front_door converse 路径与 ListBooksTool
将共用此函数（后续 Task 接线）。纯加法，尚未接进调用方。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `ListBooksTool` 改 thin wrapper

小重构：`ListBooksTool.__call__` 改成调 `list_books_text`，消除重复逻辑。`test_book_tools.py` 现有 3 个 ListBooksTool 测试是回归保障（行为不变）。

**Files:**
- Modify: `core/agent/tools/func/list_book_tool.py`（`__call__` 体改）

**Interfaces:**
- Consumes：Task 1 的 `list_books_text`。
- Produces：`ListBooksTool.__call__()` 行为不变（仍返回 `"已入库书籍：\n- 《A》（n 块）..."` 或 `"知识库当前为空。"`）。

- [ ] **Step 1: 改 `__call__`**

`core/agent/tools/func/list_book_tool.py` 整体替换为：

```python
from llama_index.core.tools import FunctionTool

from core.agent.tools import register_tool, ToolContext
from core.rag.inventory import list_books_text


@register_tool
class ListBooksTool:
    """列出当前已入库书籍清单（按 book_title 计数）。"""

    name = "list_books"
    description = "列出当前已入库书籍清单。"
    prompt_usage = "list_books() — 列出已入库书籍清单（当 book_search 反复为空、需要了解可选范围时用）。"

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def __call__(self) -> str:
        return list_books_text(self.ctx.index_manager)

    def to_function_tool(self) -> FunctionTool:
        return FunctionTool.from_defaults(
            fn=self.__call__, name=self.name, description=self.description,
        )
```

- [ ] **Step 2: 跑 ListBooksTool 回归测试**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_book_tools.py -v
```
Expected: 全部 PASS。`test_list_books_counts_titles`、`test_list_books_empty`、`test_build_book_tools_default_returns_both`（含 list_books）均绿——wrapper 行为与原内联逻辑一致。

- [ ] **Step 3: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿。

- [ ] **Step 4: 提交**

```powershell
git add core/agent/tools/func/list_book_tool.py
git commit -m "refactor: ListBooksTool delegates to list_books_text shared helper

__call__ 改成 thin wrapper 调 core.rag.inventory.list_books_text，
消除与即将接线的 front_door 之间的重复逻辑。行为不变（test_book_tools
3 个回归测试绿）。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `FrontDoorAgent` 加 list_books 元工具路径

核心 Task：schema 加 3 字段、prompt 加工具定义+红线、`__init__` 注入 `index_manager`、`run()` 加 converse+tool 的 2nd LLM 路径、新增 `_compose_tool_reply` + `_COMPOSE_PROMPT`。

**Files:**
- Modify: `core/workflow/front_door.py`（schema、prompt、`__init__`、`run`、新方法+常量）
- Test: `tests/test_front_door.py`（加 tool 路径 + 2nd + 降级测试）

**Interfaces:**
- Consumes：Task 1 的 `list_books_text`。
- Produces：`FrontDoorAgent(llm, index_manager=None)`；`FrontDoorDecision` 加 `tool` / `tool_filter` / `tool_count_only` 字段；converse+tool 时 2nd LLM 组回复。

- [ ] **Step 1: 写失败测试**

在 `tests/test_front_door.py` **末尾**追加（文件已 `from core.workflow.front_door import FrontDoorAgent, FrontDoorDecision, format_history`；补 import `list_books_text` 不需要——通过 FakeIndexManager 间接验）：

```python
# ── converse + list_books 元工具路径（Task 3）──────────────────────────


class _FakeCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class _FakeIndexManager:
    def __init__(self, metas):
        self.chroma_collection = _FakeCollection(metas)


def _agent_with_lib(llm, metas):
    """带 index_manager 的 FrontDoorAgent（元工具路径需要）。"""
    return FrontDoorAgent(llm, index_manager=_FakeIndexManager(metas))


async def test_converse_list_books_full_invokes_tool_and_composes_reply():
    # "库里都有什么" → 1st 判 converse+tool=list_books → 查库 → 2nd 组回复
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","reply":"","reason":"元查询"}',
        '已入库的有《高性能MySQL》和《Redis》两本。',
    ])
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}]
    d = await _agent_with_lib(llm, metas).run("现在库里都有什么书")
    assert d.action == "converse"
    assert "高性能MySQL" in d.reply       # 2nd 组的回复含书名
    assert llm.calls == 2                  # 1st 决策 + 2nd 组回复
    # 2nd prompt 含工具结果 + 原 query
    assert "已入库书籍" in llm.prompts[1] or "《高性能MySQL》" in llm.prompts[1]
    assert "现在库里都有什么书" in llm.prompts[1]


async def test_converse_list_books_filter_passes_filter_to_tool():
    # "有 MySQL 的书吗" → tool_filter="mysql" → 工具结果只含 MySQL 书
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","tool_filter":"mysql","reply":""}',
        '有 MySQL 相关的书：《高性能MySQL》。',
    ])
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}]
    d = await _agent_with_lib(llm, metas).run("有 MySQL 的书吗")
    assert d.action == "converse"
    assert "高性能MySQL" in d.reply
    # 2nd prompt 的工具结果不含 Redis（被 filter 过滤）
    assert "Redis" not in llm.prompts[1]
    assert "匹配「mysql」" in llm.prompts[1]


async def test_converse_list_books_count_only_returns_count():
    # "多少本" → tool_count_only=true → 工具结果只回计数
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","tool_count_only":true,"reply":""}',
        '目前库里一共有 2 本书。',
    ])
    metas = [{"book_title": "甲"}, {"book_title": "乙"}]
    d = await _agent_with_lib(llm, metas).run("现在有多少本书")
    assert d.action == "converse"
    assert "2" in d.reply
    assert "已入库 2 本" in llm.prompts[1]   # 工具结果是计数，不是列表


async def test_converse_no_tool_uses_reply_directly_no_2nd_call():
    # 纯寒暄 → tool="" → reply 直接用，不调 2nd
    llm = FakeLLM(['{"action":"converse","tool":"","reply":"你好！我是文档知识库助手～"}'])
    d = await _agent_with_lib(llm, []).run("你好")
    assert d.action == "converse"
    assert "知识库助手" in d.reply
    assert llm.calls == 1                    # 只 1st，无 2nd


async def test_converse_tool_with_filter_and_count_only():
    # "有 mysql 吗，几本" → filter + count_only 同时带
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","tool_filter":"mysql","tool_count_only":true,"reply":""}',
        '有 1 本匹配 MySQL 的书。',
    ])
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}]
    d = await _agent_with_lib(llm, metas).run("有 mysql 吗，几本")
    assert d.action == "converse"
    assert "匹配「mysql」的书有 1 本" in llm.prompts[1]


async def test_converse_tool_compose_failure_degrades_to_raw_tool_result():
    # 2nd LLM 抛错 → 降级裸 tool_result 当 reply
    class _BoomLLM:
        def __init__(self):
            self.calls = 0
            self.prompts = []
        async def acomplete(self, prompt, **kw):
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                return _Resp('{"action":"converse","tool":"list_books","reply":""}')
            raise RuntimeError("2nd 炸了")
    llm = _BoomLLM()
    metas = [{"book_title": "甲"}]
    d = await _agent_with_lib(llm, metas).run("库里有什么")
    assert d.action == "converse"
    assert "已入库书籍" in d.reply and "《甲》" in d.reply   # 裸 tool_result


async def test_converse_tool_compose_empty_degrades_to_raw_tool_result():
    # 2nd LLM 返回空 → 降级裸 tool_result
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","reply":""}',
        "",
    ])
    metas = [{"book_title": "甲"}]
    d = await _agent_with_lib(llm, metas).run("库里有什么")
    assert d.action == "converse"
    assert "已入库书籍" in d.reply and "《甲》" in d.reply


async def test_converse_tool_list_books_failure_degrades_to_placeholder():
    # list_books_text 抛错 → 占位文本进 2nd LLM
    class _BrokenCollection:
        def get(self, include=None):
            raise RuntimeError("chroma 挂了")
    class _BrokenIM:
        chroma_collection = _BrokenCollection()
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","reply":""}',
        '抱歉，我没能读取库藏清单。',
    ])
    agent = FrontDoorAgent(llm, index_manager=_BrokenIM())
    d = await agent.run("库里有什么")
    assert d.action == "converse"
    assert "未能读取库藏清单" in llm.prompts[1]   # 占位文本进了 2nd prompt
    assert "抱歉" in d.reply


async def test_dispatch_qa_ignores_tool_field():
    # dispatch_qa 即使 LLM 误填 tool，也不走工具路径
    llm = FakeLLM(['{"action":"dispatch_qa","clean_query":"MySQL锁","tool":"list_books"}'])
    d = await _agent_with_lib(llm, []).run("MySQL有哪些锁")
    assert d.action == "dispatch_qa"
    assert d.clean_query == "MySQL锁"
    assert llm.calls == 1                    # 不调 2nd


async def test_front_door_prompt_has_tool_definition_and_redline():
    llm = FakeLLM(['{"action":"converse","tool":"","reply":"hi"}'])
    await _agent_with_lib(llm, []).run("你好")
    p = llm.prompts[0]
    assert "list_books" in p                 # 工具定义进 prompt
    assert "tool_filter" in p
    assert "tool_count_only" in p
    # 红线：内容问题一律 dispatch_qa
    assert "dispatch_qa" in p
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_front_door.py -k "list_books or tool or compose or dispatch_qa_ignores or prompt_has_tool" -v
```
Expected: 全部 FAIL——`FrontDoorAgent.__init__` 不收 `index_manager`；schema 无 `tool` 字段；无 2nd 路径。

- [ ] **Step 3: 改 `front_door.py`**

`core/workflow/front_door.py` 多处改动，逐个来。

**(3a) 顶部 import 加 `list_books_text`**：

在 `from core.workflow.summarizer import SUMMARY_MARKER` 之后加：

```python
from core.rag.inventory import list_books_text
```

**(3b) 加 `_COMPOSE_PROMPT` 常量**：

在 `_FRONT_DOOR_PROMPT` 定义之后（`_FRONT_DOOR_PROMPT = """..."""` 那个大字符串之后）加：

```python
# 2nd LLM：converse+tool 路径用工具结果组自然回复。非 json_object，自然文本。
_COMPOSE_PROMPT = """用户问了关于知识库藏书的问题。下面是系统从知识库元数据查到的真实结果。请据此用一句自然、面向用户的话回复，不要机械复述数据。

铁律：
- 只能基于下面的【库藏数据】答，不得编造未列出的书。
- 简短自然，别寒暄一堆。

用户问题：{query}

库藏数据：
{data}"""
```

**(3c) `_FRONT_DOOR_PROMPT` 加工具定义 + 红线 + JSON 示例字段**：

`_FRONT_DOOR_PROMPT` 里 converse 那段（当前是）：

```
- converse：寒暄/问候/致谢/闲聊、问你是谁或能做什么这类元问题，以及【对上一轮回答的反馈、质疑、不满、调侃】（如"你逗我呢""为什么答不了""不对吧"——参考对话历史里上一轮系统的回复来判断）。reply 放面向用户的自然回复；若上一轮是拒答/没答好而本轮是不满，先如实承认再引导。
```

改为（在末尾追加元工具说明）：

```
- converse：寒暄/问候/致谢/闲聊、问你是谁或能做什么这类元问题，以及【对上一轮回答的反馈、质疑、不满、调侃】（如"你逗我呢""为什么答不了""不对吧"——参考对话历史里上一轮系统的回复来判断）。reply 放面向用户的自然回复；若上一轮是拒答/没答好而本轮是不满，先如实承认再引导。
  【元工具（仅 converse 路径可用）】若本轮是关于知识库藏书的元查询（"库里有什么""有 MySQL 的书吗""多少本"等），设 tool="list_books" + tool_filter（书名子串，大小写不敏感，如"mysql"；无过滤留空）+ tool_count_only（只要计数时 true，列清单时 false），reply 留空（系统查库后另行组织）。纯寒暄/反馈/无需库藏数据时 tool="" 照常填 reply。
  【红线】tool 只能是 list_books。绝不可用于答书里的内容问题——内容问题一律 dispatch_qa 下沉检索。
```

同个 prompt 里的 JSON 示例行（当前是）：

```
{"action":"dispatch_qa / dispatch_study_plan / converse / clarify","clean_query":"净化后的自包含 query（dispatch 时填）","reply":"面向用户的话（converse/clarify 时填）","reason":"简短理由"}
```

改为：

```
{"action":"dispatch_qa / dispatch_study_plan / converse / clarify","clean_query":"净化后的自包含 query（dispatch 时填）","reply":"面向用户的话（converse/clarify 且无需工具时填）","reason":"简短理由","tool":"list_books 或空串（仅 converse 元查询时填 list_books）","tool_filter":"书名子串过滤（tool=list_books 时填，无过滤留空）","tool_count_only":false}
```

**(3d) `FrontDoorDecision` dataclass 加 3 字段**：

```python
@dataclass
class FrontDoorDecision:
    """门口产出：action 决定 dispatch；dispatch_* 带 clean_query，converse/clarify 带 reply。"""

    action: str
    clean_query: str = ""
    reply: str = ""
    reason: str = ""
```

改为：

```python
@dataclass
class FrontDoorDecision:
    """门口产出：action 决定 dispatch；dispatch_* 带 clean_query，converse/clarify 带 reply。

    converse+tool 时 reply 由系统查库 + 2nd LLM 组回复后填入。
    """

    action: str
    clean_query: str = ""
    reply: str = ""
    reason: str = ""
    tool: str = ""
    tool_filter: str = ""
    tool_count_only: bool = False
```

**(3e) `FrontDoorDecisionModel` 加 3 字段**：

```python
class FrontDoorDecisionModel(BaseModel):
    """LLM 判定的目标 schema（json_object 不保 schema，这步 Pydantic 校验才是约束）。

    action 用 Literal 锁枚举，非法值在 model_validate 阶段被拒、走降级。
    """

    action: Literal["dispatch_qa", "dispatch_study_plan", "converse", "clarify"]
    clean_query: str = Field(default="", description="dispatch_* 的自包含 query")
    reply: str = Field(default="", description="converse/clarify 面向用户的回复")
    reason: str = Field(default="", description="简短理由")
```

改为：

```python
class FrontDoorDecisionModel(BaseModel):
    """LLM 判定的目标 schema（json_object 不保 schema，这步 Pydantic 校验才是约束）。

    action 用 Literal 锁枚举，非法值在 model_validate 阶段被拒、走降级。
    tool 仅 converse 路径用，锁枚举 list_books / 空串；绝不可加 book_search（红线）。
    """

    action: Literal["dispatch_qa", "dispatch_study_plan", "converse", "clarify"]
    clean_query: str = Field(default="", description="dispatch_* 的自包含 query")
    reply: str = Field(default="", description="converse/clarify 面向用户的回复")
    reason: str = Field(default="", description="简短理由")
    tool: Literal["list_books", ""] = Field(default="", description="converse 元工具，仅 list_books")
    tool_filter: str = Field(default="", description="书名子串过滤，大小写不敏感")
    tool_count_only: bool = Field(default=False, description="只要计数时 true")
```

**(3f) `FrontDoorAgent.__init__` 注入 `index_manager`**：

```python
class FrontDoorAgent:
    """注入 LLM，对外只暴露一个 run。单次结构化决策，便于单测（mock LLM 控输出）。"""

    def __init__(self, llm: LLM):
        self.llm = llm
```

改为：

```python
class FrontDoorAgent:
    """注入 LLM + index_manager，对外只暴露一个 run。单次结构化决策 + converse 元工具路径。

    index_manager 供 converse+list_books 查库藏元数据；None 时元工具路径降级占位文本。
    """

    def __init__(self, llm: LLM, index_manager=None):
        self.llm = llm
        self.index_manager = index_manager
```

**(3g) `run` 方法体改**：

当前 `run` 方法（约第 130–164 行）整体替换为：

```python
    async def run(
        self,
        original: str,
        memory: Optional[ChatMemoryBuffer] = None,
        book_titles: Optional[list[str]] = None,
    ) -> FrontDoorDecision:
        history = format_history(memory)
        scope = format_scope(book_titles)
        prompt = (
            _FRONT_DOOR_PROMPT.replace("{query}", original)
            .replace("{history}", history)
            .replace("{scope}", scope)
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            d = FrontDoorDecisionModel.model_validate_json(text)
            if d.action in ("dispatch_qa", "dispatch_study_plan"):
                clean = (d.clean_query or original).strip() or original
                logger.info(
                    "front_door: action=%s clean_query=%r", d.action, clean[:80]
                )
                return FrontDoorDecision(d.action, clean_query=clean, reason=d.reason)
            if d.action == "clarify":
                reply = (d.reply or "").strip() or _FALLBACK_REPLY
                logger.info("front_door: action=clarify")
                return FrontDoorDecision("clarify", reply=reply, reason=d.reason)
            # converse
            if d.tool == "list_books":
                reply = await self._converse_with_tool(original, d)
                return FrontDoorDecision(
                    "converse", reply=reply, reason=d.reason,
                    tool=d.tool, tool_filter=d.tool_filter, tool_count_only=d.tool_count_only,
                )
            # converse 无 tool：reply 直接用（空 reply 兜底）
            reply = (d.reply or "").strip() or _FALLBACK_REPLY
            logger.info("front_door: action=converse")
            return FrontDoorDecision("converse", reply=reply, reason=d.reason)
        except Exception as exc:
            # 任何失败（空返回 / 非法 JSON / schema 不符 / 网络）→ 降级 dispatch_qa + 原 query，绝不阻塞
            logger.warning("front_door 解析失败，降级 dispatch_qa + 原 query：%s", exc)
            return FrontDoorDecision("dispatch_qa", clean_query=original)

    async def _converse_with_tool(self, original: str, d: FrontDoorDecisionModel) -> str:
        """converse + list_books：查库藏元数据 → 2nd LLM 组自然回复。

        工具失败 → 占位文本进 2nd；2nd 失败/空 → 裸 tool_result 当 reply。
        """
        try:
            tool_result = list_books_text(
                self.index_manager, d.tool_filter, d.tool_count_only
            )
        except Exception as exc:
            logger.warning("front_door list_books 查询失败，用占位文本：%s", exc)
            tool_result = "（未能读取库藏清单）"
        logger.info(
            "front_door: action=converse tool=list_books filter=%r count_only=%s",
            d.tool_filter, d.tool_count_only,
        )
        return await self._compose_tool_reply(original, tool_result)

    async def _compose_tool_reply(self, original: str, tool_result: str) -> str:
        """2nd LLM：用工具结果 + 原 query 组自然回复。失败降级裸 tool_result。"""
        prompt = (
            _COMPOSE_PROMPT.replace("{query}", original)
            .replace("{data}", tool_result)
        )
        try:
            resp = await self.llm.acomplete(prompt)   # 非 json_object，自然文本
            text = str(resp).strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("front_door compose reply 失败，用裸 tool_result：%s", exc)
        return tool_result
```

- [ ] **Step 4: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_front_door.py -v
```
Expected: 全部 PASS。新 tool 路径测试 + 现有 4 出口/净化/降级测试均绿。

- [ ] **Step 5: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿。`test_doc_workflow.py` 现有测试通过 `_wf(llm)` 构造（`FrontDoorAgent(llm)` 无 index_manager），本 Task `__init__` 的 `index_manager=None` 默认保持向后兼容，不破坏它们。

- [ ] **Step 6: 提交**

```powershell
git add core/workflow/front_door.py tests/test_front_door.py
git commit -m "feat: front_door converse path gains list_books meta-tool

converse 出口条件性调 list_books 元工具答库藏元查询（有什么/有 X 吗/
多少本）：1st LLM 决策 → 执行 list_books_text → 2nd LLM 组自然回复。
schema 加 tool/tool_filter/tool_count_only；__init__ 注入 index_manager；
红线：tool 只能是 list_books，book_search 绝不可加，内容问题一律 dispatch_qa。
降级：工具失败→占位文本；2nd 失败→裸 tool_result；1st 失败→dispatch_qa（现状）。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `DocQueryWorkflow` 接线 + 端到端

把 `index_manager` 传给 `FrontDoorAgent`，加端到端测试验证"库里都有什么"走 converse+tool、不进检索。

**Files:**
- Modify: `core/workflow/doc_workflow.py`（`FrontDoorAgent(llm)` → `FrontDoorAgent(llm, index_manager)`）
- Test: `tests/test_doc_workflow.py`（加端到端测试）

**Interfaces:**
- Consumes：Task 3 的 `FrontDoorAgent(llm, index_manager)`。
- Produces：workflow 对外行为——"库里都有什么"走 converse+tool、reply 含书名、不进检索、metadata.action=converse。

- [ ] **Step 1: 写失败测试**

在 `tests/test_doc_workflow.py` **末尾**追加：

```python
# ── front_door converse + list_books 端到端（Task 4）────────────────────


class _LibCollection:
    def __init__(self, metas):
        self._metas = metas

    def get(self, include=None):
        return {"metadatas": self._metas}


class _LibIndexManager:
    def __init__(self, metas):
        self.chroma_collection = _LibCollection(metas)
        self._metas = metas

    def get_index(self):
        class _Idx:
            def as_retriever(self, **kw):
                raise AssertionError("元查询不应检索")
        return _Idx()


async def test_library_listing_routes_to_converse_tool_without_retrieval():
    # "现在库里都有什么" → front_door converse+list_books → reply 含书名、不检索
    metas = [{"book_title": "高性能MySQL"}, {"book_title": "Redis"}]
    im = _LibIndexManager(metas)
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","reply":"","reason":"元查询"}',
        '已入库的有《高性能MySQL》和《Redis》两本。',
    ])
    wf = DocQueryWorkflow(index_manager=im, llm=llm, similarity_top_k=3, timeout=10)

    retrieve_called = {"v": False}

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        retrieve_called["v"] = True
        return "不应被调用", []

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="现在库里都有什么书", memory=FakeMemory())
    assert retrieve_called["v"] is False            # 元查询不检索
    assert "高性能MySQL" in str(result.response)
    assert "Redis" in str(result.response)
    assert result.source_nodes == []
    assert result.metadata.get("action") == "converse"
    assert llm.calls == 2                            # 1st 决策 + 2nd 组回复


async def test_library_count_question_routes_to_converse_tool_count_only():
    # "多少本" → converse+list_books+count_only → reply 含计数、不检索
    metas = [{"book_title": "甲"}, {"book_title": "乙"}, {"book_title": "丙"}]
    im = _LibIndexManager(metas)
    llm = FakeLLM([
        '{"action":"converse","tool":"list_books","tool_count_only":true,"reply":""}',
        '目前库里一共有 3 本书。',
    ])
    wf = DocQueryWorkflow(index_manager=im, llm=llm, similarity_top_k=3, timeout=10)

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        raise AssertionError("元查询不应检索")

    wf.qa.retrieve = fake_retrieve

    result = await wf.run(query="现在有多少本书", memory=FakeMemory())
    assert "3" in str(result.response)
    assert result.metadata.get("action") == "converse"
```

- [ ] **Step 2: 运行测试，确认失败**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_doc_workflow.py -k "library_listing or library_count" -v
```
Expected: FAIL——`FrontDoorAgent(llm)` 无 `index_manager`，`list_books_text(None, ...)` 会抛 `AttributeError`（`None.chroma_collection`）→ 占位文本 → 2nd 组回复不含书名 → 断言 `高性能MySQL in reply` 失败。

- [ ] **Step 3: 改 `doc_workflow.py` 接线**

`core/workflow/doc_workflow.py` 第 158 行（`DocQueryWorkflow.__init__` 里）：

```python
        self.front_door = FrontDoorAgent(llm)
```

改为：

```python
        self.front_door = FrontDoorAgent(llm, index_manager)
```

- [ ] **Step 4: 运行测试，确认通过**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest tests/test_doc_workflow.py -v
```
Expected: 全部 PASS。新端到端测试 + 现有测试均绿。

- [ ] **Step 5: 跑全量测试，确认无回归**

Run:
```powershell
$env:PYTHONUTF8=1; .venv\Scripts\python.exe -m pytest -x
```
Expected: 全绿。`scripts/check_layering.py` 守卫不受影响（`core/workflow → core/rag` 是允许的，`core/rag` 不依赖 `api`）。

- [ ] **Step 6: 提交**

```powershell
git add core/workflow/doc_workflow.py tests/test_doc_workflow.py
git commit -m "feat: wire index_manager into FrontDoorAgent for converse meta-tool

DocQueryWorkflow 把 index_manager 传给 FrontDoorAgent，启用 converse 路径
的 list_books 元工具。端到端：库里都有什么 / 多少本 走 converse+tool、
reply 含真实书名、不进检索、metadata.action=converse。

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

### 1. Spec coverage

| spec 条目 | 落在哪个 Task |
|---|---|
| `list_books_text` 共享函数（filter + count_only） | Task 1 |
| `ListBooksTool` thin wrapper | Task 2 |
| `FrontDoorAgent` 注入 `index_manager` | Task 3 |
| schema 加 tool/tool_filter/tool_count_only | Task 3 |
| prompt 加工具定义 + 红线 | Task 3 |
| `run()` converse+tool 2nd LLM 路径 | Task 3 |
| `_compose_tool_reply` + `_COMPOSE_PROMPT` | Task 3 |
| 降级：工具失败→占位、2nd 失败→裸 tool_result、1st 失败→dispatch_qa | Task 3 |
| `DocQueryWorkflow` 接线 | Task 4 |
| 端到端测试 | Task 4 |
| 回归 `test_book_tools.py` | Task 2 Step 2 |
| 红线：tool 只能 list_books、book_search 绝不可加 | Task 3（prompt + schema Literal） |
| clarify 不加 tool | Task 3（run 只在 converse 分支查 tool） |
| `ListBooksTool` 不加参数 | Task 2（`__call__` 仍无参） |

无遗漏。

### 2. Placeholder scan

- 无 "TBD" / "TODO" / "implement later"。
- 每个步骤含完整代码或精确"改为"前后对照。
- 测试代码可直接运行。

### 3. Type consistency

- `list_books_text(index_manager, filter="", count_only=False) -> str` —— Task 1 定义，Task 2 `ListBooksTool.__call__` 调 `list_books_text(self.ctx.index_manager)`，Task 3 `_converse_with_tool` 调 `list_books_text(self.index_manager, d.tool_filter, d.tool_count_only)`，签名一致。
- `FrontDoorAgent(llm, index_manager=None)` —— Task 3 定义，Task 4 `FrontDoorAgent(llm, index_manager)` 调用，一致。
- `FrontDoorDecisionModel.tool: Literal["list_books", ""]` —— Task 3 schema，run 里 `d.tool == "list_books"` 判断，一致。
- `FrontDoorDecision` dataclass 加 `tool/tool_filter/tool_count_only` —— Task 3，converse+tool 返回时填入，一致。

无类型/签名漂移。
