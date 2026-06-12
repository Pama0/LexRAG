# probe-then-classify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修 judge「盲判」根因——`classify` 前先用 clean_query 探测召回，把召回信号喂给 judge，让分类以「知识库实际有什么」为准而非「LLM 认不认识词」；并给 agent 加「先检索」铁律作防御纵深。

**Architecture:** `QaCapability.classify` 加一次 probe 宽召回（容错），`_format_probe` 把命中数+top截断片段+章节分布格式化喂给 `QueryPreprocessor.run(clean_query, retrieval_context)`；`_JUDGE_PROMPT` 加召回块 + 专名铁律。`doc_workflow.preprocess` 把 `book_titles` 传给 `classify`。`QaAgent` system prompt 加「先 book_search 再答、不准检索前猜测/反问」。

**Tech Stack:** Python 3.12，LlamaIndex，DeepSeek（json_object + Pydantic），pytest。

参考 spec：`docs/superpowers/specs/2026-06-12-probe-then-classify-design.md`

---

## File Structure

- **Modify** `core/workflow/query_preprocess.py` — `run` 加 `retrieval_context` 参数；`_JUDGE_PROMPT` 加召回块 + 铁律 + 各类看召回判据。
- **Modify** `tests/test_query_preprocess.py` — 追加 retrieval_context 注入测试；现有不传参测试保持绿。
- **Modify** `core/workflow/qa_capability.py` — `classify` 加 `book_titles` + probe + `_format_probe` + 容错。
- **Modify** `tests/test_qa_capability.py` — classify probe 接线 / 容错 / `_format_probe` 测试。
- **Modify** `core/workflow/doc_workflow.py` — `preprocess` step 传 `book_titles`。
- **Modify** `tests/test_doc_workflow.py` — book_titles 透传测试。
- **Modify** `core/agent/qa_agent.py` — system prompt 加「先检索」铁律。

---

### Task 1: QueryPreprocessor 带召回上下文判定

**Files:**
- Modify: `core/workflow/query_preprocess.py`
- Test: `tests/test_query_preprocess.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_query_preprocess.py` 末尾追加（复用已有 `FakeLLM` / `_pp`）:

```python
async def test_run_injects_retrieval_context_into_prompt():
    llm = FakeLLM(['{"category": "retrievable", "rewritten_query": "openclaw 是什么"}'])
    ctx = "共命中 8 段，分布：《openclaw》第3章\n1. [《openclaw》3.2] openclaw 是一个用于……"
    await _pp(llm).run("openclaw 是什么", retrieval_context=ctx)
    assert "知识库探测召回" in llm.prompts[0]      # 召回块进了 prompt
    assert "《openclaw》第3章" in llm.prompts[0]   # 召回内容进了 prompt


async def test_run_without_retrieval_context_still_works():
    # 向后兼容：不传 retrieval_context 仍可判定（probe 失败时退回纯文本）
    llm = FakeLLM(['{"category": "retrievable", "rewritten_query": "MySQL锁"}'])
    result = await _pp(llm).run("MySQL锁")
    assert result.category == "retrievable"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_preprocess.py -k retrieval_context -q`
Expected: FAIL（`run()` 不接受 `retrieval_context` / prompt 无"知识库探测召回"）

- [ ] **Step 3: 改 query_preprocess.py**

(a) 把 `_JUDGE_PROMPT` 中这段（"第二步"那行到 retrievable 返回）:

```
第二步 判定该 query 能否直接进入检索（基于降噪后的 query）：
【可以】可确定指向具体的技术概念/章节/问题，能检索到精准、集中的内容。
特征：仅检索问题即可
返回 {"category":"retrievable","rewritten_query": "处理后的 query"}
```

替换为:

```
第二步 判定该 query 能否直接进入检索（基于降噪后的 query）。

系统已用该 query 在知识库做了一次探测检索：
【知识库探测召回】
{retrieval}

【铁律·必读】判定必须以上面的【知识库探测召回】为准，【绝不以你是否认识问题中的词为准】。知识库里全是你训练时没见过的专有名词（书名/工具名/项目名），其含义由检索决定、不由你的世界知识判断。绝不要因为"我不认识这个词"或"这个词可能多义/很难"就判 missing_info 或 other——只要召回到与问题相关且集中的内容，就是 retrievable。

【可以】retrievable：召回片段与问题相关、且集中指向单一概念/章节。
特征：仅检索问题即可
返回 {"category":"retrievable","rewritten_query": "处理后的 query"}
```

(b) 把 missing_info 那段:

```
- missing_info（信息不足）：缺了检索必需的关键限定，根本无法检索（多为指代不明且历史里也无从补全）。
  如「这个索引的应用场景是什么」——"这个索引"指代不明（全文索引？B+树索引？其他？）
```

替换为:

```
- missing_info（信息不足）：缺了检索必需的关键限定，根本无法检索；**且探测召回为空、或片段明显与问题无关**（知识库里确实没有相关内容）。多为指代不明且历史无从补全。
  如「这个索引的应用场景是什么」——"这个索引"指代不明（全文索引？B+树索引？其他？）
  注意：若召回到了相关内容，即便问题里有你不认识的专名，也不是 missing_info。
```

(c) 把 other 那段:

```
- other（高难度/开放复杂问题）：需要【跨多个主题综合、多步推理，或开放设计/权衡比较】，单轮检索难以一次答全，更适合多轮检索+推理逐步求解。
  特征：要综合多处证据、需要分析取舍、或答案随视角展开（如「综合评价 X 的架构取舍」「结合书里多个概念设计一套方案」）。
  倾向（积极）：当问题明显偏复杂综合、又不属于前三类（缺信息/角度不定/单纯并列罗列）时，判为 other 交由 agent 多轮处理；仅当问题其实能单轮检索集中命中时才回到 retrievable。
```

替换为:

```
- other（高难度/开放复杂问题）：**召回到了相关内容，但**需要【跨多个主题综合、多步推理，或开放设计/权衡比较】，单轮检索难以一次答全，更适合多轮检索+推理逐步求解。
  特征：要综合多处证据、需要分析取舍、或答案随视角展开（如「综合评价 X 的架构取舍」「结合书里多个概念设计一套方案」）。
  铁律：other 看的是【问题结构是否需多步综合】，不是【你认不认识其中的词】。「X是什么 / 讲讲X / 讲明白X」这类即便 X 是你不认识的专名，只要召回到相关内容，就归 retrievable（单一概念）或 pending_split（X 是大主题需罗列），**绝不因不认识 X 而判 other**。
```

(d) 把 `run` 方法签名与 prompt 构造:

```python
    async def run(self, clean_query: str) -> PreprocessResult:
        prompt = _JUDGE_PROMPT.replace("{query}", clean_query)
```

替换为:

```python
    async def run(
        self, clean_query: str, retrieval_context: str = ""
    ) -> PreprocessResult:
        prompt = (
            _JUDGE_PROMPT.replace("{query}", clean_query)
            .replace(
                "{retrieval}",
                retrieval_context or "（系统未能探测知识库，请仅依据问题文本判定）",
            )
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_query_preprocess.py -q`
Expected: 全部 passed（原有 + 新增 2）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/query_preprocess.py tests/test_query_preprocess.py
git commit -m "feat(workflow): QueryPreprocessor 带知识库探测召回判定（堵 judge 盲判）"
```

---

### Task 2: QaCapability.classify 加探测召回

**Files:**
- Modify: `core/workflow/qa_capability.py`
- Test: `tests/test_qa_capability.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_qa_capability.py` 末尾追加（复用已有 `_qa` / `FakeIndexManager` / `FakeCtx`）:

```python
# ── classify：probe-then-classify ───────────────────────────────────
class _PNode:
    def __init__(self, content, book="openclaw", chapter="3.2"):
        self._c = content
        self.metadata = {"book_title": book, "chapter": chapter}

    def get_content(self):
        return self._c


async def test_classify_probes_then_passes_context_to_preprocessor():
    qa = _qa(FakeIndexManager(nodes=[_PNode("openclaw 是一个工具")]))
    captured = {}

    async def fake_run(clean_query, retrieval_context=""):
        captured["ctx"] = retrieval_context
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable", clean_query)

    qa.preprocessor.run = fake_run
    await qa.classify("给我讲明白openclaw", ["openclaw"])
    assert "openclaw 是一个工具" in captured["ctx"]   # 探测片段进了召回上下文
    assert "《openclaw》" in captured["ctx"]           # 章节分布进了上下文


async def test_classify_degrades_when_probe_fails():
    qa = _qa(index_manager=None)   # 无 index → probe 抛错
    captured = {}

    async def fake_run(clean_query, retrieval_context=""):
        captured["ctx"] = retrieval_context
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable", clean_query)

    qa.preprocessor.run = fake_run
    result = await qa.classify("openclaw", ["openclaw"])
    assert captured["ctx"] == ""           # probe 失败 → 空上下文，不阻塞
    assert result.category == "retrievable"


def test_format_probe_empty_and_nonempty():
    qa = _qa()
    assert "未召回" in qa._format_probe([], None)
    out = qa._format_probe([_PNode("片段X", book="A", chapter="1.1")], None)
    assert "共命中 1 段" in out and "《A》1.1" in out and "片段X" in out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_capability.py -k "classify or format_probe" -q`
Expected: FAIL（`classify` 不接受 book_titles / 无 `_format_probe`）

- [ ] **Step 3: 改 qa_capability.py**

把现有 `classify`:

```python
    # ── 预处理：降噪 + 难度/明确性分类（不再消指代）──────────────────
    async def classify(self, clean_query: str):
        """对 clean_query 做降噪 + 分类，返回 PreprocessResult（category/rewritten/reason）。"""
        return await self.preprocessor.run(clean_query)
```

替换为:

```python
    # ── 预处理：probe-then-classify（先探测召回，再带召回判定）──────────
    async def classify(
        self, clean_query: str, book_titles: Optional[list[str]] = None
    ):
        """先用 clean_query 探测召回，把召回信号喂给 judge，堵住「盲判」。

        probe 失败（index 空/异常）→ 容错为空上下文，judge 退回纯文本判定，不阻塞。
        """
        retrieval_context = ""
        try:
            probe = await self._retrieve_nodes(clean_query, book_titles)
            retrieval_context = self._format_probe(probe, book_titles)
        except Exception as exc:
            logger.warning("classify probe 探测失败，退回纯文本判定：%s", exc)
        return await self.preprocessor.run(clean_query, retrieval_context)

    def _format_probe(self, nodes: list, book_titles) -> str:
        """探测召回 → 喂 judge 的信号：命中数 + 章节分布 + top 截断片段。"""
        if not nodes:
            return "知识库未召回到任何相关内容。"
        dist: list[str] = []
        seen: set = set()
        for n in nodes:
            meta = getattr(n, "metadata", None) or {}
            tag = f"《{meta.get('book_title', '?')}》{meta.get('chapter', '')}".strip()
            if tag not in seen:
                seen.add(tag)
                dist.append(tag)
        lines: list[str] = []
        for i, n in enumerate(nodes[:5], 1):
            meta = getattr(n, "metadata", None) or {}
            tag = f"《{meta.get('book_title', '?')}》{meta.get('chapter', '')}".strip()
            content = (
                n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            )[:150]
            lines.append(f"{i}. [{tag}] {content}")
        return f"共命中 {len(nodes)} 段，分布：{'、'.join(dist)}\n" + "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_qa_capability.py -q`
Expected: 全部 passed（原有 + 新增 3）

- [ ] **Step 5: Commit**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(workflow): classify 先探测召回再判定（probe-then-classify）"
```

---

### Task 3: doc_workflow 传 book_titles + agent 防御纵深

**Files:**
- Modify: `core/workflow/doc_workflow.py`, `core/agent/qa_agent.py`
- Test: `tests/test_doc_workflow.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_doc_workflow.py` 末尾追加:

```python
async def test_preprocess_passes_book_titles_to_classify():
    llm = FakeLLM(['{"intent": "qa", "clean_query": "openclaw是什么"}'])  # 仅 Router 调 LLM
    wf = _wf(llm)

    captured = {}

    async def fake_classify(clean_query, book_titles=None):
        captured["clean"] = clean_query
        captured["books"] = book_titles
        from core.workflow.query_preprocess import PreprocessResult
        return PreprocessResult("retrievable", clean_query)

    wf.qa.classify = fake_classify

    async def fake_retrieve(ctx, query, book_titles, preamble=""):
        return "答案", []

    wf.qa.retrieve = fake_retrieve

    await wf.run(query="openclaw是什么", memory=FakeMemory(), book_titles=["openclaw"])
    assert captured["clean"] == "openclaw是什么"
    assert captured["books"] == ["openclaw"]   # scope 透传到 probe
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_doc_workflow.py -k passes_book_titles -q`
Expected: FAIL（`classify` 收到的 book_titles 为 None，因 preprocess 未传）

- [ ] **Step 3: 改 doc_workflow.py**

把 `preprocess` step 的开头:

```python
        clean_query = await ctx.store.get("clean_query")

        result = await self.qa.classify(clean_query)
```

替换为:

```python
        clean_query = await ctx.store.get("clean_query")
        book_titles = await ctx.store.get("book_titles")

        result = await self.qa.classify(clean_query, book_titles)
```

- [ ] **Step 4: 改 qa_agent.py（防御纵深）**

把 `QA_AGENT_SYSTEM_PROMPT` 的铁律段:

```
铁律（grounding）：
- 只能基于 book_search 返回的检索片段作答，严禁用你自己的训练知识或常识脑补事实。
```

替换为:

```
铁律（grounding）：
- 【先检索再答】拿到问题必须先调用 book_search 检索，严禁在检索前就用训练知识猜测含义、或直接反问用户"你是指什么"。即便问题里有你不认识的词，也要先检索——它很可能就是知识库里的专有名词。
- 只能基于 book_search 返回的检索片段作答，严禁用你自己的训练知识或常识脑补事实。
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_doc_workflow.py -q`
Expected: 全部 passed（原有 + 新增 1）。原有测试 index_manager=None → classify probe 失败容错 → 不破。

- [ ] **Step 6: Commit**

```bash
git add core/workflow/doc_workflow.py core/agent/qa_agent.py tests/test_doc_workflow.py
git commit -m "feat(workflow): preprocess 传 book_titles 探测 + agent 先检索铁律（防御纵深）"
```

---

### Task 4: 复现验证 + 全量回归

**Files:** 无（验证 only；复现脚本一次性，不入库）

- [ ] **Step 1: 编译 + 全量回归**

Run: `.venv/Scripts/python.exe -m py_compile core/workflow/query_preprocess.py core/workflow/qa_capability.py core/workflow/doc_workflow.py core/agent/qa_agent.py`
Run: `.venv/Scripts/python.exe -m pytest -q --continue-on-collection-errors`
Expected: 新增测试全 passed；仅遗留 `test_book_rag_workflow.py` / `test_book_search_tool.py` 2 errors（范围外）。

- [ ] **Step 2: 分层守卫**

Run: `.venv/Scripts/python.exe scripts/check_layering.py`
Expected: 通过。

- [ ] **Step 3: 复现验证（关键——LLM 行为修复无法用单测确定性验证）**

前提：真实环境（chroma 已入库 openclaw 那本书 + `DEEPSEEK_API_KEY`）。用项目既有方式构造 `index_manager`（参考 `api/main.py` / `main.py` 的装配），跑：

```python
import asyncio
from configs.llm import configure_llm
# 用项目既有装配构造 index_manager（含已入库的 openclaw）
from core.workflow.qa_capability import QaCapability

llm = configure_llm()
index_manager = ...  # 项目既有构造
qa = QaCapability(index_manager, llm)

async def main():
    for q in ["给我讲明白openclaw", "openclaw是什么", "讲讲MySQL"]:
        r = await qa.classify(q, None)   # None=全部书
        print(q, "->", r.category, r.reason)

asyncio.run(main())
```

Expected:
- `给我讲明白openclaw` → **不再是 `other`**（应为 `retrievable` 或 `pending_split`，因 probe 召回到 openclaw 内容）—— **修复成功判据**。
- `openclaw是什么` → 仍 `retrievable`（不退化）。
- `讲讲MySQL` → 仍 `pending_split`（不退化）。

若仍判 `other`：检查 probe 是否真召回到 openclaw 内容（`_format_probe` 输出），及 prompt 召回块是否生效；不要盲目改 prompt，回到证据。

- [ ] **Step 4: Commit（如有验证性微调）**

```bash
git add -A
git commit -m "test(workflow): probe-then-classify 复现验证 + 全量回归"
```

---

## Self-Review Notes

- **Spec coverage:** probe-then-classify → Task 2（classify+probe）；召回粒度（命中数+top5截断150字+章节分布，不给分数）→ Task 2 `_format_probe`；judge 带召回判定 + 专名铁律 → Task 1（prompt+run）；book_titles 透传 → Task 3；防御纵深 agent 先检索 → Task 3 Step 4；probe 容错降级 → Task 2 try/except；向后兼容（retrieval_context 默认空）→ Task 1；验证策略（单测接线 + 真库复现）→ Task 2/3 单测 + Task 4 复现。复用不做（§4）→ 未触碰，spec 已记录未来路径。
- **Type consistency:** `QueryPreprocessor.run(clean_query, retrieval_context="") -> PreprocessResult`；`QaCapability.classify(clean_query, book_titles=None)`；`_format_probe(nodes, book_titles) -> str`；`doc_workflow.preprocess` 传 `book_titles`。各 Task 一致。
- **No placeholders:** 步骤含完整代码与命令（复现脚本的 index_manager 构造依赖真实装配，已标注"用项目既有方式"——这是环境依赖非占位）。
- **风险点:** ① 现有 doc_workflow 测试 index_manager=None → classify probe 抛错被容错（retrieval_context=""），不破——Task 2 容错 + Task 3 Step 5 已覆盖；② `_format_probe` 读 `n.metadata`（NodeWithScore 代理 node.metadata，与 split/assume 一致），测试用鸭子替身；③ 修复是 LLM 行为，单测只保接线，真实判定质量靠 Task 4 复现 + 后续评测，prompt 文案可能需按复现结果微调（仍以证据为准，不盲改）。
