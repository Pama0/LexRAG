# pending_split 分流（罗列 vs 综合）+ 并行检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `pending_split` 按「罗列 vs 综合」分流——综合型走真整合合成修正碎片化答案；并把扇出检索改并行。

**Architecture:** 判型放 `QueryDecomposer`（`run` 返回 `(sub_queries, mode)`）。`QaCapability.split()` 按 mode 分流：`list` 走改名后的 `_retrieve_and_concat`（现状裸拼），`synthesize` 走新增 `_retrieve_and_synthesize`（扇出检索 → 去重合并 → 一次整合合成）。两模式的扇出检索经共享 `_retrieve_all`（`asyncio.gather` + 信号量）并行。`assume` 共用 `_retrieve_and_concat`，白拿并行检索。

**Tech Stack:** Python 3.12 async、llama-index、pytest（`pytest-asyncio` 已配，测试用裸 `async def`）。

## Global Constraints

- 所有 I/O 用 `async/await`；函数签名带类型注解（CLAUDE.md）。
- 子模块内用相对导入；从项目根目录运行测试。
- 任何 LLM/检索异常都倒向现状最安全行为，绝不阻塞。
- 范围仅 `split`；不改门口分类、不改 workflow step 图、不动 eval SUT 超时。
- 开发阶段唯一验证手段＝单元测试；eval 端到端延后。
- 综合池上限常量复用 `self.rerank_candidate_k`（默认 20）。
- 检索并发上限常量 `_retrieve_concurrency = 4`。
- 运行测试：`python -m pytest <path> -v`（项目根目录）。

---

### Task 1: QueryDecomposer 升为「拆解 + 判型」，`run` 返回 `(sub_queries, mode)`

**Files:**
- Modify: `core/workflow/query_decompose.py`（`_DECOMPOSE_PROMPT`、`Decomposition`、`QueryDecomposer.run`）
- Modify: `core/workflow/qa_capability.py`（`split()` 解包 decomposer 返回值，本任务先忽略 mode）
- Test: `tests/test_query_decompose.py`（改现有断言为解包 + 加 mode 用例）
- Test: `tests/test_qa_capability.py`（split 桩 `decomposer.run` 改返回元组）

**Interfaces:**
- Produces: `QueryDecomposer.run(clean_query: str, headings: list[str], passages: list[str], max_items: int = 6) -> tuple[list[str], str]`，`mode ∈ {"list","synthesize"}`，失败返回 `([], "list")`。

- [ ] **Step 1: 改现有 decompose 测试为解包 + 加 mode 用例（先红）**

把 `tests/test_query_decompose.py` 中所有 `subs = await ...run(...)` 改为 `subs, mode = await ...run(...)`，并加新用例：

```python
async def test_run_returns_mode_list():
    llm = FakeLLM(['{"sub_queries": ["a", "b"], "mode": "list"}'])
    subs, mode = await _dec(llm).run("q", [], ["p"])
    assert subs == ["a", "b"]
    assert mode == "list"


async def test_run_returns_mode_synthesize():
    llm = FakeLLM(['{"sub_queries": ["a", "b"], "mode": "synthesize"}'])
    subs, mode = await _dec(llm).run("q", [], ["p"])
    assert mode == "synthesize"


async def test_run_missing_mode_defaults_to_list():
    llm = FakeLLM(['{"sub_queries": ["a"]}'])
    subs, mode = await _dec(llm).run("q", [], ["p"])
    assert subs == ["a"] and mode == "list"


async def test_run_invalid_mode_falls_back_to_list():
    llm = FakeLLM(['{"sub_queries": ["a"], "mode": "foo"}'])
    subs, mode = await _dec(llm).run("q", [], ["p"])
    assert subs == ["a"] and mode == "list"


async def test_run_returns_empty_list_mode_on_parse_failure():
    subs, mode = await _dec(FakeLLM(["这不是JSON"])).run("q", [], ["p"])
    assert subs == [] and mode == "list"
```

现有用例改解包（示例）：

```python
async def test_run_parses_sub_queries():
    llm = FakeLLM(['{"sub_queries": ["工具A 是什么", "工具B 怎么用"]}'])
    subs, mode = await _dec(llm).run("openclaw 的工具系统", ["3.2.1 工具A", "3.2.2 工具B"], ["正文片段"])
    assert subs == ["工具A 是什么", "工具B 怎么用"]
```

（`test_run_caps_at_max_items` / `test_run_drops_blank_sub_queries` / `test_run_returns_empty_on_parse_failure` / `test_run_returns_empty_on_empty_content` / `test_run_prompt_includes_headings_and_passages` 同样把 `subs = ` 改成 `subs, mode = `。）

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_query_decompose.py -v`
Expected: FAIL（`run` 仍返回 list，解包成 `subs, mode` 报 `ValueError: too many values to unpack` / mode 断言失败）。

- [ ] **Step 3: 改 `Decomposition` schema + prompt + `run`**

`core/workflow/query_decompose.py`，schema 加 `mode`（用宽松 `str` + 默认，避免坏 mode 拖垮整段解析）：

```python
class Decomposition(BaseModel):
    """LLM 拆解结果的目标 schema（代码侧 Pydantic 校验）。"""

    sub_queries: List[str] = Field(default_factory=list)
    mode: str = "list"  # list=各子项独立罗列；synthesize=需跨子项整合。非法值代码侧归一
```

`_DECOMPOSE_PROMPT` 末尾（「只返回 JSON」之前）插入判型说明，并把示例 JSON 加上 mode：

```python
_DECOMPOSE_PROMPT = """你是检索 query 拆解器。下面给出一个较宽的问题，以及与它相关的
【章节标题】和【召回正文片段】。请【只依据给定素材】把问题拆成若干并列的子查询，
每个子查询聚焦一个具体子项/小节/对比维度，便于逐个检索。

铁律：
- 子查询只能来自给定的章节标题或召回正文里真实出现的内容，严禁编造素材里没有的实体。
- 若问题是"对比/区别"，子查询应是各对比维度（如"X 与 Y 在适用场景上的区别"）。
- 子查询数量不超过 {max} 个；素材子项更多时，归并或取最重要的若干个。
- 每个子查询是能独立检索的完整短句。

此外判断问题类型，写入 mode 字段：
- "list"：各子项答案彼此独立、各自成段即完整（如「分别/各自」说明各自功能）。
- "synthesize"：答案必须跨子项推理——比较、讲关系、谈协作取舍、合成单一概念。
准绳：每个子项的答案能否单独成立？能→list；必须放一起才说得清→synthesize。拿不准选 list。

问题：{query}

章节标题：
{headings}

召回正文片段：
{passages}

只返回 JSON，不要其他任何内容：
{"sub_queries": ["子查询1", "子查询2", ...], "mode": "list"}"""
```

`run` 返回签名与归一逻辑：

```python
    async def run(
        self,
        clean_query: str,
        headings: List[str],
        passages: List[str],
        max_items: int = 6,
    ) -> tuple[List[str], str]:
        prompt = (
            _DECOMPOSE_PROMPT.replace("{query}", clean_query)
            .replace("{headings}", "\n".join(f"- {h}" for h in headings) or "（无）")
            .replace("{passages}", "\n---\n".join(passages) or "（无）")
            .replace("{max}", str(max_items))
        )
        try:
            resp = await self.llm.acomplete(
                prompt, response_format={"type": "json_object"}
            )
            text = _strip_fences(str(resp)).strip()
            if not text:
                raise ValueError("empty content")
            data = Decomposition.model_validate_json(text)
            subs = [s.strip() for s in data.sub_queries if s and s.strip()]
            mode = data.mode if data.mode in ("list", "synthesize") else "list"
            logger.info("decompose: 拆出 %d 个子查询 mode=%s", len(subs[:max_items]), mode)
            return subs[:max_items], mode
        except Exception as exc:
            logger.warning("decompose 失败，返回空（split 将降级单轮）：%s", exc)
            return [], "list"
```

- [ ] **Step 4: 改 `split()` 调用点解包（忽略 mode）+ qa 测试桩返回元组**

`core/workflow/qa_capability.py` `split()` 内：

```python
        sub_queries, _mode = await self.decomposer.run(
            query, headings, passages, self.max_sub_queries
        )
```

`tests/test_qa_capability.py` 中所有 `qa.decomposer.run` 桩改返回元组：

```python
    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["工具A 是什么", "工具B 怎么用"], "list"
```

```python
    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["子项1", "子项2"], "list"
```

```python
    async def empty_decompose(clean_query, headings, passages, max_items):
        return [], "list"
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_query_decompose.py tests/test_qa_capability.py -v`
Expected: PASS（含新增 mode 用例；split/assume 行为不变）。

- [ ] **Step 6: Commit**

```bash
git add core/workflow/query_decompose.py core/workflow/qa_capability.py tests/test_query_decompose.py tests/test_qa_capability.py
git commit -m "feat(decompose): run 返回 (sub_queries, mode)，judge 罗列 vs 综合"
```

---

### Task 2: 并行扇出检索 `_retrieve_all` + helper 改名 `_retrieve_and_concat`

**Files:**
- Modify: `core/workflow/qa_capability.py`（`__init__` 加并发常量；新增 `_retrieve_all`；`_retrieve_and_reduce`→`_retrieve_and_concat` 用 `_retrieve_all`；`assume`/`split` 调用点改名）
- Test: `tests/test_qa_capability.py`（加并发用例；现有 concat/assume 用例保持绿）

**Interfaces:**
- Consumes: `QueryDecomposer.run -> (list[str], str)`（Task 1）。
- Produces:
  - `QaCapability._retrieve_all(sub_queries: list[str], book_titles) -> list[list]`（顺序与入参一致，内部并发）。
  - `QaCapability._retrieve_and_concat(ctx, sections, book_titles, preamble="") -> tuple[str, list]`（原 `_retrieve_and_reduce`）。

- [ ] **Step 1: 写并发检索失败测试（先红）**

`tests/test_qa_capability.py` 末尾追加：

```python
import asyncio


async def test_retrieve_all_runs_concurrently_and_preserves_order():
    qa = _qa()
    started = 0
    release = asyncio.Event()

    async def fake_rn(query, book_titles):
        nonlocal started
        started += 1
        if started >= 2:
            release.set()      # 两个都进来才放行 → 串行会卡死，证明并发
        await release.wait()
        return [query]

    qa._retrieve_nodes = fake_rn
    out = await asyncio.wait_for(qa._retrieve_all(["a", "b"], None), timeout=1.0)
    assert out == [["a"], ["b"]]   # gather 保持入参顺序
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_qa_capability.py::test_retrieve_all_runs_concurrently_and_preserves_order -v`
Expected: FAIL（`AttributeError: '_retrieve_all'`）。

- [ ] **Step 3: 加并发常量 + `_retrieve_all` + 改名 helper + 改调用点**

`core/workflow/qa_capability.py` `__init__` 末尾加常量：

```python
        self._retrieve_concurrency = 4  # 扇出检索并发上限，防 embedding/BM25/rerank 打爆
```

`import asyncio`（文件顶部）。新增 `_retrieve_all`（放在 `_retrieve_nodes` 附近）：

```python
    async def _retrieve_all(self, sub_queries: list[str], book_titles) -> list[list]:
        """并发扇出检索：对每个子查询各检索一次，返回与入参同序的 node 列表的列表。"""
        sem = asyncio.Semaphore(self._retrieve_concurrency)

        async def _one(q: str):
            async with sem:
                return await self._retrieve_nodes(q, book_titles)

        return await asyncio.gather(*(_one(q) for q in sub_queries))
```

把 `_retrieve_and_reduce` 改名为 `_retrieve_and_concat`，检索段改用 `_retrieve_all`：

```python
    async def _retrieve_and_concat(
        self,
        ctx: Context,
        sections: list[tuple[str, str]],
        book_titles: Optional[list[str]],
        preamble: str = "",
    ) -> tuple[str, list]:
        """sections: [(分节标题, 检索/合成用子查询)]。list 模式：逐节裸拼（split / assume 共用）。

        - 扇出检索并发；逐节合成仍串行（保分节流式顺序）。
        - 先全检索（只发一次 RetrievalDone）。preamble 非空 → 答案阶段先推一个 AnswerDeltaEvent。
        """
        retrieved = await self._retrieve_all([sq for _h, sq in sections], book_titles)
        all_nodes: list = [n for ns in retrieved for n in ns]
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(all_nodes)))

        parts: list[str] = []
        if preamble:
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=preamble))
            parts.append(preamble)
        for (heading, sub_query), ns in zip(sections, retrieved):
            h = f"\n## {heading}\n"
            ctx.write_event_to_stream(AnswerDeltaEvent(delta=h))
            body = (
                await self._synthesize_stream(ctx, sub_query, ns)
                if ns
                else "（未检索到相关内容）"
            )
            parts.append(h + body)
        return "".join(parts).strip(), all_nodes
```

改两处调用点：`split()` 末尾 `return await self._retrieve_and_reduce(...)` → `_retrieve_and_concat`；`assume()` 末尾同样改名。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_qa_capability.py -v`
Expected: PASS（新并发用例 + 全部现有 split/assume/retrieve 用例绿）。

- [ ] **Step 5: Commit**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(qa): 扇出检索并行 _retrieve_all；_retrieve_and_reduce 改名 _retrieve_and_concat"
```

---

### Task 3: synthesize 模式 `_retrieve_and_synthesize` + split 分流

**Files:**
- Modify: `core/workflow/qa_capability.py`（`_node_id`/`_merge_pool`/`_retrieve_and_synthesize`；`split()` 按 mode 分流）
- Test: `tests/test_qa_capability.py`（synthesize 路由 / 单次合成 / 去重 / 原始问题 / 空池 / 单子查询降级）

**Interfaces:**
- Consumes: `_retrieve_all`、`_synthesize_stream`、`self.reranker`、`self.rerank_candidate_k`、`QueryDecomposer.run -> (list, mode)`。
- Produces: `QaCapability._retrieve_and_synthesize(ctx, original_query: str, sub_queries: list[str], book_titles) -> tuple[str, list]`。

- [ ] **Step 1: 写 synthesize 测试（先红）**

`tests/test_qa_capability.py` 追加。先加一个带 id/score 的桩节点与一个记录合成调用的 split 夹具：

```python
class _IdNode:
    def __init__(self, nid, content="正文", score=1.0):
        self.node_id = nid
        self._c = content
        self.score = score

    def get_content(self):
        return self._c


def _synth_qa(retrieve_map):
    """retrieve_map: {子查询: [节点...]}；记录 _synthesize_stream 的调用。"""
    qa = _qa()
    qa._book_chapters = lambda book_titles: []

    async def fake_retrieve_nodes(query, book_titles):
        return retrieve_map.get(query, [])

    qa._retrieve_nodes = fake_retrieve_nodes

    calls = []

    async def fake_synth(ctx, query, nodes):
        calls.append((query, list(nodes)))
        return f"[整合:{query}]"

    qa._synthesize_stream = fake_synth
    qa._synth_calls = calls
    return qa


async def test_split_synthesize_single_synthesis_over_merged_pool():
    a, b = _IdNode("1"), _IdNode("2")
    qa = _synth_qa({"locate": [a], "子查询A": [a], "子查询B": [b]})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["子查询A", "子查询B"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()  # 定位整句检索走 retrieve_map["locate"]=[a]

    answer, nodes = await qa.split(ctx, "locate", ["书"])

    # 只合成一次，且用原始问题、池含两子查询去重后的节点
    assert len(qa._synth_calls) == 1
    synth_query, synth_nodes = qa._synth_calls[0]
    assert synth_query == "locate"
    assert {n.node_id for n in synth_nodes} == {"1", "2"}
    assert "##" not in answer            # 单段连贯，无分节标题
    assert answer == "[整合:locate]"


async def test_split_synthesize_dedupes_overlapping_nodes():
    shared = _IdNode("dup")
    qa = _synth_qa({"locate": [shared], "qa": [shared], "qb": [shared, _IdNode("x")]})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["qa", "qb"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    await qa.split(ctx, "locate", ["书"])
    _q, synth_nodes = qa._synth_calls[0]
    assert [n.node_id for n in synth_nodes] == ["dup", "x"]   # 按 node_id 去重，保序


async def test_split_synthesize_emits_single_retrieval_done():
    qa = _synth_qa({"locate": [_IdNode("1")], "qa": [_IdNode("1")], "qb": [_IdNode("2")]})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["qa", "qb"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    await qa.split(ctx, "locate", ["书"])
    names = [e.__class__.__name__ for e in ctx.events]
    assert names.count("RetrievalDoneEvent") == 1


async def test_split_synthesize_empty_pool_returns_scope_hint():
    qa = _synth_qa({"locate": [_IdNode("1")], "qa": [], "qb": []})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["qa", "qb"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    answer, nodes = await qa.split(ctx, "locate", ["某本书"])
    assert nodes == []
    assert "某本书" in answer and "没有检索到" in answer


async def test_split_synthesize_single_subquery_degrades_to_single():
    only = _IdNode("only")
    qa = _synth_qa({"locate": [only], "唯一子查询": [only]})

    async def fake_decompose(clean_query, headings, passages, max_items):
        return ["唯一子查询"], "synthesize"

    qa.decomposer.run = fake_decompose
    ctx = FakeCtx()

    answer, nodes = await qa.split(ctx, "locate", ["书"])
    assert len(qa._synth_calls) == 1
    assert qa._synth_calls[0][0] == "locate"   # 仍对原始问题单次合成
    assert "##" not in answer
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_qa_capability.py -k synthesize -v`
Expected: FAIL（`split` 尚未按 mode 分流，synthesize 仍走 concat，断言 `##`/单次合成失败）。

- [ ] **Step 3: 实现 `_node_id`/`_merge_pool`/`_retrieve_and_synthesize` + split 分流**

`core/workflow/qa_capability.py` 加去重与综合 helper（放 `_retrieve_and_concat` 之后）：

```python
    @staticmethod
    def _node_id(n) -> object:
        """稳定去重键：优先 NodeWithScore.node.node_id，退回 node_id，再退回对象 id。"""
        node = getattr(n, "node", None)
        return getattr(node, "node_id", None) or getattr(n, "node_id", None) or id(n)

    def _merge_pool(self, lists: list[list]) -> list:
        """多路检索结果按 node_id 去重合并，保首次出现顺序。"""
        seen: set = set()
        out: list = []
        for ns in lists:
            for n in ns:
                k = self._node_id(n)
                if k in seen:
                    continue
                seen.add(k)
                out.append(n)
        return out

    async def _retrieve_and_synthesize(
        self,
        ctx: Context,
        original_query: str,
        sub_queries: list[str],
        book_titles: Optional[list[str]],
    ) -> tuple[str, list]:
        """synthesize 模式：扇出检索（并发）→ 去重合并 → 对原始问题一次整合合成。

        子查询只为拓宽召回面；合成用【原始问题】，让 LLM 同时看到所有子项原始片段去比较/讲关系。
        """
        retrieved = await self._retrieve_all(sub_queries, book_titles)
        pool = self._merge_pool(retrieved)
        if self.reranker:
            # 拿原始问题（非子查询）对合并池重排，截到上下文预算
            pool = await self.reranker.rerank(original_query, pool, self.rerank_candidate_k)
        else:
            pool = sorted(pool, key=lambda n: getattr(n, "score", 0) or 0, reverse=True)[
                : self.rerank_candidate_k
            ]
        ctx.write_event_to_stream(RetrievalDoneEvent(count=len(pool)))
        if not pool:
            scope = f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
            return f"在{scope}没有检索到与「{original_query}」相关的内容。", []
        answer = await self._synthesize_stream(ctx, original_query, pool)
        return answer, pool
```

`split()` 在拿到 `sub_queries, mode` 与现有「空降级」之后、`sections` 拼接之前分流（把 Task 1 的 `_mode` 改回 `mode`）：

```python
        sub_queries, mode = await self.decomposer.run(
            query, headings, passages, self.max_sub_queries
        )

        # 降级：拆不出子查询 → 整句单轮合成
        if not sub_queries:
            logger.info("split: 无子查询，降级单轮检索")
            ctx.write_event_to_stream(RetrievalDoneEvent(count=len(located)))
            if not located:
                scope = (
                    f"《{'》《'.join(book_titles)}》中" if book_titles else "知识库中"
                )
                return f"在{scope}没有检索到与「{query}」相关的内容。", []
            answer = await self._synthesize_stream(ctx, query, located)
            return answer, located

        # 综合型：扇出检索 → 去重合并 → 对原始问题一次整合合成（单子查询天然退化单轮）
        if mode == "synthesize":
            return await self._retrieve_and_synthesize(ctx, query, sub_queries, book_titles)

        # 罗列型：逐项检索 + 分节裸拼
        sections = [(sq, sq) for sq in sub_queries]
        return await self._retrieve_and_concat(ctx, sections, book_titles)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_qa_capability.py -v`
Expected: PASS（synthesize 全部用例 + list/assume 回归绿）。

- [ ] **Step 5: 全量回归**

Run: `python -m pytest tests/test_qa_capability.py tests/test_query_decompose.py tests/test_query_dimension.py -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add core/workflow/qa_capability.py tests/test_qa_capability.py
git commit -m "feat(qa): synthesize 模式 _retrieve_and_synthesize，split 按 mode 分流"
```

---

## Self-Review

- **Spec coverage:** 判型放 decomposer（Task 1）✓；synthesize=扇出+去重+单次整合合成（Task 3）✓；并行检索 gather+信号量（Task 2）✓；assume 白拿并行（Task 2 共享 helper）✓；范围仅 split ✓；边界/降级表逐条有对应测试（Task 1 mode 兜底、Task 3 空池/单子查询、现有空降级回归）✓；schema `mode` 默认兜底（Task 1）✓；eval 延后（计划无 eval 任务）✓。
- **Placeholder scan:** 无 TBD/TODO；每个改码步骤含完整代码。
- **Type consistency:** `run -> tuple[list[str], str]`（Task 1）被 Task 2/3 的 `split` 以 `sub_queries, mode` 解包一致；`_retrieve_all -> list[list]`（Task 2）被 `_retrieve_and_concat`/`_retrieve_and_synthesize` 一致消费；`_retrieve_and_synthesize` 签名 Task 3 定义即调用，名字一致。
- **已知取舍：** 综合池上限固定用 `rerank_candidate_k`；list 模式合成仍 N 次串行（本计划不碰），与 spec 一致。
