# 评测体系新增「agent 自主规划」路线 · 设计

> 一句话：给 ablation 加**第二种被测系统**——绕过 `DocQueryWorkflow` 的 intent-router + category 分类，每条 query 直接喂给有界 `FunctionAgent`（复用 `QaAgent`）自主规划检索，正面对比「显式决策路由 vs 让 agent 自己规划」。

日期：2026-06-18 ｜ 分支：`feat/probe-shape-signal`（或新开）

---

## 1. 背景与动机

当前评测（见 `docs/EVAL_OVERVIEW.md`）唯一被测系统是 `core/workflow/DocQueryWorkflow`：两层决策（IntentRouter → QaCapability 按 category 路由到 retrieve/split/assume/clarify/other）。`eval/harness/sut.py` 的 `DocQueryWorkflowSystem` 把它包成统一 `answer(query) -> RagOutput`，`compare.py` 按决策 flag 组合跑 ablation。

整套评测的核心问题是「每个工程化决策带来多少提升」。但缺一个**对照系**：如果完全不做显式路由，只给一个 agent 同样的检索工具、让它自己规划，效果如何？本设计补上这条对照路线。

`QaAgent`（`core/agent/qa_agent.py`）已经是一个有界自主 agent：FunctionAgent + `book_search`/`list_books` 工具 + `max_iterations` 边界 + source 收集，`run()` 返回 `(answer, source_nodes)`。复用它即可，改动最小、对比最纯净（不引入新 prompt/新装配作为干扰变量）。

## 2. 范围

**做**：
- 新增 SUT 适配器 `AgentSystem`，实现 `RagSystem` 协议，内部跑 `QaAgent`。
- `run_eval.aggregate` 一行语义微调，让「不产 category 的系统」分类准确率显示 N/A 而非误导的 0.00。
- `compare.py` 加一个哨兵变体，把 agent 路线作为同一张 delta 表里的一行。
- 对应单元测试。

**不做**（YAGNI）：
- 不用生产 `BookAgent`（只返回字符串、不回传 source_nodes，需接 contextvar，改动大且引入干扰变量）。
- 不为评测新写 agent / 新 prompt。
- 不让 agent 产出 category（自主规划路线本就跳过分类，强行产 category 会污染对比语义）。
- 不改 golden 数据集。

## 3. 组件设计

### 3.1 `AgentSystem`（`eval/harness/sut.py` 新增）

实现既有 `RagSystem` 协议：`async def answer(self, query: str) -> RagOutput`。

构造参数与 `DocQueryWorkflowSystem` 对齐：`(index_manager, llm, similarity_top_k=5, max_iterations=6)`。

`answer()` 流程：
1. 每次调用新建 `QaAgent(index_manager, llm, similarity_top_k, max_iterations)`（与 workflow 适配器每请求新建一致，避免跨条状态串扰）。
2. 调 `answer, sources = await qa_agent.run(ctx=_NullCtx(), query=query, book_titles=book_titles)`。
3. 映射成 `RagOutput`：
   - `response = answer`
   - `retrieved_contexts = [n.get_content() for n in sources]`（node 取文同 `map_doc_result`）
   - `outcome = "answered"` 若 `answer` 非空且 `sources` 非空，否则 `"empty"`
   - `category = ""`（agent 不产分类）
4. try/except 兜底 → `RagOutput(f"{type(e).__name__}: {e}", [], "error", "")`，与 `DocQueryWorkflowSystem` 一致。

签名 `answer(self, query, book_titles=None)` 与 `DocQueryWorkflowSystem.answer` 对齐（`book_titles` 评测里默认 None = 全库）。

### 3.2 `_NullCtx`（`eval/harness/sut.py` 新增，私有）

`QaAgent.run()` 需要一个带 `write_event_to_stream(event)` 的 ctx——生产里是 workflow ctx，用于把检索事件推给前端流。评测无 workflow，加一个最小 stub：

```python
class _NullCtx:
    def write_event_to_stream(self, event) -> None:  # noqa: D401 — no-op
        pass
```

**正确性论证**：`QaAgent.run()` 里 `stream_events` 循环只用于 `ctx.write_event_to_stream(...)` 推 `RetrievalStart/Done` 事件给前端；最终答案来自 `final = await handler`，与 ctx 无关。故 no-op ctx 不影响答案与 sources。

### 3.3 `aggregate` 分类准确率 N/A（`eval/harness/run_eval.py` 一行改动）

现状：
```python
exp = r.get("expected_category")
if exp:
    cls_total += 1
    cls_correct += int(cat == exp)
```
agent 恒 `cat=""`，golden 的 `expected_category` 非空 → `cls_total` 累加但永不命中 → 准确率算成 **0.00（全错）**，误导成「分类能力极差」，而真相是「这条路线不做分类」。

改为：
```python
if exp and cat:
    cls_total += 1
    cls_correct += int(cat == exp)
```

效果：
- **workflow**：每行都有 category，`cls_total`/命中**完全不变**，无回归。
- **agent**：`cls_total=0` → `accuracy=None` → 表里「—」，诚实表达「不做分类」。
- **副作用（可接受，且更合理）**：workflow 偶发 error 行（`outcome="error"`、`category=""`）原本计为「误分类」（拉低准确率），改后被**排除**。error 本就不该算作「分类错误」，这是修正而非退步。

### 3.4 接入 `compare.py`

`VARIANTS` 是「名字→flags dict」，喂给 `DocQueryWorkflowSystem`。agent 是另一个 SUT 类，不是 flag 组合。做法：
- 新增常量 `AGENT_VARIANT = "agent(自主规划)"`。
- `--variants` 的可选名集合里加入 `AGENT_VARIANT`（CLI help 与默认全集都含它）。
- `_run_variants` 按名分流：
  ```python
  if name == AGENT_VARIANT:
      sut = AgentSystem(index_manager, sut_llm)
  else:
      sut = DocQueryWorkflowSystem(index_manager, sut_llm, flags=VARIANTS[name])
  ```
- agent 路线作为**一行**出现在 workflow 变体下方；分类准确率列「—」，5 个 ragas 列照常，直接同台对比答案质量。

跑法示例：
```powershell
python -m eval.harness.compare --testset eval/dataset/golden.jsonl \
  --variants "全开" "agent(自主规划)"
```

## 4. 数据流

```
query ──> AgentSystem.answer
            │  新建 QaAgent
            ├─> qa_agent.run(ctx=_NullCtx(), query, book_titles=None)
            │       FunctionAgent 自主多轮 book_search/list_books（≤max_iterations）
            │       收集 source_nodes
            │   <─ (answer, sources)
            └─> RagOutput(response=answer,
                          retrieved_contexts=[node 文本...],
                          outcome=answered|empty|error,
                          category="")
                  │
                  ▼  score_row（同 workflow 路径）
            5 个 ragas 指标（faithfulness / answer_relevancy /
            context_precision / context_recall / factual_correctness）
            分类准确率 = N/A（category 空）
```

## 5. 错误处理

- agent 运行异常（含 `max_iterations` 内未收敛但 `early_stopping_method="generate"` 已兜底作答）：`QaAgent` 自身正常返回；适配器 try/except 仅兜 agent.run 抛出的异常 → `outcome="error"`，单条不中断整轮（与 `DocQueryWorkflowSystem` 一致）。
- 知识库为空 / 无命中：`QaAgent._search` 返回占位提示，`sources` 为空 → `outcome="empty"`，不进 ragas 打分（`score_row` 对非 answered 直接返回 base）。

## 6. 测试（TDD）

`tests/`（沿用现有 `map_doc_result` 等纯函数测试风格）：

1. **`AgentSystem.answer` 映射**（注入假 QaAgent，monkeypatch 或构造可替身）：
   - 正常：假 agent 返回 `("答案", [node1, node2])` → `response="答案"`、`retrieved_contexts` 长度 2、`outcome="answered"`、`category=""`。
   - 空 sources：返回 `("答案", [])` → `outcome="empty"`。
   - 异常：假 agent.run 抛 → `outcome="error"`、response 含异常类型名、`category=""`。
2. **`_NullCtx.write_event_to_stream`** 接受任意 event、no-op、不抛。
3. **`aggregate` N/A**：
   - 全空 category 行（expected_category 非空）→ `classification["accuracy"] is None`、`classification["total"]==0`。
   - 回归：含 category 的 workflow 行，准确率与改动前一致（构造命中/未命中混合，断言数值不变）。

## 7. 影响面与回归

- `DocQueryWorkflowSystem`、`compare.py` 现有变体、golden 数据集：**零行为变化**（aggregate 改动对「每行都有 category」的 workflow 恒等）。
- 新增面：`AgentSystem` + `_NullCtx`（sut.py）、`AGENT_VARIANT` 分流（compare.py）、aggregate 一行。
- 文档：完成后在 `docs/EVAL_OVERVIEW.md` §2/§7 补一句「第二被测系统：agent 自主规划路线」。

## 8. 取舍记录

- **复用 QaAgent 而非生产 BookAgent**：QaAgent 已回传 source_nodes（直接喂 ragas retrieved_contexts），且工具集与 workflow 内部检索一致 → 对比纯净；BookAgent 只返回字符串、需接 contextvar 收集 sources，改动大且引入装配差异作为干扰变量。
- **NullCtx no-op**：取巧复用 `QaAgent.run` 原样，不为评测拆出无 ctx 的运行入口（避免在领域层为评测开后门）。
- **`if exp and cat:`**：在 aggregate 内做最小语义修正，让「不产类别的系统」自然得到 N/A，而非在 compare 层为 agent 特判跳过分类列（特判会散落系统类型知识到展示层）。
