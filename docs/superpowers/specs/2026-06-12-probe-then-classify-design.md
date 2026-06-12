# probe-then-classify：判定前先探测知识库 Design

**日期:** 2026-06-12
**状态:** 设计已定，待实现（修 other 误判 bug 的根因）
**关联:** [other 有界 agent](2026-06-12-other-bounded-agent-design.md) · [项目架构](../../ARCHITECTURE.md)

## 1. Bug 与根因（已复现确认）

**现象**：选中全部书籍问「给我讲明白openclaw」，AI 不检索、直接用训练知识猜（OpenCL/OpenCLAW）反问。

**证据**：
- 日志：3 次 DeepSeek 调用 + 零检索事件 + StopEvent → 走了 `other → QaAgent`，agent 一次 LLM 就决定不检索。
- 复现：`QueryPreprocessor` 把「给我讲明白openclaw」判 `other`（2/2），reason 自陈"openclaw 可能多义、需全面讲解、高难度"；对照「openclaw是什么」判 `retrievable`、「讲讲MySQL」判 `pending_split`。

**根因**：judge（preprocess）是**盲判**——零知识库信息，纯靠 LLM 世界知识 + query 文本分类。它看到不认识的专名 `openclaw` → 用世界知识脑补"多义/高难度" → 判 `other`（绕过检索）。但 openclaw 就在用户书库里，一检索就有。**LLM 把"我不认识这个词"误当成"问题难/不明确"——judge 范式对 RAG 专有名词的系统性盲区**（`missing_info` 同病）。`other` 积极判定放大了它。

## 2. 修复：给 judge 装"知识库视力"

把「先分类后检索」改成「先探测召回、再带着召回结果分类」（probe-then-classify）。judge 的判定依据从"LLM 世界知识"换成"知识库实际召回"，从根上堵住盲判。

```
QaCapability.classify(clean_query, book_titles):
  1) probe = 一次宽召回(clean_query, scope)             ← 新增（用 clean_query，因 rewritten 是 preprocess 产物）
  2) retrieval_context = 格式化召回信号
  3) return preprocessor.run(clean_query, retrieval_context)   ← judge 带召回判定
```

五类判定全部从"猜"升级为"看库里有什么判"：retrievable=召回相关且集中；pending_split=召回散落多章节/实体；missing_info=**召回为空/明显不相关（库里真没有）**；other=**召回到但需跨多处综合**；都与"judge 认不认识词"脱钩。

## 3. 召回粒度（已定：中+，不给分数）

喂给 judge：**命中数 + top 3~5 截断片段（每段 ~150 字）+ 命中的 book/chapter 去重分布**。不给相似度分数。

理由：
- **排除只给命中数**：向量检索恒返回 top-k，命中数无判别力，区分不了"相关 vs 凑数"。
- **给片段不给分数**：judge 是 LLM，读文本判"相关性/集中度"远比读分数（尺度依 embedding 模型、不直观）靠谱；片段还能区分"召回空 vs 召回到但不相关"。
- **截断控成本**：top3~5 × 150 字 ≈ 加 600~900 字到 prompt，LLM 调用次数不变。

格式：
```
【知识库探测召回】共命中 N 段，分布：《openclaw》第3章·第5章
1. [《openclaw》3.2] openclaw 是一个……（截前150字）
...
（N=0：知识库未召回到任何相关内容）
```

## 4. probe nodes 复用：MVP 不做（决策记录）

probe 用 `clean_query`、各分支用 `rewritten_query`（preprocess 产物），**两者 query 不同**——probe 是"探测"、分支是"精检"，本是两个语义不同的检索，不该强行复用。且一次向量检索（本地 ~几十 ms）在 LLM 主导的总延迟（1~2s）里 <5%，复用省的微乎其微，却要换 ctx 跨 step 传递 + 条件复用 + 测试复杂度——成本收益倒挂、过早耦合（ARCHITECTURE §105）。

**未来路径**（若评测显示检索成瓶颈）：做**按 query 指纹的 per-request 检索缓存**（`_retrieve_nodes` 查缓存命中即返），自动解决 clean/rewritten 一致性（同 query 自动复用、不同 query 各自检索），比"专门传 probe nodes"通用干净。**不在本期实现。**

## 5. 防御纵深：agent 加"先检索"铁律

即便 probe 后仍偶有误判到 `other → QaAgent`，agent system prompt 加铁律：**拿到问题先调 book_search，严禁在检索前用训练知识猜测或反问**。双层保险。

## 6. 鲁棒性 / 降级

- probe 检索失败（index 空 / 异常）→ `try/except` 容错，`retrieval_context=""`，judge 退回纯文本判定（不阻塞，记 WARNING）。
- `QueryPreprocessor.run(clean_query, retrieval_context="")` 默认空 → 向后兼容（现有单测不传该参，行为不变）。

## 7. 验证策略（关键）

这是 **LLM 行为修复**，判定质量无法用确定性单测验证。
- **单测**：只测接线——classify 调 probe + 传 context、preprocess 用 retrieval_context（prompt 含召回块）、probe 失败容错。
- **复现脚本**（真 DeepSeek）：跑「给我讲明白openclaw」确认不再判 `other`（修复成功判据），对照「openclaw是什么」「讲讲MySQL」不退化。
- 长期：纳入 eval 评测集校准。

## 8. 改动面

- `core/workflow/query_preprocess.py`：`run` 加 `retrieval_context` 参数；`_JUDGE_PROMPT` 加召回块 + 专名铁律 + 各类"看召回"判据。
- `core/workflow/qa_capability.py`：`classify` 加 `book_titles` 参数 + probe + `_format_probe` + 容错。
- `core/workflow/doc_workflow.py`：`preprocess` step 把 `book_titles` 传给 `classify`。
- `core/agent/qa_agent.py`：system prompt 加"先检索"铁律。

## 9. 不做（YAGNI）

- probe nodes 复用 / per-request 缓存（§4）。
- probe 用独立更大的 top_k（先复用 `similarity_top_k`）。
- 召回分数喂 judge（§3）。
