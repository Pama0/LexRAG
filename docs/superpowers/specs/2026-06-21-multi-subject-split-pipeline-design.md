# 设计：多主体拆分 + 分解后可答性 —— QA 管线顺序重排

> 把"可答性判定（admit）"从**原子、分解之前**改成**逐子问题、分解之后**。
> 入口先拆"显式并列的多主体问题"，每个子问题**独立并行**走全流程，最后合并。
> 修掉"复合问题被一个原子判决整体拒答 / 整体放行"的根因——其中最坏一例：
> 库内、已召回的那半也被连坐拒答（见 Context）。
> 前身：[answerability-pregate](2026-06-21-answerability-pregate-design.md)（admit 抽成共享单元，但仍原子、仍在分解前）。

## Context

线上复现（2026-06-21）：问"给我讲讲 MySQL 和 OpenClaw 的 gateway"，系统**整体拒答**。日志：

```
front_door: action=dispatch_qa clean_query='给我讲讲MySQL和openclaw的gateway'   # 专名已保住
gate: intent=explain denoised='MySQL openclaw gateway'
admit: verdict=out_of_scope reason=问题主体实体是MySQL，但召回片段全部关于OpenClaw Gateway，MySQL不在知识库中
explain: admit 判库外，拒答 → REFUSAL_TEXT
```

**根因（代码层已定位）**：

1. **admit 的 rubric 是单主体的**（`admitter.py:40-44` 通篇"问题的主体技术实体"单数）。面对"MySQL **和** openclaw 的 gateway"，它被迫任选一个当"主体"（选了 MySQL），发现召回里全是 OpenClaw → 判"主体缺席 → 库外"。它的 reason 自相矛盾恰是铁证：召回到了 OpenClaw Gateway（说明库里**有**能答这半的内容），却把整个问题拒了。
2. **admit 原子、且在分解之前**。分解（`pending_split`）在 `classify` 下游；而本例走 explain 路（`intent=explain`），**explain 根本不分解**（`qa_capability.py:278` 对整句一次宽召回 + 一次 admit）。复合问题被当一个 blob 判。
3. **召回偏斜**："MySQL openclaw gateway" 一次宽召回，OpenClaw Gateway 是更紧凑的簇，把 MySQL 挤出 top-k → 喂给 admit 的证据本身就偏。
4. **临门一脚**：`doc_workflow.py:288` explain_branch 接住 `OutOfScope` 直接 `REFUSAL_TEXT`，把已检索到的 OpenClaw 内容全扔。

**同一根因，两个方向都会错**：上一刀（前置闸）刚修过"一个库内主体就整体放行"（MySQL+OpenCL 判 ok）；现在反过来"原子判决整体拒答"。本质都是**可答性被原子化、且先于分解**。正交轴诊断：可答性轴必须**后置于"问题分解"**——多主体问题没有单一可答性，必须逐主体判。

## Goals / Non-goals

**Goals**
- 入口新增**多主体拆分器** `QuerySplitter`：把"显式并列、话题独立、无比较词、无依赖"的复合问题拆成 ≥1 个**降噪后的自包含子问题**；不可拆 → 单元素列表（原问题）。
- **每个子问题独立走全流程**（probe → admit → 类型分类 → 执行），分类/可答性判定的轮次**并行**；最终合成**按序流式**。
- **合并阶段**：先输出 ok 子问题的答案（分节），末尾追加 missing_info 子问题的反问 + out_of_scope 子问题的"不在库"提示；全非 ok → 退化纯拒答/反问；单问题（无拆分）→ 退化为单一答案、无合并装饰。
- **溶解 gate**：降噪并入 `QuerySplitter`，意图（explain/other）并入类型分类器。
- **类型分类器**改产 4 类 `explain / compare / simple / complex`（替代 gate.intent + 旧 4 类 judge）。映射见下。
- **simple/complex 保留** + **安全网**：simple 单轮检索若证据明显不足 → 升级 agent。
- **missing_info 跨轮持久化**：记录"待补充子问题"，下一轮只重跑该子问题再并入已答部分。

**Non-goals（明确不做）**
- **拆比较/多跳/广度发散**：拆分器**只以多主体为判据**。比较类、多跳依赖、单主题广度发散一律**不拆**（理由见决策点 1/2）。
- **比较类的双侧分别检索特别优化**：本期比较类先沿用现有 assume（维度）路线；"仿 explain 做双主体分别召回再综合"的特别优化**留后续一刀**（决策点 2）。
- **agent 接管 simple**：simple 走便宜确定路径，agent 只接 complex 与"证据不足升级"（决策点 3）。
- **运行时多轮自校验**：各判定器一次判定、失败优雅降级，不加回路。

## 类别映射（旧 → 新）

| 旧 | 新 | 落点 |
|---|---|---|
| gate.intent=explain | `explain` | 讲解路线（teach 合成） |
| `ambiguous`（角度不定） | `compare`（比较/评价） | assume（维度）路线 |
| `retrievable` | `simple` | 单轮检索+合成（+证据不足升级 agent） |
| `other` + `pending_split` | `complex` | 有界 agent（自行扇出子查询/循环检索） |
| `out_of_scope` / `missing_info` | （admit verdict，分类前） | 合并阶段处理（不再即时终结） |

> `pending_split` 并入 `complex`：不再维护独立的"拆解器"分支，由 agent 自行决定扇出/循环——prompt 需增强引导其对大主题/多跳主动多查（已知缺口）。

## 架构：组件（新增 + 改造 + 溶解）

| 组件 | 变化 | 职责 |
|---|---|---|
| `QuerySplitter`（新） | 入口拆分器 | `run(clean_query) -> list[str]`：降噪 + 多主体拆分；≥1 个自包含子问题；纯文本判定（不检索） |
| `QueryClassifier`（改自 `QueryPreprocessor`） | 4 类新枚举 | `explain / compare / simple / complex`，吃 probe 证据；删 gate.intent 旧职责并入此 |
| `Admitter` | 不动 | 逐子问题判 `ok / missing_info / out_of_scope`（`admitter.py`，沿用） |
| `qa.answer`（新，capability 内编排） | 扇出 + 合并 | 对子问题列表逐个 `probe→admit→classify→执行`，并行判定、按序合成、合并装饰 |
| `QueryGate`（删） | 溶解 | 降噪→`QuerySplitter`；intent→`QueryClassifier`。文件移除 |
| `PendingClarification`（新） | 跨轮状态 | 持久化"待补充子问题 + 已答部分"，下一轮消费 |
| `DocQueryWorkflow`（瘦身） | step 图收敛 | `route → split → answer → finalize`；旧 6 个 category 分支收进 `qa.answer` |

**复用（不动）**：front_door（净化+4出口）/ `Admitter` / probe（`_probe_retrieve`）/ 单轮 `retrieve` / assume / explain / `QaAgent` / 单轮各 helper / 拒答常量 `REFUSAL_TEXT`·`REFUSAL_FALLBACK` / 流式事件三件套。各判定单元沿用约定（注入 LLM、只暴露 `run`、`json_object`+Pydantic 校验、失败降级、自带 `_strip_fences`）。

## 数据流（目标管线）

```
start → route(front_door: 净化 + 4出口)            # 不变
  ├─ converse / clarify / study_plan → 各自分支     # 不变
  └─ dispatch_qa → split → answer → finalize

split step:
  sub_qs = await qa.split_query(clean_query)         # QuerySplitter: ≥1 降噪自包含子问题
  # 若有上一轮 PendingClarification 且本轮是补充 → 解析为"补全那个待补充子问题"，
  #   sub_qs = [已答部分占位 + 补全后的子问题]（见决策点 4）

answer step  (qa.answer 内部编排):
  # 阶段一（并行，无用户可见输出）：逐子问题判定
  results = await gather(per_subq_decide(q) for q in sub_qs)
    per_subq_decide(q):
      located  = _probe_retrieve(q)                  # 不变
      evidence = _format_probe(located)              # 不变
      verdict  = await admitter.run(q, [evidence])   # 逐子问题判
      if verdict != ok: return (q, verdict, None)    # 记录，不即时终结
      category = await classifier.run(q, evidence)   # explain/compare/simple/complex
      return (q, ok, category)

  # 阶段二（按序流式）：执行 ok 子问题 + 合并装饰
  parts = []
  for (q, verdict, category) in results where verdict == ok:
      parts += stream_section(q, category)           # 见"执行分派"
  # 末尾装饰
  for missing_info 子问题: parts += 反问句(verdict.clarify_question)
  for out_of_scope 子问题: parts += "「{q}」不在知识库中" 提示
  # 退化
  if 无 ok 子问题: answer = 纯拒答/反问(单条时复用原话术)
  if len(sub_qs)==1: 不加分节标题/合并装饰，等价旧单问题路径
  记录 PendingClarification（若有 missing_info 子问题）

执行分派 stream_section(q, category):
  explain  → qa.explain 教学路线
  compare  → qa.assume  维度路线        # 本期；特别优化留后续
  simple   → qa.retrieve 单轮 + 合成；证据不足 → 升级 qa_agent.run   # 安全网
  complex  → qa_agent.run；异常 → 降级单轮 retrieve
```

## 关键决策点（评审依据）

### 1. 拆分判据：只拆"明显独立的多主体"，居中地带默认不拆

拆是**不可逆的破坏性选择**（各自独立答完就回不到跨主体对照）；不拆是可退的（两主体仍在同一合成上下文）。故规则偏保守：

**拆**（同时满足）：① 显式并列连词/标点（A 和 B、A 与 B、A、B）；② 两侧**话题不同**（"A 的 x 和 B 的 y"）或带"分别/各自"并列标记；③ 无比较/对比/区别词；④ 无依赖（后半不依赖前半的答案）。

**不拆**（任一）：比较/评价（"A 和 B 的区别/哪个好"）→ `compare`；多跳依赖 → `complex`；单主题广度发散（"怎么优化 X"）→ `complex`；**话题共享且无"分别"标记的居中句式**（"讲讲 A 和 B 的缓存机制"）→ 默认不拆，交 `compare`。

> 让 LLM 只判一个相对清晰的信号（话题是否共享 / 有无"分别"标记），比判"用户想不想对比"稳得多。误判不是灾难——拆与比较两条路**都会召回两主体资料**，区别仅在合成形态。

### 2. 比较/评价（漏洞2 收口）→ 判为不拆，走 compare(assume)

居中地带与显式比较一律不拆，归 `compare`，沿用现有 assume（归纳维度→逐维度检索）。**已知短板**：assume 按维度扇出、非按主体扇出，单维度 query 仍可能丢掉某一侧（旧 `pending_split` 正为此把"区别"判拆）。本期接受此短板；后续一刀给 compare 做"仿 explain 的双主体分别召回再对照综合"特别优化（已知缺口）。

### 3. simple/complex 保留 + 安全网（回答用户的取舍问题）

**保留**，不全交 agent。关键事实：simple/complex 不是额外 LLM 调用，只是 `QueryClassifier` 里多一个枚举值——**保留几乎零成本**。而全交 agent 对 simple 问题**无质量提升**（一轮检索就够），却涨成本（≥2 次 LLM vs 1 次）、涨首 token 延迟（先"思考"再出正文）、涨方差（可能空转/跑偏，故才有 `max_iterations` + 降级兜底）。书问答里 simple 是大头，全交 = 多数流量替少数复杂问题交税。

**安全网**（治"complex 误判成 simple"）：simple 单轮检索后，若**证据明显不足**——召回为空 / top-1 分数低于阈值（可配，初值复用既有 rerank/检索阈值口径）——则升级 `qa_agent.run`。默认走便宜路径，仅在有证据表明不够时掏 agent。（合成时模型自陈"资料不足"的文本信号判定较糊，本期只用"空/低分"的确定信号，自陈信号留后续。）

### 4. missing_info 跨轮持久化（漏洞3 收口）

`PendingClarification`（持久化到 DB，**不**进会话记忆——会话记忆只存用户原话+最终答案，见 `doc_workflow.py:363` 红线）记录：`{ session_id, answered_sections, pending_sub_questions[] }`。

- 本轮：某子问题 missing_info → 合并阶段把其反问追加到末尾，并落 `PendingClarification`。
- 下一轮：front_door 照常净化；若检测到本轮是对 pending 的补充（front_door 已能读会话历史判"对上一轮的补充/反馈"），split step 把补充解析为"补全那个 pending 子问题"，**只重跑该子问题**，再与 `answered_sections` 合并，清除 pending 记录。
- 边界：用户下一轮换了新问题（非补充）→ 丢弃 pending，正常新流程。

> 这是真状态机，是本方案最重的一块。最小可行版可先"重跑整复合问题"（简单但浪费），但既然用户确认要持久化，按上方做。

### 5. gate 溶解（漏洞5 收口）

`QueryGate` 删除：降噪能力并入 `QuerySplitter`（拆出的子问题即降噪自包含句，单问题时等于只降噪），intent(explain/other) 并入 `QueryClassifier`（explain 成为 4 类之一）。**两处 admit 合一**：原 explain 路内嵌 admit 与 classify 内嵌 admit，统一为 `qa.answer` 里逐子问题的一次 admit；explain 不再自带 admit、不再走 `OutOfScope`/`MissingInfo` 异常（那套控制流被 verdict 数据流取代）。

## 流式与并行

- **并行**只用于阶段一（probe/admit/classify，无用户可见输出），省判定往返；**合成阶段按序流式**，逐子问题一个 `## 分节`，复用 `_retrieve_and_concat` 的分节流式模式（`qa_capability.py:424`）。真正并行多个会流式的子合成与"有序 token 流"冲突，不做。
- 合并装饰（反问/不在库提示）作为最后的 `AnswerDeltaEvent` 推出。
- 多数拆分为 2 子问题，按序合成的额外延迟可接受。

## 降级（绝不阻塞，方向=放行/不拆）

| 触发 | 落点 |
|---|---|
| `QuerySplitter` 失败/空 | 单元素列表 = 原问题（不拆，最安全） |
| `Admitter` 失败/空 | 该子问题 `ok`（放行去答） |
| `QueryClassifier` 失败/空 | `simple`（最便宜确定路径） |
| simple 证据不足 | 升级 agent；agent 再异常 → 单轮 retrieve 结果照出 |
| `complex` agent 异常 | 降级单轮 retrieve（不让 complex 比单轮更脆） |
| `PendingClarification` 读写失败 | 退化为"重跑整问题"，不阻塞 |

与现有"判定器坏了不该误拒正常问题"同一哲学。`QaAgent` 库外拒答补丁仍是最后防御纵深。

## 测试（mock LLM，验解析/接线/降级，不验真 LLM 判断质量）

- `QuerySplitter`：显式独立多主体 → 拆 ≥2；比较/多跳/广度/居中共享话题 → 单元素；失败/空 → 原问题单元素；输出子问题已降噪自包含。
- `QueryClassifier`：解析 explain/compare/simple/complex；失败/空 → simple。
- `qa.answer` 编排：
  - 全 ok 多子问题 → 分节合并、顺序正确；
  - 部分 missing_info → 先答 ok 段、末尾带反问、落 PendingClarification；
  - 部分 out_of_scope → 末尾带"不在库"提示；
  - 全非 ok → 纯拒答/反问；
  - 单问题 → 无分节装饰，等价旧单路径（回归）。
- simple 安全网：召回空/低分 → 调 agent（stub retrieve 返回空 → 断言走 agent）。
- 跨轮：上一轮 PendingClarification 存在 + 本轮补充 → 只重跑该子问题、并入 answered、清 pending。
- 复现回归：构造"MySQL 和 openclaw 的 gateway"两子问题，stub openclaw 子问题召回到内容 → 断言**不**整体拒答、openclaw 段有答案、MySQL 段按其 verdict 处理。
- workflow：`route → split → answer → finalize` 接线；converse/clarify/study_plan 不变（回归）。

## 决策锁定

1. **可答性后置于分解**：admit 逐子问题判，不再原子、不再在分解前。
2. **拆分只认多主体**，居中地带默认不拆；比较/多跳/广度 → 不拆。
3. **比较走 assume**（本期），双主体分别召回的特别优化留后续。
4. **simple/complex 保留**（近零成本）+ **空/低分证据→升级 agent** 安全网；不全交 agent。
5. **gate 溶解**：降噪→splitter，intent→classifier，两处 admit 合一。
6. **missing_info 跨轮持久化**（`PendingClarification`，落 DB，不污染会话记忆）。
7. **`pending_split` 并入 `complex`**，由 agent 自行扇出/循环。

## 已知缺口（留后续）

- **compare 双主体特别优化**：仿 explain 做"分主体分别召回 → 对照综合"，治 assume 按维度扇出丢一侧的短板。
- **complex/agent prompt 增强**：引导 agent 对大主题主动扇出子查询、对多跳主动循环检索（pending_split 能力的承接）。
- **simple 自陈"资料不足"升级信号**：本期只用空/低分确定信号。
- **真实冷烟**：多主体（库内+库外混合）、比较、跨轮补充三类需 DEEPSEEK_API_KEY + 索引人读。
- **probe 复用**：阶段一每子问题各 probe 一次，子问题多时检索放大；可观测后再议是否合并/限流。

## 命名（评审时定）

- 拆分器 `QuerySplitter`（备选 `MultiSubjectSplitter`）；分类器 `QueryClassifier`（改自 `QueryPreprocessor`），枚举 `explain / compare / simple / complex`；编排方法 `qa.answer`（备选 `answer_multi`）；跨轮状态 `PendingClarification`；删 `QueryGate`。
