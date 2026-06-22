# 设计：front_door 升级为「规划器」 + scope 下沉进 admit

> 删掉 `ConversationScoper`（"拆分前按整句多主体 query 收窄、丢少数派主体的书"这一 bug 的根因载体）。
> 把**拆分**并入 front_door：front_door 从"单次结构化决策"升级为**规划器**——
> 整句消指代/规范化 → **带消歧的拆分**（有界 agent + probe 工具：先判歧义，无歧义直出、有歧义才 probe 再拆）→
> **逐子问题路由**（沿用现有 4 出口，把整句意图判定下沉到每个子问题）。
> 把**确定 scope** 从独立的预收窄阶段下沉进 **admit**：每个子问题用**自己那轮干净的全库 probe** 算主导书 scope + 同源判 ok/out_of_scope/missing_info。
> 前身：[conversation-scoper](2026-06-21-conversation-scoper-design.md)（本设计**取代并删除**它）、
> [multi-subject-split-pipeline](2026-06-21-multi-subject-split-pipeline-design.md)（admit 已逐子问题、分解后；本设计把拆分再提到 front_door 并补知识接地）。

## Context

线上复现（已确诊，见 memory `project-scoper-multisubject-bug`）：问 `给我讲讲Mysql和openclaw的gateway`（MySQL 与 openclaw **两本书都在库里**：MySQL 674 chunk + openclaw_guide-v1.2.2 519 chunk），MySQL 子问题被判 `out_of_scope`，reason="召回片段全部关于 OpenClaw，MySQL 主体缺席"。

**根因（已定位，在 scoper 不在 splitter）**：`conversation_scoper.py` 的 `ConversationScoper.run` 在 `doc_workflow.route` 里、**拆分之前**，用整句多主体 query 做 top-8 探测。"gateway" 是强 openclaw 概念 → top-8 几乎全 openclaw → `_decide` 判单一主导 → 收窄到 `['openclaw_guide-v1.2.2']`，**把 MySQL 那本书丢了**。两个子问题继承该收窄 scope，MySQL 子问题只能在 openclaw 书里 probe → 必然库外。病根是**用机械多数票、对拆分前的合并整句收窄**，牺牲少数派主体。

顺着这个 bug 往下挖，brainstorm 暴露出**同源的另两层**（都因"整句只判一次"）：

1. **修饰语作用域歧义**：`MySQL和openclaw的gateway` 的 `的gateway` 该挂谁？纯语义 splitter 无法知道"gateway 是 openclaw 的概念、MySQL 没有"——**消歧本质需要知识接地**（probe 一下 `mysql的gateway` 看 MySQL 有没有 gateway）。
2. **子问题 intent 不同**：`给我讲讲mysql和编个童话故事` 里 `编个童话故事` 压根不是文档 QA。而 intent 现在由 front_door 拿**整句判一次**，无论它选 dispatch_qa 还是 converse，都有一半被坑（童话被当失败检索拒"库里没有"，或 mysql 没好好查书答）。

三层同一病理：**scope / 消歧 / intent 都被"整句、拆分前、判一次"绑死**。修法：把拆分提到最前（front_door 内），其后所有判定都**逐子问题**做；scope 不再预收窄，而是每个子问题在 admit 阶段用自己的干净召回现挣。

## Goals / Non-goals

**Goals**
- **删 `ConversationScoper`**。其"主导书判定"逻辑（`_decide`）不删，**搬进 admit 阶段的 per-subq scope 计算**——同一套多数票，从"拆分前合并整句"挪到"拆分后单主体干净召回"，即从有害变正确。
- **拆分并入 `front_door`**：front_door 升级为规划器，对外 `run(...)` 产出**路由计划** = `[(子问题, 出口), ...]`。内部三段：① 整句消指代/规范化 → ② 带消歧的拆分 → ③ 逐子问题路由。单子问题时退化为今天的单条决策（回归）。
- **带消歧的拆分 = 有界 agent + probe 工具**：agent 先判是否有修饰语作用域歧义；**无歧义直接输出子问题（零 probe）**；**有歧义才调 probe 工具**（探存疑挂法如 `mysql的gateway`）→ 据召回证据再拆。失败/超时优雅降级到"不拆/纯文本拆"。
- **逐子问题路由沿用现有 4 出口**（`dispatch_qa / converse / clarify / study_plan`），不发明新路线类型——这就是"最小集"：只把整句意图判定**下沉到每个子问题**。非 QA 子问题（童话/写代码/闲聊）→ `converse` 婉拒"我是书籍知识助手……"，**不走检索、不判库外**。
- **scope 下沉进 admit**：每个 QA 子问题 probe（全库或手选书）→ 从召回 nodes **算主导书集合 = 该子问题的 scope** + 同源 admit 判 `ok/out_of_scope/missing_info`。execute 在算出的 scope 内检索合成。
- **`disable_scope`** 重指向"关掉 per-subq 自动 scope"（execute 用全库）；**手选书**仍是硬约束，逐子问题 execute 硬锁、不被自动 scope 覆盖、消歧/scope 的 probe 也限定在手选书内。
- **删全局 `scope_note` 透明前缀**（"我按《X》回答……"）：收窄现在逐子问题自然发生，多子问题答案本就有分节标题=子问题，无需全局声明。

**Non-goals（明确不做）**
- **不引入真·创作能力**：非 QA 子问题一律婉拒，不接 LLM 自由生成（童话/代码）。
- **不重写 front_door 的 converse 细节**：list_books 元工具、clarify 反问、对上一轮反馈的判定等照旧，只是判定粒度从整句变逐子问题。
- **不给消歧 agent 加多轮自校验回路**：agent 至多 probe 一次→重拆一次，`max_iterations` 小、失败降级，不做长循环。
- **不动 admit 的可答性 rubric 文字**（`ok/oos/missing_info` 判据已调细）；只在其外包一层"从 probe nodes 算 scope"。
- **不为消歧 probe 单独建检索栈**：复用现有 probe_retriever。

## 架构：组件（新增 + 改造 + 删除）

| 组件 | 变化 | 职责 |
|---|---|---|
| `FrontDoorPlanner`（改自 `FrontDoorAgent`） | 升级为规划器 | `run(original, memory, book_titles) -> RoutePlan`：消指代/规范化（整句）→ 带消歧拆分 → 逐子问题路由；产 `[(sub_query, action), ...]` |
| 消歧拆分（front_door 内，有界 agent + probe 工具） | 新 | 判歧义→无歧义直出 / 有歧义 probe 存疑挂法→重拆；注入 `probe_retriever`+`index_manager` |
| `Admitter` | **改**：吃 nodes、产 scope | `run(q, nodes) -> AdmitVerdict{verdict, scope, reason, clarify_question}`：判可答性 + 从召回 nodes 算主导书 scope（搬入旧 `ConversationScoper._decide`） |
| `ConversationScoper`（**删**） | 删文件 | `_decide` 逻辑搬进 admit；`run`/note/全局收窄整体移除 |
| `qa.answer`（改） | 消费计划 + per-subq scope | 接收 front_door 的路由计划；QA 子问题逐个 `probe→admit(定scope+verdict)→classify→execute(在scope内)`；非 QA 子问题用计划里的婉拒 reply 装饰；合并 |
| `DocQueryWorkflow.route`（瘦身） | 去 scoper 接线 | 删 `scoper.run`/`scope_note`/`_scope_prefix`；front_door 产计划直接下沉 |
| `QuerySplitter`（**溶解**） | 并入 front_door | 其降噪+拆分 prompt 能力吸收进 front_door 的拆分段；独立类移除 |

**复用（不动）**：`QueryClassifier`（explain/compare/simple/complex）/ probe（`_probe_retrieve`）/ 单轮 `retrieve` / `assume` / `explain` / `QaAgent` / 各 helper / 拒答常量 / 流式事件三件套 / 各判定单元约定（注入 LLM、只暴露 `run`、`json_object`+Pydantic 校验、失败降级、`_strip_fences`）。

## 数据流（目标管线）

```
start → route(FrontDoorPlanner) → split_answer(qa.answer) → finalize

FrontDoorPlanner.run（一个单元，内部三段）:
  ① 整句消指代 + 规范化（LLM，跨片段指代必须先做）→ clean_query
  ② 带消歧的拆分（有界 agent + probe 工具）:
       agent 看 clean_query：
         无修饰语作用域歧义 → 直接输出子问题列表（零 probe）
         有歧义（"A和B的X"，X 挂谁不定）→ 调 probe(存疑挂法，如 "mysql的gateway")
            → 据召回：MySQL 那本书讲不讲 gateway？
                 不讲 → 的X 只挂 B → [讲讲MySQL, openclaw的gateway]
                 讲   → 保留分配读法 [MySQL的gateway, openclaw的gateway]
       失败/超时 → 退回"不拆/纯文本拆"（原 clean_query 单元素）
  ③ 逐子问题路由（沿用 4 出口，整句意图判定下沉到每子问题）:
       讲讲mysql      → dispatch_qa
       编个童话故事    → converse（reply="我是书籍知识助手，不写童话……"）
  产出 RoutePlan = [(sub_query, action[, reply]), ...]
  （手选书 / disable_scope 透传给下游）

qa.answer（消费 RoutePlan）:
  # 阶段一（并行，无用户可见输出）：逐 QA 子问题判定
  对 action==dispatch_qa 的子问题 q 并行 decide(q):
      nodes = _probe_retrieve(q, scope=手选书 or 全库)      # 单主体、召回干净
      v = await admitter.run(q, nodes)                      # AdmitVerdict{verdict, scope, ...}
        # v.scope = 从 nodes 算的主导书集合（搬入 _decide）；disable_scope 或手选 → 跳过自动 scope
      if v.verdict != ok: return (q, v.verdict, scope=None)
      category = await classifier.run(q, evidence(nodes))
      return (q, ok, category, v.scope)

  # 阶段二（按序流式）：执行 + 合并
  parts = []
  for 子问题 in 计划顺序:
      if action==converse → parts += 婉拒 reply（来自计划）
      elif ok → parts += stream_section(q, category, scope)   # execute 在 scope 内
      # missing_info / out_of_scope 收末尾装饰
  末尾：missing_info 子问题反问 + out_of_scope 子问题"不在库"提示
  退化：无任何可答/可回内容 → 纯拒答/反问；单子问题 → 无分节装饰（回归）
```

## 关键决策点（评审依据）

### 1. scope = admit 的输出，不是预收窄的输入（bug 根治）

旧 `ConversationScoper` 把"主导书判定"用在**拆分前的合并整句**上 → 多数票被强主体（gateway/openclaw）压倒 → 丢少数派（MySQL）。**判定逻辑本身没错，错在喂给它的是被污染的合并召回。** 拆分后每个子问题是**单主体**，它自己那轮 probe 召回是干净的（`讲讲MySQL` 不再被 gateway 带偏）→ 同一套 `_decide` 多数票挪过来就对了。故：删 scoper 这个**阶段**，把 `_decide` 挪进 admit 作 per-subq scope 计算。scope 从"预先施加的约束"变成"接地的产物"，从根上消灭这一 bug 类。

### 2. 消歧 = 有界 agent + probe 工具，按需 probe（用户拍板）

为何 agent 而非固定两趟管线：歧义是少数情况，多数 query 无歧义、不该付 probe 代价。让 agent **自己判要不要 probe**——无歧义直出（零检索、零额外往返），有歧义才探存疑挂法。探的是**存疑子问题**（`mysql的gateway`）而非整句，避开了"整句 probe 被强主体主导"的老坑。知识接地只在需要时进来，且 agent 沿用 `QaAgent` 的有界模式（小 `max_iterations`、异常降级），不做长循环。降级方向：probe/agent 失败 → 退回纯文本拆/不拆，绝不阻塞。

> 对**这个** bug 而言，消歧多数靠 LLM 世界知识（"MySQL 没有 gateway"）即可，probe 主要兜"两个主体都是库内私有书、LLM 无世界知识"的情形。agent 形态让这两种情况自然分流。

### 3. 拆分提到 front_door 内（用户方案，纠正早先"splitter 独立挂 route"）

拆分功能**长在 front_door**，不是把 `QuerySplitter` 拎出来挂 route。理由：拆分、消指代、逐子问题路由三者强耦合——消指代要整句先做（指代跨片段），路由要拆完才能逐子问题判，消歧 probe 又夹在拆分中间。把它们放进一个"规划器"单元，数据在内部流转，比散在 workflow step 里拼接清晰。front_door 内部仍可拆私有 helper（normalize / split / route）各自可测，但对外是一个规划单元。

### 4. 逐子问题路由沿用 4 出口，不加新类型（"最小集"）

front_door 本就做意图判定（dispatch_qa/converse/clarify/study_plan）。拆分后只是把这套判定**下沉到每个子问题**，不发明新路线。`编个童话故事` → 现有 `converse` 出口的婉拒分支即可承接（不走检索、不判库外，避免"库里没有童话"的类别错误）。不引入真创作能力（Non-goal）。多数 query 单子问题 → 等价今天单条决策。

### 5. admit 接口从"吃 passages 文本"改成"吃召回 nodes"

要算 scope 必须看 nodes 的 `book_title` 元数据，纯 passages 文本丢了来源。故 `Admitter.run(q, nodes)`：内部既格式化片段判可答性（rubric 文字不动），又从 nodes 统计主导书。`AdmitVerdict` 加 `scope: Optional[list[str]]` 字段（None=不收窄/全库）。`explain` 路原先喂 `Admitter` 的是 passages 文本——同步改为传 nodes（explain 的宽召回本就是 nodes，截断前取元数据）。

### 6. 删全局 scope_note；手选书与 disable_scope 的语义

- **scope_note 删**：逐子问题收窄是隐式的、分节标题已表达"这节在讲哪个子问题"，无需全局"我按《X》回答"前缀。`_scope_prefix` 一并删。
- **手选书**：硬约束。front_door 的消歧 probe、qa.answer 的 per-subq probe、execute 全部限定在手选书内；admit 不再自动收窄（手选即 scope）。
- **disable_scope**（"在所有书里讲"）：关掉 per-subq 自动 scope，execute 用全库。front_door 仍产此标志，含义重指向。

## 降级（绝不阻塞，方向=放行 / 不拆 / 全库）

| 触发 | 落点 |
|---|---|
| front_door 整体失败 | 降级 `dispatch_qa` + 原 query 单子问题（同今天兜底） |
| 消歧 agent / probe 失败 | 退回纯文本拆 / 不拆（原 clean_query 单元素） |
| `Admitter` 失败/空 | 该子问题 `ok` + scope=None（放行、全库 execute） |
| scope 算出为空/噪声 | scope=None（全库，不强收窄） |
| `QueryClassifier` 失败/空 | `simple` |
| 子问题 execute 各分支异常 | 沿用既有降级（simple 单轮 / complex 降单轮 / explain 兜底，见 qa_capability） |

与现有"判定器坏了不该误拒正常问题"同一哲学。`QaAgent` 库外拒答补丁仍是最后防御纵深。

## 测试（mock LLM + mock probe，验解析/接线/降级/scope，不验真 LLM 判断质量）

- **复现回归（首要）**：`给我讲讲Mysql和openclaw的gateway`，front_door 拆出 `[讲讲MySQL, openclaw的gateway]`；stub `讲讲MySQL` 的 probe 返回 MySQL 书 nodes → admit `ok`+scope=`[MySQL]`；stub `openclaw的gateway` probe 返回 openclaw nodes → `ok`+scope=`[openclaw]`。断言：**MySQL 段有答案、不再整体/部分库外**，两段各自锁对书。
- **front_door 规划器**：
  - 无歧义多主体 → 不调 probe、直接拆 ≥2；
  - 有歧义（"A和B的X"）→ 调 probe 一次（存疑挂法）→ 据 stub 证据定挂法；
  - 单主体 → 单元素计划，等价今天单条决策（回归）；
  - 消歧 agent/probe 失败 → 退回不拆。
- **逐子问题路由**：`讲讲mysql和编个童话故事` → 计划 = `[(讲讲mysql, dispatch_qa), (编个童话故事, converse+婉拒reply)]`；断言童话段是婉拒文案、**不**出现"知识库未收录"。
- **admit 定 scope**（搬入 `_decide` 的回归）：
  - 单一主导 nodes → scope=单书；
  - 跨书概念 nodes（两书各占）→ scope=两书；
  - 弥散/噪声 → scope=None（不收窄）；
  - 手选书 → 跳过自动 scope，scope=手选；
  - admit 抛错 → ok+scope=None。
- **qa.answer 编排**：全 QA-ok 多子问题分节顺序正确、各段 execute 收到自己的 scope；混 converse 子问题装饰正确；混 missing_info/oos 末尾装饰；全非 ok/全 converse → 退化；单问题无分节（回归）。
- **scope 锁 execute**：stub execute 记录收到的 book_titles，断言 == 该子问题 admit 算出的 scope。
- **workflow 接线**：`route → split_answer → finalize`；converse/clarify/study_plan 单路径不变（回归）；`ConversationScoper` 引用全部移除（import 不残留）。
- **真实冷烟（需 DEEPSEEK_API_KEY + 索引人读）**：多主体（库内+库内 / 库内+库外混合）、修饰语歧义（gateway）、混合 intent（童话）、裸概念续问（"它的索引呢"靠消指代锚书）四类。

## 决策锁定

1. **删 `ConversationScoper`**；`_decide` 搬进 admit 作 per-subq scope；scope 是接地产物不是预收窄输入。
2. **拆分并入 front_door**，front_door 升级为产"路由计划"的规划器。
3. **消歧 = 有界 agent + probe 工具**：先判歧义，无歧义直出、有歧义才 probe 存疑挂法再拆。
4. **逐子问题路由沿用现有 4 出口**，不加新类型；非 QA → converse 婉拒，不走检索。
5. **`Admitter.run` 改吃 nodes、产 `scope`**；rubric 文字不动；explain 路同步改传 nodes。
6. **删全局 scope_note / `_scope_prefix`**；手选书硬锁；`disable_scope` 重指向"关 per-subq 自动 scope"。
7. **`QuerySplitter` 溶解进 front_door**。

## 已知缺口（留后续）

- **消歧 probe 成本/质量**：agent 判歧义本身是一次 LLM；歧义频率与 probe 命中率需冷烟后观测，再议是否值得。
- **per-subq probe 放大**：每 QA 子问题各 probe 一次，子问题多时检索放大（继承上一刀已知缺口）。
- **手选多书 + 多子问题**：手选 N 书时各子问题是否该在 N 书内再 per-subq 细分 scope，本期不做（手选即统一 scope）。
- **裸概念续问的回归保障**：scoper 删除后，"它的索引呢"完全依赖 front_door 消指代把"它"补成书内主体；需冷烟确认消指代足够稳，否则丢了 scoper 的历史兜底。
- **真实冷烟**：四类场景需人读。

## 命名（评审时定）

- 规划器 `FrontDoorPlanner`（改自 `FrontDoorAgent`，备选保留 `FrontDoorAgent` 名）；产出 `RoutePlan`（备选 `list[RoutedSubQuery]`）；`AdmitVerdict` 加 `scope` 字段；删 `ConversationScoper` / `ScopeDecision` / `QuerySplitter`。
