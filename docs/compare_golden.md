# 决策对比（baseline=baseline(全单轮)）

测试集：`eval/dataset/golden.jsonl`

| 配置 | 分类准确率 | context_recall | factual_correctness | faithfulness | answer_relevancy |
|---|---|---|---|---|---|
| baseline(全单轮) | 0.70 | 0.61 | 0.43 | 0.69 | 0.57 |
| +probe | 0.70 | 0.59 (-0.02) | 0.38 (-0.04) | 0.69 (-0.00) | 0.57 (-0.00) |
