"""路线对比 runner：对若干 SUT 路线跑同一测试集，渲染 baseline vs 各路线的 delta 表。

三条路线：naive_rag（朴素 RAG 对照组）/ workflow（生产编排）/ agent（自主规划）。
默认 baseline = naive_rag，delta 列即各路线相对朴素 RAG 的质量/成本差。
指标 = 5 ragas 质量 + 成本（时延/tokens，串行模式逐行计 token）。
"""
# 只比朴素 RAG 与 workflow：python -m eval.harness.compare --testset eval/dataset/golden.jsonl --variants workflow naive_rag
import argparse
import asyncio
import json
import os
import sys
import time
from time import perf_counter

from eval.harness.metrics import METRIC_NAMES, MetricSpec  # noqa: F401
from eval.harness.report import (
    default_result_paths,
    render_delta_table,
    write_detail_csv,
    write_detail_csv_per_variant,
)
from eval.harness.sut import RagOutput

# 「拒答类」金标准：正确行为是反问澄清 / 告知库外，而非给出可被 ragas 打分的答案。
# 按金标准 expected_category 把这两类的指标归 null（对所有被测系统一致），避免污染质量均值。
REFUSE_CATEGORIES = frozenset({"missing_info", "out_of_scope"})


def load_testset(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_dict(row) -> dict:
    """把测试集行（dict / pydantic / 带属性对象）归一成 dict。"""
    if isinstance(row, dict):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    keys = ["user_input", "reference", "reference_contexts"]
    return {k: getattr(row, k, None) for k in keys}


async def score_row(
    row: dict, sut, metric_specs: list[MetricSpec], meter=None
) -> dict:
    if meter is not None:
        meter.reset()
    t0 = perf_counter()
    out: RagOutput = await sut.answer(row["user_input"])
    latency_s = perf_counter() - t0
    base = {
        "user_input": row["user_input"],
        "reference": row.get("reference", ""),
        "response": out.response,
        "outcome": out.outcome,
        "expected_category": row.get("category", ""),
        "num_contexts": len(out.retrieved_contexts),
        "latency_s": round(latency_s, 3),
    }
    if meter is not None:
        base.update(meter.read())
    if out.outcome != "answered" or base["expected_category"] in REFUSE_CATEGORIES:
        return base
    for spec in metric_specs:
        try:
            result = await spec.metric.ascore(**spec.kwargs(row, out))
            base[spec.name] = result.value
        except Exception as e:  # noqa: BLE001 — 单指标失败不影响其他指标
            base[spec.name] = None
            base[f"{spec.name}_error"] = f"{type(e).__name__}: {e}"
    return base


def aggregate(rows: list[dict]) -> dict:
    """指标均值（仅 answered 行、忽略 None）+ outcome 分布 + 成本。"""
    outcomes: dict[str, int] = {}
    for r in rows:
        oc = r.get("outcome", "error")
        outcomes[oc] = outcomes.get(oc, 0) + 1
    answered = [r for r in rows if r.get("outcome") == "answered"]
    metric_means: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        vals = [r[name] for r in answered if r.get(name) is not None]
        metric_means[name] = (sum(vals) / len(vals)) if vals else None
    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    token_vals = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
    cost = {
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "mean_total_tokens": (sum(token_vals) / len(token_vals)) if token_vals else None,
        "total_tokens": sum(token_vals) if token_vals else None,
    }
    return {
        "total": len(rows),
        "answered": len(answered),
        "outcome_distribution": outcomes,
        "metric_means": metric_means,
        "cost": cost,
    }


# 三条 SUT 路线：naive_rag（朴素 RAG 对照组，默认 baseline）/ workflow（生产编排）/ agent（自主规划）。
# VARIANTS 仅作「可选路线名」登记表（值不再有语义，build_sut 按名分流）；
# 顺序即对比表行序，naive_rag 居首作 delta 锚。
WORKFLOW_VARIANT = "workflow"
NAIVE_VARIANT = "naive_rag"
AGENT_VARIANT = "agent"
VARIANTS = {
    NAIVE_VARIANT: None,
    WORKFLOW_VARIANT: {},
    AGENT_VARIANT: None,
}


def build_sut(name: str, index_manager, llm):
    """按路线名构造被测系统：workflow / naive_rag / agent 三选一。"""
    from eval.harness.sut import AgentSystem, DocQueryWorkflowSystem, NaiveRagSystem
    if name == WORKFLOW_VARIANT:
        return DocQueryWorkflowSystem(index_manager, llm)
    if name == NAIVE_VARIANT:
        return NaiveRagSystem(index_manager, llm)
    if name == AGENT_VARIANT:
        return AgentSystem(index_manager, llm)
    raise KeyError(name)


def resolve_baseline(baseline: str, variant_names: list[str]) -> str:
    """baseline 名不在所选变体里时回退到第一个变体（leftmost 作 delta 锚）。"""
    return baseline if baseline in variant_names else (variant_names[0] if variant_names else baseline)


async def _run_variants(testset_path, limit, names, concurrency: int = 1):
    # load_testset/score_row/aggregate 已是本模块局部函数
    from configs.embedding import configure_embedding
    from configs.llm import configure_llm
    from core.rag.data_loader import RAGIndexManager
    from eval.config import CHROMA_DIR, make_eval_embeddings, make_eval_llm
    from eval.harness.meter import attach_token_meter
    from eval.harness.metrics import build_metric_specs

    rows = load_testset(testset_path)
    if limit:
        rows = rows[:limit]
    eval_llm, eval_emb = make_eval_llm(), make_eval_embeddings()
    metric_specs = build_metric_specs(eval_llm, eval_emb)
    sut_llm = configure_llm()
    configure_embedding()
    meter = attach_token_meter(sut_llm)  # 挂在 SUT llm 上 → 只数被测系统 token（逐行 reset）
    index_manager = RAGIndexManager(persist_dir=CHROMA_DIR)

    variants = []
    detail = []  # 每条明细（带 variant 列），供 --detail 落盘
    total = len(rows)
    concurrency = max(1, concurrency)
    for vi, name in enumerate(names, 1):
        sut = build_sut(name, index_manager, sut_llm)
        # 进度走 stderr：每条 query 是真实 LLM 调用（慢），逐行刷新让控制台看得见进展，
        # 同时不污染 stdout 的 markdown 对比表。
        print(f"[变体 {vi}/{len(names)}] {name}（{total} 条, 并发 {concurrency}）",
              file=sys.stderr, flush=True)
        t0 = time.monotonic()
        if concurrency == 1:
            scored = await _score_rows_serial(rows, sut, metric_specs, meter, total, t0)
        else:
            # 并行：逐行 token 计量在并发下会串扰（meter 是 SUT llm 上的单个全局计数器），
            # 故跑前 reset 一次、跑完 read 一次取「变体级总量」，逐行不再单独计 token。
            scored = await _score_rows_parallel(
                rows, sut, metric_specs, meter, total, t0, concurrency
            )
        for s in scored:
            detail.append({"variant": name, **s})
        variants.append({"name": name, "report": aggregate(scored)})
    return variants, detail


def _progress(done: int, total: int, t0: float) -> None:
    elapsed = time.monotonic() - t0
    print(f"\r  {done}/{total}  ({elapsed:.0f}s, 均 {elapsed / done:.1f}s/条)",
          end="", file=sys.stderr, flush=True)


async def _score_rows_serial(rows, sut, metric_specs, meter, total, t0):
    """串行：保留逐行 token 计量（meter reset/read 在 score_row 内、单行独占无串扰）。"""
    # score_row 已是本模块局部函数

    scored = []
    for ri, r in enumerate(rows, 1):
        scored.append(await score_row(r, sut, metric_specs, meter=meter))
        _progress(ri, total, t0)
    print(file=sys.stderr, flush=True)  # 收尾换行，避免下个变体覆盖在同一行
    return scored


async def _score_rows_parallel(rows, sut, metric_specs, meter, total, t0, concurrency):
    """并行：信号量限流 gather；逐行不计 token，变体跑完打印一次总量。"""
    # score_row 已是本模块局部函数

    meter.reset()
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def _one(r):
        nonlocal done
        async with sem:
            res = await score_row(r, sut, metric_specs, meter=None)
        done += 1
        _progress(done, total, t0)
        return res

    scored = await asyncio.gather(*(_one(r) for r in rows))  # 结果按入参序，进度按完成序
    tok = meter.read()
    print(f"\n  变体 token 总量(并发，不逐行归因): {tok['total_tokens']}",
          file=sys.stderr, flush=True)
    return scored


def main():
    p = argparse.ArgumentParser(description="决策对比评测（ablation）")
    p.add_argument("--testset", required=True, help="测试集 jsonl（建议金标准 golden.jsonl）")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    p.add_argument("--concurrency", type=int, default=1,
                   help="每个变体内并发跑多少条 query（默认 1=串行，保留逐行 token 计量；"
                        ">1 提速但 token 仅给变体级总量，遇限流(429)调小）")
    p.add_argument("--variants", nargs="+",
                   default=list(VARIANTS),
                   choices=list(VARIANTS.keys()),
                   help=f"路线子集，可选：{list(VARIANTS.keys())}（默认两条都跑）")
    p.add_argument("--baseline", default=NAIVE_VARIANT,
                   help="作为 delta 基准的路线名（默认 naive_rag，delta 列即各路线相对朴素 RAG 的差）")
    p.add_argument("--out", default=None,
                   help="对比表 Markdown 落盘路径；缺省 eval/results/compare_<时间戳>.md")
    p.add_argument("--detail", default=None,
                   help="每条明细 CSV 落盘路径；缺省 eval/results/compare_<时间戳>_detail.csv")
    args = p.parse_args()
    args.baseline = resolve_baseline(args.baseline, args.variants)
    variants, detail = asyncio.run(
        _run_variants(args.testset, args.limit, args.variants, args.concurrency)
    )
    table = render_delta_table(variants, baseline=args.baseline)
    print(table)

    # 缺省即落盘到 eval/results（带时间戳防覆盖）；--out/--detail 可显式改路径
    default_md, default_csv = default_result_paths()
    out_path = args.out or default_md
    detail_path = args.detail or default_csv

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 决策对比（baseline={args.baseline}）\n\n")
        f.write(f"测试集：`{args.testset}`" + (f"（前 {args.limit} 条）" if args.limit else "") + "\n\n")
        f.write(table + "\n")
    print(f"\n[已存] {out_path}")

    write_detail_csv(detail, detail_path)
    print(f"[已存明细] {detail_path}（共 {len(detail)} 行 = 条数 × 变体数）")

    # 再按路线各拆一份单独明细 CSV，便于单独查看某条路线
    per_variant = write_detail_csv_per_variant(detail, detail_path)
    for variant, path in per_variant.items():
        n = sum(1 for d in detail if d.get("variant") == variant)
        print(f"[已存明细·{variant}] {path}（{n} 行）")


if __name__ == "__main__":
    main()
