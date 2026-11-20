"""命令行入口：体检、候选比较、报价快照、接受、取消、回执、复检处置、重判。

所有运行都只读 data/，写入 .local/；不会联网，也不会生成真实交易。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

from .candidates import accepted_interaction, calibrate_order_settings, compare_candidates
from .config import DEFAULT, Config
from .context import build_context, realized_summary
from .data import DataError
from .quotes import accept_with_snapshot, build_candidate_snapshot, reapproval_report
from .report import allocations_csv, consistency_checks, json_report, material_trace, text_report
from .risk import draw_scenarios
from .state import Workspace


def _parse_ids(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _config(args) -> Config:
    return DEFAULT.with_(plan_date=args.plan_date, price_valid_days=args.valid_days)


def cmd_check(args) -> int:
    from support.inspect_data import load
    manifest, tables = load()
    print(json.dumps({"项目": manifest["project"], "资料版本": manifest["version"],
                      "资料表": {name: len(rows) for name, rows in tables.items()},
                      "检查范围": "文件摘要、字段、身份与引用"}, ensure_ascii=False, indent=2))
    return 0


def cmd_context(args) -> int:
    cfg = _config(args)
    ctx = build_context(cfg, Workspace(args.root, args.local), plan_date=cfg.plan_date)
    print(json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_lots(args) -> int:
    cfg = _config(args)
    ctx = build_context(cfg, Workspace(args.root, args.local), plan_date=cfg.plan_date)
    if args.scenario:
        scenario = ctx.scenarios[args.scenario]
        print(json.dumps({lid: q.to_dict() for lid, q in scenario.items()}, ensure_ascii=False, indent=2))
        return 0
    print("| 料批 | 材质 | 来源 | 湿重 | 随机覆盖 | 状态 | 杂质均值 | 聚合物均值 | 限制 |")
    print("| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |")
    for lid, lot in sorted(ctx.lots.items()):
        print(
            f"| {lid} | {lot.material} | {lot.source} | {lot.wet_kg:.0f} | {lot.coverage_random:.0%} | "
            f"{lot.status} | {lot.impurity.mean:.4f} | "
            f"{'—' if lot.polymer is None else f'{lot.polymer.mean:.4f}'} | {'、'.join(lot.restrictions) or ''} |"
        )
    return 0


def _evaluate(ctx, cfg: Config, pending: list[str], draws):
    margins, buffers, diagnostics = calibrate_order_settings(
        ctx.orders, pending, ctx.lots, ctx.capacity, ctx.model, cfg, ctx.plan_date,
        available_wet=ctx.available_wet, draws=draws,
    )
    return compare_candidates(
        ctx.orders, pending, sorted(ctx.accepted), ctx.lots, ctx.capacity, ctx.model, cfg,
        ctx.plan_date, available_wet=ctx.available_wet, margins=margins, buffers=buffers,
        draws=draws, settings_diagnostics=diagnostics,
    )


def cmd_candidates(args) -> int:
    cfg = _config(args)
    workspace = Workspace(args.root, args.local)
    ctx = build_context(cfg, workspace, plan_date=cfg.plan_date)
    pending = _parse_ids(args.orders) or sorted(ctx.pending)
    draws = draw_scenarios(ctx.lots, ctx.model, cfg, cfg.sims_portfolio)
    evaluation = _evaluate(ctx, cfg, pending, draws)
    if args.json:
        print(json.dumps(evaluation.to_dict(cfg), ensure_ascii=False, indent=2))
        return 0
    print(f"推荐：{evaluation.recommended or '不报价'}")
    for option in evaluation.options[:args.limit]:
        gate = option.gate(cfg)
        print(
            f"- {'+'.join(option.orders) or '不接单'}：期望毛利 {option.expected_profit_cents/100:,.2f} 元，"
            f"P10 {option.simulation.profit_p10_cents/100:,.2f} 元；"
            f"缺交{'达标' if gate['缺交达标'] else '未达标'}，质量{'达标' if gate['质量达标'] else '未达标'}"
        )
    for note in evaluation.notes:
        print(f"注：{note}")
    return 0


def cmd_quote(args) -> int:
    cfg = _config(args)
    workspace = Workspace(args.root, args.local)
    knowledge = args.as_of or cfg.knowledge_cut_effective
    ctx = build_context(cfg, workspace, plan_date=cfg.plan_date, knowledge_cut=knowledge)
    pending = _parse_ids(args.orders) or sorted(ctx.pending)
    evaluation, snapshot = build_candidate_snapshot(ctx, cfg, workspace, pending=pending)
    if args.format == "json":
        print(json_report(ctx, cfg, evaluation, snapshot))
    else:
        print(text_report(ctx, cfg, evaluation, snapshot))
    if args.csv_dir:
        directory = pathlib.Path(args.csv_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{snapshot['id']}_allocations.csv").write_text(
            allocations_csv(snapshot.get("plan", {})), encoding="utf-8"
        )
    return 0 if evaluation.options else 2


def cmd_accept(args) -> int:
    cfg = _config(args)
    workspace = Workspace(args.root, args.local)
    snapshot = workspace.load_snapshot(args.snapshot)
    orders = _parse_ids(args.orders) or list(snapshot.get("pending", []))
    result = accept_with_snapshot(workspace, snapshot, orders, cfg, accepted_at=args.accepted_at)
    print(json.dumps({"接受": orders, "决策": result["decision"]["id"], "状态": result["accepted"]},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_cancel(args) -> int:
    workspace = Workspace(args.root, args.local)
    decision = workspace.register_cancellation(args.order, reason=args.reason)
    print(json.dumps({"取消": args.order, "决策": decision["id"], "说明": args.reason}, ensure_ascii=False, indent=2))
    return 0


def cmd_receipt(args) -> int:
    workspace = Workspace(args.root, args.local)
    receipt = {
        "id": args.id,
        "lot": args.lot,
        "order": args.order,
        "kind": args.kind,
        "quantity_kg": args.quantity,
        "observed_impurity": args.observed_impurity,
        "known_at": args.known_at,
    }
    receipt = {key: value for key, value in receipt.items() if value is not None}
    result = workspace.register_receipt(receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_assay(args) -> int:
    workspace = Workspace(args.root, args.local)
    result = workspace.register_assay({
        "id": args.id,
        "lot": args.lot,
        "sample_stratum": args.stratum,
        "sampling_basis": args.basis,
        "represented_wet_kg": args.represented,
        "moisture_fraction": args.moisture,
        "impurity_fraction_dry": args.impurity,
        "polymer_fraction_dry": args.polymer,
        "known_at": args.known_at,
        "supersedes": args.supersedes,
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_disposition(args) -> int:
    workspace = Workspace(args.root, args.local)
    decision = workspace.register_disposition(
        args.actual, args.disposition,
        reason=args.reason, accepted_impurity=args.observed_impurity,
        return_kg=args.return_kg,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def cmd_recheck(args) -> int:
    cfg = _config(args)
    workspace = Workspace(args.root, args.local)
    knowledge = args.as_of or cfg.knowledge_cut_effective
    ctx = build_context(cfg, workspace, plan_date=cfg.plan_date, knowledge_cut=knowledge)
    report = reapproval_report(workspace, args.snapshot, ctx, cfg, plan_date=cfg.plan_date)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["仍可兑现"] else 1


def cmd_differences(args) -> int:
    cfg = _config(args)
    workspace = Workspace(args.root, args.local)
    ctx = build_context(cfg, workspace, plan_date=cfg.plan_date)
    rows = []
    for item in workspace.list_snapshots():
        snapshot = workspace.load_snapshot(item["id"])
        if args.snapshot and item["id"] != args.snapshot:
            continue
        rows.append({"快照": item["id"], "状态": consistency_checks(ctx, snapshot)})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def cmd_trace(args) -> int:
    cfg = _config(args)
    workspace = Workspace(args.root, args.local)
    ctx = build_context(cfg, workspace, plan_date=cfg.plan_date)
    rows = material_trace(ctx, workspace)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def cmd_gaps(args) -> int:
    cfg = _config(args)
    workspace = Workspace(args.root, args.local)
    ctx = build_context(cfg, workspace, plan_date=cfg.plan_date)
    draws = draw_scenarios(ctx.lots, ctx.model, cfg, cfg.sims_portfolio)
    rows = accepted_interaction(
        ctx.orders, sorted(ctx.accepted), ctx.lots, ctx.capacity, ctx.model, cfg,
        ctx.plan_date, available_wet=ctx.available_wet, draws=draws,
    )
    print(json.dumps({
        "已接受": sorted(ctx.accepted),
        "剩余履约干重": {k: round(v, 1) for k, v in ctx.residual_remaining.items()},
        "相互影响": rows,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_realized(args) -> int:
    cfg = _config(args)
    workspace = Workspace(args.root, args.local)
    ctx = build_context(cfg, workspace, plan_date=cfg.plan_date)
    print(json.dumps(realized_summary(ctx.dataset, workspace, cfg.plan_date), ensure_ascii=False, indent=2))
    return 0


def cmd_demo(args) -> int:
    """在临时工作区跑完整流程；不写入项目 .local。"""
    import tempfile
    cfg = _config(args)
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Workspace(args.root, pathlib.Path(temporary) / ".local")
        ctx = build_context(cfg, workspace, plan_date=cfg.plan_date)
        pending = _parse_ids(args.orders) or sorted(ctx.pending)
        evaluation, snapshot = build_candidate_snapshot(ctx, cfg, workspace, pending=pending)
        print(text_report(ctx, cfg, evaluation, snapshot))
        recommended = evaluation.option_by_name(evaluation.recommended)
        if recommended is not None and set(recommended.orders) & set(pending):
            accepted = [oid for oid in recommended.orders if oid in pending]
            accept_with_snapshot(workspace, snapshot, accepted, cfg)
            ctx_later = build_context(cfg, workspace, plan_date=cfg.plan_date)
            report = reapproval_report(workspace, snapshot["id"], ctx_later, cfg,
                                       plan_date=cfg.plan_date)
            print("\n## 后续重判（演示）")
            print(json.dumps(report["重新判断"], ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recovery", description="再生料混配报价的质量风险与承销量分析")
    parser.add_argument("--root", default=None, help="项目根目录（含 data/）")
    parser.add_argument("--local", default=None, help="工作区目录，默认 .local")
    parser.add_argument("--plan-date", default=DEFAULT.plan_date, help="决策日")
    parser.add_argument("--valid-days", type=int, default=DEFAULT.price_valid_days, help="价格有效期天数")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="核对原始资料摘要与字段").set_defaults(func=cmd_check)

    p = sub.add_parser("context", help="输出当前市场上下文")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("lots", help="查看料批质量估计")
    p.add_argument("--scenario", choices=["central", "adverse", "pooled"], default=None)
    p.set_defaults(func=cmd_lots)

    p = sub.add_parser("candidates", help="比较候选方案")
    p.add_argument("--orders", nargs="*", help="待审批承购，逗号分隔或空格分隔；默认全部未接受")
    p.add_argument("--json", action="store_true")
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(func=cmd_candidates)

    p = sub.add_parser("quote", help="生成并保存报价快照")
    p.add_argument("--orders", nargs="*")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--csv-dir", default=None)
    p.add_argument("--as-of", default=None, help="知识截止时点；默认决策日结束，之后获知的资料不进入报价")
    p.set_defaults(func=cmd_quote)

    p = sub.add_parser("accept", help="接受快照中的承购")
    p.add_argument("snapshot")
    p.add_argument("--orders", nargs="*")
    p.add_argument("--accepted-at", default=None)
    p.set_defaults(func=cmd_accept)

    p = sub.add_parser("cancel", help="取消已接受但未加工的承购")
    p.add_argument("order")
    p.add_argument("--reason", default="")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("receipt", help="登记加工或客户复检回执")
    p.add_argument("--id", required=True)
    p.add_argument("--lot", required=True)
    p.add_argument("--order", required=True)
    p.add_argument("--kind", choices=["processed", "customer_assay"], required=True)
    p.add_argument("--quantity", type=float, required=True)
    p.add_argument("--observed-impurity", type=float, default=None)
    p.add_argument("--known-at", default=None)
    p.set_defaults(func=cmd_receipt)

    p = sub.add_parser("assay", help="登记新增检测")
    p.add_argument("--id", required=True)
    p.add_argument("--lot", required=True)
    p.add_argument("--stratum", required=True)
    p.add_argument("--basis", choices=["random_mass", "targeted"], required=True)
    p.add_argument("--represented", type=float, required=True)
    p.add_argument("--moisture", type=float, required=True)
    p.add_argument("--impurity", type=float, required=True)
    p.add_argument("--polymer", type=float, default=None)
    p.add_argument("--known-at", required=True)
    p.add_argument("--supersedes", default=None)
    p.set_defaults(func=cmd_assay)

    p = sub.add_parser("disposition", help="登记客户复检处置")
    p.add_argument("actual")
    p.add_argument("disposition", choices=["accept", "recheck", "dispute", "return", "pending"])
    p.add_argument("--reason", default="")
    p.add_argument("--observed-impurity", type=float, default=None)
    p.add_argument("--return-kg", type=float, default=None)
    p.set_defaults(func=cmd_disposition)

    p = sub.add_parser("recheck", help="用当前证据重新判断旧报价快照")
    p.add_argument("snapshot")
    p.add_argument("--as-of", default=None, help="按该时点的已知资料重判；默认决策日结束")
    p.set_defaults(func=cmd_recheck)

    p = sub.add_parser("differences", help="列出报价与实际差异")
    p.add_argument("--snapshot", default=None)
    p.set_defaults(func=cmd_differences)

    p = sub.add_parser("trace", help="列出单批物料去向")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("gaps", help="列出已接受承购的履约缺口与相互影响")
    p.set_defaults(func=cmd_gaps)

    p = sub.add_parser("realized", help="列出已实现加工/复检与未实现部分")
    p.set_defaults(func=cmd_realized)

    p = sub.add_parser("demo", help="在临时目录运行完整演示")
    p.add_argument("--orders", nargs="*")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (DataError, FileNotFoundError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
