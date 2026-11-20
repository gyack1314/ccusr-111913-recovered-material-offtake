"""报价快照、审批与后续重判。

快照记录“当时用了哪些证据、哪些假设、承诺了什么”。后续补检测、接受其他承购、
客户复检或回执进入时，只追加新记录并重新判断方案是否仍能兑现；旧快照不被改写。
"""
from __future__ import annotations

from datetime import datetime

from .candidates import (
    CandidateEvaluation,
    accepted_interaction,
    calibrate_order_settings,
    compare_candidates,
)
from .config import Config
from .context import MarketContext
from .risk import ScenarioDraws, draw_scenarios, requote_status
from .state import Workspace, default_price_valid_until, snapshot_id


def _evidence(lots_now: dict, used_lots: list[str]) -> dict:
    return {
        "lots": {
            lid: {
                "status": lots_now[lid].status,
                "coverage": round(lots_now[lid].coverage_random, 4),
                "impurity": round(lots_now[lid].impurity.mean, 6),
                "polymer": None if lots_now[lid].polymer is None else round(lots_now[lid].polymer.mean, 6),
                "assays": list(lots_now[lid].effective_assay_ids),
                "restrictions": list(lots_now[lid].restrictions),
                "conflicts": list(lots_now[lid].conflicts),
            }
            for lid in used_lots if lid in lots_now
        }
    }


def build_candidate_snapshot(
    ctx: MarketContext,
    cfg: Config,
    workspace: Workspace | None = None,
    pending: list[str] | None = None,
    draws: ScenarioDraws | None = None,
) -> tuple[CandidateEvaluation, dict]:
    pending = pending or sorted(ctx.pending)
    if draws is None:
        draws = draw_scenarios(ctx.lots, ctx.model, cfg, cfg.sims_portfolio)
    margins, buffers, diagnostics = calibrate_order_settings(
        ctx.orders, pending, ctx.lots, ctx.capacity, ctx.model, cfg, ctx.plan_date,
        available_wet=ctx.available_wet, draws=draws,
    )
    # 已接受承购保留其报价时的承诺裕量与缓冲；不能为了容纳新单而悄悄降低旧承诺。
    margins = {**ctx.committed_margins, **margins}
    buffers = {**ctx.committed_buffers, **buffers}
    # 若单笔校准已不可行，不应让它的“零裕量”把整组抬成看似可行；保留诊断并让
    # 联合比较按同一组校准设置重算，不可行的单会从方案中排除或在门槛中标出。
    evaluation = compare_candidates(
        ctx.orders, pending, sorted(ctx.accepted), ctx.lots, ctx.capacity, ctx.model, cfg,
        ctx.plan_date, available_wet=ctx.available_wet, margins=margins, buffers=buffers,
        draws=draws, settings_diagnostics=diagnostics,
    )
    snapshot = {
        "id": snapshot_id(pending),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "plan_date": ctx.plan_date,
        "knowledge_cut": ctx.knowledge_cut,
        "price_valid_until": default_price_valid_until(ctx.plan_date, cfg.price_valid_days),
        "pending": pending,
        "accepted": sorted(ctx.accepted),
        "orders": {oid: ctx.orders[oid].to_dict() for oid in pending if oid in ctx.orders},
        "evaluation": evaluation.to_dict(cfg),
        "scenario": {
            "common_share": draws.common_share,
            "customer_sigma": cfg.customer_assay_sigma,
            "quality_mode": cfg.quality_correlation_mode,
            "sims": len(next(iter(draws.quality.values()))),
        },
    }
    gap_rows = accepted_interaction(
        ctx.orders, sorted(ctx.accepted), ctx.lots, ctx.capacity, ctx.model, cfg,
        ctx.plan_date, available_wet=ctx.available_wet, draws=draws,
        margins={**ctx.committed_margins, **(margins or {})},
        buffers={**ctx.committed_buffers, **(buffers or {})},
    )
    snapshot["accepted_interaction"] = gap_rows

    # 快照采用的证据：待审批承购在联合计划中实际占用的料批；
    # 若推荐方案不含待审批承购，则退回到“全部待审批”的占用集合，避免快照没有证据。
    recommended = evaluation.option_by_name(evaluation.recommended)
    used_lots = sorted(
        {
            lid
            for option in evaluation.options
            for lid in option.plan.lot_use
            if set(option.orders) & set(pending)
        }
    )
    if recommended and set(recommended.orders) & set(pending):
        used_lots = sorted(recommended.plan.lot_use)
    snapshot["evidence"] = _evidence(ctx.lots, used_lots)
    if recommended:
        snapshot["plan"] = recommended.plan.to_dict()
        snapshot["verdict"] = {oid: recommended.simulation.order_risk[oid].to_dict() for oid in recommended.plan.orders}
    else:
        snapshot["plan"] = {}
        snapshot["verdict"] = {}
    snapshot["assumptions"] = cfg.assumptions()
    snapshot["quality_scenarios"] = {
        name: {lid: q.to_dict() for lid, q in scenario.items() if lid in used_lots}
        for name, scenario in ctx.scenarios.items()
    }
    sensitivity = sensitivity_grid(ctx, cfg, evaluation, draws, pending)
    snapshot["sensitivity"] = sensitivity
    if workspace is not None:
        workspace.save_snapshot(snapshot)
    return evaluation, snapshot


def sensitivity_grid(
    ctx: MarketContext,
    cfg: Config,
    evaluation: CandidateEvaluation,
    draws: ScenarioDraws,
    pending: list[str],
) -> dict:
    """数据无法区分的相关性假设：给出上下界，而不是只报一个区间。

    逐个替换关键假设并重算推荐方案的风险；不改变报价证据，只改变不确定性表达。
    """
    base = evaluation.option_by_name(evaluation.recommended)
    if base is None:
        return {"说明": "没有可行方案，无法做敏感性比较"}
    from .risk import evaluate_plan

    grid: dict[str, dict] = {}
    for share in cfg.yield_common_share_sensitivity:
        alt = draw_scenarios(
            ctx.lots, ctx.model, cfg, len(next(iter(draws.quality.values()))),
            seed=draws.seed, common_share=share,
        )
        sim = evaluate_plan(base.plan, alt, ctx.model, cfg)
        grid[f"产出共同因子_{share}"] = {
            "任一承购缺交概率": round(sim.joint_metrics["任一承购缺交概率"], 4),
            "期望毛利_分": round(sim.scenario_mean_profit_cents, 1),
            "单承购缺交概率": {oid: round(r.shortfall_probability, 4) for oid, r in sim.order_risk.items()},
        }
    for sigma in cfg.customer_assay_sigma_sensitivity:
        sim = evaluate_plan(base.plan, draws, ctx.model, cfg, customer_sigma=sigma)
        grid[f"复检误差_{sigma}"] = {
            "任一承购质量拒收概率": round(sim.joint_metrics["任一承购质量拒收概率"], 4),
            "单承购拒收概率": {oid: round(r.reject_probability, 4) for oid, r in sim.order_risk.items()},
        }
    for mode in ("adverse", "independent"):
        alt = draw_scenarios(
            ctx.lots, ctx.model, cfg, len(next(iter(draws.quality.values()))),
            seed=draws.seed, common_share=draws.common_share, quality_mode=mode,
        )
        sim = evaluate_plan(base.plan, alt, ctx.model, cfg)
        grid[f"质量相关_{mode}"] = {
            "任一承购质量拒收概率": round(sim.joint_metrics["任一承购质量拒收概率"], 4),
            "期望毛利_分": round(sim.scenario_mean_profit_cents, 1),
        }
    return {
        "推荐组合": list(base.orders),
        "情景": grid,
        "说明": "这些是模型无法从资料区分其相关性的假设；结果给出范围，不应把任一端点当作保证",
    }


def accept_with_snapshot(
    workspace: Workspace,
    snapshot: dict,
    order_ids: list[str],
    cfg: Config,
    accepted_at: str | None = None,
) -> dict:
    decision = workspace.register_acceptance(snapshot, order_ids, accepted_at)
    snapshot["accepted_at"] = workspace.state["accepted"][order_ids[0]]["accepted_at"]
    snapshot["accepted_orders"] = order_ids
    # 快照一旦保存就不再改写；接受事件单独落盘由 workspace 的状态与决策日志承担。
    return {
        "decision": decision,
        "accepted": {oid: workspace.state["accepted"][oid] for oid in order_ids},
    }


def reapproval_report(
    workspace: Workspace,
    snapshot_id_value: str,
    ctx: MarketContext,
    cfg: Config,
    plan_date: str | None = None,
    draws: ScenarioDraws | None = None,
) -> dict:
    """在补充检测或接受新承购后，用当前证据重新判断旧快照是否仍能兑现。"""
    snapshot = workspace.load_snapshot(snapshot_id_value)
    plan_date = plan_date or ctx.plan_date
    if draws is None:
        draws = draw_scenarios(ctx.lots, ctx.model, cfg, cfg.sims_portfolio)
    verdict = requote_status(snapshot, ctx.lots, ctx.available_wet, ctx.orders, cfg, ctx.capacity, ctx.model, plan_date)
    verdict.setdefault("原因", [])
    orders = snapshot.get("pending", [])
    live = [oid for oid in orders if oid in ctx.orders]
    recompute = None
    if live:
        evaluation = compare_candidates(
            ctx.orders, live, sorted(ctx.accepted), ctx.lots, ctx.capacity, ctx.model, cfg,
            plan_date, available_wet=ctx.available_wet, draws=draws,
        )
        recompute = evaluation.to_dict(cfg)
    if recompute:
        old = snapshot.get("evaluation", {}).get("推荐")
        new = recompute.get("推荐")
        if old != new:
            verdict["原因"].append(f"按当前证据重算，推荐组合由 {old} 变为 {new}；原报价承诺不能自动沿用")
        old_options = {
            tuple(option.get("承购组合")): option.get("期望毛利_分")
            for option in snapshot.get("evaluation", {}).get("方案", [])
        }
        new_options = {tuple(option.get("承购组合")): option.get("期望毛利_分") for option in recompute.get("方案", [])}
        for key in set(old_options) & set(new_options):
            before = old_options[key]
            after = new_options[key]
            if before is not None and after is not None and abs(after - before) > 5000:
                verdict["原因"].append(
                    f"同一组合 {list(key)} 的期望毛利由 {before/100:.2f} 元变为 {after/100:.2f} 元，超过 50 元容差"
                )
    verdict["需要重新审批"] = bool(verdict["原因"])
    if recompute and not verdict["需要重新审批"]:
        verdict["建议"] = "证据与风险仍在快照容差内，可按原承诺继续执行并持续录入回执"
    valid_until = snapshot.get("price_valid_until")
    expired = bool(valid_until and plan_date > valid_until)
    if expired:
        verdict["原因"].append(f"价格有效期截至 {valid_until}，决策日 {plan_date} 已过期")
    needs_reapproval = bool(verdict["需要重新审批"] or expired)
    if needs_reapproval:
        verdict["建议"] = "暂停按原快照交付；重新生成候选方案并重新审批"
    return {
        "快照": snapshot_id_value,
        "快照创建": snapshot.get("created_at"),
        "价格有效期至": valid_until,
        "已过期": expired,
        "仍可兑现": not needs_reapproval,
        "重新判断": verdict,
        "当前重算": recompute,
        "处理建议": (
            "暂停按原快照交付；重新生成候选方案并重新审批"
            if needs_reapproval
            else "证据与风险仍在快照容差内，可按原承诺继续执行并持续录入回执"
        ),
        "建议": verdict.get("建议"),
        "原则": "不能只更新屏幕上的可用吨数而保留旧承诺；必须重新评估并重新审批",
    }
