"""候选方案比较与联合情景下的机会成本。

每次审批同时评估：
* 继续接受待审批询价（enforce 该单足量）；
* 不接受该询价（baseline）；
* 若该单必须接受，则同时给出它对其他承购造成的缺交/质量代价。
多单共享原料与加工日，因此所有比较都在同一组蒙特卡洛抽样和同一联合计划上完成，
不做“按客户各自最优再加总”。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable

from .config import Config
from .planning import OrderSpec, PlanResult, QualityMargins, solve_plan
from .quality import LotQuality
from .risk import ScenarioDraws, SimulationResult, calibrate_plan, draw_scenarios, evaluate_plan
from .yield_model import YieldModel


@dataclass
class PortfolioOption:
    orders: tuple[str, ...]
    plan: PlanResult
    simulation: SimulationResult

    @property
    def contract_dry_kg(self) -> float:
        return float(sum(order.contract_dry_kg for order in self.plan.orders.values()))

    @property
    def expected_profit_cents(self) -> float:
        return self.simulation.scenario_mean_profit_cents

    def gate(self, cfg: Config) -> dict:
        short_ok = all(r.passes_shortfall for r in self.simulation.order_risk.values())
        quality_ok = all(r.passes_quality for r in self.simulation.order_risk.values())
        return {
            "缺交达标": short_ok,
            "质量达标": quality_ok,
            "全部达标": short_ok and quality_ok,
        }

    def to_dict(self, cfg: Config) -> dict:
        return {
            "承购组合": list(self.orders),
            "合同干重": round(self.contract_dry_kg, 1),
            "计划毛利_分": round(self.plan.profit_cents, 1),
            "期望毛利_分": round(self.expected_profit_cents, 1),
            "毛利P10_分": round(self.simulation.profit_p10_cents, 1),
            "启用加工日": list(self.plan.open_days),
            "缺交概率": {k: round(v.shortfall_probability, 4) for k, v in self.simulation.order_risk.items()},
            "质量拒收概率": {k: round(v.reject_probability, 4) for k, v in self.simulation.order_risk.items()},
            **{"门槛": self.gate(cfg)},
            "联合指标": {k: round(v, 4) for k, v in self.simulation.joint_metrics.items()},
        }


@dataclass
class CandidateEvaluation:
    pending: tuple[str, ...]
    accepted: tuple[str, ...]
    options: list[PortfolioOption]
    recommended: str
    marginal: dict
    notes: list[str] = field(default_factory=list)

    def option_by_name(self, name: str) -> PortfolioOption | None:
        for option in self.options:
            if "+".join(option.orders) == name or (not option.orders and name == "none"):
                return option
        key = tuple(sorted(name.split("+"))) if name and name != "none" else ()
        for option in self.options:
            if tuple(sorted(option.orders)) == key:
                return option
        return None

    def to_dict(self, cfg: Config) -> dict:
        return {
            "待审批": list(self.pending),
            "已接受": list(self.accepted),
            "推荐": self.recommended,
            "方案": [option.to_dict(cfg) for option in self.options],
            "边际影响": self.marginal,
            "说明": list(self.notes),
        }


def _simulate(plan: PlanResult, draws: ScenarioDraws, model: YieldModel, cfg: Config, quality_limit: float | None) -> SimulationResult:
    return evaluate_plan(plan, draws, model, cfg, quality_limit=quality_limit)


def calibrate_order_settings(
    orders: dict[str, OrderSpec],
    order_ids: Iterable[str],
    lots: dict[str, LotQuality],
    capacity: list[dict],
    model: YieldModel,
    cfg: Config,
    plan_date: str,
    available_wet: dict[str, float] | None = None,
    draws: ScenarioDraws | None = None,
    quality_limit: float | None = None,
) -> tuple[dict[str, QualityMargins], dict[str, float], dict[str, dict]]:
    """逐单校准质量裕量与产出缓冲，作为联合比较的共同输入。

    这里逐单校准只为得到“该单自身可承诺所需的安全边际”，不把各单结果相加；
    真正的收入、缺交与拒收仍在联合计划中重新计算。
    """
    if draws is None:
        draws = draw_scenarios(lots, model, cfg, max(400, cfg.sims_calibration // 2))
    margins: dict[str, QualityMargins] = {}
    buffers: dict[str, float] = {}
    diagnostics: dict[str, dict] = {}
    for oid in order_ids:
        if oid not in orders:
            continue
        plan, sim, log = calibrate_plan(
            {oid: orders[oid]}, lots, capacity, model, cfg, plan_date,
            available_wet=available_wet, enforce_orders={oid}, draws=draws,
            quality_limit=quality_limit,
        )
        if not plan.feasible:
            diagnostics[oid] = {"可行": False, "原因": plan.message, "松弛": plan.relaxation}
            continue
        margins[oid] = plan.margins.get(oid, QualityMargins())
        buffers[oid] = plan.buffers.get(oid, 0.0)
        record = sim.order_risk[oid]
        diagnostics[oid] = {
            "可行": True,
            "质量裕量": margins[oid].to_dict(),
            "产出缓冲": buffers[oid],
            "缺交概率": round(record.shortfall_probability, 4),
            "质量拒收概率": round(record.reject_probability, 4),
            "缺交达标": record.passes_shortfall,
            "质量达标": record.passes_quality,
            "校准记录": {k: v[-2:] for k, v in log.items()},
        }
    return margins, buffers, diagnostics


def _prepare(
    orders: dict[str, OrderSpec],
    lots: dict[str, LotQuality],
    capacity: list[dict],
    model: YieldModel,
    cfg: Config,
    plan_date: str,
    available_wet: dict[str, float] | None,
    margins: dict[str, QualityMargins],
    buffers: dict[str, float],
    enforce: set[str],
) -> PlanResult:
    return solve_plan(
        orders, lots, capacity, model, cfg, plan_date,
        available_wet=available_wet, margins=margins, buffers=buffers,
        enforce_orders=enforce,
    )


def compare_candidates(
    orders: dict[str, OrderSpec],
    pending: Iterable[str],
    accepted: Iterable[str],
    lots: dict[str, LotQuality],
    capacity: list[dict],
    model: YieldModel,
    cfg: Config,
    plan_date: str,
    available_wet: dict[str, float] | None = None,
    margins: dict[str, QualityMargins] | None = None,
    buffers: dict[str, float] | None = None,
    draws: ScenarioDraws | None = None,
    quality_limit: float | None = None,
    max_pending_subset: int = 6,
    settings_diagnostics: dict[str, dict] | None = None,
) -> CandidateEvaluation:
    """枚举“接受待审批询价的哪些子集”，其余已接受承购始终保留。

    待审批单可能相互竞争同一原料；因此按子集评估，而不是逐单相加。
    """
    pending = tuple(sorted(pending))
    accepted = tuple(sorted(accepted))
    margins = margins or {}
    buffers = buffers or {}
    if draws is None:
        draws = draw_scenarios(lots, model, cfg, cfg.sims_calibration)

    base_orders = {oid: orders[oid] for oid in accepted if oid in orders}
    options: list[PortfolioOption] = []
    infeasible: list[dict] = []
    # 当待审批单较多时，先评估单笔与全选，再对中间子集抽样，避免组合爆炸。
    subsets: list[tuple[str, ...]] = []
    n = len(pending)
    all_subsets = [combo for size in range(0, min(n, max_pending_subset) + 1) for combo in itertools.combinations(pending, size)]
    if len(all_subsets) > 40:
        # 组合数增长很快时，保留：全选、不选、每个单笔，以及按单笔期望毛利排序的
        # 贪心扩张路径。这样不会只因为枚举上限把高价值组合漏掉。
        key_subsets = {(), pending}
        for oid in pending:
            key_subsets.add((oid,))
        greedy: list[str] = []
        remaining = list(pending)
        key_subsets.add(tuple(remaining))
        while remaining and len(greedy) < len(pending):
            best = None
            for oid in remaining:
                order = orders[oid]
                score = order.revenue_cents - order.shortfall_rate() * 0.05 * order.contract_dry_kg
                if best is None or score > best[1]:
                    best = (oid, score)
            greedy.append(best[0])
            remaining.remove(best[0])
            key_subsets.add(tuple(sorted(greedy)))
        all_subsets = sorted(key_subsets, key=lambda c: (len(c), c))
    for subset in all_subsets:
        subset_orders = dict(base_orders)
        subset_orders.update({oid: orders[oid] for oid in subset})
        if not subset_orders:
            continue
        plan = _prepare(subset_orders, lots, capacity, model, cfg, plan_date, available_wet,
                        {oid: margins.get(oid, QualityMargins()) for oid in subset_orders},
                        {oid: buffers.get(oid, 0.0) for oid in subset_orders},
                        set(subset))
        if not plan.feasible:
            infeasible.append({"组合": list(subset), "原因": plan.message, "松弛": plan.relaxation})
            continue
        sim = _simulate(plan, draws, model, cfg, quality_limit)
        options.append(PortfolioOption(orders=tuple(sorted(subset_orders)), plan=plan, simulation=sim))
    subsets = [option.orders for option in options]

    if not options:
        return CandidateEvaluation(
            pending=pending, accepted=accepted, options=[], recommended="infeasible",
            marginal={}, notes=["没有任何可行组合"] + [item["原因"] for item in infeasible],
        )

    def rank_key(option: PortfolioOption):
        gate = option.gate(cfg)
        return (
            0 if gate["全部达标"] else 1,
            -option.expected_profit_cents,
        )

    options.sort(key=rank_key)
    best = options[0]
    best_names = set(best.orders)
    full = next((option for option in options if set(option.orders) == set(base_orders) | set(pending)), None)
    without_pending = next((option for option in options if set(option.orders) == set(base_orders)), None)

    marginal = {
        "推荐方案": {},
        "相对不含待审批的基线": {},
        "相对接受全部待审批": {},
    }
    if without_pending is not None and accepted:
        for oid in pending:
            if oid not in best.orders:
                continue
            # 接受该单对“已接受承购”整体风险的影响；同一联合计划内比较。
            affected = {}
            for acc in accepted:
                after = best.simulation.order_risk.get(acc)
                before = without_pending.simulation.order_risk.get(acc)
                if after is None or before is None:
                    continue
                affected[acc] = {
                    "缺交概率变化": round(after.shortfall_probability - before.shortfall_probability, 4),
                    "质量拒收概率变化": round(after.reject_probability - before.reject_probability, 4),
                    "期望产出变化_kg": round(after.mean_output_dry_kg - before.mean_output_dry_kg, 1),
                }
            marginal.setdefault("接受该单对已接受承购的影响", {})[oid] = affected
    if without_pending is not None and full is not None:
        for oid in accepted:
            base = without_pending.simulation.order_risk.get(oid)
            after = full.simulation.order_risk.get(oid)
            if base is None or after is None:
                continue
            marginal["相对不含待审批的基线"][oid] = {
                "缺交概率变化": round(after.shortfall_probability - base.shortfall_probability, 4),
                "质量拒收概率变化": round(after.reject_probability - base.reject_probability, 4),
                "期望产出变化_kg": round(after.mean_output_dry_kg - base.mean_output_dry_kg, 1),
            }
        marginal["相对不含待审批的基线"]["系统合计"] = {
            "期望毛利变化_分": round(full.expected_profit_cents - without_pending.expected_profit_cents, 1),
            "合同干重变化": round(full.contract_dry_kg - without_pending.contract_dry_kg, 1),
        }

    baseline = without_pending
    base_label = "相对不含已接受承购的空计划" if not accepted else "相对不含待审批单的基线"
    for oid, record in best.simulation.order_risk.items():
        base_record = baseline.simulation.order_risk.get(oid) if baseline else None
        marginal["推荐方案"][oid] = {
            "缺交概率": round(record.shortfall_probability, 4),
            "质量拒收概率": round(record.reject_probability, 4),
            "相对基线_缺交变化": None if base_record is None else round(record.shortfall_probability - base_record.shortfall_probability, 4),
            "相对基线_拒收变化": None if base_record is None else round(record.reject_probability - base_record.reject_probability, 4),
            "基线说明": base_label if base_record is not None else "基线与推荐方案相同，未单独衡量挤占",
        }
    if not accepted:
        marginal["相对基线的局限"] = (
            "没有已接受承购时，“不接待审批单”就是空计划；挤占只能在已接受承购之间比较，"
            "不能把空计划当成真实的替代收益"
        )

    best_gate = best.gate(cfg)
    notes = []
    if settings_diagnostics:
        marginal["校准诊断"] = settings_diagnostics
        if any(not info.get("可行", True) for info in settings_diagnostics.values()):
            notes.append("部分承购在单独校准时已不可行；联合比较中只能按受限方案处理")
    if not best_gate["全部达标"]:
        notes.append("推荐方案仍有承购超出缺交或质量风险上限；应视为受限方案，不得按足量承诺审批")
    if infeasible:
        notes.append(f"{len(infeasible)} 个组合因质量、原料或产能不可行而未列出")
    if not best_names >= set(base_orders):
        notes.append("推荐组合不含全部已接受承购，说明已接受承诺之间也存在挤占，需重新协商")
    return CandidateEvaluation(
        pending=pending, accepted=accepted, options=options,
        recommended="+".join(best.orders) if best.orders else "none",
        marginal=marginal, notes=notes,
    )


def accepted_interaction(
    orders: dict[str, OrderSpec],
    accepted: Iterable[str],
    lots: dict[str, LotQuality],
    capacity: list[dict],
    model: YieldModel,
    cfg: Config,
    plan_date: str,
    available_wet: dict[str, float] | None = None,
    draws: ScenarioDraws | None = None,
    margins: dict[str, QualityMargins] | None = None,
    buffers: dict[str, float] | None = None,
    quality_limit: float | None = None,
) -> list[dict]:
    """已接受承购之间的相互挤占：单看一单与看全部时，其他单的缺交/质量如何变化。

    这不是把客户各自最优相加，而是在同一计划里逐一比较。
    """
    accepted = tuple(sorted(accepted))
    if not accepted:
        return []
    if draws is None:
        draws = draw_scenarios(lots, model, cfg, cfg.sims_calibration)
    margins = margins or {}
    buffers = buffers or {}
    combined_plan = _prepare(
        {oid: orders[oid] for oid in accepted}, lots, capacity, model, cfg, plan_date,
        available_wet, {oid: margins.get(oid, QualityMargins()) for oid in accepted},
        {oid: buffers.get(oid, 0.0) for oid in accepted}, set(),
    )
    if not combined_plan.feasible:
        return [{"说明": "已接受承购的联合计划不可行", "承购": list(accepted), "原因": combined_plan.message}]
    combined = _simulate(combined_plan, draws, model, cfg, quality_limit)

    rows = []
    for oid in accepted:
        alone_plan = _prepare(
            {oid: orders[oid]}, lots, capacity, model, cfg, plan_date, available_wet,
            {oid: margins.get(oid, QualityMargins())}, {oid: buffers.get(oid, 0.0)}, set(accepted),
        )
        alone = _simulate(alone_plan, draws, model, cfg, quality_limit) if alone_plan.feasible else None
        others = {}
        for other in accepted:
            if other == oid:
                continue
            joint_record = combined.order_risk.get(other)
            alone_record = alone.order_risk.get(other) if alone else None
            # “只做该单”时其他承购不在计划内，交付为 0，缺交率是 100%，不是缺失值。
            others[other] = {
                "只做该单时是否计划了对方": alone_record is not None,
                "只做该单时的缺交概率": (round(alone_record.shortfall_probability, 4)
                                        if alone_record is not None else 1.0),
                "联合计划中的缺交概率": None if joint_record is None else round(joint_record.shortfall_probability, 4),
                "只做该单时的质量拒收概率": None if alone_record is None else round(alone_record.reject_probability, 4),
                "联合计划中的质量拒收概率": None if joint_record is None else round(joint_record.reject_probability, 4),
            }
        alone_record = alone.order_risk.get(oid) if alone else None
        joint_record = combined.order_risk.get(oid)
        rows.append({
            "承购": oid,
            "单独计划毛利_分": None if alone is None else round(alone.scenario_mean_profit_cents, 1),
            "联合计划毛利_分": round(combined.scenario_mean_profit_cents, 1),
            "单独计划缺交概率": None if alone_record is None else round(alone_record.shortfall_probability, 4),
            "联合计划缺交概率": None if joint_record is None else round(joint_record.shortfall_probability, 4),
            "对其他承购的影响": others,
        })
    return rows
