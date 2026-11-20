"""蒙特卡洛风险模拟与计划校准。

每个情景一次性抽取整条供应链的随机因素，再在同一组抽样上评估所有承购：
* 料批质量：由估计标准差与（情景设定的）指标相关结构共同抽取；同一料批的
  所有用量共享同一质量实现，不重复抽样“同一吨料”。
* 产出率：料批+配方的独立项 + 全厂共同因子（共同因子占比可调，用于表达
  “产出率在配方间相关”的资料不确定）。
* 客户复检：在真实质量上叠加测量误差，按整车抽样，得到质量拒收事件。
* 加工回执：已交批次的争议结论会改变剩余履约与已实现利润，但不改写当时报价证据。

校准采用“分位数起点 + 逐步增大名义缓冲”的方式：先用正态近似给出起点，
再用模拟复核实际违反率，加入由弱到强的分布尾部保守量。目标不是把区间说成
保证，而是让风险数字本身经过同一模型的独立复核。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .planning import OrderSpec, PlanResult, QualityMargins, ZERO_MARGINS, solve_plan
from .quality import LotQuality, quality_correlation
from .yield_model import YieldModel


@dataclass
class ScenarioDraws:
    quality: dict[str, np.ndarray]  # lot -> (S, 3)
    yield_saleable: dict[str, np.ndarray]  # (lot, recipe) -> (S,)
    yield_dry_output: dict[str, np.ndarray]  # (lot, recipe) -> (S,)
    seed: int
    common_share: float


@dataclass
class OrderRisk:
    order: str
    contract_dry_kg: float
    mean_output_dry_kg: float
    p10_output: float
    p50_output: float
    p90_output: float
    shortfall_probability: float
    mean_shortfall_kg: float
    mean_impurity: float | None
    p90_impurity: float | None
    mean_polymer: float | None
    p10_polymer: float | None
    reject_probability: float
    truck_fail_probability: float
    expected_rejected_trucks: float
    expected_revenue_cents: float
    expected_shortfall_penalty_cents: float
    expected_reject_penalty_cents: float
    risk_limit: float
    quality_limit: float
    passes_shortfall: bool
    passes_quality: bool
    worst_truck_impurity: float | None = None
    worst_truck_polymer: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        def r(value, digits=4):
            return None if value is None else round(float(value), digits)
        return {
            "承购": self.order,
            "合同干重": r(self.contract_dry_kg, 1),
            "产出干重均值": r(self.mean_output_dry_kg, 1),
            "产出P10": r(self.p10_output, 1),
            "产出P50": r(self.p50_output, 1),
            "产出P90": r(self.p90_output, 1),
            "缺交概率": r(self.shortfall_probability, 4),
            "平均短交_kg": r(self.mean_shortfall_kg, 1),
            "杂质均值": r(self.mean_impurity, 6),
            "杂质P90": r(self.p90_impurity, 6),
            "聚合物均值": r(self.mean_polymer, 6),
            "聚合物P10": r(self.p10_polymer, 6),
            "质量拒收概率": r(self.reject_probability, 4),
            "单车不合格概率": r(self.truck_fail_probability, 4),
            "期望拒收整车数": r(self.expected_rejected_trucks, 3),
            "期望收入_分": r(self.expected_revenue_cents, 1),
            "期望短交代价_分": r(self.expected_shortfall_penalty_cents, 1),
            "期望拒收损失_分": r(self.expected_reject_penalty_cents, 1),
            "缺交上限": self.risk_limit,
            "质量上限": self.quality_limit,
            "缺交达标": self.passes_shortfall,
            "质量达标": self.passes_quality,
            "备注": self.note,
        }


@dataclass
class SimulationResult:
    order_risk: dict[str, OrderRisk]
    scenario_mean_profit_cents: float
    profit_p10_cents: float
    profit_p50_cents: float
    profit_p90_cents: float
    probability_loss: float
    joint_metrics: dict[str, float]
    n_sims: int
    common_share: float
    customer_sigma: float
    quality_mode: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "承购": {k: v.to_dict() for k, v in self.order_risk.items()},
            "期望毛利_分": round(self.scenario_mean_profit_cents, 1),
            "毛利P10_分": round(self.profit_p10_cents, 1),
            "毛利P50_分": round(self.profit_p50_cents, 1),
            "毛利P90_分": round(self.profit_p90_cents, 1),
            "亏损概率": round(self.probability_loss, 4),
            "联合指标": {k: round(v, 5) for k, v in self.joint_metrics.items()},
            "抽样次数": self.n_sims,
            "产出率共同因子占比": self.common_share,
            "客户复检误差标准差": self.customer_sigma,
            "质量相关情景": self.quality_mode,
            "说明": list(self.notes),
        }


def _correlation_matrix(mode: str) -> np.ndarray:
    corr = quality_correlation("", mode)
    # 保证正定
    eig = np.linalg.eigvalsh(corr)
    if eig.min() <= 1e-8:
        corr = corr + np.eye(3) * (1e-6 - eig.min())
    return corr


def draw_scenarios(
    lots: dict[str, LotQuality],
    model: YieldModel,
    cfg: Config,
    n_sims: int,
    seed: int | None = None,
    common_share: float | None = None,
    quality_mode: str | None = None,
) -> ScenarioDraws:
    seed = cfg.random_seed if seed is None else seed
    common_share = cfg.yield_common_share if common_share is None else common_share
    quality_mode = cfg.quality_correlation_mode if quality_mode is None else quality_mode
    rng = np.random.default_rng(seed)
    corr = _correlation_matrix(quality_mode)
    chol = np.linalg.cholesky(corr)

    quality: dict[str, np.ndarray] = {}
    for lid, lot in lots.items():
        z = rng.standard_normal((n_sims, 3)) @ chol.T
        means = lot.quality_vector()
        sds = lot.sd_vector()
        draws = means[None, :] + z * sds[None, :]
        draws[:, 0] = np.clip(draws[:, 0], 0.0, 0.95)
        draws[:, 1] = np.clip(draws[:, 1], 0.0, 1.0)
        draws[:, 2] = np.clip(draws[:, 2], 0.0, 1.0)
        quality[lid] = draws

    common = rng.standard_normal(n_sims)
    saleable: dict[str, np.ndarray] = {}
    dry_output: dict[str, np.ndarray] = {}
    for lid, lot in lots.items():
        moisture = quality[lid][:, 0]
        impurity = quality[lid][:, 1]
        for recipe in model.recipes:
            base = model.saleable_yield_vector(moisture, impurity, recipe)
            idio = rng.standard_normal(n_sims)
            z = math.sqrt(max(common_share, 0.0)) * common + math.sqrt(max(1.0 - common_share, 0.0)) * idio
            latent = np.clip(base + model.residual_sd * z, 0.05, 1.0)
            key = f"{lid}|{recipe}"
            saleable[key] = latent
            dry_output[key] = latent * (1.0 - moisture) * lot.wet_kg
    return ScenarioDraws(quality=quality, yield_saleable=saleable, yield_dry_output=dry_output,
                         seed=seed, common_share=common_share)


def _plan_output_draws(plan: PlanResult, draws: ScenarioDraws, n_sims: int) -> dict[str, np.ndarray]:
    """把计划中的湿重分配转成每个情景下的可售干重。"""
    output: dict[str, np.ndarray] = {}
    for allocation in plan.allocations:
        key = f"{allocation.lot}|{allocation.recipe}"
        if key not in draws.yield_saleable:
            raise KeyError(f"情景缺少 {key}")
        factor = draws.yield_saleable[key]
        output.setdefault(allocation.order, np.zeros(n_sims))
        output[allocation.order] += allocation.wet_kg * (1.0 - draws.quality[allocation.lot][:, 0]) * factor
    for oid in plan.orders:
        output.setdefault(oid, np.zeros(n_sims))
    return output


def _quality_draws(
    plan: PlanResult, draws: ScenarioDraws, n_sims: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray | None], dict[str, np.ndarray]]:
    """返回 (产出干重, 订单混合聚合物, 订单混合杂质)。"""
    weight: dict[str, np.ndarray] = {}
    impurity_sum: dict[str, np.ndarray] = {}
    polymer_sum: dict[str, np.ndarray] = {}
    for allocation in plan.allocations:
        key = f"{allocation.lot}|{allocation.recipe}"
        mass = allocation.wet_kg * (1.0 - draws.quality[allocation.lot][:, 0]) * draws.yield_saleable[key]
        weight.setdefault(allocation.order, np.zeros(n_sims))
        impurity_sum.setdefault(allocation.order, np.zeros(n_sims))
        polymer_sum.setdefault(allocation.order, np.zeros(n_sims))
        weight[allocation.order] += mass
        impurity_sum[allocation.order] += mass * draws.quality[allocation.lot][:, 1]
        if allocation.lot_polymer is not None:
            polymer_sum[allocation.order] += mass * draws.quality[allocation.lot][:, 2]
    impurity_frac: dict[str, np.ndarray] = {}
    polymer_frac: dict[str, np.ndarray | None] = {}
    for oid in plan.orders:
        w = weight.get(oid)
        if w is None or np.all(w <= 0):
            impurity_frac[oid] = np.zeros(n_sims)
            polymer_frac[oid] = None
        else:
            safe = np.where(w > 0, w, 1.0)
            impurity_frac[oid] = impurity_sum[oid] / safe
            has_polymer = plan.orders[oid].min_polymer is not None
            polymer_frac[oid] = (polymer_sum[oid] / safe) if has_polymer else None
    return weight, polymer_frac, impurity_frac  # type: ignore[return-value]


def evaluate_plan(
    plan: PlanResult,
    draws: ScenarioDraws,
    model: YieldModel,
    cfg: Config,
    customer_sigma: float | None = None,
    quality_limit: float | None = None,
    n_sims: int | None = None,
) -> SimulationResult:
    customer_sigma = cfg.customer_assay_sigma if customer_sigma is None else customer_sigma
    n_sims = len(next(iter(draws.quality.values()))) if n_sims is None else n_sims
    output = _plan_output_draws(plan, draws, n_sims)
    weight, polymer, impurity = _quality_draws(plan, draws, n_sims)
    rng = np.random.default_rng(draws.seed + 9176)

    # 计划固定成本在情景中不变；变动成本随湿重固定。
    fixed_cost = plan.fixed_cost_cents
    variable_cost = plan.variable_cost_cents

    risk: dict[str, OrderRisk] = {}
    # 利润 = 合同收入 - 加工成本 - 缺交收入损失与违约 - 拒收整车收入损失与违约
    profit_total = np.full(n_sims, -fixed_cost - variable_cost, dtype=float)
    shortfall_any = np.zeros(n_sims, dtype=bool)
    reject_any = np.zeros(n_sims, dtype=bool)
    total_revenue = np.zeros(n_sims)
    total_short_penalty = np.zeros(n_sims)
    total_reject_penalty = np.zeros(n_sims)

    for oid, order in plan.orders.items():
        produced = output[oid]
        shortfall = np.maximum(0.0, order.contract_dry_kg - produced)
        shortfall_event = shortfall > cfg.shortfall_tolerance_kg
        trucks = order.trucks
        mass_per_truck = order.contract_dry_kg / max(trucks, 1)

        imp = impurity[oid]
        poly = polymer[oid]
        imp_limit = order.max_impurity
        poly_limit = order.min_polymer
        truck_fail = np.zeros(n_sims, dtype=bool)
        rejected_trucks = np.zeros(n_sims)
        # 逐车抽样：把订单混合质量按整车离散化。质量误差只在真正交付的车上发生。
        for truck in range(trucks):
            delivered = np.clip(produced - truck * mass_per_truck, 0.0, mass_per_truck)
            has_mass = delivered > 1e-6
            fail = np.zeros(n_sims, dtype=bool)
            if imp_limit is not None:
                observed = imp + rng.normal(0.0, customer_sigma, n_sims)
                fail |= (observed > imp_limit) & has_mass
            if poly_limit is not None and poly is not None:
                observed_p = poly + rng.normal(0.0, customer_sigma, n_sims)
                fail |= (observed_p < poly_limit) & has_mass
            truck_fail |= fail & has_mass
            rejected_trucks += fail & has_mass

        reject_event = truck_fail if cfg.quality_gate_any_truck else (rejected_trucks > 0)
        reject_penalty = rejected_trucks * (order.reject_penalty_cents / max(trucks, 1))
        rejected_mass = np.minimum(produced, rejected_trucks * mass_per_truck)
        revenue = (produced - shortfall - rejected_mass) * order.price_cents_per_dry_kg
        short_pen = shortfall * order.shortfall_rate()
        reject_loss = rejected_mass * order.price_cents_per_dry_kg + reject_penalty

        profit_total += revenue - short_pen - reject_loss
        total_revenue += revenue
        total_short_penalty += short_pen
        total_reject_penalty += reject_loss
        shortfall_any |= shortfall_event
        reject_any |= reject_event

        q_limit = order.risk_limit if quality_limit is None else quality_limit
        risk[oid] = OrderRisk(
            order=oid,
            contract_dry_kg=order.contract_dry_kg,
            mean_output_dry_kg=float(np.mean(produced)),
            p10_output=float(np.percentile(produced, 10)),
            p50_output=float(np.percentile(produced, 50)),
            p90_output=float(np.percentile(produced, 90)),
            shortfall_probability=float(np.mean(shortfall_event)),
            mean_shortfall_kg=float(np.mean(shortfall)),
            mean_impurity=float(np.mean(imp)) if imp_limit is not None else None,
            p90_impurity=float(np.percentile(imp, 90)) if imp_limit is not None else None,
            mean_polymer=float(np.mean(poly)) if poly is not None else None,
            p10_polymer=float(np.percentile(poly, 10)) if poly is not None else None,
            reject_probability=float(np.mean(reject_event)),
            truck_fail_probability=float(np.mean(truck_fail)),
            expected_rejected_trucks=float(np.mean(rejected_trucks)),
            expected_revenue_cents=float(np.mean(revenue)),
            expected_shortfall_penalty_cents=float(np.mean(short_pen)),
            expected_reject_penalty_cents=float(np.mean(reject_loss)),
            risk_limit=order.risk_limit,
            quality_limit=q_limit,
            passes_shortfall=bool(np.mean(shortfall_event) <= order.risk_limit + 1e-12),
            passes_quality=bool(np.mean(reject_event) <= q_limit + 1e-12),
            note="" if customer_sigma > 0 else "客户复检误差设为 0 仅用于敏感性对照",
        )

    scenario_profit = profit_total - total_short_penalty * 0  # 已计入
    return SimulationResult(
        order_risk=risk,
        scenario_mean_profit_cents=float(np.mean(profit_total)),
        profit_p10_cents=float(np.percentile(profit_total, 10)),
        profit_p50_cents=float(np.percentile(profit_total, 50)),
        profit_p90_cents=float(np.percentile(profit_total, 90)),
        probability_loss=float(np.mean(profit_total < 0)),
        joint_metrics={
            "任一承购缺交概率": float(np.mean(shortfall_any)),
            "任一承购质量拒收概率": float(np.mean(reject_any)),
            "同时缺交与拒收概率": float(np.mean(shortfall_any & reject_any)),
            "期望收入_分": float(np.mean(total_revenue)),
            "期望短交代价_分": float(np.mean(total_short_penalty)),
            "期望拒收损失_分": float(np.mean(total_reject_penalty)),
        },
        n_sims=n_sims,
        common_share=draws.common_share,
        customer_sigma=customer_sigma,
        quality_mode=cfg.quality_correlation_mode,
        notes=[
            "质量与产出不确定性按同一组抽样联合评估，所有承购共享同一情景",
            "预测区间描述的是本模型下的条件分布，不是对每车质量的保证",
        ],
    )


def calibrate_plan(
    orders: dict[str, OrderSpec],
    lots: dict[str, LotQuality],
    capacity: list[dict],
    model: YieldModel,
    cfg: Config,
    plan_date: str,
    available_wet: dict[str, float] | None = None,
    enforce_orders: set[str] | None = None,
    draws: ScenarioDraws | None = None,
    quality_limit: float | None = None,
    customer_sigma: float | None = None,
) -> tuple[PlanResult, SimulationResult, dict]:
    """求解并校准：质量裕量按订单风险上调，产出缓冲按缺交风险上调。"""
    enforce_orders = enforce_orders or set()
    if draws is None:
        draws = draw_scenarios(lots, model, cfg, cfg.sims_calibration)
    margins: dict[str, QualityMargins] = {oid: ZERO_MARGINS for oid in orders}
    buffers: dict[str, float] = {oid: 0.0 for oid in orders}
    log = {"质量校准": [], "缺交校准": []}

    plan = solve_plan(orders, lots, capacity, model, cfg, plan_date, available_wet=available_wet,
                      margins=margins, buffers=buffers, enforce_orders=enforce_orders)
    if not plan.feasible:
        return plan, None, log

    # ---- 质量裕量校准：优先使用较小的裕量。
    # 先找单步可行的最小裕量，再按反馈微调；避免把“过大的裕量导致不可行”误读为
    # 质量无法满足。若最小裕量仍不可行，说明该单在现有原料下无法同时满足数量与质量。
    def feasible_at(step: float) -> tuple[bool, PlanResult | None]:
        trial = {oid: QualityMargins(step, step) if (orders[oid].max_impurity is not None or orders[oid].min_polymer is not None) else ZERO_MARGINS
                 for oid in orders}
        candidate = solve_plan(orders, lots, capacity, model, cfg, plan_date, available_wet=available_wet,
                               margins=trial, buffers=buffers, enforce_orders=enforce_orders)
        return candidate.feasible, (candidate if candidate.feasible else None)

    lo, hi = 0.0, cfg.quality_margin_max
    chosen: PlanResult | None = None
    chosen_step = 0.0
    step_count = max(1, int(round(hi / max(cfg.quality_margin_step, 1e-9))))
    for k in range(step_count + 1):
        step = min(hi, k * cfg.quality_margin_step)
        ok, candidate = feasible_at(step)
        if ok:
            chosen, chosen_step = candidate, step
            break
        log["质量校准"].append({"说明": f"裕量 {step:.3f} 下联合计划不可行", "搜索": "继续尝试更小裕量"})
    if chosen is None:
        log["质量校准"].append({
            "说明": "任何正质量裕量下都不可行；保留零裕量并标记为受限方案",
            "建议": "需要新增合格检测、放宽质量门槛或减少同批承诺",
        })
        chosen = plan
        chosen_step = 0.0
    else:
        # 以可行最小裕量起步，逐步上调直到质量达标或下一档不可行。
        best_sim = evaluate_plan(chosen, draws, model, cfg, customer_sigma=customer_sigma, quality_limit=quality_limit)
        step = chosen_step

        def quality_ok(sim: SimulationResult) -> bool:
            return all(
                sim.order_risk[oid].reject_probability
                <= (orders[oid].risk_limit if quality_limit is None else quality_limit) + 1e-12
                for oid in orders
            )

        if quality_ok(best_sim):
            log["质量校准"].append({"说明": "最小可行裕量已满足质量风险上限，不再上调", "裕量": round(step, 4)})
        while not quality_ok(best_sim) and step + cfg.quality_margin_step <= cfg.quality_margin_max + 1e-12:
            next_step = step + cfg.quality_margin_step
            ok, candidate = feasible_at(next_step)
            if not ok:
                log["质量校准"].append({
                    "说明": f"上调到裕量 {next_step:.3f} 后不可行；保留 {step:.3f}",
                    "质量": {oid: round(best_sim.order_risk[oid].reject_probability, 4) for oid in orders},
                })
                break
            sim_next = evaluate_plan(candidate, draws, model, cfg, customer_sigma=customer_sigma, quality_limit=quality_limit)
            before = sum(best_sim.order_risk[oid].reject_probability for oid in orders)
            after = sum(sim_next.order_risk[oid].reject_probability for oid in orders)
            if after > before + 1e-9:
                log["质量校准"].append({"说明": "更大裕量未改善质量拒收，停止上调", "裕量": round(step, 4)})
                break
            chosen, best_sim, step = candidate, sim_next, next_step
            log["质量校准"].append({
                "裕量": round(step, 4),
                "拒收概率": {oid: round(best_sim.order_risk[oid].reject_probability, 4) for oid in orders},
            })
        margins = {oid: QualityMargins(chosen.margins.get(oid, ZERO_MARGINS).impurity,
                                       chosen.margins.get(oid, ZERO_MARGINS).polymer) for oid in orders}
        plan = chosen

    # ---- 缺交缓冲校准（仅对需要足量审批的承购）
    buffer_order = [oid for oid in orders if oid in enforce_orders]
    for round_index in range(cfg.shortfall_calibration_rounds):
        sim = evaluate_plan(plan, draws, model, cfg, customer_sigma=customer_sigma, quality_limit=quality_limit)
        offenders = [
            oid for oid in buffer_order
            if sim.order_risk[oid].shortfall_probability > orders[oid].risk_limit + 1e-12
            and buffers[oid] < cfg.shortfall_buffer_max - 1e-12
        ]
        if not offenders:
            break
        trial = dict(buffers)
        # 前几轮固定步长；接近终端风险时缩小步长，避免一步跨过可承诺边界。
        step = cfg.shortfall_buffer_step * (0.5 ** max(0, round_index - 2))
        step = max(step, 0.005)
        for oid in offenders:
            trial[oid] = min(cfg.shortfall_buffer_max, buffers[oid] + step)
        new_plan = solve_plan(orders, lots, capacity, model, cfg, plan_date, available_wet=available_wet,
                              margins=margins, buffers=trial, enforce_orders=enforce_orders)
        if not new_plan.feasible:
            log["缺交校准"].append({
                "说明": "同时提高全部未达标承购的产出缓冲后不可行，保留上一轮缓冲",
                "未达标": offenders,
                "缓冲": {oid: round(buffers[oid], 4) for oid in buffer_order},
            })
            break
        buffers = trial
        plan = new_plan
        log["缺交校准"].append({
            "未达标": offenders,
            "缓冲": {oid: round(buffers[oid], 4) for oid in buffer_order},
            "说明": "同步提高缓冲，避免把缺口转嫁给其他承购",
        })

    return plan, evaluate_plan(plan, draws, model, cfg, customer_sigma=customer_sigma, quality_limit=quality_limit), log


def requote_status(
    snapshot: dict,
    lots_now: dict[str, LotQuality],
    available_wet: dict[str, float],
    orders_now: dict[str, OrderSpec],
    cfg: Config,
    capacity: list[dict],
    model: YieldModel,
    plan_date: str,
) -> dict:
    """比较报价快照与当前证据；返回是否需要重新审批及原因。"""
    reasons: list[str] = []
    snapshot_lots = snapshot.get("evidence", {}).get("lots", {})
    for lid, record in snapshot_lots.items():
        current = lots_now.get(lid)
        if current is None:
            reasons.append(f"报价证据中的料批 {lid} 已不在当前资料")
            continue
        for metric, key, tol in (
            ("impurity", "杂质", cfg.requote_impurity_tolerance),
            ("polymer", "聚合物", cfg.requote_polymer_tolerance),
        ):
            old = record.get(metric)
            if old is None:
                continue
            est = getattr(current, metric)
            if est is None:
                reasons.append(f"{lid} 当前缺少{key}检测，原报价依据不再成立")
                continue
            if abs(est.mean - float(old)) > tol:
                reasons.append(f"{lid} {key}均值由 {float(old):.4f} 变为 {est.mean:.4f}，超出容差 {tol}")
        old_cov = record.get("coverage")
        if old_cov is not None and abs(current.coverage_random - float(old_cov)) > cfg.requote_coverage_tolerance:
            reasons.append(f"{lid} 随机覆盖由 {float(old_cov):.0%} 变为 {current.coverage_random:.0%}")
        if current.status != record.get("status"):
            reasons.append(f"{lid} 证据状态由 {record.get('status')} 变为 {current.status}")

    for oid in snapshot.get("orders", {}):
        if oid not in orders_now:
            reasons.append(f"承购 {oid} 已被取消或不再计划")

    snapshot_days = snapshot.get("plan", {}).get("open_days", [])
    for day in snapshot_days:
        capacity_day = next((row for row in capacity if row["day"] == day), None)
        if capacity_day is None:
            reasons.append(f"原计划加工日 {day} 已不可用")

    # 用当前证据重算同一方案：如果料批已被其他承购占用，方案失效。
    snapshot_alloc = snapshot.get("plan", {}).get("allocations", [])
    for allocation in snapshot_alloc:
        lid = allocation.get("料批") or allocation.get("lot")
        grab = allocation.get("湿重")
        if lid in available_wet and grab is not None and available_wet[lid] + 1e-6 < float(grab):
            reasons.append(
                f"料批 {lid} 可用湿重 {available_wet[lid]:.0f} kg 少于报价时的 {float(grab):.0f} kg，方案不能按原量兑现"
            )
    return {
        "需要重新审批": bool(reasons),
        "原因": reasons,
        "快照有效期至": snapshot.get("price_valid_until"),
        "建议": "重新运行候选方案比较并重新审批；不要只更新可用吨数而保留旧承诺" if reasons else "证据未超出容差，可继续按原快照执行",
    }
