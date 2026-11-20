"""承购计划优化：同一联合情景下分配料批、加工日与配方。

模型是计划层面的混合整数规划：
* 原料按料批常数列分配，同一料批可在不同承购/加工日之间拆分，但总湿重不超过可用量，
  因此单批去向可追溯，且不会把同一吨料承诺给两份合同；
* 日湿基能力共享，启用班次付固定成本；
* 每单按整车数量交付，允许短交变量，其代价率 = 单价 + 按合同量分摊的拒收损失，
  这样高违约代价的订单会自然获得优先级，无需按客户分开计算再相加；
* 质量用“可售干重加权”的订单混合均值约束，并预留可校准的裕量；
* 模型不含原料取得成本、未列明的运输/仓储/资金成本与客户复检系统偏差。

所有优化都在“同一联合情景”下完成：多单共享料批与加工日，不存在先按单最优再相加。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import Config
from .quality import LotQuality
from .yield_model import YieldModel

# 单个（承购, 加工日）组合的最小湿重，用于约束“批量过小”的方案；不是物理下限。
MIN_BATCH_WET_KG = 50.0


@dataclass(frozen=True)
class OrderSpec:
    id: str
    material: str
    demand_dry_kg: float
    deadline: str
    truck_dry_kg: float
    price_cents_per_dry_kg: float
    reject_penalty_cents: float
    banned_sources: tuple[str, ...]
    risk_limit: float
    max_impurity: float | None
    min_polymer: float | None
    accepted: bool = False
    status: str = "candidate"  # candidate / accepted / cancelled
    source_inquiry: str | None = None
    note: str = ""

    @property
    def trucks(self) -> int:
        return int(math.ceil(self.demand_dry_kg / self.truck_dry_kg - 1e-9))

    @property
    def contract_dry_kg(self) -> float:
        return self.trucks * self.truck_dry_kg

    @property
    def revenue_cents(self) -> float:
        return self.contract_dry_kg * self.price_cents_per_dry_kg

    def shortfall_rate(self) -> float:
        if self.contract_dry_kg <= 0:
            return self.price_cents_per_dry_kg
        return self.price_cents_per_dry_kg + self.reject_penalty_cents / self.contract_dry_kg

    def to_dict(self) -> dict:
        return {
            "承购": self.id,
            "材质": self.material,
            "需求干重": self.demand_dry_kg,
            "整车数": self.trucks,
            "合同干重": self.contract_dry_kg,
            "交付期限": self.deadline,
            "单价_分每干公斤": self.price_cents_per_dry_kg,
            "拒收损失_分": self.reject_penalty_cents,
            "禁用来源": list(self.banned_sources),
            "缺交概率上限": self.risk_limit,
            "最高杂质": self.max_impurity,
            "最低聚合物": self.min_polymer,
            "状态": self.status,
            "来源询价": self.source_inquiry,
            "备注": self.note,
        }


@dataclass(frozen=True)
class QualityMargins:
    impurity: float = 0.0
    polymer: float = 0.0

    def __add__(self, other: "QualityMargins") -> "QualityMargins":
        return QualityMargins(self.impurity + other.impurity, self.polymer + other.polymer)

    def to_dict(self) -> dict:
        return {"杂质裕量": round(self.impurity, 6), "聚合物裕量": round(self.polymer, 6)}


ZERO_MARGINS = QualityMargins()


@dataclass
class Allocation:
    order: str
    lot: str
    day: str
    recipe: str
    wet_kg: float
    output_dry_kg: float
    lot_moisture: float
    lot_impurity: float
    lot_polymer: float | None
    source: str
    assay_evidence: tuple[str, ...] = ()
    lot_status: str = "quotable"
    mutable: bool = True  # 是否尚未加工，可在取消时释放

    def to_dict(self) -> dict:
        return {
            "承购": self.order,
            "料批": self.lot,
            "加工日": self.day,
            "配方": self.recipe,
            "湿重": round(self.wet_kg, 2),
            "可售干重": round(self.output_dry_kg, 2),
            "含水率": round(self.lot_moisture, 6),
            "杂质": round(self.lot_impurity, 6),
            "聚合物": None if self.lot_polymer is None else round(self.lot_polymer, 6),
            "来源": self.source,
            "检测证据": list(self.assay_evidence),
            "料批状态": self.lot_status,
            "可释放": self.mutable,
        }


@dataclass
class ProductSummary:
    order: str
    produced_dry_kg: float
    contract_dry_kg: float
    shortfall_dry_kg: float
    impurity: float | None
    polymer: float | None
    lots: list[str]
    days: list[str]
    recipes: list[str]
    wet_kg: float

    def to_dict(self) -> dict:
        return {
            "承购": self.order,
            "产出干重": round(self.produced_dry_kg, 1),
            "合同干重": round(self.contract_dry_kg, 1),
            "短交干重": round(self.shortfall_dry_kg, 1),
            "混合杂质": None if self.impurity is None else round(self.impurity, 6),
            "混合聚合物": None if self.polymer is None else round(self.polymer, 6),
            "料批": list(self.lots),
            "加工日": list(self.days),
            "配方": list(self.recipes),
            "投入湿重": round(self.wet_kg, 1),
        }


@dataclass
class PlanResult:
    feasible: bool
    message: str
    orders: dict[str, OrderSpec]
    allocations: list[Allocation] = field(default_factory=list)
    products: dict[str, ProductSummary] = field(default_factory=dict)
    open_days: list[str] = field(default_factory=list)
    margins: dict[str, QualityMargins] = field(default_factory=dict)
    buffers: dict[str, float] = field(default_factory=dict)
    relaxed: bool = False
    relaxation: dict[str, dict[str, float]] = field(default_factory=dict)
    revenue_cents: float = 0.0
    variable_cost_cents: float = 0.0
    fixed_cost_cents: float = 0.0
    shortfall_penalty_cents: float = 0.0
    solver_status: int | None = None
    solver_message: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def profit_cents(self) -> float:
        return self.revenue_cents - self.variable_cost_cents - self.fixed_cost_cents - self.shortfall_penalty_cents

    @property
    def lot_use(self) -> dict[str, float]:
        used: dict[str, float] = {}
        for allocation in self.allocations:
            used[allocation.lot] = used.get(allocation.lot, 0.0) + allocation.wet_kg
        return used

    def to_dict(self) -> dict:
        return {
            "可行": self.feasible,
            "说明": self.message,
            "收入_分": round(self.revenue_cents, 1),
            "加工变动成本_分": round(self.variable_cost_cents, 1),
            "班次固定成本_分": round(self.fixed_cost_cents, 1),
            "短交代价_分": round(self.shortfall_penalty_cents, 1),
            "计划毛利_分": round(self.profit_cents, 1),
            "质量裕量": {k: v.to_dict() for k, v in self.margins.items()},
            "产出缓冲": {k: round(v, 4) for k, v in self.buffers.items()},
            "启用加工日": list(self.open_days),
            "承购": {k: v.to_dict() for k, v in self.orders.items()},
            "产出汇总": {k: v.to_dict() for k, v in self.products.items()},
            "分配": [a.to_dict() for a in self.allocations],
            "松弛量": self.relaxation,
            "求解状态": self.solver_status,
            "求解信息": self.solver_message,
            "备注": list(self.notes),
        }


def build_orders(
    inquiries: Iterable[dict],
    accepted_ids: Iterable[str] = (),
    remaining: dict[str, float] | None = None,
    statuses: dict[str, str] | None = None,
) -> dict[str, OrderSpec]:
    remaining = remaining or {}
    statuses = statuses or {}
    accepted = set(accepted_ids)
    orders = {}
    for row in inquiries:
        demand = float(remaining.get(row["id"], row["dry_kg"]))
        status = statuses.get(row["id"], "accepted" if row["id"] in accepted else "candidate")
        if status == "cancelled" or demand <= 1e-9:
            continue
        orders[row["id"]] = OrderSpec(
            id=row["id"],
            material=row["material"],
            demand_dry_kg=demand,
            deadline=row["deadline"],
            truck_dry_kg=float(row["truck_dry_kg"]),
            price_cents_per_dry_kg=float(row["cents_per_dry_kg"]),
            reject_penalty_cents=float(row["reject_penalty_cents"]),
            banned_sources=tuple(row.get("banned_sources") or ()),
            risk_limit=float(row["risk_limit"]),
            max_impurity=row.get("max_impurity"),
            min_polymer=row.get("minimum_polymer"),
            accepted=status == "accepted",
            status=status,
            source_inquiry=row["id"],
        )
    return orders


def _lot_can_serve(order: OrderSpec, lot: LotQuality, cfg: Config) -> bool:
    if lot.material != order.material:
        return False
    if lot.source in order.banned_sources:
        return False
    if lot.status == "unquotable":
        return False
    if lot.forbidden_tags and not cfg.allow_forbidden_tags:
        return False
    if order.min_polymer is not None and lot.polymer is None:
        return False
    return True


def solve_plan(
    orders: dict[str, OrderSpec],
    lots: dict[str, LotQuality],
    capacity: list[dict],
    model: YieldModel,
    cfg: Config,
    plan_date: str,
    available_wet: dict[str, float] | None = None,
    margins: dict[str, QualityMargins] | None = None,
    buffers: dict[str, float] | None = None,
    enforce_orders: set[str] | None = None,
    relaxed: bool = False,
    recipe_choice: dict[str, str] | None = None,
) -> PlanResult:
    """求解一个联合计划。

    available_wet: 料批当前可用湿重（已扣减已加工量）；缺省使用料批全重。
    enforce_orders: 必须足量计划的承购（短交变量固定为 0）；通常是待审批的新询价。
    relaxed: 允许质量违约，用松弛量量化不可满足程度。
    """
    margins = margins or {}
    buffers = buffers or {}
    enforce_orders = enforce_orders or set()
    recipe_choice = recipe_choice or {}
    available_wet = available_wet or {lid: lot.wet_kg for lid, lot in lots.items()}

    day_rows = sorted([row for row in capacity if row["day"] >= plan_date], key=lambda r: r["day"])
    if not orders:
        return PlanResult(True, "没有需要计划的承购", orders={})
    if not day_rows:
        return PlanResult(False, "决策日之后没有可用加工日", orders=orders)

    serviceable = {
        oid: [lid for lid, lot in lots.items() if _lot_can_serve(order, lot, cfg) and available_wet.get(lid, 0.0) > 1e-6]
        for oid, order in orders.items()
    }
    missing = [oid for oid, lids in serviceable.items() if not lids]
    if missing:
        return PlanResult(False, "以下承购没有可用料批: " + "、".join(missing), orders=orders)

    recipes_all = sorted(model.recipes)
    recipe_index = {r: i for i, r in enumerate(recipes_all)}
    combos: list[tuple[str, str, int, int]] = []
    for oid, order in orders.items():
        for lid in serviceable[oid]:
            lot = lots[lid]
            received_day = lot.received_at[:10]
            for di, day in enumerate(day_rows):
                if day["day"] > order.deadline or day["day"] < received_day:
                    continue
                allowed = [r for r in day["recipes"] if r in recipe_index]
                if oid in recipe_choice:
                    allowed = [r for r in allowed if r == recipe_choice[oid]]
                for recipe in allowed:
                    combos.append((oid, lid, di, recipe_index[recipe]))
    if not combos:
        return PlanResult(False, "没有满足交期、来源与配方条件的料批组合", orders=orders)

    n_combo = len(combos)
    n_day = len(day_rows)
    order_ids = list(orders)
    n_order = len(order_ids)
    order_index = {oid: i for i, oid in enumerate(order_ids)}

    off_x = 0
    off_y = n_combo
    off_s = n_combo + n_day
    off_p = off_s + n_order
    base_var = off_p + n_order
    n_extra = n_order if relaxed else 0

    a = np.zeros(n_combo)
    imp = np.zeros(n_combo)
    poly = np.full(n_combo, np.nan)
    varcost = np.zeros(n_combo)
    for i, (oid, lid, di, ri) in enumerate(combos):
        lot = lots[lid]
        recipe = recipes_all[ri]
        m, im = lot.moisture.mean, lot.impurity.mean
        a[i] = max(0.0, (1.0 - m) * model.saleable_yield(m, im, recipe))
        imp[i] = im
        if lot.polymer is not None:
            poly[i] = lot.polymer.mean
        varcost[i] = model.cost_per_wet_kg.get(recipe, float(np.mean(list(model.cost_per_wet_kg.values()))))

    c = np.zeros(base_var + n_extra)
    c[off_x:off_x + n_combo] = varcost
    for di, day in enumerate(day_rows):
        c[off_y + di] = float(day["cost_cents"])
    revenue_total = 0.0
    for oid, order in orders.items():
        oi = order_index[oid]
        c[off_s + oi] = order.shortfall_rate()
        revenue_total += order.revenue_cents
    if relaxed:
        c[base_var:] = 1e8


    # ---- 质量门槛可行性预检：若某订单所有可用料批都超出裕量，直接给出受限结论。
    precheck: dict[str, dict[str, float]] = {}
    for oid, order in orders.items():
        idx = np.array([i for i, combo in enumerate(combos) if combo[0] == oid], dtype=int)
        if idx.size == 0:
            continue
        margin = margins.get(oid, ZERO_MARGINS)
        if order.max_impurity is not None and float(order.max_impurity) - margin.impurity <= float(np.min(imp[idx])) + 1e-12:
            precheck[oid] = {
                "可用料批最低杂质": round(float(np.min(imp[idx])), 6),
                "门槛": round(float(order.max_impurity) - margin.impurity, 6),
                "结论": "现有可用料批的杂质全部高于订单门槛，纯混配无法满足",
            }
        if order.min_polymer is not None:
            usable = idx[np.isfinite(poly[idx])]
            if usable.size and float(order.min_polymer) + margin.polymer >= float(np.max(poly[usable])) - 1e-12:
                precheck[oid] = {
                    "可用料批最高聚合物": round(float(np.max(poly[usable])), 6),
                    "门槛": round(float(order.min_polymer) + margin.polymer, 6),
                    "结论": "现有可用料批的聚合物全部低于订单门槛，纯混配无法满足",
                }
    if precheck and not relaxed:
        details = "；".join(f"{oid}: {info['结论']}" for oid, info in precheck.items())
        return PlanResult(False, "质量门槛不可满足。" + details, orders=orders, margins=dict(margins),
                          buffers=dict(buffers), relaxation={oid: info for oid, info in precheck.items()})
    integrality = np.zeros(base_var + n_extra)
    integrality[off_y:off_y + n_day] = 1.0
    lower = np.zeros(base_var + n_extra)
    upper = np.full(base_var + n_extra, np.inf)
    upper[off_y:off_y + n_day] = 1.0
    for oid in enforce_orders:
        if oid in order_index:
            upper[off_s + order_index[oid]] = 0.0

    constraints: list[LinearConstraint] = []

    def add(row: np.ndarray, lo: float, hi: float) -> None:
        constraints.append(LinearConstraint(row, lo, hi))

    # 料批可用湿重
    for lid, lot in lots.items():
        idx = np.array([i for i, combo in enumerate(combos) if combo[1] == lid], dtype=int)
        if idx.size == 0:
            continue
        row = np.zeros(base_var + n_extra)
        row[off_x + idx] = 1.0
        add(row, -np.inf, float(available_wet.get(lid, lot.wet_kg)))

    # 日能力
    for di, day in enumerate(day_rows):
        idx = np.array([i for i, combo in enumerate(combos) if combo[2] == di], dtype=int)
        if idx.size == 0:
            continue
        row = np.zeros(base_var + n_extra)
        row[off_x + idx] = 1.0
        row[off_y + di] = -float(day["wet_kg"])
        add(row, -np.inf, 0.0)

    # 产出定义
    for oid in order_ids:
        oi = order_index[oid]
        idx = np.array([i for i, combo in enumerate(combos) if combo[0] == oid], dtype=int)
        row = np.zeros(base_var + n_extra)
        row[off_p + oi] = 1.0
        row[off_x + idx] = -a[idx]
        add(row, 0.0, 0.0)

    # 最小批量：启用一个（承购, 加工日）就必须实际加工；避免用大量微量分配
    # 把不同料批“名义上”拼进同一订单，却未反映换批/清洗成本。
    # 维度扩充只针对 n_extra（松弛）之后的部分，避免把松弛变量错位。
    off_z = base_var + n_extra
    z_index: dict[tuple[str, int], int] = {}
    for oid in order_ids:
        for di in range(n_day):
            z_index[(oid, di)] = off_z + len(z_index)
    n_total = off_z + len(z_index)

    def resize(rows: list[np.ndarray], width: int) -> None:
        for row in rows:
            if row.size < width:
                row.resize(width, refcheck=False)

    def extend_constraint(constraint: LinearConstraint) -> LinearConstraint:
        matrix = constraint.A
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.array(matrix)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] < n_total:
            dense = np.zeros((matrix.shape[0], n_total))
            dense[:, : matrix.shape[1]] = matrix
            return LinearConstraint(dense, constraint.lb, constraint.ub)
        return constraint

    c = np.concatenate([c, np.zeros(len(z_index))])
    integrality = np.concatenate([integrality, np.ones(len(z_index))])
    lower = np.concatenate([lower, np.zeros(len(z_index))])
    upper = np.concatenate([upper, np.ones(len(z_index))])
    # 维度扩充：把已有约束补齐到 n_total
    constraints = [extend_constraint(constraint) for constraint in constraints]
    if MIN_BATCH_WET_KG > 0:
        for oid in order_ids:
            for di in range(n_day):
                zi = z_index[(oid, di)]
                idx = np.array([i for i, combo in enumerate(combos) if combo[0] == oid and combo[2] == di], dtype=int)
                if idx.size == 0:
                    continue
                row = np.zeros(n_total)
                row[off_x + idx] = 1.0
                row[zi] = -MIN_BATCH_WET_KG
                add(row, 0.0, np.inf)
    n_var = n_total
    bounds = Bounds(lower, upper)


    # 数量：p + s >= 合同干重 * (1 + buffer)
    for oid, order in orders.items():
        oi = order_index[oid]
        row = np.zeros(n_var)
        row[off_p + oi] = 1.0
        row[off_s + oi] = 1.0
        add(row, order.contract_dry_kg * (1.0 + float(buffers.get(oid, 0.0))), np.inf)

    # 质量：按混合后的订单均值约束。允许“单项超标但可被其他料批拉回”的料批参与混配，
    # 是否可行由联合约束决定，而不是逐批先验淘汰。
    for oid, order in orders.items():
        oi = order_index[oid]
        idx = np.array([i for i, combo in enumerate(combos) if combo[0] == oid], dtype=int)
        if idx.size == 0:
            continue
        margin = margins.get(oid, ZERO_MARGINS)
        if order.max_impurity is not None:
            row = np.zeros(n_var)
            row[off_x + idx] = a[idx] * imp[idx]
            row[off_p + oi] = -(float(order.max_impurity) - margin.impurity)
            if relaxed:
                row[base_var + oi] = -1.0
            add(row, -np.inf, 0.0)
        if order.min_polymer is not None:
            usable = idx[np.isfinite(poly[idx])]
            if usable.size != idx.size:
                return PlanResult(False, f"{oid} 的候选料批缺少聚合物检测", orders=orders)
            row = np.zeros(n_var)
            row[off_x + usable] = -a[usable] * poly[usable]
            row[off_p + oi] = float(order.min_polymer) + margin.polymer
            if relaxed:
                row[base_var + oi] = -1.0
            add(row, -np.inf, 0.0)

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        options={"time_limit": cfg.mip_time_limit, "mip_rel_gap": cfg.mip_rel_gap, "disp": False},
    )

    if result.x is None:
        message = "联合计划不可行" if result.status == 2 else f"求解失败: {result.message}"
        if relaxed:
            message = "即使放松质量约束仍不可行"
        return PlanResult(False, message, orders=orders, margins=dict(margins), buffers=dict(buffers),
                          relaxed=relaxed, solver_status=int(result.status), solver_message=str(result.message))

    sol = result.x
    allocations: list[Allocation] = []
    for i, (oid, lid, di, ri) in enumerate(combos):
        wet = float(sol[off_x + i])
        if wet <= 1e-6:
            continue
        lot = lots[lid]
        recipe = recipes_all[ri]
        allocations.append(
            Allocation(
                order=oid, lot=lid, day=day_rows[di]["day"], recipe=recipe, wet_kg=wet,
                output_dry_kg=wet * float(a[i]),
                lot_moisture=lot.moisture.mean, lot_impurity=lot.impurity.mean,
                lot_polymer=lot.polymer.mean if lot.polymer is not None else None,
                source=lot.source, assay_evidence=lot.effective_assay_ids, lot_status=lot.status,
            )
        )

    products: dict[str, ProductSummary] = {}
    variable_cost = float(np.dot(c[off_x:off_x + n_combo], sol[off_x:off_x + n_combo]))
    fixed_cost = float(np.dot([float(d["cost_cents"]) for d in day_rows], sol[off_y:off_y + n_day]))
    shortfall_penalty = 0.0
    for oid, order in orders.items():
        oi = order_index[oid]
        rows = [al for al in allocations if al.order == oid]
        produced = float(sol[off_p + oi])
        short = float(sol[off_s + oi])
        if rows and produced > 1e-9:
            impurity = float(sum(al.output_dry_kg * al.lot_impurity for al in rows) / produced)
            polymer_rows = [al for al in rows if al.lot_polymer is not None]
            polymer = (
                float(sum(al.output_dry_kg * al.lot_polymer for al in polymer_rows) / sum(al.output_dry_kg for al in polymer_rows))
                if polymer_rows and sum(al.output_dry_kg for al in polymer_rows) > 1e-9 else None
            )
        else:
            impurity = polymer = None
        products[oid] = ProductSummary(
            order=oid, produced_dry_kg=produced, contract_dry_kg=order.contract_dry_kg,
            shortfall_dry_kg=short, impurity=impurity, polymer=polymer,
            lots=sorted({al.lot for al in rows}), days=sorted({al.day for al in rows}),
            recipes=sorted({al.recipe for al in rows}), wet_kg=float(sum(al.wet_kg for al in rows)),
        )
        shortfall_penalty += short * order.shortfall_rate()

    relaxation = {}
    if relaxed:
        for oid, order in orders.items():
            slack = float(sol[base_var + order_index[oid]])
            if slack > 1e-6:
                relaxation[oid] = {
                    "质量松弛": round(slack, 6),
                    "含义": "现有原料与产能下，至少需要放宽的杂质/聚合物质量差距",
                }
    open_days = [day_rows[di]["day"] for di in range(n_day) if sol[off_y + di] > 0.5]
    feasible = result.x is not None and result.status in (0, 1)
    message = "最优" if result.status == 0 else ("达到时间限制但有可行解" if result.status == 1 else str(result.message))

    notes = []
    for oid, order in orders.items():
        if abs(order.contract_dry_kg - order.demand_dry_kg) > 1e-6:
            notes.append(
                f"{oid} 需求 {order.demand_dry_kg:g} kg 不是整车 {order.truck_dry_kg:g} kg 的整数倍，按合同量 {order.contract_dry_kg:g} kg 计划"
            )
    restricted = sorted({al.lot for al in allocations if al.lot_status != "quotable"})
    if restricted:
        notes.append("使用受限料批: " + "、".join(restricted) + "（覆盖或检测基础受限，风险已计入不确定性）")

    return PlanResult(
        feasible=feasible, message=message, orders=orders, allocations=allocations,
        products=products, open_days=open_days, margins=dict(margins), buffers=dict(buffers),
        relaxed=relaxed, relaxation=relaxation, revenue_cents=revenue_total,
        variable_cost_cents=variable_cost, fixed_cost_cents=fixed_cost,
        shortfall_penalty_cents=shortfall_penalty, solver_status=int(result.status),
        solver_message=str(result.message), notes=notes,
    )
