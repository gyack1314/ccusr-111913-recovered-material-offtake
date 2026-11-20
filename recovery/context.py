"""市场上下文：在逐个承购评估前先划定共享资源（原料、加工日、班次）。

业务问题不是“每单各自最优再相加”，而是：
1. 已有承购与待审批询价竞争同一批料和同一个加工日；
2. 联合情景下必须同时看收入、加工成本、质量拒收与缺交；
3. 一单提高质量可能导致另一单缺料，代价要一并呈现。

因此本模块先构造“当前可用资源视图”，再为候选方案提供统一的优化输入。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .data import Dataset, parse_ts
from .planning import OrderSpec, QualityMargins, build_orders
from .quality import LotQuality, MaterialQualityStats, estimate_all, material_quality_stats
from .state import Workspace
from .yield_model import YieldModel, fit_yield_model


@dataclass
class MarketContext:
    dataset: Dataset
    lots: dict[str, LotQuality]
    orders: dict[str, OrderSpec]
    capacity: list[dict]
    model: YieldModel
    stats: dict[str, MaterialQualityStats]
    accepted: set[str]
    pending: set[str]
    cancelled: set[str]
    available_wet: dict[str, float]
    residual_remaining: dict[str, float]
    plan_date: str
    knowledge_cut: str
    notes: list[str]
    scenarios: dict[str, dict[str, LotQuality]] = field(default_factory=dict)
    plan_date_knowledge: str = ""
    realized: dict = field(default_factory=dict)
    committed_margins: dict[str, QualityMargins] = field(default_factory=dict)
    committed_buffers: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "决策日": self.plan_date,
            "知识截止": self.knowledge_cut,
            "已接受": sorted(self.accepted),
            "待审批": sorted(self.pending),
            "已取消": sorted(self.cancelled),
            "可用湿重": {k: round(v, 1) for k, v in self.available_wet.items() if v > 1e-9},
            "剩余履约干重": {k: round(v, 1) for k, v in self.residual_remaining.items()},
            "加工日": [row["day"] for row in self.capacity],
            "质量情景": {
                name: {
                    "料批数": len(scenario),
                    "不可报价或受限": sorted(
                        lid for lid, q in scenario.items() if q.status != "quotable"
                    ),
                    "杂质均值差_相对中心": {
                        lid: round(q.impurity.mean - self.lots[lid].impurity.mean, 6)
                        for lid, q in scenario.items()
                        if lid in self.lots and abs(q.impurity.mean - self.lots[lid].impurity.mean) > 1e-9
                    },
                }
                for name, scenario in self.scenarios.items()
            },
            "已实现": self.realized,
            "说明": list(self.notes),
        }


def consumed_by_lot(dataset: Dataset, workspace: Workspace | None = None) -> dict[str, float]:
    """已加工投入（湿重近似）。processed 回执表示原料已离开可用库存。"""
    consumed: dict[str, float] = {}
    receipts = list(dataset.rows("actuals"))
    if workspace is not None:
        appended_ids = {row["id"] for row in receipts if row.get("__source") == "appended"}
        for row in workspace.appended_rows("actuals"):
            if row["id"] not in appended_ids:
                receipts.append(row)
    for row in receipts:
        if row.get("kind") == "processed":
            consumed[row["lot"]] = consumed.get(row["lot"], 0.0) + float(row["quantity_kg"])
    return consumed


def _accepted_processed_wet(dataset: Dataset, order_id: str) -> float:
    return sum(
        float(row["quantity_kg"])
        for row in dataset.rows("actuals")
        if row.get("kind") == "processed" and row.get("order") == order_id
    )


def _is_active(row: dict, plan_date: str) -> bool:
    known = row.get("known_at")
    return known is None or parse_ts(known) <= parse_ts(f"{plan_date}T23:59:59+08:00")


def realized_summary(dataset: Dataset, workspace: Workspace, plan_date: str) -> dict:
    """已实现：已加工投入 + 客户复检测试结果 + 处置；未实现：剩余交付。"""
    processed = [row for row in dataset.rows("actuals") if row.get("kind") == "processed"]
    assays = [row for row in dataset.rows("actuals") if row.get("kind") == "customer_assay"]
    dispositions = workspace.state.get("dispositions", {})
    lots = {row["id"]: row for row in dataset.rows("lots")}
    realized = []
    for row in processed:
        lot = lots.get(row.get("lot"))
        realized.append({
            "回执": row["id"], "承购": row.get("order"), "料批": row.get("lot"),
            "投入湿重": float(row["quantity_kg"]),
            "料批材质": None if lot is None else lot["material"],
            "已获知": row.get("known_at"),
            "决策日可见": _is_active(row, plan_date),
        })
    disputes = []
    for row in assays:
        disposition = dispositions.get(row["id"])
        if disposition is not None and disposition.get("disposition") == "return":
            realization = "客户退货：该数量不计入已实现收入，也不回到原批次可用库存"
        elif disposition is not None and disposition.get("disposition") == "accept":
            realization = "客户接受：按实际等级计入已实现收入"
        elif disposition is not None and disposition.get("disposition") == "recheck":
            realization = "等待重检：暂不确认该部分收入"
        elif disposition is not None and disposition.get("disposition") == "dispute":
            realization = "争议中：暂不确认该部分收入，且不回库存"
        else:
            realization = "尚未登记处置：按待确认处理"
        disputes.append({
            "回执": row["id"], "承购": row.get("order"), "料批": row.get("lot"),
            "客户实测杂质": row.get("observed_impurity"),
            "处置": None if disposition is None else disposition.get("disposition"),
            "处置说明": None if disposition is None else disposition.get("reason"),
            "退货_kg": None if disposition is None else disposition.get("return_kg"),
            "收入确认": realization,
            "已获知": row.get("known_at"),
            "决策日可见": _is_active(row, plan_date),
        })
    return {
        "已加工回执": realized,
        "客户复检与处置": disputes,
        "说明": "已实现利润只包括已加工并已知结果的部分；客户复检处置改变剩余履约与已实现利润，但报价证据不被改写",
    }


def committed_settings(workspace: Workspace | None, accepted: set[str]) -> tuple[dict, dict]:
    """从已接受承购的报价快照中读取当时承诺的质量裕量与产出缓冲。

    这些是承诺条件，不因后续重新报价而被悄悄取消；新报价必须在同一约束下比较。
    """
    margins: dict[str, QualityMargins] = {}
    buffers: dict[str, float] = {}
    if workspace is None:
        return margins, buffers
    for oid in accepted:
        record = workspace.state.get("accepted", {}).get(oid)
        if not record:
            continue
        try:
            snapshot = workspace.load_snapshot(record["snapshot"])
        except Exception:
            continue
        plan = snapshot.get("plan", {})
        margin = (plan.get("质量裕量") or {}).get(oid)
        if margin:
            margins[oid] = QualityMargins(float(margin.get("杂质裕量", 0.0)), float(margin.get("聚合物裕量", 0.0)))
        buffer = (plan.get("产出缓冲") or {}).get(oid)
        if buffer is not None:
            buffers[oid] = float(buffer)
    return margins, buffers


def build_context(
    cfg: Config,
    workspace: Workspace | None = None,
    data_root=None,
    plan_date: str | None = None,
    knowledge_cut: str | None = None,
) -> MarketContext:
    if workspace is None:
        workspace = Workspace(data_root)
        # 工作区可能尚未创建 .local；load 仍会退回原始资料。
    plan_date = plan_date or cfg.plan_date
    knowledge_cut = knowledge_cut or cfg.knowledge_cut_effective

    dataset = Dataset.load(workspace.root)
    if knowledge_cut:
        dataset = dataset.known_by(knowledge_cut)
    for table in ("assays", "actuals"):
        rows = workspace.appended_rows(table)
        if rows:
            dataset = dataset.append(table, rows)
    dataset.validate()

    scenarios = {
        "central": estimate_all(dataset, cfg, "central", knowledge_cut=None),
        "adverse": estimate_all(dataset, cfg, "adverse", knowledge_cut=None),
        "pooled": estimate_all(dataset, cfg, "pooled", knowledge_cut=None),
    }
    lots = scenarios["central"]
    # 主计划采用中心估计；敏感性情景区另行计算，避免把最乐观界拼成好料。
    materials = sorted({lot["material"] for lot in dataset.rows("lots")})
    assays_by_lot: dict[str, list[dict]] = {lot["id"]: [] for lot in dataset.rows("lots")}
    for assay in dataset.rows("assays"):
        assays_by_lot.setdefault(assay["lot"], []).append(assay)
    stats = material_quality_stats(dataset.rows("lots"), assays_by_lot, materials)

    model = fit_yield_model(dataset.rows("trials"), cfg)
    inquiry_rows = {row["id"]: row for row in dataset.rows("inquiries")}
    accepted = set(workspace.state.get("accepted", {})) if workspace else set()
    cancelled = set(workspace.state.get("cancelled", {})) if workspace else set()

    consumed = consumed_by_lot(dataset, workspace)
    available_wet = {
        lid: max(0.0, lot.wet_kg - consumed.get(lid, 0.0))
        for lid, lot in lots.items()
    }

    remaining: dict[str, float] = {}
    delivery_notes: dict[str, str] = {}
    for oid in accepted:
        if oid in cancelled:
            continue
        inquiry = inquiry_rows.get(oid)
        if inquiry is None:
            continue
        processed = _accepted_processed_wet(dataset, oid)
        # 已加工湿重按订单原料平均含水折算为干重；无检测时保守按 0.15 估计并标注。
        relevant = [
            lots[row["lot"]] for row in dataset.rows("actuals")
            if row.get("order") == oid and row.get("kind") == "processed" and row["lot"] in lots
        ]
        moisture = sum(lot.moisture.mean for lot in relevant) / len(relevant) if relevant else 0.15
        delivered = processed * (1.0 - moisture)
        delivered_note = None if relevant else "该单已加工料批没有检测，含水按 15% 假设折算"
        # 已完成的客户复检若被判退货，则对应数量不冲减剩余履约；接受/重检按已确认
        # 交付处理。这里只能按回执数量近似，且明确写入备注。
        accepted_assay_kg = 0.0
        returned_kg = 0.0
        for row in dataset.rows("actuals"):
            if row.get("order") != oid or row.get("kind") != "customer_assay":
                continue
            disposition = (workspace.state.get("dispositions", {}) or {}).get(row["id"], {})
            if disposition.get("disposition") in ("accept", "recheck"):
                accepted_assay_kg += float(row["quantity_kg"])
            if disposition.get("disposition") == "return":
                returned_kg += float(disposition.get("return_kg") or row["quantity_kg"])
        delivered += accepted_assay_kg * (1.0 - 0.15) - returned_kg
        delivered = max(0.0, delivered)
        remaining[oid] = max(0.0, float(inquiry["dry_kg"]) - delivered)
        if delivered_note:
            delivery_notes[oid] = delivered_note

    statuses = {}
    for oid in inquiry_rows:
        if oid in cancelled:
            statuses[oid] = "cancelled"
        elif oid in accepted:
            statuses[oid] = "accepted"
        else:
            statuses[oid] = "candidate"
    orders = build_orders(dataset.rows("inquiries"), accepted_ids=accepted, remaining=remaining, statuses=statuses)
    pending = {oid for oid in orders if oid not in accepted}

    capacity = [row for row in dataset.rows("capacity") if row["day"] >= plan_date]
    committed_margins, committed_buffers = committed_settings(workspace, accepted)
    notes = []
    for oid, note in delivery_notes.items():
        notes.append(f"{oid}：{note}")
    if workspace is None:
        notes.append("未加载工作区状态，仅按原始资料给出候选，不代表任何已接受承诺")
    if available_wet and any(value <= 0 for value in available_wet.values()):
        notes.append("部分料批已全部加工，不再计入可用原料")
    notes.append("主计划采用中心质量估计；定向取样偏坏与指标共变的不利情景在风险比较中单独给出")
    notes.append("已实现与未实现分开报告：已加工投入不回库存，客户复检结论只影响剩余履约与已实现利润")
    notes.append("客户复检回执的 quantity_kg 按客户抽检数量近似交付量，未提供检测时按 15% 含水折算，属于报告假设")
    return MarketContext(
        dataset=dataset,
        lots=lots,
        orders=orders,
        capacity=capacity,
        model=model,
        stats=stats,
        accepted=accepted - cancelled,
        pending=pending,
        cancelled=cancelled,
        available_wet=available_wet,
        residual_remaining=remaining,
        plan_date=plan_date,
        knowledge_cut=knowledge_cut,
        notes=notes,
        scenarios=scenarios,
        plan_date_knowledge=knowledge_cut,
        realized=realized_summary(dataset, workspace, plan_date) if workspace else {},
        committed_margins=committed_margins,
        committed_buffers=committed_buffers,
    )
