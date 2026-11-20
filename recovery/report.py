"""把分析结果组织为可审批的文本/JSON/CSV。

报告显式区分：
* 事实（来自资料）
* 模型估计（含区间与方法边界）
* 业务判断（门槛、风险承担范围）
* 未计入项
"""
from __future__ import annotations

import csv
import io
import json

from .candidates import CandidateEvaluation
from .config import Config
from .context import MarketContext
from .state import Workspace

METHOD_BOUNDARIES = [
    "质量估计采用随机质量取样层均值；未覆盖部分按同批层内差异或同材质批间差异建模，不是对每车质量的保证。",
    "定向取样只用于压力情景与审核，不进入中心估计；若定向层偏坏，报告会单列提示而不是悄悄抬高等级。",
    "产出率回归只使用开发折试验，且按 trial_lot 分组交叉验证；留出折仅用于回测，回测误差已计入风险。",
    "同一 trial_lot 的平行试验不是独立样板；配方差异常小于批间波动，报告不预设配方胜出。",
    "预测区间描述模型条件下的分布，不是对客户复检结果的承诺，也不替代合同质量条款。",
    "客户复检测量误差为敏感性假设；资料只有一个争议案例，不能据此标定检测系统偏差。",
    "未计入原料取得成本、未列明的运输、仓储、资金成本及税费。",
    "资料为合成数据，不能用于真实交易、认证或政策判断。",
]


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _fmt_money(cents: float | None) -> str:
    if cents is None:
        return "—"
    return f"{cents / 100:,.2f} 元"


def _fmt_number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def text_report(
    ctx: MarketContext,
    cfg: Config,
    evaluation: CandidateEvaluation,
    snapshot: dict,
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# 再生料混配报价：候选方案与风险比较")
    add("")
    add(f"- 决策日：{ctx.plan_date}；知识截止：{ctx.knowledge_cut}（之后获知的检测与回执未进入本次报价）")
    add(f"- 快照编号：{snapshot['id']}；价格有效期至：{snapshot['price_valid_until']}")
    add(f"- 待审批：{', '.join(evaluation.pending) or '无'}；已接受：{', '.join(evaluation.accepted) or '无'}")
    add("")

    add("## 结论")
    add("")
    add(f"推荐组合：**{evaluation.recommended or '不报价'}**")
    recommended = evaluation.option_by_name(evaluation.recommended)
    if recommended is None:
        add("没有可行组合；逐单说明如下。")
    else:
        gate = recommended.gate(cfg)
        add(
            f"该组合合同干重 {_fmt_number(recommended.contract_dry_kg)} kg，"
            f"计划毛利 {_fmt_money(recommended.plan.profit_cents)}，"
            f"情景期望毛利 {_fmt_money(recommended.expected_profit_cents)}"
            f"（P10 {_fmt_money(recommended.simulation.profit_p10_cents)}，"
            f"P90 {_fmt_money(recommended.simulation.profit_p90_cents)}）。"
        )
        add(
            f"门槛：缺交 {'达标' if gate['缺交达标'] else '未达标'}，"
            f"质量 {'达标' if gate['质量达标'] else '未达标'}。"
        )
        if not gate["全部达标"]:
            add("结论只能作为受限方案：不能按足量、按期、质量全部承诺审批。")
    add("")

    add("## 候选方案比较（同一联合情景）")
    add("")
    add("| 组合 | 合同干重 kg | 期望毛利 | 毛利 P10 | 缺交达标 | 质量达标 |")
    add("| --- | ---: | ---: | ---: | --- | --- |")
    for option in evaluation.options[:12]:
        gate = option.gate(cfg)
        add(
            f"| {'+'.join(option.orders) or '不接单'} | {_fmt_number(option.contract_dry_kg)} | "
            f"{_fmt_money(option.expected_profit_cents)} | {_fmt_money(option.simulation.profit_p10_cents)} | "
            f"{'是' if gate['缺交达标'] else '否'} | {'是' if gate['质量达标'] else '否'} |"
        )
    add("")
    add("所有组合都用同一组料批、加工日与抽样；不是按客户分别最优后相加。")
    add("")

    add("## 待审批承购的风险")
    add("")
    if recommended is not None:
        add("| 承购 | 合同干重 | 产出均值 | 产出 P10 | 缺交概率 | 杂质均值/P90 | 聚合物均值/P10 | 质量拒收 |")
        add("| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |")
        for oid, risk in recommended.simulation.order_risk.items():
            imp = "—" if risk.mean_impurity is None else f"{risk.mean_impurity:.4f}/{risk.p90_impurity:.4f}"
            poly = "—" if risk.mean_polymer is None else f"{risk.mean_polymer:.4f}/{risk.p10_polymer:.4f}"
            add(
                f"| {oid} | {_fmt_number(risk.contract_dry_kg)} | {_fmt_number(risk.mean_output_dry_kg)} | "
                f"{_fmt_number(risk.p10_output)} | {_fmt_pct(risk.shortfall_probability)} | {imp} | {poly} | "
                f"{_fmt_pct(risk.reject_probability)} |"
            )
        add("")

    if evaluation.marginal:
        add("## 挤占与机会成本")
        add("")
        crowding = evaluation.marginal.get("接受该单对已接受承购的影响", {})
        for pending_order, affected in crowding.items():
            for acc, values in affected.items():
                add(
                    f"- 若接受 {pending_order}：已接受 {acc} 的缺交概率变化 {_fmt_pct(values['缺交概率变化'])}，"
                    f"质量拒收变化 {_fmt_pct(values['质量拒收概率变化'])}，"
                    f"期望产出变化 {_fmt_number(values['期望产出变化_kg'])} kg"
                )
        recommended_rows = evaluation.marginal.get("推荐方案", {})
        if recommended_rows:
            for oid, row in recommended_rows.items():
                change_short = row.get("相对基线_缺交变化")
                change_reject = row.get("相对基线_拒收变化")
                if change_short is None and change_reject is None:
                    add(
                        f"- {oid}：缺交 {_fmt_pct(row['缺交概率'])}，质量拒收 {_fmt_pct(row['质量拒收概率'])}；"
                        f"{row.get('基线说明', '无可用基线')}"
                    )
                else:
                    add(
                        f"- {oid}：缺交 {_fmt_pct(row['缺交概率'])}，质量拒收 {_fmt_pct(row['质量拒收概率'])}；"
                        f"{row.get('基线说明', '相对基线')}："
                        f"缺交 {_fmt_pct(change_short)}，拒收 {_fmt_pct(change_reject)}"
                    )
        if evaluation.marginal.get("相对基线的局限"):
            add(f"- 注意：{evaluation.marginal['相对基线的局限']}")
        aggregate = evaluation.marginal.get("相对不含待审批的基线", {}).get("系统合计")
        if aggregate:
            add(
                f"- 系统合计：期望毛利变化 {_fmt_money(aggregate['期望毛利变化_分'])}，"
                f"合同干重变化 {_fmt_number(aggregate['合同干重变化'])} kg"
            )
        add("")

    if snapshot.get("accepted_interaction"):
        add("## 已接受承购之间的相互影响")
        add("")
        for row in snapshot["accepted_interaction"]:
            if "原因" in row:
                add(f"- {row['说明']}：{row['原因']}")
                continue
            add(f"- {row['承购']}：单独计划期望毛利 {_fmt_money(row['单独计划毛利_分'])}"
                f"（缺交 {_fmt_pct(row.get('单独计划缺交概率'))}）；"
                f"与其余已接受承购联合时 {_fmt_money(row['联合计划毛利_分'])}"
                f"（缺交 {_fmt_pct(row.get('联合计划缺交概率'))}）")
            for other, values in row["对其他承购的影响"].items():
                if not values["只做该单时是否计划了对方"]:
                    add(
                        f"  - 若只做 {row['承购']}，{other} 根本不在计划内（等价于 100% 缺交）；"
                        f"联合计划下其缺交为 {_fmt_pct(values['联合计划中的缺交概率'])}"
                    )
                else:
                    add(
                        f"  - 若只做 {row['承购']}，{other} 的缺交为 "
                        f"{_fmt_pct(values['只做该单时的缺交概率'])}；联合计划下为 "
                        f"{_fmt_pct(values['联合计划中的缺交概率'])}"
                    )
        add("")

    add("## 证据与适用范围")
    add("")
    for oid, order in recommended.plan.orders.items() if recommended else []:
        rows = [a for a in recommended.plan.allocations if a.order == oid]
        add(f"### {oid}")
        add(f"- 原始询价：{order.source_inquiry}；单价 {order.price_cents_per_dry_kg:.0f} 分/干公斤；"
            f"交期 {order.deadline}；整车 {order.truck_dry_kg:.0f} kg")
        for allocation in rows:
            lot = ctx.lots[allocation.lot]
            add(
                f"- 料批 {allocation.lot}（{lot.material}，来源 {lot.source}）：加工日 {allocation.day}，"
                f"配方 {allocation.recipe}，湿重 {allocation.wet_kg:.0f} kg，"
                f"检测证据 {', '.join(allocation.assay_evidence)}，"
                f"覆盖 {lot.coverage_random:.0%}，状态 {lot.status}"
            )
            for restriction in lot.restrictions:
                add(f"  - 限制：{restriction}")
        add("")

    sensitivity = snapshot.get("sensitivity", {})
    if sensitivity.get("情景"):
        add("## 敏感性（资料无法区分的相关性假设）")
        add("")
        add("| 情景 | 任一承购缺交 | 任一承购质量拒收 | 期望毛利 |")
        add("| --- | ---: | ---: | ---: |")
        for name, values in sensitivity["情景"].items():
            add(
                f"| {name} | {_fmt_pct(values.get('任一承购缺交概率'))} | "
                f"{_fmt_pct(values.get('任一承购质量拒收概率'))} | "
                f"{_fmt_money(values.get('期望毛利_分'))} |"
            )
        add("")
        add(sensitivity.get("说明", ""))
        add("")

    add("## 用途限制与受限料批")
    add("")
    limited = [lot for lot in ctx.lots.values() if lot.status != "quotable" or lot.forbidden_tags]
    if not limited:
        add("- 本视图没有受限或带用途限制标签的料批；这不代表未来新增料批也适用。")
    else:
        for lot in limited:
            tags = "、".join(lot.forbidden_tags) if lot.forbidden_tags else "无"
            add(f"- {lot.lot_id}：状态 {lot.status}，标签 {tags}；限制 {'；'.join(lot.restrictions) or '无'}")
    add("")

    add("## 假设与未计入项")
    add("")
    for key, value in cfg.assumptions().items():
        add(f"- {key}：{value}")
    add("")

    add("## 方法边界")
    add("")
    for item in METHOD_BOUNDARIES:
        add(f"- {item}")
    add("")
    return "\n".join(lines)


def json_report(
    ctx: MarketContext,
    cfg: Config,
    evaluation: CandidateEvaluation,
    snapshot: dict,
) -> str:
    payload = {
        "市场上下文": ctx.to_dict(),
        "候选比较": evaluation.to_dict(cfg),
        "快照": snapshot,
        "假设": cfg.assumptions(),
        "方法边界": METHOD_BOUNDARIES,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def allocations_csv(plan: dict) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["承购", "料批", "加工日", "配方", "湿重", "可售干重", "含水率", "杂质", "聚合物", "来源", "检测证据", "料批状态"])
    for row in plan.get("分配", []):
        writer.writerow([
            row["承购"], row["料批"], row["加工日"], row["配方"], row["湿重"], row["可售干重"],
            row["含水率"], row["杂质"], row["聚合物"], row["来源"],
            ";".join(row["检测证据"]), row["料批状态"],
        ])
    return buffer.getvalue()


def material_trace(ctx: MarketContext, workspace: Workspace | None = None) -> list[dict]:
    """单批物料去向：计划占用、已加工、剩余，并可追溯到承购与检测。

    有工作区时读取最近一次快照的计划分配；没有快照时只显示可用量与已加工量，
    并明确标注“无报价计划”，避免把未承诺的可用量当成已占用。
    """
    planned: dict[str, dict[str, float]] = {}
    snapshot_id_value = None
    snapshot_plan: dict = {}
    if workspace is not None and workspace.snapshot_dir.exists():
        snapshots = sorted(workspace.snapshot_dir.glob("*.json"))
        if snapshots:
            snapshot = json.loads(snapshots[-1].read_text(encoding="utf-8"))
            snapshot_id_value = snapshot.get("id")
            snapshot_plan = snapshot.get("plan", {})
            for row in snapshot_plan.get("分配", []):
                planned.setdefault(row["料批"], {})
                planned[row["料批"]][row["承购"]] = planned[row["料批"]].get(row["承购"], 0.0) + float(row["湿重"])
    processed: dict[str, float] = {}
    for row in ctx.dataset.rows("actuals"):
        if row.get("kind") == "processed":
            processed[row["lot"]] = processed.get(row["lot"], 0.0) + float(row["quantity_kg"])
    rows = []
    for lot_id, lot in sorted(ctx.lots.items()):
        rows.append({
            "料批": lot_id,
            "材质": lot.material,
            "来源": lot.source,
            "湿重": lot.wet_kg,
            "已加工湿重": round(processed.get(lot_id, 0.0), 1),
            "可用湿重": round(ctx.available_wet.get(lot_id, 0.0), 1),
            "最近报价占用": {k: round(v, 1) for k, v in planned.get(lot_id, {}).items()},
            "报价快照": snapshot_id_value,
            "状态": lot.status,
            "覆盖": round(lot.coverage_random, 4),
            "限制": lot.restrictions,
            "有效检测": list(lot.effective_assay_ids),
        })
    return rows


def consistency_checks(ctx: MarketContext, snapshot: dict) -> list[dict]:
    """报价与实际差异的检查项；返回可行动的差异清单。"""
    checks: list[dict] = []
    # 1) 快照证据与当前证据
    for lot_id, record in snapshot.get("evidence", {}).get("lots", {}).items():
        current = ctx.lots.get(lot_id)
        if current is None:
            checks.append({"类型": "报价证据缺失", "料批": lot_id, "说明": "快照引用的料批已不在当前资料"})
            continue
        if abs(current.impurity.mean - float(record.get("impurity", current.impurity.mean))) > 0.001:
            checks.append({
                "类型": "杂质证据变化",
                "料批": lot_id,
                "报价时": record.get("impurity"),
                "当前": round(current.impurity.mean, 6),
                "说明": "新检测进入后中心估计移动，应重新审批报价",
            })
        if record.get("polymer") is not None and current.polymer is not None:
            if abs(current.polymer.mean - float(record["polymer"])) > 0.005:
                checks.append({
                    "类型": "聚合物证据变化",
                    "料批": lot_id,
                    "报价时": record["polymer"],
                    "当前": round(current.polymer.mean, 6),
                    "说明": "成分证据变化可能改变适用等级",
                })
        if current.status != record.get("status"):
            checks.append({
                "类型": "证据状态变化",
                "料批": lot_id,
                "报价时": record.get("status"),
                "当前": current.status,
                "说明": "覆盖或检测基础变化，需重新判断是否仍能报价",
            })
    # 2) 加工回执与计划分配
    planned = {}
    for row in snapshot.get("plan", {}).get("分配", []):
        planned.setdefault((row["承购"], row["料批"]), 0.0)
        planned[(row["承购"], row["料批"])] += float(row["湿重"])
    actual = {}
    for row in ctx.dataset.rows("actuals"):
        if row.get("kind") == "processed":
            key = (row.get("order"), row["lot"])
            actual[key] = actual.get(key, 0.0) + float(row["quantity_kg"])
    for key, value in actual.items():
        if key not in planned:
            checks.append({"类型": "计划外加工", "承购": key[0], "料批": key[1], "数量_kg": value,
                           "说明": "实际加工不在报价计划中，可能挤占其他承诺"})
    for key, value in planned.items():
        processed = actual.get(key, 0.0)
        if processed > value + 1e-6:
            checks.append({"类型": "超计划加工", "承购": key[0], "料批": key[1],
                           "计划_kg": round(value, 1), "实际_kg": round(processed, 1)})
    return checks
