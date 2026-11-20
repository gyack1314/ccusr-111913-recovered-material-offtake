"""料批质量估计：分层取样、检测基础、替代关系与覆盖不足。

要点
----
* 湿基含水率与干基杂质/成分分开处理，绝不跨基础相加。
* 随机质量取样（random_mass）用于无偏估计；定向取样（targeted）不拥有随机代表性，
  只用于审核与压力情景，不能改善中心估计。
* 覆盖不足时，未覆盖部分的不确定性由同料批层内差异或同材质批间差异补足，
  不把“检测到的最有利一层”当成整批。
* 一份复测只有在取样层、取样基础和代表范围相容时才替代旧检测；
  否则保留两条证据并标记冲突。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .config import Config

METRICS = ("moisture", "impurity", "polymer")
METRIC_FIELD = {
    "moisture": "moisture_fraction",
    "impurity": "impurity_fraction_dry",
    "polymer": "polymer_fraction_dry",
}
# 对客户约束而言“变差”的方向：+1 表示越高越差，-1 表示越低越差。
ADVERSE_DIRECTION = {"moisture": 1, "impurity": 1, "polymer": -1}


@dataclass(frozen=True)
class MetricEstimate:
    name: str
    mean: float
    sd: float
    coverage: float
    n_random: int
    n_targeted: int
    evidence: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "指标": self.name,
            "均值": round(self.mean, 6),
            "标准差": round(self.sd, 6),
            "覆盖": round(self.coverage, 4),
            "随机层数": self.n_random,
            "定向层数": self.n_targeted,
            "证据": list(self.evidence),
            "提示": list(self.warnings),
        }


@dataclass
class LotQuality:
    lot_id: str
    material: str
    source: str
    wet_kg: float
    received_at: str
    moisture: MetricEstimate
    impurity: MetricEstimate
    polymer: MetricEstimate | None
    coverage_random: float
    coverage_including_targeted: float
    status: str  # quotable / restricted / unquotable
    restrictions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    forbidden_tags: list[str] = field(default_factory=list)
    effective_assay_ids: tuple[str, ...] = ()

    @property
    def metrics(self) -> dict[str, MetricEstimate]:
        result = {"moisture": self.moisture, "impurity": self.impurity}
        if self.polymer is not None:
            result["polymer"] = self.polymer
        return result

    def mean(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"料批 {self.lot_id} 没有 {name} 检测")
        return value.mean

    def sd(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise ValueError(f"料批 {self.lot_id} 没有 {name} 检测")
        return value.sd

    @property
    def dry_kg(self) -> float:
        return self.wet_kg * (1.0 - self.moisture.mean)

    def quality_vector(self) -> np.ndarray:
        """[moisture, impurity, polymer]，无聚合物时为 nan。"""
        polymer = self.polymer.mean if self.polymer is not None else np.nan
        return np.array([self.moisture.mean, self.impurity.mean, polymer], float)

    def sd_vector(self) -> np.ndarray:
        polymer = self.polymer.sd if self.polymer is not None else 0.0
        return np.array([self.moisture.sd, self.impurity.sd, polymer], float)

    def to_dict(self) -> dict:
        return {
            "料批": self.lot_id,
            "材质": self.material,
            "来源": self.source,
            "湿重": self.wet_kg,
            "接收时间": self.received_at,
            "干重估计": round(self.dry_kg, 1),
            "随机覆盖": round(self.coverage_random, 4),
            "含定向覆盖": round(self.coverage_including_targeted, 4),
            "状态": self.status,
            "限制": list(self.restrictions),
            "冲突": list(self.conflicts),
            "用途限制": list(self.forbidden_tags),
            "有效检测": list(self.effective_assay_ids),
            "指标": {k: v.to_dict() for k, v in self.metrics.items()},
        }


@dataclass
class MaterialQualityStats:
    material: str
    within_sd: dict[str, float]
    between_sd: dict[str, float]
    targeted_gap_mean: dict[str, float]
    targeted_gap_sd: dict[str, float]
    n_lots: int


# ---------------------------------------------------------------------------
# 替代关系


def resolve_assays(assays: list[dict], wet_kg: float) -> tuple[list[dict], list[dict], list[str]]:
    """处理 supersedes。

    返回 (active, dropped, conflicts)。
    替代成立需要：同一料批、同取样层、同取样基础、代表范围不小于旧检测、
    获知时间不早于旧检测。否则视为不相容，两者都保留并记录冲突。
    """
    by_id = {row["id"]: row for row in assays}
    conflicts: list[str] = []
    dropped: set[str] = set()
    for row in assays:
        target = row.get("supersedes")
        if not target:
            continue
        old = by_id.get(target)
        if old is None:
            conflicts.append(f"{row['id']} 声明替代不存在的检测 {target}")
            continue
        reasons = []
        if old["lot"] != row["lot"]:
            reasons.append("料批不同")
        if old.get("sample_stratum") != row.get("sample_stratum"):
            reasons.append("取样层不同")
        if old.get("sampling_basis") != row.get("sampling_basis"):
            reasons.append("取样基础不同")
        old_range = float(old.get("represented_wet_kg") or 0.0)
        new_range = float(row.get("represented_wet_kg") or 0.0)
        if new_range + 1e-9 < old_range:
            reasons.append(f"代表范围缩小（{new_range:g} < {old_range:g} kg）")
        if str(row.get("known_at") or "") < str(old.get("known_at") or ""):
            reasons.append("获知时间早于被替代检测")
        if reasons:
            conflicts.append(
                f"{row['id']} 不能替代 {target}：" + "、".join(reasons) + "；两条证据同时保留"
            )
        else:
            dropped.add(target)
    # 链式替代：若 A 兼容替代 B、C 兼容替代 A，则 B 也应失效。
    compatible = {}
    for row in assays:
        target = row.get("supersedes")
        if not target:
            continue
        if not any(message.startswith(row["id"] + " 不能替代") for message in conflicts):
            compatible[target] = row["id"]
    changed = True
    while changed:
        changed = False
        for target, replacement in compatible.items():
            if replacement in dropped and target not in dropped:
                dropped.add(target)
                changed = True
    active = [row for row in assays if row["id"] not in dropped]
    dropped_rows = [row for row in assays if row["id"] in dropped]
    declared = sum(float(row.get("represented_wet_kg") or 0.0) for row in active)
    if declared > wet_kg * 1.05:
        conflicts.append(
            f"有效检测代表范围合计 {declared:.0f} kg 超过可见湿重 {wet_kg:.0f} kg 的 105%；"
            "可能是新增检测未声明替代而重复计算，覆盖按上限截断"
        )
    return active, dropped_rows, conflicts


# ---------------------------------------------------------------------------
# 材质层面的差异统计


def material_quality_stats(lots: list[dict], assays_by_lot: dict[str, list[dict]], materials: Iterable[str]) -> dict[str, MaterialQualityStats]:
    result = {}
    for material in materials:
        within: dict[str, list[float]] = {m: [] for m in METRICS}
        between: dict[str, list[float]] = {m: [] for m in METRICS}
        gaps: dict[str, list[float]] = {m: [] for m in METRICS}
        n_lots = 0
        for lot in lots:
            if lot["material"] != material:
                continue
            rows = assays_by_lot.get(lot["id"], [])
            random_rows = [r for r in rows if r["sampling_basis"] == "random_mass"]
            targeted_rows = [r for r in rows if r["sampling_basis"] == "targeted"]
            if len(random_rows) >= 2:
                n_lots += 1
            for metric in METRICS:
                values = [float(r[METRIC_FIELD[metric]]) for r in random_rows if r.get(METRIC_FIELD[metric]) is not None]
                if len(values) >= 2:
                    within[metric].append(float(np.var(values, ddof=1)))
                if len(values) >= 1:
                    between[metric].append(float(np.mean(values)))
                tv = [float(r[METRIC_FIELD[metric]]) for r in targeted_rows if r.get(METRIC_FIELD[metric]) is not None]
                if values and tv:
                    gaps[metric].append(float(np.mean(tv) - np.mean(values)))
        within_sd = {}
        between_sd = {}
        gap_mean = {}
        gap_sd = {}
        for metric in METRICS:
            # 层内方差：对每批的层内方差取均值再开方。
            within_sd[metric] = float(np.sqrt(np.mean(within[metric]))) if within[metric] else 0.0
            between_sd[metric] = float(np.std(between[metric], ddof=1)) if len(between[metric]) >= 2 else 0.0
            gap_mean[metric] = float(np.mean(gaps[metric])) if gaps[metric] else 0.0
            gap_sd[metric] = float(np.std(gaps[metric], ddof=1)) if len(gaps[metric]) >= 2 else abs(gap_mean[metric])
        result[material] = MaterialQualityStats(
            material=material,
            within_sd=within_sd,
            between_sd=between_sd,
            targeted_gap_mean=gap_mean,
            targeted_gap_sd=gap_sd,
            n_lots=n_lots,
        )
    return result


# ---------------------------------------------------------------------------
# 单批估计


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if weights.sum() <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def estimate_metric(
    metric: str,
    random_rows: list[dict],
    targeted_rows: list[dict],
    wet_kg: float,
    material_stats: MaterialQualityStats,
    scenario: str,
    cfg: Config,
) -> tuple[MetricEstimate | None, list[str]]:
    field_name = METRIC_FIELD[metric]
    warnings: list[str] = []

    def stratum_weight(row: dict) -> float:
        """干基指标按代表干重加权；湿基指标按代表湿重加权，不跨基础平均。"""
        wet = float(row.get("represented_wet_kg") or 0.0)
        if metric in ("impurity", "polymer"):
            moisture = row.get("moisture_fraction")
            if moisture is not None:
                return wet * max(0.0, 1.0 - float(moisture))
        return wet

    random_values = [
        (float(r[field_name]), stratum_weight(r), r["id"])
        for r in random_rows if r.get(field_name) is not None
    ]
    targeted_values = [
        (float(r[field_name]), stratum_weight(r), r["id"])
        for r in targeted_rows if r.get(field_name) is not None
    ]
    if not random_values and not targeted_values:
        return None, warnings

    # 覆盖用代表湿重计算，确保不同指标的覆盖口径一致。
    covered_wet = sum(
        float(r.get("represented_wet_kg") or 0.0) for r in random_rows if r.get(field_name) is not None
    )
    coverage = min(1.0, covered_wet / wet_kg) if wet_kg > 0 else 0.0
    evidence = tuple(rid for _, _, rid in random_values)
    n_random = len(random_values)
    within_sd = material_stats.within_sd.get(metric, 0.0)
    between_sd = material_stats.between_sd.get(metric, 0.0)

    if random_values:
        values = np.array([v for v, _, _ in random_values], float)
        weights = np.array([w if w > 0 else 1.0 for _, w, _ in random_values], float)
        covered_mean = _weighted_mean(values, weights)
        if n_random >= 2:
            stratum_var = float(np.average((values - covered_mean) ** 2, weights=weights))
            if coverage > 0:
                # 归一化到“单层”的方差水平
                stratum_var = stratum_var * max(1.0, n_random / max(n_random - 1.0, 1.0))
            heterogeneity = float(np.sqrt(max(stratum_var, 0.0)))
            if heterogeneity == 0.0:
                heterogeneity = within_sd
        else:
            heterogeneity = max(within_sd, between_sd * 0.5)
            warnings.append("随机层少于 2，层内差异借用同材质统计")
        # 覆盖部分的均值误差 + 未覆盖部分与覆盖部分的差异
        missing = max(0.0, 1.0 - coverage)
        mean_sd = float(np.sqrt((coverage ** 2) * (heterogeneity ** 2) / max(n_random, 1) + (missing ** 2) * (heterogeneity ** 2)))
        mean = covered_mean
        if scenario == "adverse" and targeted_values and missing > 0:
            directed = _weighted_mean(
                np.array([v for v, _, _ in targeted_values], float),
                np.array([w if w > 0 else 1.0 for _, w, _ in targeted_values], float),
            )
            if ADVERSE_DIRECTION[metric] * (directed - covered_mean) > 0:
                mean = coverage * covered_mean + missing * directed
        elif scenario == "pooled":
            combined = random_values + targeted_values
            mean = _weighted_mean(
                np.array([v for v, _, _ in combined], float),
                np.array([w if w > 0 else 1.0 for _, w, _ in combined], float),
            )
            mean_sd = heterogeneity / np.sqrt(max(len(combined), 1))
        basis_note = "随机质量取样（未覆盖部分按同层差异建模）"
    else:
        # 只有定向取样：不能给出无偏等级。以定向均值为中心，按批间差异放大，
        # 并施加同材质观察到的定向偏差（不利方向）。
        values = np.array([v for v, _, _ in targeted_values], float)
        weights = np.array([w if w > 0 else 1.0 for _, w, _ in targeted_values], float)
        directed_mean = _weighted_mean(values, weights)
        gap = material_stats.targeted_gap_mean.get(metric, 0.0)
        adverse_gap = max(0.0, ADVERSE_DIRECTION[metric] * gap)
        mean = directed_mean + ADVERSE_DIRECTION[metric] * adverse_gap
        mean_sd = float(np.sqrt(between_sd ** 2 + material_stats.targeted_gap_sd.get(metric, 0.0) ** 2))
        coverage = 0.0
        evidence = tuple(rid for _, _, rid in targeted_values)
        warnings.append("仅有定向取样：不具随机代表性，只能作为受限估计")
        basis_note = "定向取样（受限）"
    if coverage < cfg.coverage_restricted:
        warnings.append(f"随机覆盖 {coverage:.0%} 低于 {cfg.coverage_restricted:.0%}，未覆盖部分仅由同材质差异建模")
    return (
        MetricEstimate(
            name=metric,
            mean=float(min(1.0, max(0.0, mean))),
            sd=float(max(mean_sd, 1e-6)),
            coverage=coverage,
            n_random=n_random,
            n_targeted=len(targeted_values),
            evidence=evidence,
            warnings=tuple(warnings + [basis_note]),
        ),
        warnings,
    )


def estimate_lot(
    lot: dict,
    assays: list[dict],
    material_stats: MaterialQualityStats,
    cfg: Config,
    scenario: str = "central",
) -> LotQuality:
    active, dropped, conflicts = resolve_assays(assays, float(lot["wet_kg"]))
    random_rows = [r for r in active if r["sampling_basis"] == "random_mass"]
    targeted_rows = [r for r in active if r["sampling_basis"] == "targeted"]
    if dropped:
        conflicts.append("已按相容替代关系移除旧检测: " + "、".join(sorted(r["id"] for r in dropped)))
    if not active:
        conflicts.append("没有可用检测")
    wet = float(lot["wet_kg"])
    coverage_random = min(1.0, sum(float(r.get("represented_wet_kg") or 0.0) for r in random_rows) / wet) if wet else 0.0
    coverage_total = min(1.0, coverage_random + sum(float(r.get("represented_wet_kg") or 0.0) for r in targeted_rows) / wet) if wet else 0.0
    # 代表范围声明本身不能超过可见湿重；超出的部分只按上限截断，不视为新增吨数。

    estimates: dict[str, MetricEstimate | None] = {}
    warnings: list[str] = []
    for metric in METRICS:
        estimate, metric_warnings = estimate_metric(
            metric, random_rows, targeted_rows, wet, material_stats, scenario, cfg
        )
        estimates[metric] = estimate
        warnings.extend(metric_warnings)

    # 聚合物只对塑料适用；纸类没有该项时为 None。
    if estimates["polymer"] is None and lot["material"] == "paper":
        pass

    restrictions: list[str] = []
    if coverage_random < cfg.coverage_unquotable or not random_rows:
        status = "unquotable"
        restrictions.append(f"随机覆盖 {coverage_random:.0%} 低于不可报价阈值 {cfg.coverage_unquotable:.0%}")
    elif coverage_random < cfg.coverage_restricted or len(random_rows) < cfg.min_random_strata:
        status = "restricted"
        restrictions.append(f"随机覆盖 {coverage_random:.0%} 或层数 {len(random_rows)} 不足，只能保留受限方案")
    else:
        status = "quotable"
    if estimates["moisture"] is None or estimates["impurity"] is None:
        status = "unquotable"
        restrictions.append("缺少含水率或杂质检测，无法换算干基")
    if lot["material"] == "plastic" and estimates["polymer"] is None:
        status = "restricted" if status == "quotable" else status
        restrictions.append("塑料批缺少聚合物成分检测，不能承诺最低成分")
    if conflicts:
        if status == "quotable":
            status = "restricted"
        restrictions.append("检测证据存在冲突或覆盖重复，按受限处理")
    if lot.get("forbidden_tags"):
        restrictions.append("已知用途限制: " + "、".join(lot["forbidden_tags"]))

    # 定向取样与随机均值的差异审核
    for metric in METRICS:
        est = estimates[metric]
        if est is None or not targeted_rows or not random_rows or est.n_targeted == 0:
            continue
        tv = [float(r[METRIC_FIELD[metric]]) for r in targeted_rows if r.get(METRIC_FIELD[metric]) is not None]
        rv = [float(r[METRIC_FIELD[metric]]) for r in random_rows if r.get(METRIC_FIELD[metric]) is not None]
        if not tv or not rv:
            continue
        gap = float(np.mean(tv) - np.mean(rv))
        limit = max(
            cfg.targeted_gap_se_mult * material_stats.targeted_gap_sd.get(metric, 0.0),
            cfg.targeted_gap_within_mult * material_stats.within_sd.get(metric, 0.0),
        )
        if abs(gap) > max(limit, 1e-9) and ADVERSE_DIRECTION[metric] * gap > 0:
            message = (
                f"{metric} 定向层比随机层更差 {gap:+.4f}（阈值 {limit:.4f}），未覆盖部分可能偏差；中心估计不采用定向层"
            )
            warnings.append(message)
            note = f"定向取样不利提示：{message}"
            if note not in restrictions:
                restrictions.append(note)

    moisture = estimates["moisture"]
    impurity = estimates["impurity"]
    polymer = estimates["polymer"]
    assert moisture is not None or status == "unquotable"
    # 构造占位估计，保证结构完整；调用方必须检查状态。
    if moisture is None:
        moisture = MetricEstimate("moisture", 0.0, 1.0, 0.0, 0, len(targeted_rows), ())
    if impurity is None:
        impurity = MetricEstimate("impurity", 1.0, 1.0, 0.0, 0, len(targeted_rows), ())

    return LotQuality(
        lot_id=lot["id"],
        material=lot["material"],
        source=lot["source"],
        wet_kg=wet,
        received_at=lot["received_at"],
        moisture=moisture,
        impurity=impurity,
        polymer=polymer,
        coverage_random=coverage_random,
        coverage_including_targeted=coverage_total,
        status=status,
        restrictions=restrictions,
        conflicts=conflicts,
        forbidden_tags=list(lot.get("forbidden_tags") or []),
        effective_assay_ids=tuple(r["id"] for r in active),
    )


def estimate_all(
    dataset,
    cfg: Config,
    scenario: str = "central",
    knowledge_cut: str | None = None,
) -> dict[str, LotQuality]:
    """估计全部料批。knowledge_cut 为 None 时使用 cfg 的知识截止。"""
    if knowledge_cut is not None:
        dataset = dataset.known_by(knowledge_cut)
    lots = dataset.rows("lots")
    assays_by_lot: dict[str, list[dict]] = {lot["id"]: [] for lot in lots}
    for assay in dataset.rows("assays"):
        assays_by_lot.setdefault(assay["lot"], []).append(assay)
    materials = sorted({lot["material"] for lot in lots})
    stats = material_quality_stats(lots, assays_by_lot, materials)
    return {
        lot["id"]: estimate_lot(lot, assays_by_lot.get(lot["id"], []), stats[lot["material"]], cfg, scenario)
        for lot in lots
    }


def quality_correlation(material: str, mode: str = "adverse") -> np.ndarray:
    """三指标相关矩阵（moisture, impurity, polymer）。

    资料只有约 12 个塑料批均值，无法稳定识别三项指标的相关性；因此默认采用
    “不利共变”情景：杂质偏高与聚合物偏低同时发生。independent 情景用于敏感性。
    """
    if mode == "independent":
        corr = np.eye(3)
        return corr
    # adverse
    corr = np.array(
        [
            [1.0, 0.30, -0.20],
            [0.30, 1.0, -0.60],
            [-0.20, -0.60, 1.0],
        ]
    )
    return corr


def correlation_from_data(lots: dict[str, LotQuality], material: str) -> dict:
    """报告可观察的批均值相关，仅用于说明“数据无法区分”。"""
    rows = [q for q in lots.values() if q.material == material]
    out = {"n_lots": len(rows), "note": "批间相关样本很少，不作为报价依据"}
    vecs = {"moisture": [q.moisture.mean for q in rows], "impurity": [q.impurity.mean for q in rows]}
    if all(q.polymer is not None for q in rows):
        vecs["polymer"] = [q.polymer.mean for q in rows]  # type: ignore[union-attr]
    names = list(vecs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if len(vecs[a]) >= 3:
                out[f"{a}-{b}"] = round(float(np.corrcoef(vecs[a], vecs[b])[0, 1]), 3)
    return out


def scenario_comparison(
    dataset, cfg: Config, knowledge_cut: str | None = None
) -> dict[str, dict[str, LotQuality]]:
    return {
        "central": estimate_all(dataset, cfg, "central", knowledge_cut),
        "adverse": estimate_all(dataset, cfg, "adverse", knowledge_cut),
        "pooled": estimate_all(dataset, cfg, "pooled", knowledge_cut),
    }
