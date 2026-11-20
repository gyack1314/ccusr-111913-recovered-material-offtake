"""集中管理可调假设。

所有阈值都是本应用的业务假设，不是资料本身给出的结论。修改时应在报告的
“方法与适用边界”中同步说明，并记录到报价快照的 assumptions 中。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # 决策日与知识截止：默认只使用决策日结束前已知的检测与回执。
    plan_date: str = "2026-09-18"
    knowledge_cut: str | None = None  # None 表示 plan_date 当日 23:59:59+08:00

    # 报价价格有效期（自然日），到期后必须重新审批。
    price_valid_days: int = 2

    # 证据覆盖阈值（随机质量取样覆盖的湿重比例）。
    coverage_restricted: float = 0.60
    coverage_unquotable: float = 0.35

    # 单层随机检测数低于该值时标记为受限（层内误差无法由本批估计）。
    min_random_strata: int = 2

    # 定向取样与随机均值差异的告警阈值：max(该倍数 * 标准误, 该倍数 * 层内标准差)
    targeted_gap_se_mult: float = 3.0
    targeted_gap_within_mult: float = 2.0

    # 客户复检的测量误差（干基质量分数的标准差）。资料只有一个争议案例，
    # 不能据此标定；因此作为敏感性假设，而不是拟合结果。
    customer_assay_sigma: float = 0.003
    customer_assay_sigma_sensitivity: tuple[float, ...] = (0.0, 0.003, 0.010)

    # 质量门槛：按订单层面“任一整车复检不合格”的概率控制。
    quality_gate_any_truck: bool = True
    # 质量指标相关情景：adverse 表示杂质偏高与聚合物偏低共同出现；independent 用于敏感性。
    quality_correlation_mode: str = "adverse"

    # 质量拒收概率上限的缺省值；None 表示沿用各承购自己的 risk_limit。
    quality_reject_limit: float | None = None

    # 缺交判定容差（干公斤）。
    shortfall_tolerance_kg: float = 1.0

    # 产出率不确定性的共同因子占比（1 表示各订单/配方完全相关）。
    # 资料没有重复的“同料批/同配方/不同加工日”记录，无法识别共享程度；
    # 默认取 0.5 作为中性假设，并在敏感性分析中给出 0 与 1 的上下界。
    yield_common_share: float = 0.5
    yield_common_share_sensitivity: tuple[float, ...] = (0.0, 0.5, 1.0)

    # 缺交风险的额外名义缓冲（在校准起点之外）。0.0 表示按正态分位数。
    shortfall_buffer_start: float = 0.0
    shortfall_buffer_step: float = 0.05
    shortfall_buffer_max: float = 0.60
    shortfall_calibration_rounds: int = 10

    # 质量风险缓冲的校准步长与上限（质量分数）。
    quality_margin_step: float = 0.002
    quality_margin_max: float = 0.030
    quality_calibration_rounds: int = 10

    # 蒙特卡洛抽样次数。
    sims_calibration: int = 2500
    sims_portfolio: int = 1200
    sims_final: int = 8000
    random_seed: int = 20260918

    # 已知用途限制标签：资料没有客户用途字段，默认不把带标签批次纳入可报价方案。
    allow_forbidden_tags: bool = False

    # 报价重新判断阈值：报价快照与当前证据的均值差超过阈值即视为过期。
    requote_impurity_tolerance: float = 0.002
    requote_polymer_tolerance: float = 0.010
    requote_coverage_tolerance: float = 0.05

    # 优化求解。
    mip_time_limit: float = 20.0
    mip_rel_gap: float = 1e-4

    def with_(self, **changes) -> "Config":
        return replace(self, **changes)

    @property
    def knowledge_cut_effective(self) -> str:
        return self.knowledge_cut or f"{self.plan_date}T23:59:59+08:00"

    def assumptions(self) -> dict:
        return {
            "决策日": self.plan_date,
            "知识截止": self.knowledge_cut_effective,
            "价格有效期天数": self.price_valid_days,
            "覆盖受限阈值": self.coverage_restricted,
            "覆盖不可报价阈值": self.coverage_unquotable,
            "客户复检误差标准差": self.customer_assay_sigma,
            "复检误差敏感性": list(self.customer_assay_sigma_sensitivity),
            "质量门槛": "订单任一整车复检不合格概率" if self.quality_gate_any_truck else "单车复检不合格概率",
            "质量相关情景": self.quality_correlation_mode,
            "缺交判定容差_kg": self.shortfall_tolerance_kg,
            "产出率共同因子占比": self.yield_common_share,
            "产出率相关性敏感性": list(self.yield_common_share_sensitivity),
            "用途限制标签默认处理": "排除" if not self.allow_forbidden_tags else "允许但标注",
            "未计入": "原料取得成本、未列明的运输/仓储/资金成本、客户复检系统偏差",
        }


DEFAULT = Config()
