"""从工艺试验估计配方产出率、成本与不确定性。

资料限制
--------
* trial_lot 是独立试验批；同一 trial_lot 下不同配方是同一批料的平行试验，
  不能当作独立样板重复计数。交叉验证按 trial_lot 分组。
* fold=holdout 的试验只用于回测，不能进入拟合。
* recipe 只代表批准的工艺类别；拟合结果不应被读成任意配方的物理规律。
* 产出率的批间差异远大于含水率、杂质能解释的部分，因此模型给出中心趋势，
  剩余不确定性以“共同因子 + 独立项”的分布保留，供敏感性分析。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config


@dataclass
class YieldModel:
    recipes: tuple[str, ...]
    coefficients: np.ndarray  # [intercept, is_R1, is_R2, moisture, impurity]
    residual_sd: float
    residual_rmse: float
    residual_mae: float
    holdout_rmse: float | None
    holdout_mae: float | None
    holdout_bias: float | None
    holdout_n: int
    cv_rmse: float
    cv_mae: float
    cost_per_wet_kg: dict[str, float]
    cost_per_wet_sd: dict[str, float]
    byproduct_fraction: dict[str, float]
    byproduct_fraction_sd: dict[str, float]
    n_development_lots: int
    notes: list[str] = field(default_factory=list)

    def _row(self, moisture: float, impurity: float, recipe: str) -> np.ndarray:
        return np.array(
            [1.0, 1.0 if recipe == "R1" else 0.0, 1.0 if recipe == "R2" else 0.0, moisture, impurity],
            float,
        )

    def saleable_yield(self, moisture: float, impurity: float, recipe: str) -> float:
        """可售干重 / 投入干重。上限截断 1，避免把试验噪声当成质量增益。"""
        value = float(self._row(moisture, impurity, recipe) @ self.coefficients)
        return float(min(1.0, max(0.05, value)))

    def saleable_yield_vector(self, moisture, impurity, recipe: str) -> np.ndarray:
        """向量化版本，供蒙特卡洛抽样使用。"""
        moisture = np.asarray(moisture, float)
        impurity = np.asarray(impurity, float)
        value = (
            self.coefficients[0]
            + (self.coefficients[1] if recipe == "R1" else self.coefficients[2] if recipe == "R2" else 0.0)
            + self.coefficients[3] * moisture
            + self.coefficients[4] * impurity
        )
        return np.clip(value, 0.05, 1.0)

    def recipe_effect(self, recipe: str) -> float:
        """相对于 R0 的中心差异（在同等含水/杂质下）。"""
        if recipe == "R0":
            return 0.0
        if recipe == "R1":
            return float(self.coefficients[1])
        if recipe == "R2":
            return float(self.coefficients[2])
        raise KeyError(recipe)

    def to_dict(self) -> dict:
        return {
            "配方": list(self.recipes),
            "系数": {
                "截距": round(float(self.coefficients[0]), 6),
                "R1相对R0": round(float(self.coefficients[1]), 6),
                "R2相对R0": round(float(self.coefficients[2]), 6),
                "含水率": round(float(self.coefficients[3]), 6),
                "杂质": round(float(self.coefficients[4]), 6),
            },
            "残差标准差": round(self.residual_sd, 5),
            "残差RMSE": round(self.residual_rmse, 5),
            "按试验批交叉验证RMSE": round(self.cv_rmse, 5),
            "按试验批交叉验证MAE": round(self.cv_mae, 5),
            "留出折RMSE": None if self.holdout_rmse is None else round(self.holdout_rmse, 5),
            "留出折MAE": None if self.holdout_mae is None else round(self.holdout_mae, 5),
            "留出折偏差": None if self.holdout_bias is None else round(self.holdout_bias, 5),
            "留出折记录数": self.holdout_n,
            "开发折试验批数": self.n_development_lots,
            "每湿吨加工变动成本_分": {k: round(v, 2) for k, v in self.cost_per_wet_kg.items()},
            "副产物比例": {k: round(v, 5) for k, v in self.byproduct_fraction.items()},
            "说明": list(self.notes),
        }


def _yield_value(trial: dict) -> float:
    dry_input = float(trial["input_wet_kg"]) * (1.0 - float(trial["moisture"]))
    if dry_input <= 0:
        raise ValueError(f"试验 {trial['id']} 的干基投入非正")
    return float(trial["saleable_dry_kg"]) / dry_input


def _design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    for trial in rows:
        recipe = trial["recipe"]
        x.append([1.0, 1.0 if recipe == "R1" else 0.0, 1.0 if recipe == "R2" else 0.0, float(trial["moisture"]), float(trial["impurity"])])
        y.append(_yield_value(trial))
    return np.array(x, float), np.array(y, float)


def fit_yield_model(trials: list[dict], cfg: Config, use_holdout_for_fit: bool = False) -> YieldModel:
    if not trials:
        raise ValueError("没有工艺试验记录，无法建立产出率模型")
    recipes = tuple(sorted({t["recipe"] for t in trials}))
    development = [t for t in trials if t.get("fold") != "holdout"]
    holdout = [t for t in trials if t.get("fold") == "holdout"]
    if use_holdout_for_fit:
        development = list(trials)
        holdout = []
    if not development:
        raise ValueError("没有开发折试验记录")
    x, y = _design(development)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coef
    dof = max(len(y) - x.shape[1], 1)
    residual_sd = float(np.sqrt(np.sum(residual ** 2) / dof))
    residual_rmse = float(np.sqrt(np.mean(residual ** 2)))
    residual_mae = float(np.mean(np.abs(residual)))

    # 按 trial_lot 分组的交叉验证，避免同批平行试验泄露。
    lots = sorted({t["trial_lot"] for t in development})
    cv_errors: list[float] = []
    for lot in lots:
        train = [t for t in development if t["trial_lot"] != lot]
        test = [t for t in development if t["trial_lot"] == lot]
        if not train or not test:
            continue
        x_train, y_train = _design(train)
        coef_train, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        x_test, y_test = _design(test)
        cv_errors.extend((y_test - x_test @ coef_train).tolist())
    cv_rmse = float(np.sqrt(np.mean(np.square(cv_errors)))) if cv_errors else float("nan")
    cv_mae = float(np.mean(np.abs(cv_errors))) if cv_errors else float("nan")

    holdout_rmse = holdout_mae = holdout_bias = None
    if holdout:
        x_h, y_h = _design(holdout)
        err = y_h - x_h @ coef
        holdout_rmse = float(np.sqrt(np.mean(err ** 2)))
        holdout_mae = float(np.mean(np.abs(err)))
        holdout_bias = float(np.mean(err))

    cost_per_wet: dict[str, float] = {}
    cost_sd: dict[str, float] = {}
    byproduct: dict[str, float] = {}
    byproduct_sd: dict[str, float] = {}
    for recipe in recipes:
        rows = [t for t in development if t["recipe"] == recipe]
        if not rows:
            continue
        per_kg = np.array([float(t["cost_cents"]) / float(t["input_wet_kg"]) for t in rows])
        bp = np.array([float(t["byproduct_kg"]) / float(t["input_wet_kg"]) for t in rows])
        cost_per_wet[recipe] = float(np.mean(per_kg))
        cost_sd[recipe] = float(np.std(per_kg, ddof=1)) if len(per_kg) > 1 else 0.0
        byproduct[recipe] = float(np.mean(bp))
        byproduct_sd[recipe] = float(np.std(bp, ddof=1)) if len(bp) > 1 else 0.0

    notes = [
        "产出率按可售干重/投入干重拟合，避免湿基与干基混算",
        "同一 trial_lot 的平行试验不是独立样板；交叉验证按 trial_lot 分组",
        "留出折不参与拟合，仅用于回测；回测误差已计入风险范围",
        "配方只代表批准类别，不据此宣称某配方必然胜出",
        f"残差标准差约 {residual_sd:.3f}，远大于配方间中心差异时，配方排序不应被单独用来决策",
    ]
    return YieldModel(
        recipes=recipes,
        coefficients=coef,
        residual_sd=residual_sd,
        residual_rmse=residual_rmse,
        residual_mae=residual_mae,
        holdout_rmse=holdout_rmse,
        holdout_mae=holdout_mae,
        holdout_bias=holdout_bias,
        holdout_n=len(holdout),
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
        cost_per_wet_kg=cost_per_wet,
        cost_per_wet_sd=cost_sd,
        byproduct_fraction=byproduct,
        byproduct_fraction_sd=byproduct_sd,
        n_development_lots=len(lots),
        notes=notes,
    )


def recipe_difference_report(model: YieldModel) -> list[dict]:
    """配方两两差异及其与残差不确定性的比较。"""
    recipes = list(model.recipes)
    rows = []
    for i, a in enumerate(recipes):
        for b in recipes[i + 1:]:
            if a == "R0":
                diff = model.recipe_effect(b)
            elif b == "R0":
                diff = -model.recipe_effect(a)
            else:
                diff = model.recipe_effect(b) - model.recipe_effect(a)
            se = model.residual_sd * np.sqrt(2.0)
            rows.append(
                {
                    "配方A": a,
                    "配方B": b,
                    "中心差异": round(diff, 5),
                    "残差对比标准误": round(float(se), 5),
                    "差异/标准误": round(float(diff / se), 3) if se else None,
                    "结论": "数据无法区分" if se and abs(diff) < 2 * se else "存在可分辨差异",
                }
            )
    return rows


def counterfactual_holdout(model: YieldModel, trials: list[dict]) -> dict:
    """把每个留出折试验与同 trial_lot 的其他配方比较，说明“按批”而非“按配方”的波动。"""
    holdout = [t for t in trials if t.get("fold") == "holdout"]
    if not holdout:
        return {"记录数": 0, "说明": "没有留出折记录"}
    by_lot: dict[str, list[dict]] = {}
    for trial in holdout:
        by_lot.setdefault(trial["trial_lot"], []).append(trial)
    within = []
    for lot, rows in by_lot.items():
        if len(rows) >= 2:
            vals = [_yield_value(t) for t in rows]
            within.append(max(vals) - min(vals))
    errors = []
    for trial in holdout:
        predicted = model.saleable_yield(float(trial["moisture"]), float(trial["impurity"]), trial["recipe"])
        errors.append(_yield_value(trial) - predicted)
    return {
        "记录数": len(holdout),
        "留出折平均绝对误差": round(float(np.mean(np.abs(errors))), 5),
        "留出折RMSE": round(float(np.sqrt(np.mean(np.square(errors)))), 5),
        "同批不同配方最大差_中位数": round(float(np.median(within)), 5) if within else None,
        "说明": "同一试验批在不同配方下的差异与预测误差同量级，不能把某个配方的单次高产出视为稳定优势",
    }
