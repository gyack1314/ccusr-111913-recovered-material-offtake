"""联合情景与敏感性测试：共同抽样、资金与拒收口径、上下界。"""
import unittest

from recovery.config import DEFAULT
from recovery.data import Dataset
from recovery.planning import build_orders, solve_plan, QualityMargins
from recovery.quality import estimate_all
from recovery.risk import calibrate_plan, draw_scenarios, evaluate_plan
from recovery.yield_model import fit_yield_model


class RiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = DEFAULT
        cls.dataset = Dataset.load()
        cls.lots = estimate_all(cls.dataset, cls.cfg, knowledge_cut=cls.cfg.knowledge_cut_effective)
        cls.model = fit_yield_model(cls.dataset.rows("trials"), cls.cfg)
        cls.orders = build_orders(cls.dataset.rows("inquiries"))
        cls.draws = draw_scenarios(cls.lots, cls.model, cls.cfg, 800, seed=7)

    def test_common_share_controls_joint_shortfall(self):
        plan = solve_plan({k: self.orders[k] for k in ("O1", "O3")}, self.lots,
                          self.dataset.rows("capacity"), self.model, self.cfg, "2026-09-18",
                          enforce_orders={"O1", "O3"})
        self.assertTrue(plan.feasible)
        low = draw_scenarios(self.lots, self.model, self.cfg, 800, seed=7, common_share=0.0)
        high = draw_scenarios(self.lots, self.model, self.cfg, 800, seed=7, common_share=1.0)
        sim_low = evaluate_plan(plan, low, self.model, self.cfg)
        sim_high = evaluate_plan(plan, high, self.model, self.cfg)
        self.assertGreaterEqual(
            sim_low.joint_metrics["任一承购缺交概率"] + 1e-9,
            sim_high.joint_metrics["任一承购缺交概率"],
        )

    def test_profit_counts_rejected_mass(self):
        plan = solve_plan({"O1": self.orders["O1"]}, self.lots, self.dataset.rows("capacity"),
                          self.model, self.cfg, "2026-09-18", enforce_orders={"O1"})
        no_reject = evaluate_plan(plan, self.draws, self.model, self.cfg, customer_sigma=0.0,
                                  quality_limit=None)
        high_reject = evaluate_plan(plan, self.draws, self.model, self.cfg, customer_sigma=0.5)
        # 客户复检误差极大时，拒收损失应上升而不是保持为 0。
        self.assertGreaterEqual(
            high_reject.joint_metrics["期望拒收损失_分"],
            no_reject.joint_metrics["期望拒收损失_分"],
        )

    def test_calibration_meets_shortfall_limit_or_reports_why(self):
        plan, sim, log = calibrate_plan(
            {"O1": self.orders["O1"]}, self.lots, self.dataset.rows("capacity"),
            self.model, self.cfg, "2026-09-18", draws=self.draws, enforce_orders={"O1"},
        )
        self.assertTrue(plan.feasible)
        record = sim.order_risk["O1"]
        if not record.passes_shortfall:
            self.assertTrue(log["缺交校准"], "未达标时必须在日志中说明原因")

    def test_quality_margin_reduces_rejection(self):
        plan_zero = solve_plan({"O1": self.orders["O1"]}, self.lots, self.dataset.rows("capacity"),
                               self.model, self.cfg, "2026-09-18", enforce_orders={"O1"})
        plan_margin = solve_plan({"O1": self.orders["O1"]}, self.lots, self.dataset.rows("capacity"),
                                 self.model, self.cfg, "2026-09-18", enforce_orders={"O1"},
                                 margins={"O1": QualityMargins(0.012, 0.012)})
        self.assertTrue(plan_zero.feasible and plan_margin.feasible)
        a = evaluate_plan(plan_zero, self.draws, self.model, self.cfg)
        b = evaluate_plan(plan_margin, self.draws, self.model, self.cfg)
        self.assertLessEqual(b.order_risk["O1"].reject_probability, a.order_risk["O1"].reject_probability + 1e-6)

    def test_forecast_is_distribution_not_guarantee(self):
        plan = solve_plan({"O1": self.orders["O1"]}, self.lots, self.dataset.rows("capacity"),
                          self.model, self.cfg, "2026-09-18", enforce_orders={"O1"})
        sim = evaluate_plan(plan, self.draws, self.model, self.cfg)
        record = sim.order_risk["O1"]
        # 报出 P10 与缺交概率，说明这不是对每车的保证。
        self.assertLess(record.p10_output, record.p90_output)
        self.assertGreaterEqual(record.shortfall_probability, 0.0)


if __name__ == "__main__":
    unittest.main()
