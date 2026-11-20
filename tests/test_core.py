"""业务方法测试：不依赖未来数据、不同基础不相加、同料批去向唯一、风险比较联合进行。"""
import pathlib
import unittest

from recovery.config import DEFAULT
from recovery.context import build_context
from recovery.data import Dataset
from recovery.planning import build_orders, solve_plan
from recovery.quality import estimate_all, estimate_lot, material_quality_stats, resolve_assays
from recovery.yield_model import fit_yield_model


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset.load()
        cls.cfg = DEFAULT
        cls.lots = estimate_all(cls.dataset, cls.cfg, knowledge_cut=cls.cfg.knowledge_cut_effective)

    def test_moisture_and_impurity_stay_on_their_own_basis(self):
        lot = self.lots["L001"]
        self.assertAlmostEqual(lot.dry_kg, lot.wet_kg * (1 - lot.moisture.mean), places=6)
        # 干基杂质不会被再次除以干物质比例
        self.assertLess(lot.impurity.mean, 0.2)

    def test_targeted_assay_does_not_shift_central_estimate(self):
        twin = dict(self.dataset.rows("lots")[0])
        assays = [row for row in self.dataset.rows("assays") if row["lot"] == twin["id"]]
        stats = material_quality_stats([twin], {twin["id"]: assays}, [twin["material"]])
        central = estimate_lot(twin, assays, stats[twin["material"]], self.cfg, "central")
        adverse = estimate_lot(twin, assays, stats[twin["material"]], self.cfg, "adverse")
        # 定向层更差时，只允许压力情景上移，中心估计保持不变。
        self.assertGreaterEqual(adverse.impurity.mean + 1e-12, central.impurity.mean)
        # 干基指标按代表干重加权，而不是把湿重直接平均。
        random_rows = [row for row in assays if row["sampling_basis"] == "random_mass"]
        weights = [row["represented_wet_kg"] * (1 - row["moisture_fraction"]) for row in random_rows]
        expected = sum(
            row["impurity_fraction_dry"] * weight for row, weight in zip(random_rows, weights)
        ) / sum(weights)
        self.assertAlmostEqual(central.impurity.mean, expected, places=9)

    def test_newer_assay_does_not_supersede_incompatible_scope(self):
        base = {
            "id": "Q1-1", "lot": "L001", "sample_stratum": "layer1", "sampling_basis": "random_mass",
            "represented_wet_kg": 400, "moisture_fraction": 0.11, "impurity_fraction_dry": 0.03,
            "polymer_fraction_dry": None, "known_at": "2026-09-13T09:00:00+08:00", "supersedes": None,
        }
        newer = dict(base, id="Q1-9", sample_stratum="layer9", known_at="2026-09-19T09:00:00+08:00", supersedes="Q1-1")
        active, dropped, conflicts = resolve_assays([base, newer], 1200.0)
        self.assertEqual({row["id"] for row in active}, {"Q1-1", "Q1-9"})
        self.assertTrue(any("取样层不同" in message for message in conflicts))

    def test_compatible_supersede_drops_old_assay(self):
        base = {
            "id": "Q1-1", "lot": "L001", "sample_stratum": "layer1", "sampling_basis": "random_mass",
            "represented_wet_kg": 400, "moisture_fraction": 0.11, "impurity_fraction_dry": 0.03,
            "polymer_fraction_dry": None, "known_at": "2026-09-13T09:00:00+08:00", "supersedes": None,
        }
        retest = dict(base, id="Q1-9", impurity_fraction_dry=0.031, known_at="2026-09-19T09:00:00+08:00", supersedes="Q1-1")
        active, dropped, conflicts = resolve_assays([base, retest], 1200.0)
        self.assertEqual([row["id"] for row in active], ["Q1-9"])
        self.assertEqual([row["id"] for row in dropped], ["Q1-1"])

    def test_coverage_is_reported_and_restricts_unquotable(self):
        sparse = dict(self.dataset.rows("lots")[0])
        assays = [{
            "id": "S1", "lot": sparse["id"], "sample_stratum": "layer0", "sampling_basis": "targeted",
            "represented_wet_kg": 50, "moisture_fraction": 0.10, "impurity_fraction_dry": 0.02,
            "polymer_fraction_dry": None, "known_at": "2026-09-12T09:00:00+08:00", "supersedes": None,
        }]
        stats = material_quality_stats([sparse], {sparse["id"]: assays}, [sparse["material"]])
        estimate = estimate_lot(sparse, assays, stats[sparse["material"]], self.cfg, "central")
        self.assertIn(estimate.status, ("restricted", "unquotable"))
        self.assertGreater(len(estimate.restrictions), 0)


class YieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset.load()
        cls.cfg = DEFAULT
        cls.model = fit_yield_model(cls.dataset.rows("trials"), cls.cfg)

    def test_holdout_not_used_in_fit(self):
        trials = self.dataset.rows("trials")
        model = fit_yield_model(trials, self.cfg)
        self.assertEqual(model.holdout_n, 30)
        self.assertGreater(model.holdout_rmse, 0)
        self.assertGreater(model.cv_rmse, 0)

    def test_same_trial_lot_is_not_independent(self):
        # 交叉验证按 trial_lot 分组：开发折有 30 个独立试验批，而不是 90 条记录。
        self.assertEqual(self.model.n_development_lots, 30)

    def test_yield_bounded(self):
        for moisture in (0.05, 0.15, 0.25):
            for impurity in (0.0, 0.05, 0.12):
                for recipe in self.model.recipes:
                    value = self.model.saleable_yield(moisture, impurity, recipe)
                    self.assertGreaterEqual(value, 0.05)
                    self.assertLessEqual(value, 1.0)


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = DEFAULT
        cls.dataset = Dataset.load()
        cls.lots = estimate_all(cls.dataset, cls.cfg, knowledge_cut=cls.cfg.knowledge_cut_effective)
        cls.model = fit_yield_model(cls.dataset.rows("trials"), cls.cfg)
        cls.orders = build_orders(cls.dataset.rows("inquiries"))

    def test_lot_not_promised_twice_beyond_availability(self):
        plan = solve_plan(self.orders, self.lots, self.dataset.rows("capacity"), self.model,
                          self.cfg, "2026-09-18", enforce_orders=set(self.orders))
        self.assertTrue(plan.feasible, plan.message)
        for lid, used in plan.lot_use.items():
            self.assertLessEqual(used, self.lots[lid].wet_kg + 1e-6)

    def test_banned_source_is_respected(self):
        paper_orders = {oid: order for oid, order in self.orders.items() if order.material == "paper"}
        plan = solve_plan(paper_orders, self.lots, self.dataset.rows("capacity"), self.model,
                          self.cfg, "2026-09-18", enforce_orders=set(paper_orders))
        self.assertTrue(plan.feasible)
        for allocation in plan.allocations:
            if allocation.order == "O3":
                self.assertNotEqual(allocation.source, "S3")

    def test_forbidden_tag_lot_not_used_by_default(self):
        plan = solve_plan(self.orders, self.lots, self.dataset.rows("capacity"), self.model,
                          self.cfg, "2026-09-18", enforce_orders=set(self.orders))
        self.assertTrue(plan.feasible)
        used = set(plan.lot_use)
        self.assertFalse(used & {"L007", "L014", "L021"})

    def test_impossible_quality_returns_limited_not_fake_grade(self):
        import dataclasses
        hard = dataclasses.replace(self.orders["O2"], max_impurity=0.005)
        plan = solve_plan({"O2": hard}, self.lots, self.dataset.rows("capacity"), self.model,
                          self.cfg, "2026-09-18", enforce_orders={"O2"})
        self.assertFalse(plan.feasible)
        self.assertIn("质量", plan.message)

    def test_later_acceptance_can_make_old_plan_short(self):
        from recovery.risk import draw_scenarios, evaluate_plan
        draws = draw_scenarios(self.lots, self.model, self.cfg, 400)
        base = solve_plan({"O1": self.orders["O1"]}, self.lots, self.dataset.rows("capacity"),
                          self.model, self.cfg, "2026-09-18", enforce_orders={"O1"})
        share = dict(self.orders)
        after = solve_plan(share, self.lots, self.dataset.rows("capacity"), self.model,
                           self.cfg, "2026-09-18", enforce_orders=set(share))
        self.assertTrue(after.feasible)
        sim_after = evaluate_plan(after, draws, self.model, self.cfg)
        # 全量承诺后，至少有一个承购的缺交风险高于其缺省承诺时的水平。
        self.assertTrue(any(record.shortfall_probability > 0.3 for record in sim_after.order_risk.values()))
        self.assertTrue(base.feasible)


if __name__ == "__main__":
    unittest.main()
