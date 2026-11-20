"""本地决策状态：接受、取消、客户复检处置、加工回执与报价快照。

设计原则
--------
* data/ 下的原始资料只读；运行时状态写入 .local/（或指定工作目录），可审计、可回放。
* 报价快照一经生成不可改写；后续检测、回执、接受或复检只会追加决策，并单独比较差异。
* 已加工数量是沉没投入，不随取消退回原批次；未加工的承诺可按规则释放。
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import uuid
from datetime import datetime, timedelta

from .data import DataError, TZ, as_business_date, parse_ts

STATE_VERSION = 1
DISPOSITIONS = ("pending", "accept", "recheck", "dispute", "return")


def _now() -> str:
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def _atomic_write(path: pathlib.Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


class Workspace:
    """一个可写的业务工作区。

    root/data 只读原始资料；root/.local 保存追加资料、决策日志与快照。
    """

    def __init__(self, root: str | pathlib.Path | None = None, local: str | pathlib.Path | None = None):
        self.root = pathlib.Path(root or pathlib.Path(__file__).resolve().parents[1])
        self.local = pathlib.Path(local) if local is not None else self.root / ".local"
        self.state_path = self.local / "state.json"
        self.snapshot_dir = self.local / "snapshots"
        self.decision_log = self.local / "decisions.jsonl"
        self.state = self._load_state()

    # ---- 状态读写 -------------------------------------------------------
    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {
                "version": STATE_VERSION,
                "created_at": _now(),
                "accepted": {},
                "cancelled": {},
                "dispositions": {},
                "receipts": [],
                "decisions": [],
            }
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            raise DataError(f"状态版本不兼容: {state.get('version')}")
        for key, default in (
            ("accepted", {}), ("cancelled", {}), ("dispositions", {}),
            ("receipts", []), ("decisions", []),
        ):
            state.setdefault(key, copy.deepcopy(default))
        return state

    def save(self) -> None:
        _atomic_write(self.state_path, json.dumps(self.state, ensure_ascii=False, indent=2))

    def _record(self, kind: str, payload: dict) -> dict:
        decision = {"id": uuid.uuid4().hex[:12], "type": kind, "at": _now(), "payload": payload}
        self.state["decisions"].append(decision)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.decision_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
        return decision

    # ---- 追加资料 -------------------------------------------------------
    def appended_rows(self, table: str) -> list[dict]:
        path = self.local / f"appended_{table}.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def append_rows(self, table: str, rows: list[dict]) -> pathlib.Path:
        path = self.local / f"appended_{table}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    # ---- 快照 -----------------------------------------------------------
    def save_snapshot(self, snapshot: dict) -> pathlib.Path:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.snapshot_dir / f"{snapshot['id']}.json"
        if path.exists():
            raise DataError(f"快照已存在，不能覆盖: {snapshot['id']}")
        _atomic_write(path, json.dumps(snapshot, ensure_ascii=False, indent=2))
        return path

    def load_snapshot(self, snapshot_id: str) -> dict:
        path = self.snapshot_dir / f"{snapshot_id}.json"
        if not path.exists():
            raise DataError(f"找不到快照: {snapshot_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_snapshots(self) -> list[dict]:
        if not self.snapshot_dir.exists():
            return []
        items = []
        for path in sorted(self.snapshot_dir.glob("*.json")):
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            items.append({
                "id": snapshot.get("id"),
                "created_at": snapshot.get("created_at"),
                "orders": list(snapshot.get("orders", {})),
                "price_valid_until": snapshot.get("price_valid_until"),
                "verdict": snapshot.get("verdict"),
                "accepted": bool(snapshot.get("accepted_at")),
            })
        return items

    # ---- 决策 -----------------------------------------------------------
    def register_acceptance(self, snapshot: dict, order_ids: list[str], accepted_at: str | None = None) -> dict:
        accepted_at = accepted_at or _now()
        for oid in order_ids:
            if oid in self.state["accepted"]:
                raise DataError(f"承购 {oid} 已经接受，不能重复承诺")
            if oid in self.state["cancelled"]:
                raise DataError(f"承购 {oid} 已取消")
        decision = self._record("accept", {"snapshot": snapshot["id"], "orders": order_ids})
        for oid in order_ids:
            self.state["accepted"][oid] = {
                "snapshot": snapshot["id"],
                "accepted_at": accepted_at,
                "price_valid_until": snapshot.get("price_valid_until"),
                "decision": decision["id"],
            }
        snapshot["accepted_at"] = accepted_at
        snapshot["accepted_orders"] = order_ids
        self.save()
        return decision

    def register_cancellation(self, order_id: str, reason: str = "") -> dict:
        if order_id not in self.state["accepted"]:
            raise DataError(f"承购 {order_id} 尚未接受，不能取消")
        if order_id in self.state["cancelled"]:
            raise DataError(f"承购 {order_id} 已取消")
        decision = self._record("cancel", {"order": order_id, "reason": reason})
        self.state["cancelled"][order_id] = {"at": _now(), "reason": reason, "decision": decision["id"]}
        self.save()
        return decision

    def register_receipt(self, receipt: dict) -> dict:
        required = {"lot", "order", "kind", "quantity_kg"}
        missing = required - set(receipt)
        if missing:
            raise DataError("回执缺少字段: " + "、".join(sorted(missing)))
        if receipt["kind"] not in ("processed", "customer_assay"):
            raise DataError(f"未知回执类型: {receipt['kind']}")
        if receipt["order"] not in self.state["accepted"]:
            raise DataError(f"回执对应的承购 {receipt['order']} 尚未接受")
        receipt = dict(receipt)
        receipt.setdefault("id", f"R{len(self.appended_rows('actuals')) + 1:03d}")
        receipt.setdefault("observed_impurity", None)
        receipt.setdefault("known_at", _now())
        existing = {row["id"] for row in self.appended_rows("actuals")}
        if receipt["id"] in existing:
            raise DataError(f"回执身份重复: {receipt['id']}")
        decision = self._record("receipt", dict(receipt))
        self.append_rows("actuals", [receipt])
        if receipt["kind"] == "processed":
            self.state["receipts"].append({
                "actual": receipt["id"], "lot": receipt["lot"], "order": receipt["order"],
                "wet_kg": float(receipt["quantity_kg"]), "at": receipt["known_at"],
                "decision": decision["id"],
            })
        self.save()
        return receipt

    def register_disposition(self, actual_id: str, disposition: str, **details) -> dict:
        if disposition not in DISPOSITIONS:
            raise DataError(f"未知处置: {disposition}")
        actuals = {row["id"]: row for row in self.appended_rows("actuals")}
        actual = actuals.get(actual_id)
        if actual is None:
            raise DataError(f"找不到客户复检回执: {actual_id}")
        if actual["kind"] != "customer_assay":
            raise DataError("只有客户复检回执可以登记接受、重检或争议")
        decision = self._record("disposition", {"actual": actual_id, "disposition": disposition, **details})
        self.state["dispositions"][actual_id] = {
            "disposition": disposition,
            "at": _now(),
            "decision": decision["id"],
            **details,
        }
        self.save()
        return decision

    def register_assay(self, assay: dict) -> dict:
        required = {"id", "lot", "sample_stratum", "sampling_basis", "represented_wet_kg",
                    "moisture_fraction", "impurity_fraction_dry", "known_at"}
        missing = required - set(assay)
        if missing:
            raise DataError("检测缺少字段: " + "、".join(sorted(missing)))
        if assay["sampling_basis"] not in ("random_mass", "targeted"):
            raise DataError(f"未知取样基础: {assay['sampling_basis']}")
        assay = dict(assay)
        assay.setdefault("polymer_fraction_dry", None)
        assay.setdefault("supersedes", None)
        existing = {row["id"] for row in self.appended_rows("assays")}
        if assay["id"] in existing:
            raise DataError(f"检测身份重复: {assay['id']}")
        decision = self._record("assay", dict(assay))
        self.append_rows("assays", [assay])
        self.save()
        return {"assay": assay, "decision": decision["id"]}


def business_days_between(later: str, earlier: str) -> int:
    """两个业务日之间的自然日差；用于有效期和交期比较。"""
    later_day = as_business_date(later)
    earlier_day = as_business_date(earlier)
    return (
        parse_ts(f"{later_day}T00:00:00+08:00") - parse_ts(f"{earlier_day}T00:00:00+08:00")
    ).days


def default_price_valid_until(plan_date: str, days: int) -> str:
    date = parse_ts(f"{as_business_date(plan_date)}T00:00:00+08:00").date()
    return (date + timedelta(days=days)).isoformat()


def snapshot_id(order_ids: list[str]) -> str:
    stamp = datetime.now(TZ).strftime("%Y%m%dT%H%M%S")
    suffix = hashlib.sha1(("|".join(sorted(order_ids)) + stamp).encode("utf-8")).hexdigest()[:6]
    return f"Q{stamp}-{suffix}"
