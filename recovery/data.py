"""资料读取、时间过滤与身份校验。

原始资料来自 data/，不做就地修改。追加检测或回执写入 .local/ 下的追加文件，
通过 merge 后在内存中形成当前视图；原始记录始终保留，报告的“证据”字段指向具体记录身份。
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from datetime import datetime, timezone, timedelta
from typing import Iterable

TZ = timezone(timedelta(hours=8))
TABLE_NAMES = ("lots", "assays", "trials", "inquiries", "capacity", "actuals")


class DataError(ValueError):
    """资料不一致，不能继续计算。"""


def parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - 防御性
        raise DataError(f"无法解析时间: {value}") from exc
    if parsed.tzinfo is None:
        raise DataError(f"时间缺少时区偏移: {value}")
    return parsed


def as_business_date(value: str) -> str:
    """把日期或时间字符串归一到当地业务日 YYYY-MM-DD。"""
    if len(value) == 10:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    return parse_ts(value).astimezone(TZ).date().isoformat()


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{path.name} 第 {number} 行不是合法 JSON") from exc
        if not isinstance(row, dict):
            raise DataError(f"{path.name} 第 {number} 行不是对象")
        rows.append(row)
    return rows


class Dataset:
    """当前业务视图。tables 中的记录可带 __source，用于报告证据来源。"""

    def __init__(self, tables: dict[str, list[dict]], manifest: dict | None = None):
        self.tables = copy.deepcopy(tables)
        self.manifest = manifest or {}

    # ---- 构造 -----------------------------------------------------------
    @classmethod
    def load(cls, root: str | pathlib.Path | None = None, verify_digest: bool = True) -> "Dataset":
        root = pathlib.Path(root or pathlib.Path(__file__).resolve().parents[1])
        manifest = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
        tables = {}
        for table in manifest["tables"]:
            path = root / table["path"]
            raw = path.read_bytes()
            if verify_digest:
                digest = hashlib.sha256(raw).hexdigest()
                if digest != table["sha256"]:
                    raise DataError(f"资料文件摘要不匹配: {table['name']}")
            rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            for row in rows:
                row.setdefault("__source", "original")
            tables[table["name"]] = rows
        return cls(tables, manifest)

    @classmethod
    def from_tables(cls, tables: dict[str, list[dict]]) -> "Dataset":
        return cls(tables)

    def copy(self) -> "Dataset":
        return Dataset(self.tables, self.manifest)

    # ---- 追加与合并 ------------------------------------------------------
    def append(self, table: str, rows: Iterable[dict]) -> "Dataset":
        if table not in TABLE_NAMES:
            raise DataError(f"未知资料表: {table}")
        new = self.copy()
        known = {row.get("id") for row in new.tables[table]}
        for row in rows:
            row = dict(row)
            if "id" not in row:
                raise DataError(f"{table} 追加记录缺少 id")
            if row["id"] in known:
                raise DataError(f"{table} 追加记录身份重复: {row['id']}")
            row.setdefault("__source", "appended")
            new.tables[table].append(row)
            known.add(row["id"])
        new.validate()
        return new

    def append_file(self, table: str, path: str | pathlib.Path) -> "Dataset":
        return self.append(table, _read_jsonl(pathlib.Path(path)))

    def save_append(self, directory: str | pathlib.Path, table: str, rows: Iterable[dict]) -> pathlib.Path:
        """把追加记录写入 .local 目录（JSON Lines，逐行追加），返回文件路径。"""
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"appended_{table}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                payload = {k: v for k, v in row.items() if k != "__source"}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    # ---- 校验 -----------------------------------------------------------
    def validate(self) -> None:
        lots = {row["id"] for row in self.tables["lots"]}
        inquiries = {row["id"] for row in self.tables["inquiries"]}
        assay_ids = {row["id"] for row in self.tables["assays"]}
        for row in self.tables["assays"]:
            if row["lot"] not in lots:
                raise DataError(f"检测引用了不存在的料批: {row['id']}")
            if row.get("supersedes") is not None and row["supersedes"] not in assay_ids:
                raise DataError(f"替代关系指向不存在的检测: {row['id']}")
            if row["sampling_basis"] not in ("random_mass", "targeted"):
                raise DataError(f"未知取样基础: {row['id']}")
            represented = row.get("represented_wet_kg")
            if represented is not None and float(represented) < 0:
                raise DataError(f"代表范围非法: {row['id']}")
            for key in ("moisture_fraction", "impurity_fraction_dry", "polymer_fraction_dry"):
                value = row.get(key)
                if value is not None and not 0 <= float(value) <= 1:
                    raise DataError(f"质量分数超出 [0,1]: {row['id']}.{key}")
            if row["sampling_basis"] == "targeted" and row.get("represented_wet_kg", 0) > 0:
                # targeted 仍可声明范围，但该范围不拥有随机代表性；此处只记录不拒绝。
                pass
        seen = set()
        for row in self.tables["assays"]:
            if row["id"] in seen:
                raise DataError(f"检测身份重复: {row['id']}")
            seen.add(row["id"])
        for row in self.tables["actuals"]:
            if row["lot"] not in lots:
                raise DataError(f"回执引用了不存在的料批: {row['id']}")
            if row.get("order") is not None and row["order"] not in inquiries:
                raise DataError(f"回执引用了不存在的承购: {row['id']}")
            if row["kind"] not in ("processed", "customer_assay"):
                raise DataError(f"未知回执类型: {row['id']}")

    # ---- 查询 -----------------------------------------------------------
    def rows(self, table: str) -> list[dict]:
        return self.tables[table]

    def row(self, table: str, key: str) -> dict:
        for row in self.tables[table]:
            if row["id"] == key:
                return row
        raise KeyError(key)

    def known_by(self, cut: str) -> "Dataset":
        """只保留 known_at 不晚于截止时间的检测、试验与回执。"""
        cut_dt = parse_ts(cut)
        tables = copy.deepcopy(self.tables)
        for table in ("assays", "trials", "actuals"):
            tables[table] = [
                row for row in tables[table]
                if row.get("known_at") is None or parse_ts(row["known_at"]) <= cut_dt
            ]
        return Dataset(tables, self.manifest)

    def known_up_to(self, date: str) -> "Dataset":
        return self.known_by(f"{as_business_date(date)}T23:59:59+08:00")

    def filter_material(self, material: str) -> "Dataset":
        tables = copy.deepcopy(self.tables)
        lot_ids = {row["id"] for row in tables["lots"] if row["material"] == material}
        tables["lots"] = [row for row in tables["lots"] if row["id"] in lot_ids]
        tables["assays"] = [row for row in tables["assays"] if row["lot"] in lot_ids]
        tables["actuals"] = [row for row in tables["actuals"] if row["lot"] in lot_ids]
        tables["inquiries"] = [row for row in tables["inquiries"] if row["material"] == material]
        return Dataset(tables, self.manifest)
