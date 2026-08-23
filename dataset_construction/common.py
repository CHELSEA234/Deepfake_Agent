"""Shared, dependency-free helpers for dataset-construction CLIs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePath
from typing import Any, Iterable


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read records from JSON, JSONL, or CSV."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(value)
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            for container_key in ("records", "data", "items"):
                if isinstance(value.get(container_key), list):
                    value = value[container_key]
                    break
            else:
                value = [value]
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"{path}: expected an object or an array of objects")
        return value
    raise ValueError(f"{path}: supported extensions are .json, .jsonl, .ndjson, and .csv")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_name_values(values: Iterable[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects NAME=VALUE, got {value!r}")
        name, item = value.split("=", 1)
        name, item = name.strip(), item.strip()
        if not name or not item:
            raise ValueError(f"{option} expects non-empty NAME=VALUE, got {value!r}")
        if name in result:
            raise ValueError(f"{option}: duplicate name {name!r}")
        result[name] = item
    return result


def match_key(value: Any, mode: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if mode == "basename":
        return PurePath(text).name
    if mode == "full":
        return text
    raise ValueError(f"unknown key mode: {mode}")


def build_index(
    rows: Iterable[dict[str, Any]],
    field: str,
    key_mode: str,
    source: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        key = match_key(row.get(field), key_mode)
        if not key:
            raise ValueError(f"{source}: record {number} has no value for {field!r}")
        if key in index:
            raise ValueError(f"{source}: duplicate join key {key!r}")
        index[key] = row
    return index


def stable_id(image: str, conversations: list[dict[str, str]]) -> str:
    payload = json.dumps(
        {"image": image, "conversations": conversations},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
