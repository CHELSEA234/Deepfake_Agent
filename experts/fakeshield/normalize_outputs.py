#!/usr/bin/env python3
"""Normalize FakeShield DTE-FDM JSONL without importing the upstream project."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REAL_PATTERNS = (
    r"\bnot (?:been )?tampered\b",
    r"\bnot manipulated\b",
    r"\bauthentic\b",
    r"\btaken directly (?:by|from) (?:a |the )?camera\b",
)
FAKE_PATTERNS = (
    r"\bhas been tampered\b",
    r"\btampered with\b",
    r"\bmanipulated\b",
    r"\bforg(?:ed|ery)\b",
    r"\bfake\b",
)


def probability(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, result))


def infer_label(text: str) -> str:
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in REAL_PATTERNS):
        return "real"
    if any(re.search(pattern, lowered) for pattern in FAKE_PATTERNS):
        return "fake"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FakeShield DTE-FDM JSONL to the public expert schema."
    )
    parser.add_argument("--input", required=True, type=Path, help="Upstream DTE-FDM JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Normalized JSONL.")
    parser.add_argument(
        "--keep-image-path",
        action="store_true",
        help="Keep upstream image paths; basename-only output is the privacy-safe default.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on malformed records.")
    return parser.parse_args()


def normalize(record: dict[str, Any], index: int, keep_path: bool) -> dict[str, Any]:
    image = str(record.get("image", ""))
    text = str(record.get("outputs", record.get("text", ""))).strip()
    p_yes = probability(record.get("p_yes"))
    p_no = probability(record.get("p_no"))
    fake_probability = p_yes
    if fake_probability is None and p_no is not None:
        fake_probability = 1.0 - p_no
    return {
        "id": str(record.get("id", index)),
        "image": image if keep_path else Path(image).name,
        "expert": "fakeshield",
        "label": infer_label(text),
        "fake_probability": fake_probability,
        "explanation": text,
        "metadata": {},
    }


def main() -> None:
    args = parse_args()
    written = skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record is not a JSON object")
                destination.write(
                    json.dumps(normalize(record, line_number, args.keep_image_path)) + "\n"
                )
                written += 1
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                if args.strict:
                    raise SystemExit(f"Invalid record on line {line_number}: {error}") from error
                skipped += 1
    print(f"Wrote {written} normalized records to {args.output}; skipped {skipped}")


if __name__ == "__main__":
    main()
