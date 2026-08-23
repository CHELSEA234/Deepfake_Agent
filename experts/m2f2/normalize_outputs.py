#!/usr/bin/env python3
"""Normalize M2F2-Det detection/explanation JSONL into a stable schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REAL_PATTERNS = (
    r"\bimage (?:is|appears|looks) (?:to be )?(?:a )?real\b",
    r"\bauthentic\b",
    r"\breal person\b",
)
FAKE_PATTERNS = (
    r"\bimage (?:is|appears|looks) (?:to be )?(?:a )?fake\b",
    r"\bmanipulat(?:ed|ion)\b",
    r"\bai[- ]generated\b",
    r"\bcomputer[- ]generated\b",
    r"\bdeepfake\b",
)


def parse_score(value: Any) -> float | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, score))


def infer_label(text: str) -> str:
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in REAL_PATTERNS):
        return "real"
    if any(re.search(pattern, lowered) for pattern in FAKE_PATTERNS):
        return "fake"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert M2F2-Det JSONL to the public expert schema."
    )
    parser.add_argument("--input", required=True, type=Path, help="Upstream output JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Normalized JSONL.")
    parser.add_argument(
        "--score-means",
        choices=("real", "fake"),
        default="real",
        help="Meaning of the upstream score field (default: real).",
    )
    parser.add_argument(
        "--keep-image-path",
        action="store_true",
        help="Keep upstream paths; basename-only output is the privacy-safe default.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on malformed records.")
    return parser.parse_args()


def normalize(
    record: dict[str, Any], index: int, score_means: str, keep_path: bool
) -> dict[str, Any]:
    image = str(record.get("image", ""))
    text = str(record.get("text", record.get("outputs", ""))).strip()
    score = parse_score(record.get("score"))
    fake_probability = None
    if score is not None:
        fake_probability = score if score_means == "fake" else 1.0 - score
    return {
        "id": str(record.get("id", record.get("key", index))),
        "image": image if keep_path else Path(image).name,
        "expert": "m2f2",
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
                normalized = normalize(
                    record, line_number, args.score_means, args.keep_image_path
                )
                destination.write(json.dumps(normalized) + "\n")
                written += 1
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                if args.strict:
                    raise SystemExit(f"Invalid record on line {line_number}: {error}") from error
                skipped += 1
    print(f"Wrote {written} normalized records to {args.output}; skipped {skipped}")


if __name__ == "__main__":
    main()
