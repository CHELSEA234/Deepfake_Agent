#!/usr/bin/env python3
"""Convert paired semantic-expert JSONL reports to LLaVA adjudicator data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair GPT/Gemini reports and create LLaVA conversation JSON."
    )
    parser.add_argument("--gpt", required=True, type=Path, help="GPT report JSONL.")
    parser.add_argument("--gemini", required=True, type=Path, help="Gemini report JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON.")
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional label CSV. Without it, assistant turns are omitted.",
    )
    parser.add_argument("--label-path-column", default="Filename")
    parser.add_argument("--label-column", default="GT")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--split", help="Keep only this split (case-insensitive).")
    parser.add_argument(
        "--match",
        choices=("exact", "basename"),
        default="exact",
        help="How report and label image paths are joined.",
    )
    parser.add_argument(
        "--image-path-mode",
        choices=("original", "basename", "relative"),
        default="original",
        help="How image paths are stored in output.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Root removed when --image-path-mode relative is used.",
    )
    parser.add_argument(
        "--box-source",
        choices=("gpt", "gemini", "zero"),
        default="gpt",
        help="Witness box used in supervised Fake answers.",
    )
    parser.add_argument(
        "--answer-format",
        choices=("compact", "json", "label"),
        default="compact",
        help="Supervised answer: 'Fake; [box]', JSON, or label only.",
    )
    parser.add_argument(
        "--no-image-token",
        action="store_true",
        help="Do not prefix the human prompt with <image>.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping missing, failed, or unlabeled records.",
    )
    return parser.parse_args()


def key_for(path: str, match: str) -> str:
    return Path(path).name if match == "basename" else path


def read_reports(path: Path, match: str) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                report = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            image_path = report.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"{path}:{line_number}: missing image_path")
            key = key_for(image_path, match)
            if key in reports:
                raise ValueError(f"{path}:{line_number}: duplicate image key {key!r}")
            reports[key] = report
    return reports


def read_labels(args: argparse.Namespace) -> dict[str, str]:
    if not args.labels:
        return {}
    labels: dict[str, str] = {}
    with args.labels.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {args.label_path_column, args.label_column}
        if args.split:
            required.add(args.split_column)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"label CSV lacks column(s): {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            if args.split and row[args.split_column].casefold() != args.split.casefold():
                continue
            label = row[args.label_column].strip().title()
            if label not in {"Real", "Fake"}:
                if args.strict:
                    raise ValueError(
                        f"{args.labels}:{row_number}: invalid label {label!r}"
                    )
                continue
            key = key_for(row[args.label_path_column].strip(), args.match)
            if key in labels:
                raise ValueError(f"{args.labels}:{row_number}: duplicate key {key!r}")
            labels[key] = label
    return labels


def clean_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep only portable witness fields in a stable order."""
    return {
        "verdict": report.get("verdict", "N/A"),
        "claim": report.get("claim", "No claim provided."),
        "target_box": report.get("target_box"),
    }


def output_image_path(path: str, args: argparse.Namespace) -> str:
    if args.image_path_mode == "original":
        return path
    if args.image_path_mode == "basename":
        return Path(path).name
    if args.image_root is None:
        raise ValueError("--image-root is required for relative image paths")
    try:
        return str(Path(path).resolve().relative_to(args.image_root.resolve()))
    except ValueError as exc:
        raise ValueError(f"{path!r} is outside --image-root") from exc


def target_box(
    label: str, gpt: dict[str, Any], gemini: dict[str, Any], source: str
) -> list[int | float]:
    if label == "Real" or source == "zero":
        return [0, 0, 0, 0]
    report = gpt if source == "gpt" else gemini
    box = report.get("target_box")
    if (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
    ):
        return box
    return [0, 0, 0, 0]


def format_answer(label: str, box: list[int | float], answer_format: str) -> str:
    if answer_format == "label":
        return label
    if answer_format == "json":
        return json.dumps(
            {"verdict": label, "target_box": box},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return f"{label}; {json.dumps(box, separators=(',', ':'))}"


def build_item(
    source_path: str,
    gpt: dict[str, Any],
    gemini: dict[str, Any],
    label: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    witness_a = json.dumps(clean_report(gpt), ensure_ascii=False, separators=(",", ":"))
    witness_b = json.dumps(
        clean_report(gemini), ensure_ascii=False, separators=(",", ":")
    )
    prefix = "" if args.no_image_token else "<image>\n"
    prompt = (
        f"{prefix}You are the final forensic adjudicator. Review the image and both "
        "independent witness reports, resolve disagreements using visible evidence, "
        "and determine whether the image is Real or Fake.\n"
        f"Witness A (GPT): {witness_a}\n"
        f"Witness B (Gemini): {witness_b}"
    )
    image_path = output_image_path(source_path, args)
    conversations: list[dict[str, str]] = [{"from": "human", "value": prompt}]
    if label is not None:
        box = target_box(label, gpt, gemini, args.box_source)
        conversations.append(
            {
                "from": "gpt",
                "value": format_answer(label, box, args.answer_format),
            }
        )
    return {
        "id": hashlib.sha256(image_path.encode("utf-8")).hexdigest(),
        "image": image_path,
        "conversations": conversations,
    }


def main() -> int:
    args = parse_args()
    if args.split and not args.labels:
        raise ValueError("--split requires --labels")
    gpt_reports = read_reports(args.gpt, args.match)
    gemini_reports = read_reports(args.gemini, args.match)
    labels = read_labels(args)

    items: list[dict[str, Any]] = []
    skipped = 0
    for key, gpt in gpt_reports.items():
        gemini = gemini_reports.get(key)
        label = labels.get(key) if args.labels else None
        reason = None
        if gemini is None:
            reason = "missing Gemini report"
        elif gpt.get("error") or gemini.get("error"):
            reason = "witness report contains an error"
        elif args.labels and label is None:
            reason = "missing label"
        if reason:
            if args.strict:
                raise ValueError(f"{key!r}: {reason}")
            print(f"warning: skip {key!r}: {reason}", file=sys.stderr)
            skipped += 1
            continue
        items.append(
            build_item(str(gpt["image_path"]), gpt, gemini, label, args)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(items)} item(s) to {args.output}; skipped={skipped}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
