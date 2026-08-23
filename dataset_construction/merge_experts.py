#!/usr/bin/env python3
"""Join three expert result files and produce an aggregator LLaVA dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import build_index, parse_name_values, read_rows, stable_id, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Merge exactly three JSON/JSONL/CSV expert outputs into LLaVA JSON."
    )
    result.add_argument(
        "--expert", action="append", required=True, metavar="NAME=PATH",
        help="Expert name and input path; repeat exactly three times.",
    )
    result.add_argument(
        "--image-field", action="append", default=[], metavar="NAME=FIELD",
        help="Image field for one expert (default: image).",
    )
    result.add_argument(
        "--description-field", action="append", default=[], metavar="NAME=FIELD",
        help="Description field for one expert (default: outputs).",
    )
    result.add_argument(
        "--score-field", action="append", default=[], metavar="NAME=FIELD",
        help="Score field for one expert (default: score).",
    )
    result.add_argument("--metadata", type=Path, required=True, help="Label metadata file.")
    result.add_argument("--metadata-image-field", default="image")
    result.add_argument("--metadata-label-field", default="label")
    result.add_argument(
        "--key-mode", choices=("full", "basename"), default="full",
        help="Join by complete normalized image value or basename.",
    )
    result.add_argument(
        "--label-map", action="append", default=[], metavar="SOURCE=TARGET",
        help="Case-insensitive label mapping, e.g. 0=real; repeat as needed.",
    )
    result.add_argument(
        "--allowed-label", action="append", default=["real", "fake"],
        help="Allowed normalized target label; repeat to add labels.",
    )
    result.add_argument(
        "--missing", choices=("error", "drop"), default="error",
        help="How to handle images absent from an expert or metadata.",
    )
    result.add_argument(
        "--prompt-style", choices=("structured", "historical"), default="structured",
        help="Structured is robust to ablation; historical preserves Method 1/2/3 syntax.",
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("<s>", "").replace("</s>", "").split())


def normalize_label(value: Any, mapping: dict[str, str]) -> str:
    label = str(value or "").strip().lower()
    return mapping.get(label, label)


def make_prompt(
    names: list[str],
    rows: list[dict[str, Any]],
    description_fields: dict[str, str],
    score_fields: dict[str, str],
    style: str,
) -> str:
    if style == "historical":
        lines = [
            "<image>",
            "Here are three methods' descriptions and scores for the same image. "
            "Provide a final real/fake answer.",
        ]
        for number, (name, row) in enumerate(zip(names, rows), 1):
            description = clean_text(row.get(description_fields.get(name, "outputs")))
            score = clean_text(row.get(score_fields.get(name, "score"))) or "N/A"
            lines.append(f"Method {number} ({name}): {description} (Score: {score})")
        return "\n".join(lines)

    lines = [
        "<image>",
        "Three forensic experts analyzed this image. Provide a final real/fake answer.",
    ]
    for name, row in zip(names, rows):
        description = clean_text(row.get(description_fields.get(name, "outputs")))
        score = clean_text(row.get(score_fields.get(name, "score"))) or "N/A"
        lines.extend(
            [f"[Expert: {name}]", f"Description: {description}", f"Score: {score}"]
        )
    return "\n".join(lines)


def main() -> int:
    args = parser().parse_args()
    try:
        experts = parse_name_values(args.expert, "--expert")
        if len(experts) != 3:
            raise ValueError(f"--expert must be supplied exactly three times, got {len(experts)}")
        image_fields = parse_name_values(args.image_field, "--image-field")
        description_fields = parse_name_values(args.description_field, "--description-field")
        score_fields = parse_name_values(args.score_field, "--score-field")
        label_map = {
            key.lower(): value.lower()
            for key, value in parse_name_values(args.label_map, "--label-map").items()
        }
        unknown_names = (
            set(image_fields) | set(description_fields) | set(score_fields)
        ) - set(experts)
        if unknown_names:
            raise ValueError(f"field options refer to unknown experts: {sorted(unknown_names)}")

        indexes = {}
        for name, path_text in experts.items():
            path = Path(path_text)
            field = image_fields.get(name, "image")
            indexes[name] = build_index(read_rows(path), field, args.key_mode, str(path))
        metadata = build_index(
            read_rows(args.metadata), args.metadata_image_field, args.key_mode, str(args.metadata)
        )

        names = list(experts)
        all_keys = set(metadata)
        for index in indexes.values():
            all_keys |= set(index)
        if args.missing == "error":
            missing_messages = []
            for source_name, index in [*indexes.items(), ("metadata", metadata)]:
                missing = sorted(all_keys - set(index))
                if missing:
                    missing_messages.append(
                        f"{source_name} is missing {len(missing)} key(s), e.g. {missing[:3]}"
                    )
            if missing_messages:
                raise ValueError("; ".join(missing_messages))
            candidate_keys = all_keys
        else:
            candidate_keys = set(metadata)
            for index in indexes.values():
                candidate_keys &= set(index)

        allowed = {label.strip().lower() for label in args.allowed_label}
        records = []
        for key in sorted(candidate_keys):
            expert_rows = [indexes[name][key] for name in names]
            label = normalize_label(metadata[key].get(args.metadata_label_field), label_map)
            if label not in allowed:
                raise ValueError(
                    f"metadata key {key!r}: label {label!r} is not in {sorted(allowed)}"
                )
            image = str(expert_rows[0].get(image_fields.get(names[0], "image"), "")).strip()
            prompt = make_prompt(
                names, expert_rows, description_fields, score_fields, args.prompt_style
            )
            conversations = [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": label},
            ]
            records.append({
                "id": stable_id(image, conversations),
                "image": image,
                "conversations": conversations,
            })
        write_json(args.output, records)
        print(f"Wrote {len(records)} records to {args.output}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
