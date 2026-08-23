#!/usr/bin/env python3
"""Build a portable image/question JSONL manifest for M2F2-Det inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROMPT = "Determine whether this face image is real or fake and explain the evidence."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an M2F2-Det image/question manifest from an image directory."
    )
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory of input images.")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSONL file.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Question for each image.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively.")
    parser.add_argument(
        "--extensions",
        default=".bmp,.jpeg,.jpg,.png,.webp",
        help="Comma-separated image extensions.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute image paths instead of paths relative to image-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.image_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Image directory does not exist: {args.image_dir}")
    extensions = {
        item if item.startswith(".") else f".{item}"
        for item in (part.strip().lower() for part in args.extensions.split(","))
        if item
    }
    candidates = root.rglob("*") if args.recursive else root.glob("*")
    images = sorted(
        path for path in candidates if path.is_file() and path.suffix.lower() in extensions
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, image in enumerate(images):
            image_value = str(image if args.absolute_paths else image.relative_to(root))
            record = {"id": str(index), "image": image_value, "text": args.prompt}
            stream.write(json.dumps(record) + "\n")
    print(f"Wrote {len(images)} questions to {args.output}")


if __name__ == "__main__":
    main()
