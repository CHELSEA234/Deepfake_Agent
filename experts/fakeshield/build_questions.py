#!/usr/bin/env python3
"""Build the JSONL question file consumed by FakeShield DTE-FDM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

DEFAULT_PROMPT = (
    "Is this image authentic, or has it been manipulated? Inspect object edges, "
    "perspective, semantics, lighting, and other visual artifacts, then explain "
    "the evidence for your conclusion."
)
DEFAULT_EXTENSIONS = ".bmp,.jpeg,.jpg,.png,.tif,.tiff,.webp"


def iter_images(root: Path, recursive: bool, extensions: set[str]) -> Iterable[Path]:
    candidates = root.rglob("*") if recursive else root.glob("*")
    yield from sorted(
        path for path in candidates if path.is_file() and path.suffix.lower() in extensions
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create FakeShield-compatible question JSONL from an image directory."
    )
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory of input images.")
    parser.add_argument("--output", required=True, type=Path, help="Destination JSONL file.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Question stored in each text field.")
    parser.add_argument(
        "--extensions",
        default=DEFAULT_EXTENSIONS,
        help="Comma-separated, case-insensitive image extensions.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search image-dir recursively.")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute image paths. Relative paths are safer and are the default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.image_dir.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Image directory does not exist: {args.image_dir}")

    extensions = {
        extension if extension.startswith(".") else f".{extension}"
        for extension in (item.strip().lower() for item in args.extensions.split(","))
        if extension
    }
    images = list(iter_images(root, args.recursive, extensions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for image in images:
            image_value = str(image if args.absolute_paths else image.relative_to(root))
            stream.write(json.dumps({"image": image_value, "text": args.prompt}) + "\n")
    print(f"Wrote {len(images)} questions to {args.output}")


if __name__ == "__main__":
    main()
