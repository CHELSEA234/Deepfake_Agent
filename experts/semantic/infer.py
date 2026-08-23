#!/usr/bin/env python3
"""Run GPT-4o or Gemini semantic-forensics inference over local images."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


DEFAULT_PROMPT = """You are a forensic image analyst. Assess whether the image is authentic or manipulated using visible semantic and image-level evidence (for example, impossible objects or anatomy, inconsistent text, lighting, shadows, reflections, edges, perspective, noise, or duplicated patterns). Do not identify people or infer sensitive personal attributes. Return one JSON object with exactly these keys: "verdict" ("Fake" or "Real"), "claim" (a concise three- or four-sentence explanation), and "target_box" ([x_min, y_min, width, height] in image pixels for the strongest evidence; use [0, 0, 0, 0] for Real)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query GPT-4o or Gemini for semantic image-forensics reports."
    )
    parser.add_argument("images", nargs="*", type=Path, help="Image paths to analyze.")
    parser.add_argument(
        "--image-list",
        type=Path,
        help="UTF-8 text file containing one image path per line.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument(
        "--provider", required=True, choices=("openai", "gemini"), help="API provider."
    )
    parser.add_argument(
        "--model",
        help="Model name (defaults: gpt-4o or gemini-2.0-flash).",
    )
    parser.add_argument("--prompt-file", type=Path, help="Optional system prompt text.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append and skip image paths already present in the output.",
    )
    return parser.parse_args()


def collect_images(args: argparse.Namespace) -> list[Path]:
    images = list(args.images)
    if args.image_list:
        for line in args.image_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                images.append(Path(line))
    if not images:
        raise ValueError("provide at least one image or --image-list")
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"{len(missing)} image(s) not found: {preview}")
    return images


def normalize_report(raw: str) -> dict[str, Any]:
    """Parse and validate the shared witness-report schema."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    data = json.loads(text)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")

    verdict = str(data.get("verdict", "")).strip().title()
    claim = data.get("claim")
    box = data.get("target_box")
    if verdict not in {"Real", "Fake"}:
        raise ValueError("verdict must be Real or Fake")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(not isinstance(value, (int, float)) for value in box)
    ):
        raise ValueError("target_box must contain four numbers")
    return {"verdict": verdict, "claim": claim.strip(), "target_box": box}


def openai_analyzer(
    model: str, prompt: str, max_tokens: int
) -> Callable[[Path], str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("install dependencies from requirements.txt") from exc

    client = OpenAI(api_key=api_key)

    def analyze(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image using only visible evidence.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{encoded}",
                                "detail": "auto",
                            },
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    return analyze


def gemini_analyzer(
    model: str, prompt: str, max_tokens: int
) -> Callable[[Path], str]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("install dependencies from requirements.txt") from exc

    client = genai.Client(api_key=api_key)

    def analyze(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        response = client.models.generate_content(
            model=model,
            contents=[
                "Analyze this image using only visible evidence.",
                types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
            ],
            config=types.GenerateContentConfig(
                system_instruction=prompt,
                temperature=0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text or ""

    return analyze


def completed_paths(output: Path) -> set[str]:
    if not output.exists():
        return set()
    completed: set[str] = set()
    with output.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if record.get("image_path"):
                    completed.add(str(record["image_path"]))
            except json.JSONDecodeError:
                continue
    return completed


def main() -> int:
    args = parse_args()
    if args.max_retries < 1:
        raise ValueError("--max-retries must be at least 1")
    images = collect_images(args)
    prompt = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file
        else DEFAULT_PROMPT
    )
    model = args.model or (
        "gpt-4o" if args.provider == "openai" else "gemini-2.0-flash"
    )
    analyzer = (
        openai_analyzer(model, prompt, args.max_tokens)
        if args.provider == "openai"
        else gemini_analyzer(model, prompt, args.max_tokens)
    )

    done = completed_paths(args.output) if args.resume else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    failures = 0
    written = 0
    with args.output.open(mode, encoding="utf-8") as stream:
        for index, path in enumerate(images, start=1):
            image_path = str(path)
            if image_path in done:
                print(f"[{index}/{len(images)}] skip {image_path}", file=sys.stderr)
                continue
            print(f"[{index}/{len(images)}] analyze {image_path}", file=sys.stderr)
            record: dict[str, Any] = {"image_path": image_path}
            for attempt in range(1, args.max_retries + 1):
                try:
                    record.update(normalize_report(analyzer(path)))
                    break
                except Exception as exc:
                    if attempt == args.max_retries:
                        record.update(
                            {"error": type(exc).__name__, "reason": str(exc)}
                        )
                        failures += 1
                    else:
                        time.sleep(args.retry_delay * attempt)
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            stream.flush()
            written += 1

    print(
        f"Wrote {written} record(s) to {args.output}; failures={failures}",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
