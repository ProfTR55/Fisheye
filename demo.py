"""Demo script for running the calibration-first pipeline on a real fisheye image."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _find_default_image() -> Path:
    candidates = [
        Path("fisheye1.jpg"),
        Path("fisheye2.jpg"),
    ]
    for p in candidates:
        if p.exists():
            return p
    jpgs = sorted(Path(".").glob("*.jpg")) + sorted(Path(".").glob("*.jpeg"))
    if not jpgs:
        raise FileNotFoundError("No demo image found. Pass --image explicitly.")
    return jpgs[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run demo rectification on one image.")
    parser.add_argument("--image", default=None, help="Input image path")
    parser.add_argument("--out-dir", default="demo_output", help="Output directory")
    parser.add_argument("--max-iters", type=int, default=300, help="Max optimization iterations")
    parser.add_argument("--output-fov", type=float, default=100.0, help="Output horizontal FOV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image) if args.image else _find_default_image()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_img = out_dir / "rectified_demo.jpg"
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "cli.py",
        "--input",
        str(image_path),
        "--output",
        str(output_img),
        "--save-debug-dir",
        str(debug_dir),
        "--auto-lines",
        "--crop",
        "auto",
        "--max-iters",
        str(args.max_iters),
        "--output-fov",
        str(args.output_fov),
    ]
    subprocess.run(cmd, check=True)

    summary_path = debug_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print("Demo complete.")
        print(f"Rectified image: {output_img}")
        print(f"Estimated params: {summary.get('estimated_params', {})}")
        print(f"Final metrics: {summary.get('final_metrics', {})}")
    else:
        print(f"Demo complete, but summary not found at {summary_path}")


if __name__ == "__main__":
    main()
