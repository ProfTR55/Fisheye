"""Command line entry point for Zernike-first fisheye calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

if "--annotate-manual" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import confidence_report, zernike_coefficient_summary
from .hough_bootstrap import HoughBootstrapConfig, run_hough_bootstrap
from .lines import (
    AutoLineConfig,
    annotate_lines_interactive,
    draw_line_overlay,
    extract_line_constraints_auto,
    load_manual_lines,
    save_lines_json,
)
from .loss import LossConfig, evaluate_objective
from .model import SUPPORTED_MODELS, RadialFisheyeModel
from .optimize import OptimizeConfig, calibrate_from_lines
from .rectify import RectifyConfig, render_rectified


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zernike-first single-image fisheye calibration.")
    parser.add_argument("--input", required=True, help="Path to input fisheye image")
    parser.add_argument("--output", required=True, help="Path to output rectified image")
    parser.add_argument("--debug-dir", required=True, help="Directory for debug artifacts")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="zernike4")
    parser.add_argument("--auto-lines", action="store_true", help="Enable automatic line extraction")
    parser.add_argument("--manual-lines", default=None, help="Path to manual line annotation JSON")
    parser.add_argument(
        "--annotate-manual",
        default=None,
        help="Open a click UI, save manual line annotations to this JSON path, then use them.",
    )
    parser.add_argument(
        "--annotate-only",
        action="store_true",
        help="Only create manual annotations, then exit without calibration.",
    )
    parser.add_argument("--hough-bootstrap", action="store_true", help="Enable rectified-space Hough bootstrap")
    parser.add_argument("--compare-models", action="store_true", help="Run poly4 vs selected Zernike comparison")
    parser.add_argument("--validation-frac", type=float, default=0.25)
    parser.add_argument("--validation-seed", type=int, default=29)
    parser.add_argument("--output-fov", type=float, default=100.0)
    parser.add_argument("--output-width", type=int, default=None)
    parser.add_argument("--output-height", type=int, default=None)
    parser.add_argument("--crop", choices=["none", "auto"], default="auto")
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--alternating-rounds", type=int, default=4)
    parser.add_argument("--multi-start", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--projection-priors", action="store_true", help="Add fisheye projection-family initial starts")
    parser.add_argument(
        "--min-edge-angle",
        type=float,
        default=55.0,
        help="Soft lower bound for g(1), in degrees. Raise for wider fisheye correction.",
    )
    parser.add_argument(
        "--max-edge-angle",
        type=float,
        default=150.0,
        help="Soft upper bound for g(1), in degrees.",
    )
    parser.add_argument(
        "--theta-bound-reg",
        type=float,
        default=20.0,
        help="Regularization strength for edge-angle soft bounds.",
    )
    parser.add_argument(
        "--smooth-reg",
        type=float,
        default=0.35,
        help="Regularization strength for radial curve smoothness.",
    )
    parser.add_argument(
        "--coeff-reg",
        type=float,
        default=0.08,
        help="L2 regularization strength for radial coefficients.",
    )
    parser.add_argument("--auto-max-lines", type=int, default=40)
    parser.add_argument("--auto-min-quality", type=float, default=0.22)
    parser.add_argument("--auto-local-min-quality", type=float, default=0.12)
    parser.add_argument(
        "--auto-edge-preference",
        type=float,
        default=0.18,
        help="Ranking bonus for auto-line candidates farther from image center.",
    )
    parser.add_argument("--save-maps", action="store_true")
    return parser.parse_args()


def _split_line_groups(
    line_groups: List[np.ndarray], validation_frac: float, seed: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    n = len(line_groups)
    frac = float(np.clip(validation_frac, 0.0, 0.8))
    if n < 4 or frac <= 0.0:
        return line_groups, [], list(range(n)), []
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(np.clip(round(n * frac), 1, n - 3))
    val_idx = sorted(int(i) for i in idx[:n_val])
    train_idx = sorted(int(i) for i in idx[n_val:])
    return [line_groups[i] for i in train_idx], [line_groups[i] for i in val_idx], train_idx, val_idx


def _save_g_plot(path: Path, models: Dict[str, RadialFisheyeModel]) -> None:
    plt.figure(figsize=(8, 5))
    for label, model in models.items():
        r, theta = model.curve(300)
        plt.plot(r, np.rad2deg(theta), label=label)
    plt.xlabel("normalized radius")
    plt.ylabel("theta (deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _save_debug_images(debug_dir: Path, auto_debug: Dict[str, object]) -> None:
    for key in ("gray", "edges", "segment_mask", "support"):
        value = auto_debug.get(key)
        if isinstance(value, np.ndarray) and value.ndim in (2, 3):
            cv2.imwrite(str(debug_dir / f"auto_{key}.png"), value)


def _serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    return obj


def _run_model(
    family: str,
    image_shape: Tuple[int, int],
    train_lines: List[np.ndarray],
    validation_lines: List[np.ndarray],
    optimize_cfg: OptimizeConfig,
    loss_cfg: LossConfig,
) -> Dict[str, object]:
    result = calibrate_from_lines(
        image_shape=image_shape,
        line_groups=train_lines,
        model_family=family,
        optimize_cfg=optimize_cfg,
        loss_cfg=loss_cfg,
    )
    validation_metrics = (
        evaluate_objective(result.model, image_shape, validation_lines, loss_cfg)
        if validation_lines
        else None
    )
    return {
        "result": result,
        "validation_metrics": validation_metrics,
        "confidence": confidence_report(
            result.model, image_shape, result.line_groups_used, result.final_metrics, validation_metrics
        ),
    }


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")
    if args.annotate_only and not args.annotate_manual:
        raise ValueError("--annotate-only requires --annotate-manual")

    line_groups: List[np.ndarray] = []
    auto_debug: Dict[str, object] = {}
    if args.annotate_manual:
        annotate_lines_interactive(image, args.annotate_manual)
        if args.annotate_only:
            print(f"Saved manual annotations to: {args.annotate_manual}")
            return
        line_groups.extend(load_manual_lines(args.annotate_manual))
    if args.manual_lines:
        line_groups.extend(load_manual_lines(args.manual_lines))
    if args.auto_lines or not (args.manual_lines or args.annotate_manual):
        auto_cfg = AutoLineConfig(
            max_lines=args.auto_max_lines,
            min_quality_score=args.auto_min_quality,
            local_min_quality_score=args.auto_local_min_quality,
            edge_preference=args.auto_edge_preference,
        )
        auto_lines, auto_debug = extract_line_constraints_auto(image, auto_cfg)
        line_groups.extend(auto_lines)
        _save_debug_images(debug_dir, auto_debug)
    if not line_groups:
        raise RuntimeError("No line constraints available. Use --auto-lines, --manual-lines, or --annotate-manual.")

    save_lines_json(line_groups, debug_dir / "line_constraints_all.json")
    cv2.imwrite(str(debug_dir / "line_constraints_overlay.png"), draw_line_overlay(image, line_groups, label_indices=True))
    train_lines, validation_lines, train_idx, validation_idx = _split_line_groups(
        line_groups, args.validation_frac, args.validation_seed
    )
    save_lines_json(train_lines, debug_dir / "line_constraints_train.json")
    if validation_lines:
        save_lines_json(validation_lines, debug_dir / "line_constraints_validation.json")

    optimize_cfg = OptimizeConfig(
        max_iters=args.max_iters,
        alternating_rounds=args.alternating_rounds,
        multi_start=args.multi_start,
        random_seed=args.random_seed,
        use_projection_priors=args.projection_priors,
        verbose=0,
    )
    loss_cfg = LossConfig(
        min_theta_at_edge=np.deg2rad(args.min_edge_angle),
        max_theta_at_edge=np.deg2rad(args.max_edge_angle),
        lambda_theta_bounds=args.theta_bound_reg,
        lambda_smooth=args.smooth_reg,
        lambda_coeff_l2=args.coeff_reg,
    )
    rectify_cfg = RectifyConfig(
        output_fov_deg=args.output_fov,
        output_width=args.output_width,
        output_height=args.output_height,
        crop_mode=args.crop,
    )

    primary = _run_model(args.model, image.shape[:2], train_lines, validation_lines, optimize_cfg, loss_cfg)
    result = primary["result"]
    hough_report = None
    if args.hough_bootstrap:
        result, hough_report, train_lines = run_hough_bootstrap(
            image,
            result,
            train_lines,
            validation_lines,
            args.model,
            optimize_cfg,
            loss_cfg,
            rectify_cfg,
            HoughBootstrapConfig(),
        )
        primary["result"] = result
        primary["validation_metrics"] = (
            evaluate_objective(result.model, image.shape[:2], validation_lines, loss_cfg)
            if validation_lines
            else None
        )
        primary["confidence"] = confidence_report(
            result.model,
            image.shape[:2],
            result.line_groups_used,
            result.final_metrics,
            primary["validation_metrics"],
        )

    comparison: Dict[str, object] = {}
    comparison_models = {args.model: result.model}
    if args.compare_models:
        families = ["poly4", args.model] if args.model != "poly4" else ["poly4", "zernike4"]
        for family in dict.fromkeys(families):
            run = primary if family == args.model else _run_model(
                family, image.shape[:2], train_lines, validation_lines, optimize_cfg, loss_cfg
            )
            comp_result = run["result"]
            comparison[family] = {
                "final_metrics": comp_result.final_metrics,
                "validation_metrics": run["validation_metrics"],
                "confidence": run["confidence"],
                "best_init_label": comp_result.best_init_label,
                "estimated_params": comp_result.model.to_dict(image.shape[:2]),
            }
            comparison_models[family] = comp_result.model
        (debug_dir / "poly4_vs_zernike.json").write_text(
            json.dumps(_serializable(comparison), indent=2), encoding="utf-8"
        )

    render = render_rectified(image, result.model, rectify_cfg)
    out_img = render["rectified_crop"] if args.crop == "auto" else render["rectified"]
    cv2.imwrite(str(output_path), out_img)
    cv2.imwrite(str(debug_dir / "rectified_full.png"), render["rectified"])
    cv2.imwrite(str(debug_dir / "rectified_crop.png"), render["rectified_crop"])
    cv2.imwrite(str(debug_dir / "valid_mask.png"), render["valid_mask"])
    if args.save_maps:
        np.save(debug_dir / "map_x.npy", render["map_x"])
        np.save(debug_dir / "map_y.npy", render["map_y"])

    _save_g_plot(debug_dir / "estimated_g_curve.png", comparison_models)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "num_constraints": len(line_groups),
        "num_constraints_train": len(train_lines),
        "num_constraints_validation": len(validation_lines),
        "train_indices": train_idx,
        "validation_indices": validation_idx,
        "estimated_params": result.model.to_dict(image.shape[:2]),
        "final_metrics": result.final_metrics,
        "validation_metrics": primary["validation_metrics"],
        "confidence": primary["confidence"],
        "zernike": zernike_coefficient_summary(result.model),
        "hough_bootstrap": None if hough_report is None else hough_report.__dict__,
        "comparison": comparison,
        "rectification": {
            "crop_box": render["crop_box"],
            "meta": render["meta"],
        },
    }
    (debug_dir / "summary.json").write_text(json.dumps(_serializable(summary), indent=2), encoding="utf-8")
    print(f"Saved rectified image to: {output_path}")
    print(f"Saved debug summary to: {debug_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
