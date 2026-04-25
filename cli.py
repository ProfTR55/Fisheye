"""Command line entry point for calibration-first fisheye rectification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lines import (
    AutoLineConfig,
    annotate_lines_interactive,
    draw_line_overlay,
    extract_line_constraints_auto,
    load_manual_lines,
    save_lines_json,
)
from loss import LossConfig
from optimize import OptimizeConfig, calibrate_from_lines
from rectify import RectifyConfig, render_rectified


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibration-first fisheye rectification from straight-line constraints."
    )
    parser.add_argument("--input", required=True, help="Path to input fisheye image")
    parser.add_argument("--output", required=True, help="Path to output rectified image")
    parser.add_argument("--save-debug-dir", required=True, help="Directory to save debug artifacts")

    parser.add_argument("--manual-lines", default=None, help="Path to manual line annotation JSON")
    parser.add_argument(
        "--annotate-manual",
        default=None,
        help="Create manual line annotations by clicking and save to this JSON path",
    )
    parser.add_argument("--auto-lines", action="store_true", help="Enable automatic line extraction")

    parser.add_argument("--output-fov", type=float, default=100.0, help="Horizontal output FOV in degrees")
    parser.add_argument("--crop", choices=["none", "auto"], default="auto", help="Output crop mode")
    parser.add_argument("--max-iters", type=int, default=500, help="Max optimization iterations")
    parser.add_argument("--model", default="poly4", choices=["poly4"], help="Radial model family")
    parser.add_argument("--alternating-rounds", type=int, default=4, help="Alternating optimization rounds")
    parser.add_argument("--multi-start", type=int, default=3, help="Number of randomized solver restarts")
    parser.add_argument(
        "--start-angle-span",
        type=float,
        default=20.0,
        help="Half-angle randomization span in degrees around initial guess for multi-start.",
    )
    parser.add_argument(
        "--center-jitter-frac",
        type=float,
        default=0.04,
        help="Start center jitter scale as fraction of image size for multi-start.",
    )
    parser.add_argument(
        "--anisotropy-jitter",
        type=float,
        default=0.08,
        help="Start anisotropy jitter for sx/sy in multi-start.",
    )
    parser.add_argument(
        "--tangential-jitter",
        type=float,
        default=0.02,
        help="Start jitter for tangential p1/p2 in multi-start.",
    )
    parser.add_argument("--random-seed", type=int, default=13, help="Random seed for multi-start init")
    parser.add_argument("--save-maps", action="store_true", help="Save undistortion map .npy arrays")
    parser.add_argument("--output-width", type=int, default=None, help="Output width in pixels")
    parser.add_argument("--output-height", type=int, default=None, help="Output height in pixels")
    parser.add_argument(
        "--aggressive-straightness",
        action="store_true",
        help="Prioritize straightness more aggressively (weaker smoothness/L2, stronger edge weighting).",
    )
    parser.add_argument(
        "--edge-weight-alpha",
        type=float,
        default=1.6,
        help="Extra weight strength for edge-region points in line loss.",
    )
    parser.add_argument(
        "--edge-weight-power",
        type=float,
        default=2.0,
        help="Power for edge-region weighting profile.",
    )
    parser.add_argument(
        "--smooth-reg",
        type=float,
        default=0.35,
        help="Smoothness regularization weight on g''.",
    )
    parser.add_argument(
        "--coeff-reg",
        type=float,
        default=0.08,
        help="L2 regularization weight on polynomial coefficients.",
    )
    parser.add_argument(
        "--center-reg",
        type=float,
        default=8.0,
        help="Center prior weight toward image center.",
    )
    parser.add_argument(
        "--trim-quantile",
        type=float,
        default=0.96,
        help="Outlier trimming quantile in alternating rounds.",
    )
    parser.add_argument(
        "--min-edge-angle",
        type=float,
        default=55.0,
        help="Minimum allowed theta at normalized radius 1.0 (degrees).",
    )
    parser.add_argument(
        "--max-edge-angle",
        type=float,
        default=150.0,
        help="Maximum allowed theta at normalized radius 1.0 (degrees).",
    )
    parser.add_argument(
        "--anisotropy-reg",
        type=float,
        default=3.0,
        help="Regularization weight to keep sx/sy near 1.",
    )
    parser.add_argument(
        "--tangential-reg",
        type=float,
        default=4.0,
        help="Regularization weight to keep tangential p1/p2 near 0.",
    )
    return parser.parse_args()


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _save_auto_debug(debug_dir: Path, auto_debug: Dict[str, np.ndarray]) -> None:
    for key in ["gray", "edges", "segment_mask", "support"]:
        if key in auto_debug:
            cv2.imwrite(str(debug_dir / f"auto_{key}.png"), auto_debug[key])


def _save_sample_points_plot(
    image_bgr: np.ndarray, line_groups: List[np.ndarray], path: Path
) -> None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 9))
    plt.imshow(rgb)
    cmap = plt.get_cmap("tab20")
    for i, pts in enumerate(line_groups):
        c = cmap(i % 20)
        plt.scatter(pts[:, 0], pts[:, 1], s=5, color=c, alpha=0.75)
    plt.title("Sampled line constraint points (fisheye image)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_g_plot(path: Path, model) -> None:
    r, theta = model.curve(300)
    dtheta = model.g_prime(r)
    plt.figure(figsize=(8, 5))
    plt.plot(r, np.rad2deg(theta), label=r"$g(\hat{\rho})$", lw=2.0)
    plt.plot(r, np.rad2deg(dtheta), label=r"$g'(\hat{\rho})$ (deg/radial unit)", lw=1.5)
    plt.xlabel(r"$\hat{\rho}$")
    plt.ylabel("Angle (degrees)")
    plt.title("Estimated radial mapping")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_comparison_plot(
    image_bgr: np.ndarray,
    rectified_vis_bgr: np.ndarray,
    rectified_full_shape: Tuple[int, int],
    debug_lines: List[Dict[str, np.ndarray]],
    f_out: float,
    crop_box: Tuple[int, int, int, int],
    path: Path,
) -> None:
    y0, _, x0, _ = crop_box
    orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rect_rgb = cv2.cvtColor(rectified_vis_bgr, cv2.COLOR_BGR2RGB)
    # When crop is applied, coordinates are shifted by crop origin.
    full_h, full_w = rectified_full_shape
    cx_shift = 0.5 * (full_w - 1) - x0
    cy_shift = 0.5 * (full_h - 1) - y0

    plt.figure(figsize=(15, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(orig_rgb)
    ax1.set_title("Original fisheye: sampled curved supports")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(rect_rgb)
    ax2.set_title("Rectified: supports should become straight")
    ax2.axis("off")

    cmap = plt.get_cmap("tab20")
    for i, item in enumerate(debug_lines):
        c = cmap(i % 20)
        img_pts = item["img_points"]
        rect_pts = item["rect_points"]
        ax1.scatter(img_pts[:, 0], img_pts[:, 1], s=4, color=c, alpha=0.75)
        px = rect_pts[:, 0] * f_out + cx_shift
        py = rect_pts[:, 1] * f_out + cy_shift
        ax2.scatter(px, py, s=4, color=c, alpha=0.75)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = _parse_args()
    if args.model != "poly4":
        raise ValueError(f"Unsupported model: {args.model}")
    if args.min_edge_angle >= args.max_edge_angle:
        raise ValueError("--min-edge-angle must be smaller than --max-edge-angle")

    input_path = Path(args.input)
    output_path = Path(args.output)
    debug_dir = Path(args.save_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    if args.annotate_manual:
        annotate_lines_interactive(image_bgr, args.annotate_manual)

    line_groups: List[np.ndarray] = []
    auto_debug: Dict[str, np.ndarray] = {}

    if args.manual_lines:
        manual_lines = load_manual_lines(args.manual_lines)
        line_groups.extend(manual_lines)

    use_auto = bool(args.auto_lines) or not bool(args.manual_lines)
    if use_auto:
        auto_lines, auto_debug = extract_line_constraints_auto(image_bgr, AutoLineConfig())
        line_groups.extend(auto_lines)
        _save_auto_debug(debug_dir, auto_debug)

    if not line_groups:
        raise RuntimeError("No line constraints found. Use --manual-lines or --annotate-manual.")

    overlay = draw_line_overlay(image_bgr, line_groups, radius=2)
    cv2.imwrite(str(debug_dir / "line_constraints_overlay.png"), overlay)
    _save_sample_points_plot(
        image_bgr=image_bgr,
        line_groups=line_groups,
        path=debug_dir / "line_sampled_points.png",
    )
    save_lines_json(line_groups, debug_dir / "line_constraints_used.json")

    optimize_cfg = OptimizeConfig(
        max_iters=args.max_iters,
        alternating_rounds=args.alternating_rounds,
        outlier_trim_quantile=float(np.clip(args.trim_quantile, 0.7, 0.999)),
        multi_start=max(1, int(args.multi_start)),
        start_half_angle_span_deg=max(0.0, float(args.start_angle_span)),
        start_center_jitter_frac=max(0.0, float(args.center_jitter_frac)),
        start_anisotropy_jitter=max(0.0, float(args.anisotropy_jitter)),
        start_tangential_jitter=max(0.0, float(args.tangential_jitter)),
        random_seed=int(args.random_seed),
        verbose=2,
    )
    loss_cfg = LossConfig(
        lambda_smooth=args.smooth_reg,
        lambda_coeff_l2=args.coeff_reg,
        lambda_center=args.center_reg,
        edge_weight_alpha=args.edge_weight_alpha,
        edge_weight_power=args.edge_weight_power,
        min_theta_at_edge=np.deg2rad(args.min_edge_angle),
        max_theta_at_edge=np.deg2rad(args.max_edge_angle),
        lambda_anisotropy=args.anisotropy_reg,
        lambda_tangential=args.tangential_reg,
    )
    if args.aggressive_straightness:
        loss_cfg.lambda_smooth = max(0.02, 0.35 * loss_cfg.lambda_smooth)
        loss_cfg.lambda_coeff_l2 = max(0.01, 0.35 * loss_cfg.lambda_coeff_l2)
        loss_cfg.lambda_center = max(1.0, 0.5 * loss_cfg.lambda_center)
        loss_cfg.lambda_anisotropy = max(0.5, 0.5 * loss_cfg.lambda_anisotropy)
        loss_cfg.lambda_tangential = max(0.2, 0.5 * loss_cfg.lambda_tangential)
        loss_cfg.edge_weight_alpha = max(loss_cfg.edge_weight_alpha, 2.4)
        loss_cfg.edge_weight_power = max(loss_cfg.edge_weight_power, 2.2)
        loss_cfg.min_theta_at_edge = max(loss_cfg.min_theta_at_edge, np.deg2rad(70.0))
        optimize_cfg.outlier_trim_quantile = max(optimize_cfg.outlier_trim_quantile, 0.975)
    calib = calibrate_from_lines(
        image_shape=image_bgr.shape[:2],
        line_groups=line_groups,
        optimize_cfg=optimize_cfg,
        loss_cfg=loss_cfg,
    )

    rectify_cfg = RectifyConfig(
        output_fov_deg=args.output_fov,
        output_width=args.output_width,
        output_height=args.output_height,
        crop_mode=args.crop,
    )
    render = render_rectified(image_bgr, calib.model, cfg=rectify_cfg)

    if args.crop == "auto":
        out_img = render["rectified_crop"]
    else:
        out_img = render["rectified"]
    cv2.imwrite(str(output_path), out_img)

    cv2.imwrite(str(debug_dir / "rectified_full.png"), render["rectified"])
    cv2.imwrite(str(debug_dir / "rectified_crop.png"), render["rectified_crop"])
    cv2.imwrite(str(debug_dir / "valid_mask.png"), render["valid_mask"])
    cv2.imwrite(str(debug_dir / "valid_mask_crop.png"), render["valid_mask_crop"])

    if args.save_maps:
        np.save(debug_dir / "map_x.npy", render["map_x"])
        np.save(debug_dir / "map_y.npy", render["map_y"])
        np.save(debug_dir / "map_x_crop.npy", render["map_x_crop"])
        np.save(debug_dir / "map_y_crop.npy", render["map_y_crop"])

    _save_g_plot(debug_dir / "estimated_g_curve.png", calib.model)
    _save_comparison_plot(
        image_bgr=image_bgr,
        rectified_vis_bgr=out_img,
        rectified_full_shape=render["rectified"].shape[:2],
        debug_lines=calib.debug_lines,
        f_out=float(render["meta"]["f_out"]),
        crop_box=render["crop_box"],
        path=debug_dir / "original_vs_rectified_lines.png",
    )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "debug_dir": str(debug_dir),
        "image_shape": list(image_bgr.shape[:2]),
        "num_constraints_initial": len(line_groups),
        "num_constraints_final": len(calib.line_groups_used),
        "model": "poly4",
        "estimated_params": calib.model.to_dict(image_shape=image_bgr.shape[:2]),
        "final_metrics": calib.final_metrics,
        "history": calib.history,
        "start_summaries": calib.start_summaries,
        "rectification": {
            "output_fov_deg": args.output_fov,
            "crop_mode": args.crop,
            "crop_box_y0y1x0x1": render["crop_box"],
            "meta": render["meta"],
        },
        "loss_config": _to_serializable(loss_cfg.__dict__),
        "optimize_config": _to_serializable(optimize_cfg.__dict__),
        "aggressive_straightness": bool(args.aggressive_straightness),
        "manual_lines_path": args.manual_lines,
        "auto_lines_used": use_auto,
    }
    summary_path = debug_dir / "summary.json"
    summary_path.write_text(json.dumps(_to_serializable(summary), indent=2), encoding="utf-8")

    print(f"Saved rectified image to: {output_path}")
    print(f"Saved debug artifacts to: {debug_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
