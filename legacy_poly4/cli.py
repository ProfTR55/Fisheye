"""Command line entry point for calibration-first fisheye rectification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
if "--annotate-manual" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from confidence import ConfidenceConfig, assess_confidence
from confidence import report_to_dict as _confidence_report_to_dict
from lines import (
    AutoLineConfig,
    annotate_lines_interactive,
    draw_line_overlay,
    extract_line_constraints_auto,
    load_manual_lines,
    save_lines_json,
)
from loss import LossConfig, evaluate_objective, straightness_residuals
from optimize import OptimizeConfig, calibrate_from_lines
from rectify import RectifyConfig, render_rectified
from vp_grouping import VPGroupingConfig, group_lines_by_vp
from vp_grouping import report_to_dict as _vp_report_to_dict


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
    parser.add_argument(
        "--annotate-only",
        action="store_true",
        help="Only create manual annotations, then exit without calibration.",
    )
    parser.add_argument("--auto-lines", action="store_true", help="Enable automatic line extraction")
    parser.add_argument(
        "--exclude-line-indices",
        default="",
        help="Comma/space separated zero-based line constraint indices to exclude after extraction.",
    )
    parser.add_argument(
        "--auto-min-quality",
        type=float,
        default=0.28,
        help="Minimum automatic line quality score; raise to reject more weak/odd candidates.",
    )
    parser.add_argument(
        "--auto-local-min-quality",
        type=float,
        default=0.20,
        help="Lower quality floor used only as a regional backup for under-covered areas.",
    )
    parser.add_argument(
        "--auto-max-lines",
        type=int,
        default=40,
        help="Maximum number of automatic line candidates kept after quality scoring.",
    )
    parser.add_argument(
        "--auto-coverage-grid",
        type=int,
        default=4,
        help="Grid size for regional automatic line coverage, e.g. 4 means 4x4 cells.",
    )
    parser.add_argument(
        "--auto-min-lines-per-cell",
        type=int,
        default=1,
        help="Minimum candidate lines to try to keep in each occupied coverage cell.",
    )
    parser.add_argument(
        "--auto-max-lines-per-cell",
        type=int,
        default=4,
        help="Maximum candidate lines kept per coverage cell.",
    )
    parser.add_argument(
        "--auto-min-center-distance",
        type=float,
        default=0.10,
        help="Minimum distance between selected auto-line centers as a fraction of max image dimension.",
    )
    parser.add_argument(
        "--auto-monotonic-curvature-min",
        type=float,
        default=0.10,
        help=(
            "Minimum monotonic-curvature score for automatic candidates. "
            "Raise to reject more sign-flipping curves; lower to keep more noisy/near-straight supports."
        ),
    )
    parser.add_argument(
        "--auto-monotonic-curvature-weight",
        type=float,
        default=0.18,
        help="Weight of monotonic-curvature consistency in the automatic line quality score.",
    )

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
    parser.add_argument(
        "--validation-frac",
        type=float,
        default=0.0,
        help="Fraction of line constraints held out from calibration and used only for validation.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=29,
        help="Random seed used when splitting line constraints into train/validation groups.",
    )
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
    parser.add_argument(
        "--no-line-normalization",
        action="store_true",
        help="Disable per-line residual normalization; longer sampled lines then have more influence.",
    )
    parser.add_argument(
        "--loss-balance-grid",
        type=int,
        default=4,
        help="Grid size for balancing loss contribution across image regions.",
    )
    parser.add_argument(
        "--loss-balance-strength",
        type=float,
        default=1.0,
        help="Regional loss balancing strength; 0 disables, 1 gives each occupied cell similar influence.",
    )
    parser.add_argument(
        "--no-projection-priors",
        action="store_true",
        help=(
            "Disable physical projection priors (equidistant/equisolid/stereographic) "
            "as multi-start initializations. Falls back to randomized linear starts only."
        ),
    )
    parser.add_argument(
        "--projection-prior-families",
        default="equidistant,equisolid,stereographic",
        help=(
            "Comma-separated projection families used as multi-start priors. "
            "Supported: equidistant, equisolid, stereographic, orthographic."
        ),
    )
    parser.add_argument(
        "--projection-prior-half-angles",
        default="95,110,125",
        help=(
            "Comma-separated half-angle values (degrees) sampled for each projection prior."
        ),
    )
    parser.add_argument(
        "--vp-refinement",
        action="store_true",
        help=(
            "After initial calibration, cluster rectified line directions into "
            "vanishing-point groups, drop outliers, and run a second calibration "
            "pass. Accepts the refined model only if it improves training RMSE."
        ),
    )
    parser.add_argument(
        "--vp-max-clusters",
        type=int,
        default=3,
        help="Maximum vanishing-point clusters to fit (Manhattan world: <=3).",
    )
    parser.add_argument(
        "--vp-inlier-deg",
        type=float,
        default=6.0,
        help="Maximum angular residual (degrees) for a line to count as a VP inlier.",
    )
    parser.add_argument(
        "--vp-outlier-deg",
        type=float,
        default=14.0,
        help="Angular residual (degrees) above which a line is treated as a VP outlier.",
    )
    parser.add_argument(
        "--vp-min-lines",
        type=int,
        default=6,
        help="Skip VP refinement if fewer than this many usable lines are present.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with non-zero status if calibration confidence falls below the fail "
            "threshold. Useful for batch jobs that should reject unreliable results."
        ),
    )
    parser.add_argument(
        "--confidence-warn",
        type=float,
        default=0.55,
        help="Confidence score warn threshold; below this a warning is logged.",
    )
    parser.add_argument(
        "--confidence-fail",
        type=float,
        default=0.30,
        help="Confidence score fail threshold; below this --strict triggers a non-zero exit.",
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
    if "quality_rows" in auto_debug:
        quality_payload = {
            "quality_columns": _to_serializable(auto_debug.get("quality_columns", [])),
            "quality_rows": _to_serializable(auto_debug.get("quality_rows", [])),
            "selected_columns": _to_serializable(auto_debug.get("selected_columns", [])),
            "selected_rows": _to_serializable(auto_debug.get("selected_rows", [])),
        }
        (debug_dir / "auto_line_quality.json").write_text(
            json.dumps(quality_payload, indent=2), encoding="utf-8"
        )


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


def _parse_index_list(raw: str | None) -> List[int]:
    if raw is None or not raw.strip():
        return []
    indices: List[int] = []
    for token in raw.replace(",", " ").split():
        idx = int(token)
        if idx < 0:
            raise ValueError("--exclude-line-indices must contain non-negative indices.")
        indices.append(idx)
    return sorted(set(indices))


def _exclude_line_groups(
    line_groups: List[np.ndarray], excluded_indices: List[int]
) -> Tuple[List[np.ndarray], List[int]]:
    if not excluded_indices:
        return line_groups, []
    excluded = set(excluded_indices)
    invalid = [idx for idx in excluded_indices if idx >= len(line_groups)]
    if invalid:
        raise ValueError(
            f"Excluded line indices out of range: {invalid}; valid range is 0..{len(line_groups) - 1}"
        )
    kept = [line for idx, line in enumerate(line_groups) if idx not in excluded]
    dropped = [idx for idx in range(len(line_groups)) if idx in excluded]
    return kept, dropped


def _split_line_groups(
    line_groups: List[np.ndarray], validation_frac: float, seed: int
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    n = len(line_groups)
    frac = float(np.clip(validation_frac, 0.0, 0.9))
    if frac <= 0.0 or n < 3:
        return line_groups, [], list(range(n)), []

    val_count = int(round(n * frac))
    val_count = int(np.clip(val_count, 1, n - 1))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = sorted(int(i) for i in perm[:val_count])
    train_idx = sorted(int(i) for i in perm[val_count:])
    train = [line_groups[i] for i in train_idx]
    validation = [line_groups[i] for i in val_idx]
    return train, validation, train_idx, val_idx


def _line_rmse_px(metrics: Dict[str, float] | None, f_out: float) -> float | None:
    if not metrics:
        return None
    rmse = metrics.get("raw_line_rmse", metrics["line_rmse"])
    return float(rmse * f_out)


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

    if args.annotate_only and not args.annotate_manual:
        raise ValueError("--annotate-only requires --annotate-manual")

    if args.annotate_manual:
        annotate_lines_interactive(image_bgr, args.annotate_manual)
        if args.annotate_only:
            print(f"Saved manual annotations to: {args.annotate_manual}")
            return

    line_groups: List[np.ndarray] = []
    auto_debug: Dict[str, np.ndarray] = {}

    if args.manual_lines:
        manual_lines = load_manual_lines(args.manual_lines)
        line_groups.extend(manual_lines)

    use_auto = bool(args.auto_lines) or not bool(args.manual_lines)
    if use_auto:
        auto_min_quality = float(np.clip(args.auto_min_quality, 0.0, 1.0))
        auto_local_min_quality = float(
            np.clip(args.auto_local_min_quality, 0.0, auto_min_quality)
        )
        auto_lines, auto_debug = extract_line_constraints_auto(
            image_bgr,
            AutoLineConfig(
                max_lines=max(1, int(args.auto_max_lines)),
                min_quality_score=auto_min_quality,
                local_min_quality_score=auto_local_min_quality,
                coverage_grid_size=max(1, int(args.auto_coverage_grid)),
                min_lines_per_cell=max(0, int(args.auto_min_lines_per_cell)),
                max_lines_per_cell=max(1, int(args.auto_max_lines_per_cell)),
                min_center_distance_frac=max(0.0, float(args.auto_min_center_distance)),
                monotonic_curvature_min=float(
                    np.clip(args.auto_monotonic_curvature_min, 0.0, 1.0)
                ),
                monotonic_curvature_weight=max(
                    0.0, float(args.auto_monotonic_curvature_weight)
                ),
            ),
        )
        line_groups.extend(auto_lines)
        _save_auto_debug(debug_dir, auto_debug)

    if not line_groups:
        raise RuntimeError("No line constraints found. Use --manual-lines or --annotate-manual.")

    all_line_groups = list(line_groups)
    save_lines_json(all_line_groups, debug_dir / "line_constraints_all_before_exclude.json")
    cv2.imwrite(
        str(debug_dir / "line_constraints_all_labeled_overlay.png"),
        draw_line_overlay(image_bgr, all_line_groups, radius=2, label_indices=True),
    )

    excluded_indices = _parse_index_list(args.exclude_line_indices)
    line_groups, dropped_indices = _exclude_line_groups(line_groups, excluded_indices)
    if not line_groups:
        raise RuntimeError("All line constraints were excluded.")

    overlay = draw_line_overlay(image_bgr, line_groups, radius=2)
    cv2.imwrite(str(debug_dir / "line_constraints_overlay.png"), overlay)
    cv2.imwrite(
        str(debug_dir / "line_constraints_labeled_overlay.png"),
        draw_line_overlay(image_bgr, line_groups, radius=2, label_indices=True),
    )
    _save_sample_points_plot(
        image_bgr=image_bgr,
        line_groups=line_groups,
        path=debug_dir / "line_sampled_points.png",
    )
    save_lines_json(line_groups, debug_dir / "line_constraints_used.json")

    train_line_groups, validation_line_groups, train_indices, validation_indices = _split_line_groups(
        line_groups=line_groups,
        validation_frac=args.validation_frac,
        seed=args.validation_seed,
    )
    save_lines_json(train_line_groups, debug_dir / "line_constraints_train.json")
    if validation_line_groups:
        save_lines_json(validation_line_groups, debug_dir / "line_constraints_validation.json")
        cv2.imwrite(
            str(debug_dir / "line_constraints_train_overlay.png"),
            draw_line_overlay(image_bgr, train_line_groups, radius=2),
        )
        cv2.imwrite(
            str(debug_dir / "line_constraints_validation_overlay.png"),
            draw_line_overlay(image_bgr, validation_line_groups, radius=3),
        )
        _save_sample_points_plot(
            image_bgr=image_bgr,
            line_groups=train_line_groups,
            path=debug_dir / "line_sampled_points_train.png",
        )
        _save_sample_points_plot(
            image_bgr=image_bgr,
            line_groups=validation_line_groups,
            path=debug_dir / "line_sampled_points_validation.png",
        )

    prior_families = tuple(
        s.strip().lower() for s in args.projection_prior_families.split(",") if s.strip()
    )
    prior_half_angles = tuple(
        float(s.strip()) for s in args.projection_prior_half_angles.split(",") if s.strip()
    )
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
        use_projection_priors=not bool(args.no_projection_priors),
        projection_prior_families=prior_families if prior_families else (),
        projection_prior_half_angles_deg=prior_half_angles if prior_half_angles else (),
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
        normalize_line_residuals=not bool(args.no_line_normalization),
        spatial_balance_grid_size=max(1, int(args.loss_balance_grid)),
        spatial_balance_strength=max(0.0, float(args.loss_balance_strength)),
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
        line_groups=train_line_groups,
        optimize_cfg=optimize_cfg,
        loss_cfg=loss_cfg,
    )

    rectify_cfg = RectifyConfig(
        output_fov_deg=args.output_fov,
        output_width=args.output_width,
        output_height=args.output_height,
        crop_mode=args.crop,
    )

    vp_report: Dict[str, object] = {"enabled": False}
    if args.vp_refinement:
        vp_cfg = VPGroupingConfig(
            max_clusters=max(1, int(args.vp_max_clusters)),
            angle_inlier_deg=max(0.5, float(args.vp_inlier_deg)),
            angle_outlier_deg=max(float(args.vp_inlier_deg) + 0.5, float(args.vp_outlier_deg)),
            min_lines_for_grouping=max(3, int(args.vp_min_lines)),
        )
        vp_initial = group_lines_by_vp(
            line_groups=train_line_groups,
            model=calib.model,
            image_shape=image_bgr.shape[:2],
            cfg=vp_cfg,
        )
        outlier_idx = [i for i, flag in enumerate(vp_initial.is_outlier) if flag]
        kept_lines = [
            line for i, line in enumerate(train_line_groups) if i not in set(outlier_idx)
        ]
        baseline_train_metrics = evaluate_objective(
            calib.model, image_bgr.shape[:2], train_line_groups, loss_cfg
        )
        baseline_rmse = float(
            baseline_train_metrics.get(
                "raw_line_rmse", baseline_train_metrics["line_rmse"]
            )
        )
        attempted = (
            not vp_initial.skipped
            and len(kept_lines) >= max(3, int(args.vp_min_lines))
            and len(kept_lines) < len(train_line_groups)
        )
        accepted = False
        refined_rmse = None
        if attempted:
            calib_refined = calibrate_from_lines(
                image_shape=image_bgr.shape[:2],
                line_groups=kept_lines,
                optimize_cfg=optimize_cfg,
                loss_cfg=loss_cfg,
            )
            refined_train_metrics = evaluate_objective(
                calib_refined.model, image_bgr.shape[:2], train_line_groups, loss_cfg
            )
            refined_rmse = float(
                refined_train_metrics.get(
                    "raw_line_rmse", refined_train_metrics["line_rmse"]
                )
            )
            if refined_rmse < baseline_rmse * 0.995:
                calib = calib_refined
                train_line_groups = kept_lines
                accepted = True
                # Re-render with the new model.
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
        vp_report = {
            "enabled": True,
            "attempted": bool(attempted),
            "accepted": bool(accepted),
            "baseline_train_rmse": float(baseline_rmse),
            "refined_train_rmse": float(refined_rmse) if refined_rmse is not None else None,
            "num_outliers_flagged": len(outlier_idx),
            "outlier_indices": list(outlier_idx),
            "initial_grouping": _vp_report_to_dict(vp_initial),
        }
        save_lines_json(
            train_line_groups, debug_dir / "line_constraints_train_after_vp.json"
        )
        cv2.imwrite(
            str(debug_dir / "line_constraints_train_after_vp_overlay.png"),
            draw_line_overlay(image_bgr, train_line_groups, radius=2),
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

    train_all_metrics = evaluate_objective(
        calib.model, image_bgr.shape[:2], train_line_groups, loss_cfg
    )
    validation_metrics = (
        evaluate_objective(calib.model, image_bgr.shape[:2], validation_line_groups, loss_cfg)
        if validation_line_groups
        else None
    )
    validation_debug_lines: List[Dict[str, np.ndarray]] = []
    if validation_line_groups:
        _, _, validation_debug_lines = straightness_residuals(
            calib.model, image_bgr.shape[:2], validation_line_groups, loss_cfg
        )
    f_out = float(render["meta"]["f_out"])

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
    if validation_debug_lines:
        _save_comparison_plot(
            image_bgr=image_bgr,
            rectified_vis_bgr=out_img,
            rectified_full_shape=render["rectified"].shape[:2],
            debug_lines=validation_debug_lines,
            f_out=f_out,
            crop_box=render["crop_box"],
            path=debug_dir / "original_vs_rectified_validation_lines.png",
        )

    confidence_cfg = ConfidenceConfig(
        confidence_warn=float(np.clip(args.confidence_warn, 0.0, 1.0)),
        confidence_fail=float(np.clip(args.confidence_fail, 0.0, 1.0)),
    )
    confidence = assess_confidence(
        model=calib.model,
        image_shape=image_bgr.shape[:2],
        line_groups_used=calib.line_groups_used,
        train_metrics=train_all_metrics,
        validation_metrics=validation_metrics,
        cfg=confidence_cfg,
    )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "debug_dir": str(debug_dir),
            "image_shape": list(image_bgr.shape[:2]),
        "num_constraints_initial": len(line_groups),
        "num_constraints_before_exclude": len(all_line_groups),
        "excluded_line_indices": dropped_indices,
        "num_constraints_train": len(train_line_groups),
            "num_constraints_validation": len(validation_line_groups),
            "num_constraints_final": len(calib.line_groups_used),
            "model": "poly4",
            "estimated_params": calib.model.to_dict(image_shape=image_bgr.shape[:2]),
            "best_init_label": calib.best_init_label,
            "final_metrics": calib.final_metrics,
            "train_all_metrics": train_all_metrics,
            "validation_metrics": validation_metrics,
            "validation": {
                "enabled": bool(validation_line_groups),
                "validation_frac_requested": args.validation_frac,
                "validation_seed": args.validation_seed,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "train_line_rmse_px": _line_rmse_px(train_all_metrics, f_out),
                "validation_line_rmse_px": _line_rmse_px(validation_metrics, f_out),
            },
            "history": calib.history,
            "start_summaries": calib.start_summaries,
            "rectification": {
            "output_fov_deg": args.output_fov,
            "crop_mode": args.crop,
            "crop_box_y0y1x0x1": render["crop_box"],
            "meta": render["meta"],
        },
        "confidence": _confidence_report_to_dict(confidence),
        "vp_refinement": _to_serializable(vp_report),
        "loss_config": _to_serializable(loss_cfg.__dict__),
        "optimize_config": _to_serializable(optimize_cfg.__dict__),
        "aggressive_straightness": bool(args.aggressive_straightness),
        "manual_lines_path": args.manual_lines,
        "auto_lines_used": use_auto,
        "auto_line_quality": {
            "min_quality_score": float(np.clip(args.auto_min_quality, 0.0, 1.0)),
            "local_min_quality_score": float(
                np.clip(
                    args.auto_local_min_quality,
                    0.0,
                    float(np.clip(args.auto_min_quality, 0.0, 1.0)),
                )
            ),
            "max_lines": max(1, int(args.auto_max_lines)),
            "coverage_grid_size": max(1, int(args.auto_coverage_grid)),
            "min_lines_per_cell": max(0, int(args.auto_min_lines_per_cell)),
            "max_lines_per_cell": max(1, int(args.auto_max_lines_per_cell)),
            "min_center_distance_frac": max(0.0, float(args.auto_min_center_distance)),
            "monotonic_curvature_min": float(
                np.clip(args.auto_monotonic_curvature_min, 0.0, 1.0)
            ),
            "monotonic_curvature_weight": max(
                0.0, float(args.auto_monotonic_curvature_weight)
            ),
            "raw_candidates": int(len(auto_debug.get("quality_rows", [])))
            if auto_debug
            else 0,
            "selected_auto_lines": int(len(auto_debug.get("selected_rows", [])))
            if auto_debug
            else 0,
        },
    }
    summary_path = debug_dir / "summary.json"
    summary_path.write_text(json.dumps(_to_serializable(summary), indent=2), encoding="utf-8")

    print(f"Saved rectified image to: {output_path}")
    print(f"Saved debug artifacts to: {debug_dir}")
    print(f"Saved summary to: {summary_path}")
    print(
        f"Calibration confidence: {confidence.score:.3f} ({confidence.level}) — "
        f"best init: {calib.best_init_label}"
    )
    if confidence.warnings:
        for msg in confidence.warnings:
            print(f"  warning: {msg}", file=sys.stderr)

    if args.strict and not confidence.is_acceptable:
        print(
            f"Strict mode: confidence {confidence.score:.3f} below fail threshold "
            f"{confidence_cfg.confidence_fail:.2f}; exiting non-zero.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
