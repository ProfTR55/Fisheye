# Calibration-First Fisheye Rectification (Single Image, Line Constraints)

This project estimates an unknown fisheye radial model from one image using straight-line constraints, then rectifies the image into a rectilinear view.

Primary objective: make world-straight lines appear straight after rectification.

## What is implemented

- Single-image calibration from line/plumb-line constraints
- Flexible learned radial mapping (`poly4`) instead of fixed fisheye formulas
- Optional anisotropic radial scaling (`sx`, `sy`) for edge behavior improvements
- Optional tangential/decentering terms (`p1`, `p2`)
- Calibration-first pipeline (estimate model, then render)
- Automatic curved line support extraction
- Manual line fallback via JSON annotations and optional click tool
- Rectification with output FOV control and optional auto-crop
- Debug artifacts:
  - line overlays on original image
  - sampled line points
  - estimated radial function plot
  - original-vs-rectified line comparison
  - summary JSON
  - optional undistortion maps (`map_x`, `map_y`)
- Tests:
  - synthetic recovery test
  - real image smoke test (if sample image exists)

## Model and math

For input pixel `(u, v)` and center `(cx, cy)`:

- `x_s = (u-cx)/sx`
- `y_s = (v-cy)/sy`
- Undo tangential distortion iteratively (`p1`, `p2`) to get `(x_u, y_u)`
- `rho = sqrt(x_u^2 + y_u^2)`
- `phi = atan2(y_u, x_u)`
- `rho_hat = rho / rho_max` where `rho_max` is center-to-farthest-corner distance

Radial angle mapping:

- `theta = g(rho_hat)`
- `g(r) = a1*r + a2*r^2 + a3*r^3 + a4*r^4`

Ray:

- `X = sin(theta) cos(phi)`
- `Y = sin(theta) sin(phi)`
- `Z = cos(theta)`

Rectified normalized plane (`f_out = 1` form used in loss):

- `x = X / Z = tan(theta) cos(phi)`
- `y = Y / Z = tan(theta) sin(phi)`

## Optimization objective

For each constraint line `i` with sampled points `p_ij`:

1. Undistort `p_ij -> q_ij = (x_ij, y_ij)` using current parameters.
2. Fit best 2D line in rectified plane (PCA/SVD).
3. Minimize point-to-line distances.

Main residuals:

- all signed orthogonal distances in rectified space

Regularization residuals:

- monotonicity penalty on `g'(r)` (enforces `g' > 0` softly)
- smoothness penalty on `g''(r)`
- soft center prior toward image center
- soft bounds on `g(1)` (avoid absurd edge angle)
- light coefficient L2 damping
- soft anisotropy prior toward `sx=1`, `sy=1`
- soft tangential prior toward `p1=0`, `p2=0`

Solver:

- `scipy.optimize.least_squares` with robust `soft_l1` loss
- alternating rounds with outlier trimming and refit
- multi-start initialization; best objective is selected automatically

## File structure

- `model.py`: fisheye model + forward/inverse projection helpers
- `loss.py`: straightness loss + regularizers
- `lines.py`: auto extraction + manual annotation/load
- `optimize.py`: calibration routine
- `rectify.py`: undistortion map + rendering + auto-crop
- `cli.py`: command-line entry point
- `demo.py`: real-image demo runner
- `tests/`: synthetic and real-image tests

## Install

```bash
python -m pip install numpy scipy opencv-python matplotlib
```

## Run

```bash
python cli.py --input image.jpg --output rectified.jpg --save-debug-dir debug/
```

Useful options:

```bash
python cli.py \
  --input image.jpg \
  --output rectified.jpg \
  --save-debug-dir debug/ \
  --auto-lines \
  --manual-lines annotations.json \
  --output-fov 100 \
  --crop auto \
  --max-iters 500 \
  --model poly4 \
  --aggressive-straightness \
  --save-maps
```

Straightness tuning options:

- `--aggressive-straightness`: stronger edge weighting, weaker smooth/L2 priors
- `--edge-weight-alpha`, `--edge-weight-power`: control edge-priority in line loss
- `--smooth-reg`, `--coeff-reg`, `--center-reg`: regularization strength controls
- `--trim-quantile`: outlier trimming aggressiveness between alternating rounds
- `--min-edge-angle`, `--max-edge-angle`: bounds for `g(1)` in degrees

Multi-start options:

- `--multi-start`: number of randomized starts
- `--start-angle-span`: randomization span around initial half-angle
- `--center-jitter-frac`: center jitter fraction
- `--anisotropy-jitter`: initial `sx/sy` jitter
- `--tangential-jitter`: initial `p1/p2` jitter
- `--random-seed`: reproducible multi-start seeds

Tangential options:

- `--tangential-reg`: regularization weight for `p1/p2`

Manual annotation (optional):

```bash
python cli.py --input image.jpg --output rectified.jpg --save-debug-dir debug/ --annotate-manual annotations.json
python cli.py --input image.jpg --output rectified.jpg --save-debug-dir debug/ --manual-lines annotations.json
```

Annotation JSON format:

```json
{
  "lines": [
    [[u1, v1], [u2, v2], [u3, v3]],
    [[u1, v1], [u2, v2], [u3, v3], [u4, v4]]
  ]
}
```

## Demo

Runs the full pipeline on `fisheye1.jpg` (or first available jpg):

```bash
python demo.py --out-dir demo_output --max-iters 300 --output-fov 100
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Limitations

- Auto line extraction is heuristic; cluttered scenes can still require manual lines.
- `poly4` is flexible but can underfit/overfit extreme lenses; monotonic spline variants are a natural next step.
- Current auto-crop is centered and validity-driven, not globally optimal by area.
- Perspective framing is FOV-driven; advanced framing (vanishing-point aware) is not yet included.

## Next improvements

- Add monotonic spline / cumulative-positive parameterization.
- Add robust line grouping with graph optimization and semantic filtering.
- Add optional joint optimization of output framing and FOV.
- Add baseline comparisons against equidistant/equisolid/stereographic for diagnostics.
