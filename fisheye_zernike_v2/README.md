# Zernike-First Fisheye Calibration V2

Clean research prototype for single-image fisheye calibration from straight-line
constraints.

The default radial model is Zernike-parametrized:

```text
theta(r) = c1 R_1^1(r) + c3 R_3^1(r) + c5 R_5^1(r) + c7 R_7^1(r)
```

`poly4` remains available as a baseline under the same loss and optimizer.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m fisheye_zernike.cli \
  --input ../data/raw/fisheye1.jpg \
  --output rectified.jpg \
  --debug-dir debug \
  --auto-lines \
  --model zernike4 \
  --compare-models
```

Optional Hough bootstrap:

```bash
python -m fisheye_zernike.cli \
  --input ../data/raw/fisheye1.jpg \
  --output rectified_hough.jpg \
  --debug-dir debug_hough \
  --auto-lines \
  --hough-bootstrap
```

## Main Options

- `--model zernike4|zernike6|poly4` selects the radial model.
- `--auto-lines` extracts line constraints automatically.
- `--manual-lines lines.json` loads manual polylines.
- `--annotate-manual lines.json` opens a click UI and saves manual polylines.
- `--annotate-only` saves annotations and exits before calibration.
- `--compare-models` writes `poly4_vs_zernike.json`.
- `--hough-bootstrap` runs rectified-space Hough refinement and keeps it only
  if validation or training metric improves.
- `--validation-frac` controls train/validation line split.
- `--projection-priors` adds equidistant/equisolid/stereographic initial
  states for slower but broader optimization.
- `--min-edge-angle` / `--max-edge-angle` constrain the estimated fisheye edge
  angle; raising the minimum is useful when line constraints do not reach the
  image edge.
- `--auto-local-min-quality` and `--auto-edge-preference` can keep more
  peripheral automatic line candidates in sparse scenes.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Manual Annotation

Create annotations only:

```bash
python -m fisheye_zernike.cli \
  --input ../data/raw/fisheye1.jpg \
  --output unused.jpg \
  --debug-dir debug_manual_pick \
  --annotate-manual ../data/annotations/manual_lines.json \
  --annotate-only
```

Then calibrate from those lines:

```bash
python -m fisheye_zernike.cli \
  --input ../data/raw/fisheye1.jpg \
  --output rectified_manual.jpg \
  --debug-dir debug_manual \
  --manual-lines ../data/annotations/manual_lines.json \
  --model zernike4 \
  --compare-models
```
