# Fisheye Academic Stitcher

`acedemic.py` is a single-file, research-oriented panorama stitching pipeline for **single fisheye images**.

It generates a wide panorama by sampling multiple virtual perspective patches from fisheye space, warping them into a panorama canvas, and blending them with multiband fusion.

## Highlights

- Single-fisheye input support with automatic lens-region selection (`auto`, `left`, `right`, `top`, `bottom`, `full`)
- Adaptive patch center placement (`adaptive` or `uniform`)
- Bilinear sampling from fisheye space
- Bilinear splat-based warping into panorama canvas
- Gain matching in overlap regions
- Multiband blending with Laplacian/Gaussian pyramids
- Optional hole filling for uncovered pixels
- Debug export mode for per-patch diagnostics

## How It Works

1. Detect active fisheye region and crop to a square working area.
2. Build virtual rectilinear patch rays for the requested FOV.
3. Rotate rays by target yaw/pitch.
4. Project world rays back into fisheye coordinates.
5. Sample fisheye pixels with bilinear interpolation.
6. Warp each patch into panorama coordinates via bilinear splatting.
7. Blend patches progressively using gain matching + multiband blending.
8. Save final panorama and optional debug outputs.

## Requirements

- Python 3.9+
- `numpy`
- `Pillow`

Install:

```bash
pip install numpy pillow
```

## Quick Start

```bash
python acedemic.py --input fisheye1.jpg --output pano.png
```

Quality-focused example:

```bash
python acedemic.py ^
  --input fisheye1.jpg ^
  --output pano_hq.png ^
  --out-w 3200 --out-h 1200 ^
  --patch-count 10 ^
  --patch-fov-x 85 --patch-fov-y 120 ^
  --center-mode adaptive ^
  --multiband-levels 4
```

Debug export example:

```bash
python acedemic.py --input fisheye2.jpg --save-debug-dir debug_out
```

## CLI Arguments

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `--input` | str | required | Input fisheye image path |
| `--output` | str | `academic_stitch.png` | Output panorama path |
| `--lens` | str | `auto` | Lens mode: `auto/full/left/right/top/bottom` |
| `--crop-pad` | float | `1.12` | Padding ratio while square-cropping active lens area |
| `--fish-fov` | float | `180.0` | Fisheye field of view (degrees) |
| `--out-w` | int | `3200` | Output panorama width |
| `--out-h` | int | `1200` | Output panorama height |
| `--span` | float | `175.0` | Panorama yaw span (degrees) |
| `--pitch-center` | float | `0.0` | Panorama center pitch (degrees) |
| `--pitch-span` | float | `120.0` | Panorama pitch span (degrees) |
| `--patch-count` | int | `8` | Number of virtual patches |
| `--patch-w` | int | `0` | Patch width (`0` = auto) |
| `--patch-fov-x` | float | `90.0` | Patch horizontal FOV |
| `--patch-fov-y` | float | `120.0` | Patch vertical FOV |
| `--center-mode` | str | `adaptive` | Patch center distribution: `adaptive` or `uniform` |
| `--multiband-levels` | int | `4` | Pyramid levels for multiband blending |
| `--confidence-gamma` | float | `1.0` | Confidence shaping factor (currently used for diagnostics) |
| `--save-debug-dir` | str | `""` | Save intermediate patch/warp/alpha images |
| `--fill-holes` | flag | off | Fill uncovered pixels row-wise |

## Output

The script prints:

- Selected lens mode
- Processed fisheye dimensions
- Patch centers and patch parameters
- Coverage percentage
- Per-patch diagnostics (`srcCov`, `warpCov`, `conf`)

## Notes

- This pipeline is optimized for **single fisheye wide-view panoramas**, not full 360x180 dual-fisheye stitching.
- Increasing `--patch-count` and output resolution usually improves detail but increases runtime.
- If you see patch/block artifacts, try increasing `--patch-count` and/or lowering `--patch-fov-x`.

## File

- Main script: `acedemic.py`

## License

Add your preferred license (MIT, Apache-2.0, etc.) before publishing.
