# Single-Image Fisheye Rectification

A classical computer-vision tool that **calibrates an unknown fisheye lens from
a single image** using only the straight lines already present in the scene, and
rectifies the image into a distortion-free rectilinear view. No calibration
target, no training data.

![Input fisheye, rectification, and a phone reference](assets/hero.jpg)

*Left:* raw fisheye input. *Middle:* rectified output. *Right:* a normal phone
capture of the same scene for visual reference.

---

## What it does

Fisheye lenses bend straight world lines into curves. This tool finds the
distortion model that makes those lines straight again (the classical
*plumb-line* principle), estimating:

- the radial distortion map (interchangeable generic basis),
- the optical centre,
- anisotropic scales and tangential terms,

then renders a rectified image. It works from a **single image**, with **no
calibration target** and **no learned model**.

![Before / after on several scenes](assets/showcase.jpg)

---

## Repository layout

```
fisheye_zernike_v2/
└── fisheye_zernike/
    ├── model.py        # radial fisheye model + forward/inverse projection
    ├── loss.py         # straight-line loss + regularizers
    ├── lines.py        # automatic / manual line extraction
    ├── optimize.py     # multi-start calibration
    ├── hough_bootstrap.py  # rectified-space Hough refinement
    ├── diagnostics.py  # confidence reporting
    ├── rectify.py      # undistortion map + rendering
    └── cli.py          # command-line entry point
legacy_poly4/           # earlier poly4 pipeline (kept for reference)
```

## Install

```bash
python -m pip install -r fisheye_zernike_v2/requirements.txt
# numpy, scipy, opencv-python, matplotlib
```

## Usage

```bash
cd fisheye_zernike_v2

python -m fisheye_zernike.cli \
  --input  path/to/fisheye.jpg \
  --output rectified.jpg \
  --debug-dir debug \
  --auto-lines \
  --model zernike4
```

Manual line annotation (for sparse or cluttered scenes):

```bash
python -m fisheye_zernike.cli --input path/to/fisheye.jpg \
  --output unused.jpg --debug-dir debug_pick \
  --annotate-manual lines.json --annotate-only
```

## Tests

```bash
cd fisheye_zernike_v2
python -m unittest discover -s tests -v
```

## License

Released for research and educational use.
