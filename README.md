# Uniform Detection with YOLO

This project contains an Ultralytics YOLO object-detection dataset and a Python
inference script for identifying whether a person is wearing a uniform.

The detector has two classes:

- `in_uniform`
- `not_uniform`

## Project contents

```text
.
├── Uniform.pdf
├── infer_yolov11.py          # Image, video, directory, URL, and webcam inference
├── unidataset_yolov8_best.pt # Bundled trained checkpoint
└── unidataset/
    ├── data.yaml             # YOLO dataset configuration
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── staging/              # Staged dataset files
```

Each label file uses the YOLO bounding-box format:

```text
class_id center_x center_y width height
```

Coordinates are normalized to the image width and height.

## Requirements

- Python 3.8 or newer
- An Ultralytics-compatible YOLO checkpoint
- `ultralytics` for image, video, and directory inference
- `opencv-python` for webcam inference

Install the main dependency:

```bash
python -m pip install ultralytics
```

For webcam inference, also install OpenCV:

```bash
python -m pip install opencv-python
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python
```

## Running inference

Run inference on a single image:

```bash
python infer_yolov11.py \
  --model ./unidataset_yolov8_best.pt \
  --source ./unidataset/images/test/example.jpg
```

Run inference on every supported file in a directory:

```bash
python infer_yolov11.py \
  --model ./unidataset_yolov8_best.pt \
  --source ./unidataset/images/test
```

Run live inference from the default webcam:

```bash
python infer_yolov11.py \
  --model ./unidataset_yolov8_best.pt \
  --source 0
```

Press `q` or `Esc` in the video window to stop webcam inference.

Run inference with a stricter confidence threshold and save JSON detections:

```bash
python infer_yolov11.py \
  --model ./unidataset_yolov8_best.pt \
  --source ./unidataset/images/test \
  --conf 0.35 \
  --save-json
```

The script also accepts video files and URLs as `--source` values.

## Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--model PATH` | `/home/Sterben/unidataset_yolov11n_best.pt` | YOLO checkpoint to load |
| `--source SOURCE` | `0` | Image, directory, video, URL, or webcam index |
| `--output PATH` | `runs/detect` | Directory for annotated outputs and JSON |
| `--conf FLOAT` | `0.25` | Minimum confidence from `0` to `1` |
| `--iou FLOAT` | `0.7` | Non-maximum suppression IoU threshold |
| `--imgsz INT` | `736` | Inference image size in pixels |
| `--device DEVICE` | automatic | Device such as `cpu`, `0`, or `0,1` |
| `--save-json` | disabled | Save detections to `detections.json` |
| `--show` | disabled | Display annotated results in a window |
| `--no-save` | disabled | Do not save annotated media |

The default model path in the script refers to a checkpoint outside this
repository. When running from a fresh checkout or copy of this folder, pass
`--model ./unidataset_yolov8_best.pt` or provide another checkpoint explicitly.

## Outputs

For non-webcam inference, annotated media is written below:

```text
runs/detect/predict/
```

When `--save-json` is used, the script writes:

```text
runs/detect/detections.json
```

Each detection record has this shape:

```json
{
  "class_id": 0,
  "class_name": "in_uniform",
  "confidence": 0.91,
  "xyxy": [left, top, right, bottom]
}
```

The `xyxy` values are pixel coordinates in the source image.

## Dataset configuration

The dataset configuration is in [`unidataset/data.yaml`](unidataset/data.yaml).
It defines the train, validation, and test image directories and the two class
names.

The current YAML file contains an absolute `path` value. If the project is
moved to another machine, update that value to the new dataset location, or
use paths relative to the YAML file before training or validation.

## Troubleshooting

### Model file not found

Pass the checkpoint explicitly and verify that it exists:

```bash
python infer_yolov11.py --model ./unidataset_yolov8_best.pt --source ./unidataset/images/test
```

### Missing dependency

Install the package named in the error:

```bash
python -m pip install ultralytics opencv-python
```

OpenCV is only required for webcam mode.

### Webcam cannot be opened

Check that the camera is connected and try another camera index, for example:

```bash
python infer_yolov11.py --model ./unidataset_yolov8_best.pt --source 1
```

## Notes

- The script performs inference only; it does not train or evaluate the model.
- The checkpoint must be compatible with the installed Ultralytics version.
- GPU inference can be selected with `--device 0`; use `--device cpu` to force
  CPU inference.
