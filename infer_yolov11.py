#!/usr/bin/env python3
"""Run inference with an Ultralytics YOLO11 bounding-box model.

Install dependencies before running:
    python -m pip install ultralytics

Examples:
    python infer_yolov11.py
    python infer_yolov11.py --source 1 --conf 0.35
    python infer_yolov11.py --source image.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/home/Sterben/unidataset_yolov11n_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO11 object detection live from a webcam or on media."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to the YOLO checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Image, directory, video, URL, or webcam index (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/detect"),
        help="Directory where annotated media is saved (default: runs/detect).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=736,
        help="Inference image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, such as cpu, 0, or 0,1 (default: auto-select).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Write detections to detections.json in the output directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated results in a window.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images or videos.",
    )
    return parser.parse_args()


def detection_records(result: Any) -> list[dict[str, Any]]:
    """Convert one Ultralytics result into JSON-serializable detections."""
    if result.boxes is None:
        return []

    names = result.names
    records: list[dict[str, Any]] = []
    for box, confidence, class_id in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
    ):
        class_index = int(class_id)
        records.append(
            {
                "class_id": class_index,
                "class_name": names[class_index],
                "confidence": float(confidence),
                "xyxy": [float(value) for value in box],
            }
        )
    return records


def is_webcam_source(source: str) -> bool:
    """Return whether source identifies a local webcam index."""
    return source.isdigit()


def run_webcam(model: Any, args: argparse.Namespace) -> int:
    """Capture frames, run detection, and display results until q or Esc."""
    try:
        import cv2
    except ImportError:
        print(
            "Missing webcam dependency. Install it with: "
            "python -m pip install opencv-python",
            file=sys.stderr,
        )
        return 2

    camera_index = int(args.source)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"Could not open webcam {camera_index}.", file=sys.stderr)
        return 1

    print("Live inference started. Press q or Esc in the video window to stop.")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                print("Could not read a frame from the webcam.", file=sys.stderr)
                return 1

            predict_options: dict[str, Any] = {
                "source": frame,
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "verbose": False,
            }
            if args.device is not None:
                predict_options["device"] = args.device

            result = model.predict(**predict_options)[0]
            annotated_frame = result.plot()
            cv2.imshow("YOLO11 Live Inference", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except Exception as exc:
        print(f"Webcam inference failed: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()

    if not args.model.is_file():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 2
    if not 0.0 <= args.conf <= 1.0:
        print("--conf must be between 0 and 1.", file=sys.stderr)
        return 2
    if not 0.0 <= args.iou <= 1.0:
        print("--iou must be between 0 and 1.", file=sys.stderr)
        return 2
    if args.imgsz <= 0:
        print("--imgsz must be greater than 0.", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Missing dependency. Install it with: python -m pip install ultralytics",
            file=sys.stderr,
        )
        return 2

    model = YOLO(str(args.model))
    if is_webcam_source(args.source):
        return run_webcam(model, args)

    predict_options: dict[str, Any] = {
        "source": args.source,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "save": not args.no_save,
        "show": args.show,
        "project": str(args.output),
        "name": "predict",
        "exist_ok": True,
        "verbose": True,
    }
    if args.device is not None:
        predict_options["device"] = args.device

    try:
        results = model.predict(**predict_options)
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    if args.save_json:
        args.output.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "source": str(result.path),
                "detections": detection_records(result),
            }
            for result in results
        ]
        json_path = args.output / "detections.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Detections written to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
