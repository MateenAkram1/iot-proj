"""Run one-shot CNN+LSTM inference from a single Firebase-style IMU sample."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from inference import (  # noqa: E402
    extract_six_axis,
    model_indicates_fall,
    predict_fall_from_flat,
)

# Tile mode only: Firebase-style keys (same order as CSV columns).
DEFAULT_VITALS = {
    "accelX": 6.99,
    "accelY": -0.57,
    "accelZ": -7.28,
    "gyroX": -2.75,
    "gyroY": -3.23,
    "gyroZ": 2.62,
    "fallDetected": False,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Run fall model on one IMU sample.")
    p.add_argument("--accelX", type=float, default=DEFAULT_VITALS["accelX"])
    p.add_argument("--accelY", type=float, default=DEFAULT_VITALS["accelY"])
    p.add_argument("--accelZ", type=float, default=DEFAULT_VITALS["accelZ"])
    p.add_argument("--gyroX", type=float, default=DEFAULT_VITALS["gyroX"])
    p.add_argument("--gyroY", type=float, default=DEFAULT_VITALS["gyroY"])
    p.add_argument("--gyroZ", type=float, default=DEFAULT_VITALS["gyroZ"])
    args = p.parse_args()

    vitals = {
        "accelX": args.accelX,
        "accelY": args.accelY,
        "accelZ": args.accelZ,
        "gyroX": args.gyroX,
        "gyroY": args.gyroY,
        "gyroZ": args.gyroZ,
        "fallDetected": False,
    }
    print("Input vitals (IMU):")
    print(
        json.dumps(
            {k: vitals[k] for k in ("accelX", "accelY", "accelZ", "gyroX", "gyroY", "gyroZ")},
            indent=2,
        )
    )
    row = extract_six_axis(vitals)
    print("\n6-vector order [accelX, accelY, accelZ, gyroX, gyroY, gyroZ]:")
    print(row.tolist())

    preds = predict_fall_from_flat(row)
    fall = model_indicates_fall(preds)

    print("\nModel output:")
    print(json.dumps(preds, indent=2))
    print(f"\nRule (fall -> PATCH fallDetected): model_indicates_fall = {fall}")


if __name__ == "__main__":
    main()
