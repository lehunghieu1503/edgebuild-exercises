#!/usr/bin/env python3
"""Chạy sample.onnx trên từng thiết bị OpenVINO (CPU, GPU, NPU), in top-1 và tên thiết bị."""
import sys
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    core = ov.Core()
    onnx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "sample.onnx"
    model = core.read_model(onnx_path)
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)

    print(f"openvino {ov.__version__}")
    ref = None
    failed = False
    for dev in core.available_devices:
        name = core.get_property(dev, "FULL_DEVICE_NAME")
        try:
            logits = core.compile_model(model, dev)(x)[0][0]
        except Exception as e:  # thiết bị có mặt nhưng không chạy được model
            print(f"{dev}: FAIL ({name}): {str(e).splitlines()[0]}")
            failed = True
            continue
        if ref is None:
            ref = logits
        print(f"{dev}: top-1 {int(logits.argmax())}, max |diff vs {core.available_devices[0]}| = "
              f"{float(np.abs(logits - ref).max()):.2e} ({name})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
