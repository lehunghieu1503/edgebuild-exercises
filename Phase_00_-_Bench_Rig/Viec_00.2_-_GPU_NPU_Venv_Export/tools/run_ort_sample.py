#!/usr/bin/env python3
"""Export resnet18 ra ONNX, chạy một ảnh bằng ONNX Runtime, in top-1."""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path, default=ROOT / "data" / "dog.jpg")
    ap.add_argument("--onnx", type=Path, default=ROOT / "build" / "sample.onnx")
    ap.add_argument("--reuse", action="store_true", help="bỏ qua export nếu file onnx đã có")
    args = ap.parse_args()

    if not args.image.exists():
        args.image.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_URL, args.image)

    args.onnx.parent.mkdir(parents=True, exist_ok=True)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()
    x = weights.transforms()(Image.open(args.image).convert("RGB")).unsqueeze(0)

    if not (args.reuse and args.onnx.exists()):
        torch.onnx.export(model, (x,), str(args.onnx), input_names=["input"], output_names=["logits"])
    onnx.checker.check_model(onnx.load(str(args.onnx)))

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(None, {"input": x.numpy()})[0][0]
    with torch.no_grad():
        ref_logits = model(x)[0].numpy()

    ort_top1 = int(ort_logits.argmax())
    ref_top1 = int(ref_logits.argmax())
    prob = np.exp(ort_logits - ort_logits.max())
    prob /= prob.sum()
    max_diff = float(np.abs(ort_logits - ref_logits).max())

    print(f"onnx: {args.onnx} (torch {torch.__version__}, onnxruntime {ort.__version__})")
    print(f"top-1: {ort_top1} {weights.meta['categories'][ort_top1]!r} p={prob[ort_top1]:.4f}")
    print(f"torch top-1: {ref_top1}, max |logit diff| = {max_diff:.2e}")
    return 0 if ort_top1 == ref_top1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
