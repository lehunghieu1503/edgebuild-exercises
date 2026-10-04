"""
tensor_basics_torch.py
The same ideas as the from-scratch version, but using real PyTorch tensors.
PyTorch's torch.Tensor is exactly "flat buffer + shape + strides + dtype + device".

Install:  pip install torch
Run:      python tensor_basics_torch.py
"""

import torch


def print_meta(name, t):
    print(f"{name}: shape={tuple(t.shape)}  strides={t.stride()}  "
          f"dtype={t.dtype}  device={t.device}  numel={t.numel()}  "
          f"contiguous={t.is_contiguous()}")


def main():
    # (a) The tensor from the slide: one 224x224 RGB image
    img = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    print_meta("img", img)
    print(f"  bytes in buffer = {img.numel() * img.element_size()}\n")

    # (b) A tiny 2x3 image in HWC layout, uint8, like what OpenCV gives you
    hwc = torch.tensor([
        [[10, 110, 210], [11, 111, 211], [12, 112, 212]],   # row 0
        [[13, 113, 213], [14, 114, 214], [15, 115, 215]],   # row 1
    ], dtype=torch.uint8)  # shape (H=2, W=3, C=3)
    print_meta("hwc", hwc)
    print("  underlying buffer:", hwc.flatten().tolist(), "\n")

    # (c) HWC -> CHW with permute. NO data is copied yet:
    #     PyTorch only rearranges the strides, so the tensor is "non-contiguous".
    chw_view = hwc.permute(2, 0, 1)  # (C, H, W)
    print_meta("chw_view", chw_view)
    print("  same memory as hwc?", chw_view.data_ptr() == hwc.data_ptr(), "\n")

    # .contiguous() actually copies into planar order: RRRRRR GGGGGG BBBBBB
    # This copy is what our hand-written hwc_to_nchw loop did.
    chw = chw_view.contiguous()
    print_meta("chw", chw)
    print("  underlying buffer:", chw.flatten().tolist(), "\n")

    # (d) Add batch dimension, convert dtype, normalize -> network input
    x = chw.unsqueeze(0).float() / 255.0  # (1, 3, 2, 3), float32 in [0, 1]
    print_meta("x", x)

    # (e) Same element, two views: indexing vs. raw buffer position
    n, c, h, w = 0, 1, 1, 2  # green channel, row 1, col 2
    s = x.stride()
    off = n * s[0] + c * s[1] + h * s[2] + w * s[3]
    print(f"x[0][G][1][2] = {round(x[n, c, h, w].item() * 255)}, "
          f"lives at buffer[{off}] = {round(x.flatten()[off].item() * 255)}  (expected 115)\n")

    # (f) view/reshape only changes metadata, not data (for contiguous tensors)
    flat = x.view(-1)
    print_meta("flat", flat)
    print("  same memory as x?", flat.data_ptr() == x.data_ptr(), "\n")

    # (g) Device: move the buffer to the GPU if one exists
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_dev = x.to(device)
    print_meta("x_dev", x_dev)


if __name__ == "__main__":
    main()