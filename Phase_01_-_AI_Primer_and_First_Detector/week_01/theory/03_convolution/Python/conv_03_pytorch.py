"""
conv_03_pytorch.py
The same convolution demo as conv_03.cpp / conv_03_non_pytorch.py, with real PyTorch.
The 7 nested loops become one call: F.conv2d / nn.Conv2d.

Install:  pip install torch
Run:      python conv_03_pytorch.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class Lcg:
    """Same tiny generator as the from-scratch versions, so all three print the same numbers."""

    def __init__(self, seed):
        self.state = seed

    def next(self):  # uniform in [-0.5, 0.5)
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / 16777216.0 - 0.5

    def tensor(self, *shape):
        n = torch.Size(shape).numel()
        return torch.tensor([self.next() for _ in range(n)], dtype=torch.float32).view(*shape)


def print_channel(title, t):
    """t has shape (N=1, C=1, H, W)"""
    print(f"  {title}  ({t.shape[2]} x {t.shape[3]})")
    for row in t[0, 0]:
        print("   " + "".join(f"{v:3.0f}" for v in row))


def main():
    # (a) One kernel sliding over a 6x6 image: dark on the left, bright on the right.
    #     PyTorch wants a batch dimension: image is (N, C, H, W), kernel is (C_out, C_in, k, k).
    print("(a) Slide a 3x3 kernel over a 6x6 image")
    image = torch.zeros(1, 1, 6, 6)
    image[..., 3:] = 1.0
    print_channel("image", image)

    vertical_edge = torch.tensor([[-1.0, 0.0, 1.0]] * 3).view(1, 1, 3, 3)
    horizontal_edge = vertical_edge.transpose(2, 3).contiguous()

    window = image[0, 0, 0:3, 1:4]
    print(f"  window at (row 0, col 1) * kernel, summed = {(window * vertical_edge[0, 0]).sum():.0f}"
          "  -> output[0][1]")
    print_channel("output of the vertical-edge kernel: non-zero exactly where dark meets bright",
                  F.conv2d(image, vertical_edge))
    print_channel("output of the horizontal-edge kernel: this image has none",
                  F.conv2d(image, horizontal_edge))
    print()

    # (b) kernel, stride, padding decide the output size
    print("(b) Output size: H_out = (H + 2*pad - k) / stride + 1")
    print("    H    k  stride  pad   formula   actual")
    for stride, pad in [(1, 0), (1, 1), (2, 0), (2, 1)]:
        formula = (6 + 2 * pad - 3) // stride + 1
        actual = F.conv2d(image, vertical_edge, stride=stride, padding=pad).shape[2]
        print(f"  {6:3d}  {3:3d}  {stride:6d}  {pad:3d}  {formula:8d}  {actual:7d}")
    for k, stride, pad, note in [(3, 2, 1, "stride 2 halves the image"),
                                 (7, 2, 3, "first conv of ResNet")]:
        formula = (224 + 2 * pad - k) // stride + 1
        actual = nn.Conv2d(3, 8, k, stride=stride, padding=pad)(torch.zeros(1, 3, 224, 224)).shape[2]
        print(f"  {224:3d}  {k:3d}  {stride:6d}  {pad:3d}  {formula:8d}  {actual:7d}   ({note})")
    print()

    # (c) A real-sized layer. nn.Conv2d is a layer object that OWNS its weight and bias.
    print("(c) Conv 3 -> 16 channels, kernel 3x3, stride 1, pad 1, on a 3x32x32 image")
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    # Overwrite PyTorch's random init with the same numbers the from-scratch versions use
    rng = Lcg(2024)
    rgb = rng.tensor(1, 3, 32, 32)
    with torch.no_grad():
        conv.weight.copy_(rng.tensor(16, 3, 3, 3))
        conv.bias.copy_(rng.tensor(16))
        out = conv(rgb)

    print(f"  weight shape = {tuple(conv.weight.shape)}   bias shape = {tuple(conv.bias.shape)}")
    print(f"  params = 16*3*3*3 + 16 = {sum(p.numel() for p in conv.parameters())}")
    print(f"  input  shape = {tuple(rgb.shape)}   (with the batch dimension N=1 in front)")
    print(f"  output shape = {tuple(out.shape)}")
    print(f"  out[0][0][0] = {out[0, 0, 0, 0]:+.4f}   out[7][16][16] = {out[0, 7, 16, 16]:+.4f}   "
          f"out[15][31][31] = {out[0, 15, 31, 31]:+.4f}")


if __name__ == "__main__":
    main()
