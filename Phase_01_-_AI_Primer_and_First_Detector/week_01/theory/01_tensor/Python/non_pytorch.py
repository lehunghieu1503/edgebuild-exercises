"""
tensor_basics_scratch.py
A tensor built from scratch in pure Python (no NumPy, no PyTorch).
Same idea as the C++ version: a flat 1D buffer + metadata (shape, strides, dtype, device).

Run:  python tensor_basics_scratch.py
"""

from array import array
from math import prod


class Tensor:
    def __init__(self, shape, dtype="float32", device="cpu"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device  # only a label here; data always stays in RAM

        # The "float buffer[N]": array('f') stores real 32-bit floats, like C++ float
        self.data = array("f", [0.0]) * prod(self.shape)

        # Row-major strides: last dimension is contiguous (stride 1)
        # shape (1, 3, 224, 224) -> strides (150528, 50176, 224, 1)
        strides = []
        stride = 1
        for dim in reversed(self.shape):
            strides.append(stride)
            stride *= dim
        self.strides = tuple(reversed(strides))

    def numel(self):
        return len(self.data)

    def offset(self, idx):
        """Turn a multi-dimensional index into a position in the 1D buffer."""
        if len(idx) != len(self.shape):
            raise IndexError("rank mismatch")
        off = 0
        for i, s, st in zip(idx, self.shape, self.strides):
            if not 0 <= i < s:
                raise IndexError("index out of range")
            off += i * st
        return off

    # Lets you write t[0, 1, 1, 2] and t[0, 1, 1, 2] = 0.5
    def __getitem__(self, idx):
        return self.data[self.offset(idx)]

    def __setitem__(self, idx, value):
        self.data[self.offset(idx)] = value

    def print_meta(self, name):
        print(f"{name}: shape={self.shape}  strides={self.strides}  "
              f"dtype={self.dtype}  device={self.device}  numel={self.numel()}")


def hwc_to_nchw(hwc, H, W, C):
    """
    OpenCV-style image (uint8, interleaved RGBRGB...) -> NCHW float tensor.
      HWC  position of (h, w, c):  (h * W + w) * C + c
      NCHW position of (0, c, h, w): ((0 * C + c) * H + h) * W + w
    """
    t = Tensor((1, C, H, W))
    for h in range(H):
        for w in range(W):
            for c in range(C):
                px = hwc[(h * W + w) * C + c]
                t[0, c, h, w] = px / 255.0  # normalize to [0, 1]
    return t


def main():
    # (a) The tensor from the slide: one 224x224 RGB image
    img = Tensor((1, 3, 224, 224))
    img.print_meta("img")
    print(f"  bytes in buffer = {img.numel() * img.data.itemsize}\n")

    # (b) A tiny 2x3 image so we can see the memory layout
    H, W, C = 2, 3, 3
    hwc = bytes([
        10, 110, 210,  11, 111, 211,  12, 112, 212,   # row 0
        13, 113, 213,  14, 114, 214,  15, 115, 215,   # row 1
    ])  # R = 10..15, G = 110..115, B = 210..215

    print("HWC buffer (interleaved, like OpenCV):")
    print(" ", *hwc, "\n")

    x = hwc_to_nchw(hwc, H, W, C)
    x.print_meta("x")
    print("NCHW buffer (planar: R plane, then G plane, then B plane):")
    print(" ", *(round(v * 255) for v in x.data), "\n")

    # (c) Same element, two views: index math vs. raw buffer position
    off = x.offset((0, 1, 1, 2))  # green channel, row 1, col 2
    print(f"x[0][G][1][2] lives at buffer[{off}] = {round(x.data[off] * 255)}  (expected 115)")


if __name__ == "__main__":
    main()