"""
conv_03_non_pytorch.py
Convolution from scratch in pure Python (no NumPy, no PyTorch). Same structure as conv_03.cpp.

Convolution = slide a small window (the kernel) over the image. At every position,
multiply each pixel by the matching weight and add everything up -> ONE number.

  kernel  k : how wide the window is
  stride    : how many pixels the window moves per step
  padding   : how many zero pixels are added around the border
  H_out = (H + 2*pad - k) // stride + 1
  params = C_out * C_in * k * k + C_out

Run:  python conv_03_non_pytorch.py
"""


# ---------------------------------------------------------------------------
# 1. Tensor in CHW layout (one image, no batch dimension)
# ---------------------------------------------------------------------------
class Tensor:
    def __init__(self, C, H, W):
        self.C, self.H, self.W = C, H, W
        self.data = [0.0] * (C * H * W)

    def at(self, c, h, w):
        return self.data[(c * self.H + h) * self.W + w]

    def set(self, c, h, w, value):
        self.data[(c * self.H + h) * self.W + w] = value


class Conv:
    """The weights of one conv layer: C_out kernels of size C_in x k x k, plus one bias each."""

    def __init__(self, c_out, c_in, k):
        self.c_out, self.c_in, self.k = c_out, c_in, k
        self.weight = [0.0] * (c_out * c_in * k * k)  # layout (C_out, C_in, k, k)
        self.bias = [0.0] * c_out

    def w(self, co, ci, kh, kw):
        return self.weight[((co * self.c_in + ci) * self.k + kh) * self.k + kw]

    def num_params(self):
        return len(self.weight) + len(self.bias)


class Lcg:
    """Tiny deterministic random generator, so C++ and Python produce the same numbers."""

    def __init__(self, seed):
        self.state = seed

    def next(self):  # uniform in [-0.5, 0.5)
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / 16777216.0 - 0.5


# ---------------------------------------------------------------------------
# 2. The convolution itself
# ---------------------------------------------------------------------------
def conv_out_size(size, k, stride, pad):
    return (size + 2 * pad - k) // stride + 1


def conv2d(x, conv, stride, pad):
    """Note: like PyTorch, the kernel is NOT flipped (signal-processing books flip it)."""
    k = conv.k
    y = Tensor(conv.c_out, conv_out_size(x.H, k, stride, pad), conv_out_size(x.W, k, stride, pad))

    for co in range(conv.c_out):          # every output channel has its own kernel
        for oh in range(y.H):             # slide down
            for ow in range(y.W):         # slide right
                total = conv.bias[co]
                for ci in range(conv.c_in):       # the window covers ALL input channels
                    for kh in range(k):
                        for kw in range(k):
                            # which input pixel sits under this kernel cell
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if ih < 0 or ih >= x.H or iw < 0 or iw >= x.W:
                                continue          # in the padding: pixel is 0
                            total += x.at(ci, ih, iw) * conv.w(co, ci, kh, kw)
                y.set(co, oh, ow, total)
    return y


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def print_channel(title, t):
    print(f"  {title}  ({t.H} x {t.W})")
    for h in range(t.H):
        print("   " + "".join(f"{t.at(0, h, w):3.0f}" for w in range(t.W)))


def main():
    # (a) One kernel sliding over a 6x6 image: dark on the left, bright on the right
    print("(a) Slide a 3x3 kernel over a 6x6 image")
    image = Tensor(1, 6, 6)
    for h in range(6):
        for w in range(3, 6):
            image.set(0, h, w, 1.0)
    print_channel("image", image)

    vertical_edge = Conv(1, 1, 3)
    vertical_edge.weight = [-1, 0, 1,
                            -1, 0, 1,
                            -1, 0, 1]
    horizontal_edge = Conv(1, 1, 3)
    horizontal_edge.weight = [-1, -1, -1,
                               0,  0,  0,
                               1,  1,  1]

    # The arithmetic at ONE window position: output[0][1]
    print("  window at (row 0, col 1)   kernel       products")
    total = 0.0
    for kh in range(3):
        pixels = [image.at(0, kh, 1 + kw) for kw in range(3)]
        weights = [vertical_edge.w(0, 0, kh, kw) for kw in range(3)]
        products = [p * w + 0.0 for p, w in zip(pixels, weights)]  # + 0 turns "-0" into "0"
        total += sum(products)
        print("   " + "".join(f"{v:3.0f}" for v in pixels) + " " * 16
              + "".join(f"{v:3.0f}" for v in weights) + " " * 4
              + "".join(f"{v:3.0f}" for v in products))
    print(f"  sum of the 9 products + bias = {total:.0f}  -> output[0][1]")

    print_channel("output of the vertical-edge kernel: non-zero exactly where dark meets bright",
                  conv2d(image, vertical_edge, 1, 0))
    print_channel("output of the horizontal-edge kernel: this image has none",
                  conv2d(image, horizontal_edge, 1, 0))
    print()

    # (b) kernel, stride, padding decide the output size
    print("(b) Output size: H_out = (H + 2*pad - k) / stride + 1")
    print("    H    k  stride  pad   formula   actual")
    for stride, pad in [(1, 0), (1, 1), (2, 0), (2, 1)]:
        formula = conv_out_size(6, 3, stride, pad)
        actual = conv2d(image, vertical_edge, stride, pad).H
        print(f"  {6:3d}  {3:3d}  {stride:6d}  {pad:3d}  {formula:8d}  {actual:7d}")
    print(f"  {224:3d}  {3:3d}  {2:6d}  {1:3d}  {conv_out_size(224, 3, 2, 1):8d}"
          "        -   (stride 2 halves the image)")
    print(f"  {224:3d}  {7:3d}  {2:6d}  {3:3d}  {conv_out_size(224, 7, 2, 3):8d}"
          "        -   (first conv of ResNet)\n")

    # (c) A real-sized layer: 3 input channels (RGB), 16 output channels, 3x3 kernel
    print("(c) Conv 3 -> 16 channels, kernel 3x3, stride 1, pad 1, on a 3x32x32 image")
    rng = Lcg(2024)
    rgb = Tensor(3, 32, 32)
    rgb.data = [rng.next() for _ in rgb.data]
    conv = Conv(16, 3, 3)
    conv.weight = [rng.next() for _ in conv.weight]
    conv.bias = [rng.next() for _ in conv.bias]

    out = conv2d(rgb, conv, 1, 1)
    print(f"  weight shape = ({conv.c_out}, {conv.c_in}, {conv.k}, {conv.k})   "
          f"bias shape = ({len(conv.bias)})")
    print(f"  params = 16*3*3*3 + 16 = {conv.num_params()}")
    print(f"  input  shape = ({rgb.C}, {rgb.H}, {rgb.W})")
    print(f"  output shape = ({out.C}, {out.H}, {out.W})")
    print(f"  multiply-adds = 16*32*32 outputs x 3*3*3 each = "
          f"{len(out.data) * conv.c_in * conv.k * conv.k}")
    print(f"  out[0][0][0] = {out.at(0, 0, 0):+.4f}   out[7][16][16] = {out.at(7, 16, 16):+.4f}   "
          f"out[15][31][31] = {out.at(15, 31, 31):+.4f}")


if __name__ == "__main__":
    main()
