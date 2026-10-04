"""
layers_04_pytorch.py
The same demo as layers_04.cpp / layers_04_non_pytorch.py, with real PyTorch layers:
nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d, nn.Linear, stacked with nn.Sequential.

Install:  pip install torch
Run:      python layers_04_pytorch.py
"""

from collections import OrderedDict

import torch
from torch import nn


class Lcg:
    """Same tiny generator as the from-scratch versions, so all three print the same numbers."""

    def __init__(self, seed):
        self.state = seed

    def next(self):  # uniform in [-0.5, 0.5)
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / 16777216.0 - 0.5

    def tensor(self, *shape, offset=0.0):
        n = torch.Size(shape).numel()
        values = [offset + self.next() for _ in range(n)]
        return torch.tensor(values, dtype=torch.float32).view(*shape)


def print_values(name, t, fmt="+.3f"):
    print(f"  {name} [ " + " ".join(f"{v:{fmt}}" for v in t.flatten().tolist()) + " ]")


def print_channel(title, t):
    """t has shape (N=1, C=1, H, W)"""
    print(f"  {title}  ({t.shape[2]} x {t.shape[3]})")
    for row in t[0, 0]:
        print("   " + "".join(f"{v:3.0f}" for v in row))


def main():
    # (a) ReLU
    print("(a) ReLU: negatives become 0")
    r = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
    print_values("in ", r, "+.1f")
    print_values("out", nn.ReLU()(r), "+.1f")
    print()

    # (b) MaxPool
    print("(b) MaxPool 2x2: keep the largest value of every 2x2 block")
    p = torch.tensor([[1.0, 3, 2, 0],
                      [4, 2, 1, 1],
                      [0, 1, 5, 6],
                      [2, 0, 7, 8]]).view(1, 1, 4, 4)
    print_channel("in ", p)
    print_channel("out", nn.MaxPool2d(kernel_size=2)(p))
    print("  16 numbers -> 4 numbers: the next layer has 4x less work\n")

    # (c) BatchNorm: two channels on wildly different scales.
    #     .eval() = inference mode: use the stored statistics (running_mean / running_var).
    print("(c) BatchNorm: put every channel on the same scale")
    b = torch.tensor([1000.0, 2000, 3000, 4000, 1, 2, 3, 4]).view(1, 2, 2, 2)
    bn = nn.BatchNorm2d(2).eval()
    bn.running_mean.copy_(torch.tensor([2500.0, 2.5]))
    bn.running_var.copy_(torch.tensor([1250000.0, 1.25]))
    with torch.no_grad():
        normed = bn(b)
    for c in range(2):
        before = " ".join(f"{v:.0f}" for v in b[0, c].flatten())
        after = " ".join(f"{v:+.3f}" for v in normed[0, c].flatten())
        print(f"  channel {c}  before [ {before} ]  after [ {after} ]")
    print(f"  parameters (learned): {[name for name, _ in bn.named_parameters()]}"
          "   <- gamma, beta")
    print(f"  buffers (statistics): {[name for name, _ in bn.named_buffers()]}\n")

    # (d) A CNN: two "conv -> BatchNorm -> ReLU -> pool" blocks, then Linear
    print("(d) A CNN: 1x8x8 image -> 3 class scores")
    model = nn.Sequential(OrderedDict([
        ("conv1", nn.Conv2d(1, 4, kernel_size=3, padding=1)),
        ("bn1", nn.BatchNorm2d(4)),
        ("relu1", nn.ReLU()),
        ("pool1", nn.MaxPool2d(2)),
        ("conv2", nn.Conv2d(4, 8, kernel_size=3, padding=1)),
        ("bn2", nn.BatchNorm2d(8)),
        ("relu2", nn.ReLU()),
        ("pool2", nn.MaxPool2d(2)),
        ("flatten", nn.Flatten()),
        ("fc", nn.Linear(8 * 2 * 2, 3)),
    ])).eval()

    # Overwrite PyTorch's random init with the same numbers the from-scratch versions use
    rng = Lcg(7)
    image = rng.tensor(1, 1, 8, 8)
    with torch.no_grad():
        for conv, bn in ((model.conv1, model.bn1), (model.conv2, model.bn2)):
            conv.weight.copy_(rng.tensor(*conv.weight.shape))
            conv.bias.copy_(rng.tensor(*conv.bias.shape))
            bn.weight.copy_(rng.tensor(*bn.weight.shape, offset=1.0))   # gamma
            bn.bias.copy_(rng.tensor(*bn.bias.shape))                   # beta
            bn.running_mean.copy_(rng.tensor(*bn.running_mean.shape))
            bn.running_var.copy_(rng.tensor(*bn.running_var.shape, offset=1.0))
        model.fc.weight.copy_(rng.tensor(*model.fc.weight.shape))
        model.fc.bias.copy_(rng.tensor(*model.fc.bias.shape))

    print("  layer            output shape     params")
    x = image
    print(f"  {'input':<16} {str(tuple(x.shape)):<16} {0:6d}")
    with torch.no_grad():
        for name, layer in model.named_children():
            x = layer(x)   # this loop is all that nn.Sequential does
            params = sum(p.numel() for p in layer.parameters())
            print(f"  {name:<16} {str(tuple(x.shape)):<16} {params:6d}")
    print(f"  total params = {sum(p.numel() for p in model.parameters())}")
    print_values("scores", x, "+.4f")


if __name__ == "__main__":
    main()
