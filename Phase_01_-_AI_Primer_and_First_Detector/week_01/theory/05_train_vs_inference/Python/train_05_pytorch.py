"""
train_05_pytorch.py
The same demo as train_05.cpp / train_05_non_pytorch.py, with real PyTorch.
The backward pass you wrote by hand there is ONE call here: loss.backward().

  Train     : forward -> loss -> zero_grad -> backward -> step, repeated
  Inference : model.eval() + torch.no_grad(), forward only

Install:  pip install torch
Run:      python train_05_pytorch.py
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

    def tensor(self, *shape):
        n = torch.Size(shape).numel()
        return torch.tensor([self.next() for _ in range(n)], dtype=torch.float32).view(*shape)


def line_images():
    images, labels = [], []
    for k in range(3):
        v = torch.zeros(3, 3)
        v[:, k] = 1.0  # column k is lit
        h = torch.zeros(3, 3)
        h[k, :] = 1.0  # row k is lit
        images += [v.flatten(), h.flatten()]
        labels += [0, 1]  # 0 = vertical, 1 = horizontal
    return torch.stack(images), torch.tensor(labels)


def count_correct(model, images, labels):
    with torch.no_grad():
        return int((model(images).argmax(dim=1) == labels).sum())


def main():
    # (a) One weight. requires_grad=True tells PyTorch: "record what happens to w,
    #     I will ask for d(loss)/d(w)". That recording + backward() is called autograd.
    print("(a) Training ONE weight: find w so that w * x matches y = 3 * x")
    xs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    ys = 3.0 * xs
    w = torch.tensor(0.0, requires_grad=True)

    print("  step      w     loss  gradient")
    for step in range(7):
        loss = ((w * xs - ys) ** 2).mean()
        loss.backward()                 # fills w.grad
        print(f"  {step:4d}  {w.item():.4f}  {loss.item():7.4f}  {w.grad.item():8.4f}")
        with torch.no_grad():           # the update itself must not be recorded
            w -= 0.05 * w.grad
        w.grad.zero_()                  # gradients accumulate: clear them every step
    print("  w walked from 0 to 3 by itself: that is \"learning\"\n")

    # (b) The same network, the same starting weights, the same data
    print("(b) TRAIN: Linear(9->6) -> ReLU -> Linear(6->2), starting from random weights")
    model = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(9, 6)),
        ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(6, 2)),
    ]))
    rng = Lcg(5)
    with torch.no_grad():
        for layer in (model.fc1, model.fc2):
            layer.weight.copy_(rng.tensor(*layer.weight.shape))
            layer.bias.copy_(rng.tensor(*layer.bias.shape))

    images, labels = line_images()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    cross_entropy = nn.CrossEntropyLoss()   # softmax + (-log of the correct class), averaged

    model.train()                           # training mode
    steps = 200
    print(f"  before training: {count_correct(model, images, labels)}/{len(labels)} correct")
    for step in range(steps + 1):
        correct = count_correct(model, images, labels)

        logits = model(images)              # forward
        loss = cross_entropy(logits, labels)  # measure how wrong
        optimizer.zero_grad()               # clear old gradients
        loss.backward()                     # backpropagation: gradient of every weight
        optimizer.step()                    # w -= lr * gradient

        if step % 40 == 0:
            print(f"  step {step:3d}  loss = {loss.item():.4f}  correct = {correct}/{len(labels)}")
    print()

    # (c) Inference. Two switches, both matter:
    #     model.eval()    -> layers like BatchNorm and Dropout switch to their inference behavior
    #     torch.no_grad() -> stop recording for backward: less memory, less time
    print("(c) INFERENCE: weights frozen, forward pass only")
    model.eval()
    with torch.no_grad():
        logits = model(images[:2])
        probs = logits.softmax(dim=1)
    for i in range(2):
        print(f"  image {i} ({'vertical' if labels[i] == 0 else 'horizontal'}): "
              f"logits = [{logits[i, 0]:+.3f} {logits[i, 1]:+.3f}]  "
              f"probabilities = [{probs[i, 0]:.3f} {probs[i, 1]:.3f}]")
    print()

    # (d) What "recording for backward" looks like
    print("(d) What each job keeps")
    recorded = model(images[:1])
    with torch.no_grad():
        plain = model(images[:1])
    print(f"  normal forward : requires_grad={recorded.requires_grad}  "
          f"grad_fn={type(recorded.grad_fn).__name__}   <- remembers how it was computed")
    print(f"  under no_grad(): requires_grad={plain.requires_grad}  grad_fn={plain.grad_fn}")
    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
    print(f"  train     memory: {n_params} weights + {n_grads} gradients + recorded activations")
    print(f"  inference memory: {n_params} weights")


if __name__ == "__main__":
    main()
