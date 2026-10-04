"""
train_05_non_pytorch.py
Train and inference from scratch in pure Python (no NumPy, no PyTorch).
Same structure as train_05.cpp.

  Train     : show labeled examples, measure how wrong the network is (the LOSS),
              then nudge every weight so the loss goes down. Which way to nudge is
              the GRADIENT, computed by BACKPROPAGATION. Repeat many times.
  Inference : the weights are frozen. Just run the input through the network.

This file trains the 9 -> 6 -> 2 "vertical or horizontal line" network of
02_layer_weight from random weights, with no library at all.

Run:  python train_05_non_pytorch.py
"""

from math import exp, log


class Lcg:
    """Tiny deterministic random generator, so C++ and Python produce the same numbers."""

    def __init__(self, seed):
        self.state = seed

    def next(self):  # uniform in [-0.5, 0.5)
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / 16777216.0 - 0.5


# ---------------------------------------------------------------------------
# 1. The simplest possible training: ONE weight
# ---------------------------------------------------------------------------
# Model: y = w * x.  The data was made with w = 3, the model has to find that out.
XS = [1.0, 2.0, 3.0, 4.0]
YS = [3.0, 6.0, 9.0, 12.0]


def loss_one_weight(w):
    """Loss = how wrong we are = mean of (prediction - target)^2"""
    return sum((w * x - y) ** 2 for x, y in zip(XS, YS)) / len(XS)


def grad_one_weight(w):
    """
    Gradient = d(loss)/d(w) = how much the loss changes when w grows a little.
    Here calculus gives it directly: mean of 2 * (w*x - y) * x
    """
    return sum(2.0 * (w * x - y) * x for x, y in zip(XS, YS)) / len(XS)


# ---------------------------------------------------------------------------
# 2. The network: Linear(9 -> 6) -> ReLU -> Linear(6 -> 2)
# ---------------------------------------------------------------------------
IN, HID, OUT = 9, 6, 2


class Net:
    def __init__(self):
        self.w1, self.b1 = [0.0] * (HID * IN), [0.0] * HID
        self.w2, self.b2 = [0.0] * (OUT * HID), [0.0] * OUT

    def num_params(self):
        return len(self.w1) + len(self.b1) + len(self.w2) + len(self.b2)


def forward(net, x):
    """
    Forward pass: returns (hidden, logits). It never changes the weights.
    Inference only needs `logits`; training must also keep `hidden`,
    because the backward pass reads it.
    """
    hidden = []
    for i in range(HID):
        total = net.b1[i]
        for j in range(IN):
            total += net.w1[i * IN + j] * x[j]
        hidden.append(max(total, 0.0))  # ReLU
    logits = []
    for o in range(OUT):
        total = net.b2[o]
        for i in range(HID):
            total += net.w2[o * HID + i] * hidden[i]
        logits.append(total)
    return hidden, logits


def softmax(logits):
    """Softmax turns the scores into probabilities that add up to 1."""
    biggest = max(logits)
    p = [exp(v - biggest) for v in logits]
    total = sum(p)
    return [v / total for v in p]


# ---------------------------------------------------------------------------
# 3. One training step = forward, loss, backward (gradients), update
# ---------------------------------------------------------------------------
def train_step(net, batch, lr):
    """This is the ONLY function that modifies the weights."""
    grad = Net()  # one gradient per weight, same shapes as the weights, starts at 0
    loss = 0.0
    n = len(batch)

    for image, label in batch:
        # forward
        hidden, logits = forward(net, image)
        p = softmax(logits)

        # loss (cross-entropy): 0 when the correct class gets probability 1, large when near 0
        loss += -log(p[label]) / n

        # backward = backpropagation: walk the layers in REVERSE order and ask each one
        # "how does the loss change if your output changes?" (the chain rule)
        d_logits = list(p)  # d(loss)/d(logits) = p - one_hot(label)
        d_logits[label] -= 1.0
        d_logits = [v / n for v in d_logits]

        d_hidden = [0.0] * HID
        for o in range(OUT):  # through Linear(6 -> 2)
            grad.b2[o] += d_logits[o]
            for i in range(HID):
                grad.w2[o * HID + i] += d_logits[o] * hidden[i]
                d_hidden[i] += net.w2[o * HID + i] * d_logits[o]
        for i in range(HID):  # through ReLU, then Linear(9 -> 6)
            if hidden[i] <= 0.0:
                continue  # ReLU was off: nothing flows back
            grad.b1[i] += d_hidden[i]
            for j in range(IN):
                grad.w1[i * IN + j] += d_hidden[i] * image[j]

    # update: move every weight a small step AGAINST its gradient
    for w, g in ((net.w1, grad.w1), (net.b1, grad.b1), (net.w2, grad.w2), (net.b2, grad.b2)):
        for i in range(len(w)):
            w[i] -= lr * g[i]
    return loss


# ---------------------------------------------------------------------------
# 4. Data and evaluation
# ---------------------------------------------------------------------------
def line_images():
    samples = []
    for k in range(3):
        v, h = [0.0] * 9, [0.0] * 9
        for j in range(3):
            v[j * 3 + k] = 1.0  # column k is lit
            h[k * 3 + j] = 1.0  # row k is lit
        samples += [(v, 0), (h, 1)]  # 0 = vertical, 1 = horizontal
    return samples


def count_correct(net, samples):
    correct = 0
    for image, label in samples:
        _, logits = forward(net, image)
        correct += int(logits[1] > logits[0]) == label
    return correct


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------
def main():
    # (a) One weight: loss, gradient, update
    print("(a) Training ONE weight: find w so that w * x matches y = 3 * x")
    w, h = 0.0, 0.01
    numeric = (loss_one_weight(w + h) - loss_one_weight(w - h)) / (2 * h)
    print(f"  gradient at w=0: by formula = {grad_one_weight(w):.2f}, "
          f"by nudging w a little = {numeric:.2f}")
    print("  step      w     loss  gradient")
    for step in range(7):
        grad = grad_one_weight(w)
        print(f"  {step:4d}  {w:.4f}  {loss_one_weight(w):7.4f}  {grad:8.4f}")
        w -= 0.05 * grad  # learning rate 0.05: negative gradient -> w must grow
    print("  w walked from 0 to 3 by itself: that is \"learning\"\n")

    # (b) The same loop on a network: 74 weights, gradients from backpropagation
    print("(b) TRAIN: Linear(9->6) -> ReLU -> Linear(6->2), starting from random weights")
    samples = line_images()
    net = Net()
    rng = Lcg(5)
    for v in (net.w1, net.b1, net.w2, net.b2):
        v[:] = [rng.next() for _ in v]

    steps, lr = 200, 0.5
    print(f"  before training: {count_correct(net, samples)}/{len(samples)} correct")
    for step in range(steps + 1):
        correct = count_correct(net, samples)
        loss = train_step(net, samples, lr)
        if step % 40 == 0:
            print(f"  step {step:3d}  loss = {loss:.4f}  correct = {correct}/{len(samples)}")
    print()

    # (c) Inference: weights frozen, forward only
    print("(c) INFERENCE: weights frozen, forward pass only")
    for i in range(2):
        image, label = samples[i]
        _, logits = forward(net, image)
        p = softmax(logits)
        print(f"  image {i} ({'vertical' if label == 0 else 'horizontal'}): "
              f"logits = [{logits[0]:+.3f} {logits[1]:+.3f}]  "
              f"probabilities = [{p[0]:.3f} {p[1]:.3f}]")
    print()

    print("(d) What each job needed")
    passes = (steps + 1) * len(samples)
    print(f"  train    : {steps + 1} steps x {len(samples)} images = {passes} forward + "
          f"{passes} backward passes, {net.num_params()} weights changed {steps + 1} times")
    print(f"             memory: {net.num_params()} weights + {net.num_params()} gradients "
          "+ activations kept for backward")
    print("  inference: 1 forward pass per image, 0 weights changed")
    print(f"             memory: {net.num_params()} weights")


if __name__ == "__main__":
    main()
