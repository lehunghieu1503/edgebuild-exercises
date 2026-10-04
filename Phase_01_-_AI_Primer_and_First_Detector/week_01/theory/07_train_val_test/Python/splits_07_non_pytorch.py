"""
splits_07_non_pytorch.py
Train / val / test and overfitting, in pure Python (no NumPy, no PyTorch).
Same structure as splits_07.cpp.

  train : used to adjust the weights
  val   : used to choose the configuration (here: how many epochs to train)
  test  : used ONCE at the very end, to report the result

Overfitting: the train loss keeps going down while the val loss goes UP.
The network is memorizing the training images instead of learning the rule.

The data: noisy 3x3 "vertical or horizontal line" images, and 1 label in 8 is
wrong on purpose (like real annotation mistakes). The network is the one from
05_train_vs_inference, with a bigger hidden layer so it has room to memorize.

Run:  python splits_07_non_pytorch.py     (pure Python loops: takes about a second)
"""

from math import exp, log


class Lcg:
    """Tiny deterministic random generator, so C++ and Python produce the same numbers."""

    def __init__(self, seed):
        self.state = seed

    def next_bits(self):  # 24 random bits
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state >> 8

    def next(self):  # uniform in [-0.5, 0.5)
        return self.next_bits() / 16777216.0 - 0.5

    def next_int(self, n):  # integer in [0, n)
        return self.next_bits() % n


# ---------------------------------------------------------------------------
# 1. Data: make it, shuffle it, split it
# ---------------------------------------------------------------------------
IN, HID, OUT = 9, 24, 2


def make_dataset(rng, count):
    data = []
    for i in range(count):
        label = i % 2  # 0 = vertical, 1 = horizontal
        k = (i // 2) % 3  # which column / row is lit
        image = [0.0] * IN
        for j in range(3):
            image[j * 3 + k if label == 0 else k * 3 + j] = 1.0
        image = [pixel + rng.next() for pixel in image]  # noise: no two images are equal
        if rng.next_int(8) == 0:  # 1 label in 8 is wrong
            label = 1 - label
        data.append((image, label))
    return data


def shuffled_indices(rng, count):
    """
    Shuffle BEFORE splitting, otherwise the three sets could hold different kinds of images.
    Shuffle with a fixed seed, so the split is the same on every run.
    """
    idx = list(range(count))
    for i in range(count - 1, 0, -1):
        j = rng.next_int(i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


# ---------------------------------------------------------------------------
# 2. The network and its training step (explained in 05_train_vs_inference)
# ---------------------------------------------------------------------------
class Net:
    def __init__(self):
        self.w1, self.b1 = [0.0] * (HID * IN), [0.0] * HID
        self.w2, self.b2 = [0.0] * (OUT * HID), [0.0] * OUT

    def copy(self):
        other = Net()
        other.w1, other.b1 = list(self.w1), list(self.b1)
        other.w2, other.b2 = list(self.w2), list(self.b2)
        return other


def forward(net, x):
    hidden = []
    for i in range(HID):
        total = net.b1[i]
        for j in range(IN):
            total += net.w1[i * IN + j] * x[j]
        hidden.append(max(total, 0.0))
    logits = []
    for o in range(OUT):
        total = net.b2[o]
        for i in range(HID):
            total += net.w2[o * HID + i] * hidden[i]
        logits.append(total)
    return hidden, logits


def softmax(logits):
    biggest = max(logits)
    p = [exp(v - biggest) for v in logits]
    total = sum(p)
    return [v / total for v in p]


def evaluate(net, dataset):
    """Forward only: measures, never changes the weights. Used for val and test."""
    loss, correct = 0.0, 0
    for image, label in dataset:
        _, logits = forward(net, image)
        loss += -log(softmax(logits)[label])
        correct += int(logits[1] > logits[0]) == label
    return loss / len(dataset), 100.0 * correct / len(dataset)  # accuracy in percent


def train_epoch(net, dataset, lr):
    """
    One epoch = one pass over the whole train set, then one weight update.
    Returns the train loss / accuracy measured during that pass.
    """
    grad = Net()
    loss, correct = 0.0, 0
    n = len(dataset)

    for image, label in dataset:
        hidden, logits = forward(net, image)
        p = softmax(logits)
        loss += -log(p[label]) / n
        correct += int(logits[1] > logits[0]) == label

        d_logits = list(p)
        d_logits[label] -= 1.0
        d_logits = [v / n for v in d_logits]

        d_hidden = [0.0] * HID
        for o in range(OUT):
            grad.b2[o] += d_logits[o]
            for i in range(HID):
                grad.w2[o * HID + i] += d_logits[o] * hidden[i]
                d_hidden[i] += net.w2[o * HID + i] * d_logits[o]
        for i in range(HID):
            if hidden[i] <= 0.0:
                continue
            grad.b1[i] += d_hidden[i]
            for j in range(IN):
                grad.w1[i * IN + j] += d_hidden[i] * image[j]

    for w, g in ((net.w1, grad.w1), (net.b1, grad.b1), (net.w2, grad.w2), (net.b2, grad.b2)):
        for i in range(len(w)):
            w[i] -= lr * g[i]
    return loss, 100.0 * correct / n


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def main():
    # (a) One dataset -> three sets that never share an image
    print("(a) Split: shuffle once, then cut into three sets")
    rng = Lcg(7)
    n_train, n_val, n_test = 80, 100, 100
    data = make_dataset(rng, n_train + n_val + n_test)
    idx = shuffled_indices(rng, len(data))

    train = [data[i] for i in idx[:n_train]]
    val = [data[i] for i in idx[n_train:n_train + n_val]]
    test = [data[i] for i in idx[n_train + n_val:]]

    print(f"  {len(data)} images -> train {len(train)} / val {len(val)} / test {len(test)}")
    print("  first train indices: " + " ".join(str(i) for i in idx[:8]) + " ...")
    print("  (train is tiny here on purpose, so overfitting shows up fast. Real projects\n"
          "   keep most data for train: CIFAR-10 this week is 45000 / 5000 / 10000)\n")

    # (b) Train on TRAIN, watch VAL after every epoch
    print("(b) Train on train, measure on val after every epoch")
    net = Net()
    for v in (net.w1, net.b1, net.w2, net.b2):
        v[:] = [rng.next() for _ in v]

    epochs = 300
    best_net = net.copy()  # the "checkpoint": a copy of the weights
    best_val, best_epoch = (1e9, 0.0), 0

    print("  epoch  train loss  train acc  val loss  val acc")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(net, train, 0.5)
        last_val = evaluate(net, val)
        if last_val[0] < best_val[0]:  # val decides which weights we keep
            best_val, best_epoch = last_val, epoch
            best_net = net.copy()
        if epoch == 1 or epoch % 30 == 0:
            print(f"  {epoch:5d}  {train_loss:10.3f}  {train_acc:8.1f}%  "
                  f"{last_val[0]:8.3f}  {last_val[1]:6.1f}%")
    print("  train loss only goes down; val loss goes down, then UP: overfitting")
    print("  train acc reaches 100% although 1 label in 8 is wrong: it memorized them\n")

    # (c) val chooses the configuration
    print("(c) Val chooses how many epochs to train")
    print(f"  lowest val loss at epoch {best_epoch}: val loss {best_val[0]:.3f}, "
          f"val acc {best_val[1]:.1f}%   <- keep these weights")
    print(f"  after all {epochs} epochs      : val loss {last_val[0]:.3f}, "
          f"val acc {last_val[1]:.1f}%\n")

    # (d) test is touched once, with the chosen weights
    print("(d) Test: used once, at the very end, to report")
    test_loss, test_acc = evaluate(best_net, test)
    print(f"  test loss {test_loss:.3f}, test acc {test_acc:.1f}%  (weights of epoch {best_epoch})")
    print("  val was used to choose, so it is no longer a fair judge. test never influenced anything.")


if __name__ == "__main__":
    main()
