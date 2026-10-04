"""
splits_07_pytorch.py
The same demo as splits_07.cpp / splits_07_non_pytorch.py, with real PyTorch.

  train : used to adjust the weights          -> the only set that sees loss.backward()
  val   : used to choose the configuration    -> measured under torch.no_grad()
  test  : used ONCE at the very end           -> measured under torch.no_grad()

Install:  pip install torch
Run:      python splits_07_pytorch.py
"""

import copy
from collections import OrderedDict

import torch
from torch import nn


class Lcg:
    """Same tiny generator as the from-scratch versions, so all three print the same numbers."""

    def __init__(self, seed):
        self.state = seed

    def next_bits(self):  # 24 random bits
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state >> 8

    def next(self):  # uniform in [-0.5, 0.5)
        return self.next_bits() / 16777216.0 - 0.5

    def next_int(self, n):  # integer in [0, n)
        return self.next_bits() % n

    def tensor(self, *shape):
        n = torch.Size(shape).numel()
        return torch.tensor([self.next() for _ in range(n)], dtype=torch.float32).view(*shape)


def make_dataset(rng, count):
    """Noisy 3x3 line images, flattened to 9 numbers. 1 label in 8 is wrong on purpose."""
    images, labels = [], []
    for i in range(count):
        label = i % 2  # 0 = vertical, 1 = horizontal
        k = (i // 2) % 3
        image = torch.zeros(3, 3)
        if label == 0:
            image[:, k] = 1.0
        else:
            image[k, :] = 1.0
        image = image.flatten() + rng.tensor(9)  # noise: no two images are equal
        if rng.next_int(8) == 0:
            label = 1 - label
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def shuffled_indices(rng, count):
    idx = list(range(count))
    for i in range(count - 1, 0, -1):
        j = rng.next_int(i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def evaluate(model, loss_fn, images, labels):
    """Forward only: measures, never changes the weights. Used for val and test."""
    with torch.no_grad():
        logits = model(images)
        loss = loss_fn(logits, labels).item()
        correct = int((logits.argmax(dim=1) == labels).sum())
    return loss, 100.0 * correct / len(labels)  # accuracy in percent


def main():
    # (a) One dataset -> three sets that never share an image.
    #     In a real project this is one line:
    #       train, val, test = torch.utils.data.random_split(
    #           dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))
    #     Done by hand here so the split is identical to the from-scratch versions.
    print("(a) Split: shuffle once, then cut into three sets")
    rng = Lcg(7)
    n_train, n_val, n_test = 80, 100, 100
    images, labels = make_dataset(rng, n_train + n_val + n_test)
    idx = shuffled_indices(rng, len(labels))

    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    x_train, y_train = images[train_idx], labels[train_idx]
    x_val, y_val = images[val_idx], labels[val_idx]
    x_test, y_test = images[test_idx], labels[test_idx]

    print(f"  {len(labels)} images -> train {len(y_train)} / val {len(y_val)} / test {len(y_test)}")
    print("  first train indices: " + " ".join(str(i) for i in train_idx[:8]) + " ...")
    shared = set(train_idx) & set(val_idx) | set(train_idx) & set(test_idx) | set(val_idx) & set(test_idx)
    print(f"  images shared between sets: {len(shared)}")
    print("  (train is tiny here on purpose, so overfitting shows up fast. Real projects\n"
          "   keep most data for train: CIFAR-10 this week is 45000 / 5000 / 10000)\n")

    # (b) Train on TRAIN, watch VAL after every epoch
    print("(b) Train on train, measure on val after every epoch")
    model = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(9, 24)),
        ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(24, 2)),
    ]))
    with torch.no_grad():
        for layer in (model.fc1, model.fc2):
            layer.weight.copy_(rng.tensor(*layer.weight.shape))
            layer.bias.copy_(rng.tensor(*layer.bias.shape))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 300
    best_state = copy.deepcopy(model.state_dict())  # the "checkpoint": a copy of the weights
    best_val, best_epoch = (1e9, 0.0), 0

    print("  epoch  train loss  train acc  val loss  val acc")
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = 100.0 * int((logits.argmax(dim=1) == y_train).sum()) / n_train

        model.eval()
        last_val = evaluate(model, loss_fn, x_val, y_val)
        if last_val[0] < best_val[0]:  # val decides which weights we keep
            best_val, best_epoch = last_val, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 30 == 0:
            print(f"  {epoch:5d}  {loss.item():10.3f}  {train_acc:8.1f}%  "
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
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, loss_fn, x_test, y_test)
    print(f"  test loss {test_loss:.3f}, test acc {test_acc:.1f}%  (weights of epoch {best_epoch})")
    print("  val was used to choose, so it is no longer a fair judge. test never influenced anything.")


if __name__ == "__main__":
    main()
