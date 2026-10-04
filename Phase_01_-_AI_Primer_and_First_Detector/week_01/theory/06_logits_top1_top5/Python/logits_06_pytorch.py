"""
logits_06_pytorch.py
The same demo as logits_06.cpp / logits_06_non_pytorch.py, with real PyTorch:
argmax, topk and softmax are one call each, and work on a whole batch at once.

Install:  pip install torch
Run:      python logits_06_pytorch.py
"""

import torch


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


CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def main():
    # (a) One image, 10 classes -> 10 logits
    print("(a) One image -> 10 logits (one per class)")
    logits = torch.tensor([-1.2, 0.3, 2.1, 5.7, 1.0, 4.9, 0.2, 1.8, -0.5, -2.0])
    print("  logits = [ " + " ".join(f"{v:+.1f}" for v in logits) + " ]")
    winner = int(logits.argmax())
    print(f"  argmax = {winner} -> the network says \"{CLASSES[winner]}\"")

    probs = logits.softmax(dim=0)
    top = logits.topk(5)   # .values and .indices, best first
    print("  top 5:  rank  index  class       logit  probability")
    for r, (value, i) in enumerate(zip(top.values, top.indices), 1):
        print(f"          {r:4d}  {int(i):5d}  {CLASSES[i]:<10}  {value:+5.1f}  {probs[i]:.3f}")
    print(f"  probabilities add up to {probs.sum():.3f}; the logits add up to {logits.sum():.1f}\n")

    # (b) Eight images with known labels -> top-1 and top-5 accuracy.
    #     A batch of logits has shape (N images, C classes).
    print("(b) 8 labeled images -> top-1 and top-5")
    labels = torch.tensor([3, 5, 0, 8, 1, 9, 2, 6])
    boost = torch.tensor([3.0, 3.0, 2.0, 1.2, 1.2, 0.6, 0.6, -1.5])
    rng = Lcg(11)
    batch = 4.0 * rng.tensor(8, 10)
    batch[torch.arange(8), labels] += boost
    print(f"  batch shape = {tuple(batch.shape)}")

    top5 = batch.topk(5, dim=1).indices        # (8, 5): the 5 best classes of every image
    hits = top5 == labels[:, None]             # (8, 5) booleans: where is the correct class?
    top1_hit = hits[:, 0]                      # correct class is in first place
    top5_hit = hits.any(dim=1)                 # correct class is anywhere in the 5

    print("  image  label       predicted   top-1  top-5")
    for i in range(8):
        print(f"  {i:5d}  {CLASSES[labels[i]]:<10}  {CLASSES[top5[i, 0]]:<10}  "
              f"{'hit' if top1_hit[i] else '-':<5}  {'hit' if top5_hit[i] else '-'}")
    print(f"  top-1 accuracy = {top1_hit.float().mean():.3f}")
    print(f"  top-5 accuracy = {top5_hit.float().mean():.3f}   (always >= top-1)\n")

    # (c) ImageNet-sized output: 1000 classes -> 1000 logits per image
    print("(c) A 1000-class network returns 1000 logits per image")
    big = 20.0 * rng.tensor(1000)
    top = big.topk(5)
    print(f"  number of logits = {big.numel()}")
    print("  top 5 indices =" + "".join(f" {int(i)} ({v:+.2f})" for v, i in zip(top.values, top.indices)))

    try:
        from torchvision.models import resnet18
    except ImportError:
        return
    net = resnet18(weights=None).eval()   # architecture only: random weights, nothing downloaded
    with torch.no_grad():
        out = net(torch.zeros(4, 3, 224, 224))
    print(f"  a real network: resnet18 on 4 images returns shape {tuple(out.shape)}"
          "  (untrained here, so the values mean nothing)")


if __name__ == "__main__":
    main()
