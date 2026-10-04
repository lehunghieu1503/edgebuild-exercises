"""
logits_06_non_pytorch.py
Reading the output of a classification network, in pure Python (no NumPy, no PyTorch).
Same structure as logits_06.cpp.

  logits : the raw numbers the last layer returns, one per class.
           The largest one is the class the network picks.
  top-1  : fraction of images where the highest-scoring class is the correct one
  top-5  : fraction of images where the correct class is among the 5 highest scores

The logits here are made up: this file is about READING them, not producing them.

Run:  python logits_06_non_pytorch.py
"""

from math import exp


class Lcg:
    """Tiny deterministic random generator, so C++ and Python produce the same numbers."""

    def __init__(self, seed):
        self.state = seed

    def next(self):  # uniform in [-0.5, 0.5)
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / 16777216.0 - 0.5


CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# ---------------------------------------------------------------------------
# 1. Reading one logits vector
# ---------------------------------------------------------------------------
def argmax(logits):
    """The prediction: index of the largest logit"""
    return max(range(len(logits)), key=lambda i: logits[i])


def top_k(logits, k):
    """Indices of the k largest logits, best first"""
    return sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:k]


def softmax(logits):
    """
    Logits are not probabilities (they can be negative, they do not add up to 1).
    Softmax converts them. It keeps the order, so the winner stays the winner.
    """
    biggest = max(logits)
    p = [exp(v - biggest) for v in logits]
    total = sum(p)
    return [v / total for v in p]


# ---------------------------------------------------------------------------
# 2. Accuracy over many images
# ---------------------------------------------------------------------------
def rank_of(logits, label):
    """
    Rank of the correct class: 1 = it has the highest score, 2 = one class beats it, ...
    No sorting needed: just count how many classes score higher.
    """
    return sum(v > logits[label] for v in logits) + 1


def top_k_accuracy(batch, labels, k):
    """top-k accuracy = fraction of images whose correct class has rank <= k"""
    hits = sum(rank_of(logits, label) <= k for logits, label in zip(batch, labels))
    return hits / len(batch)


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def main():
    # (a) One image, 10 classes -> 10 logits
    print("(a) One image -> 10 logits (one per class)")
    logits = [-1.2, 0.3, 2.1, 5.7, 1.0, 4.9, 0.2, 1.8, -0.5, -2.0]
    print("  logits = [ " + " ".join(f"{v:+.1f}" for v in logits) + " ]")
    print(f"  argmax = {argmax(logits)} -> the network says \"{CLASSES[argmax(logits)]}\"")

    probs = softmax(logits)
    print("  top 5:  rank  index  class       logit  probability")
    for r, i in enumerate(top_k(logits, 5), 1):
        print(f"          {r:4d}  {i:5d}  {CLASSES[i]:<10}  {logits[i]:+5.1f}  {probs[i]:.3f}")
    print()

    # (b) Eight images with known labels -> top-1 and top-5 accuracy
    print("(b) 8 labeled images -> top-1 and top-5")
    labels = [3, 5, 0, 8, 1, 9, 2, 6]
    # How strongly each made-up image "looks like" its true class
    boost = [3.0, 3.0, 2.0, 1.2, 1.2, 0.6, 0.6, -1.5]
    rng = Lcg(11)
    batch = []
    for label, b in zip(labels, boost):
        row = [4.0 * rng.next() for _ in range(10)]
        row[label] += b
        batch.append(row)

    print("  image  label       predicted   rank of label  top-1  top-5")
    for i, (row, label) in enumerate(zip(batch, labels)):
        rank = rank_of(row, label)
        print(f"  {i:5d}  {CLASSES[label]:<10}  {CLASSES[argmax(row)]:<10}  {rank:13d}  "
              f"{'hit' if rank <= 1 else '-':<5}  {'hit' if rank <= 5 else '-'}")
    print(f"  top-1 accuracy = {top_k_accuracy(batch, labels, 1):.3f}")
    print(f"  top-5 accuracy = {top_k_accuracy(batch, labels, 5):.3f}   (always >= top-1)\n")

    # (c) ImageNet-sized output: 1000 classes -> 1000 logits per image
    print("(c) A 1000-class network returns 1000 logits per image")
    big = [20.0 * rng.next() for _ in range(1000)]
    print(f"  number of logits = {len(big)}")
    print("  top 5 indices =" + "".join(f" {i} ({big[i]:+.2f})" for i in top_k(big, 5)))
    print("  a table index -> class name turns the winning index into a word")


if __name__ == "__main__":
    main()
