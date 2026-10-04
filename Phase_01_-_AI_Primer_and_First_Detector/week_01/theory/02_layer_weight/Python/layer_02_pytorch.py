"""
layer_02_pytorch.py
Layers and weights with real PyTorch. This is the "training side":
it builds the network, lets the weights be LEARNED from data, then writes
the model file that layer_02.cpp and layer_02_non_pytorch.py load.

  layer      = a function "tensor in -> tensor out" + the numbers it computes with
  weights    = those numbers (learned from data, nobody types them)
  network    = a chain of layers
  model file = the chain's configuration + the value of every weight

Install:  pip install torch
Run:      python layer_02_pytorch.py
"""

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

MODEL_FILE = Path(__file__).resolve().parent.parent / "line_model.txt"
CLASSES = ["vertical", "horizontal"]


def line_images():
    """Six 3x3 images, flattened to 9 numbers: 3 vertical lines, 3 horizontal lines."""
    images, labels = [], []
    for k in range(3):
        v = torch.zeros(3, 3)
        v[:, k] = 1.0  # column k is lit
        h = torch.zeros(3, 3)
        h[k, :] = 1.0  # row k is lit
        images += [v.flatten(), h.flatten()]
        labels += [0, 1]
    return torch.stack(images), torch.tensor(labels)


def print_summary(model):
    print("  #  name   type    in -> out  params")
    total = 0
    for i, (name, layer) in enumerate(model.named_children(), 1):
        n = sum(p.numel() for p in layer.parameters())
        total += n
        if isinstance(layer, nn.Linear):
            io = f"{layer.in_features} -> {layer.out_features}"
            note = f"  ({layer.out_features}*{layer.in_features} weights + {layer.out_features} biases)"
        else:
            io, note = "same", "  (no weights: it only computes)"
        print(f"  {i}  {name:<6} {type(layer).__name__.lower():<7} {io:<10} {n}{note}")
    print(f"  total params = {total}  (= {total * 4} bytes as float32)")


def evaluate(model, images, labels):
    correct = 0
    with torch.no_grad():  # inference only: no gradients needed
        for img, label in zip(images, labels):
            scores = model(img)
            pred = int(scores.argmax())
            correct += pred == int(label)
            rows = ["".join("#" if p > 0.5 else "." for p in img[r * 3:r * 3 + 3]) for r in range(3)]
            verdict = "OK" if pred == int(label) else f"WRONG (it is {CLASSES[label]})"
            print(f"  {rows[0]}   scores: vertical={scores[0]:+.3f}  horizontal={scores[1]:+.3f}")
            print(f"  {rows[1]}   -> {CLASSES[pred]}  {verdict}")
            print(f"  {rows[2]}")
    print(f"  correct: {correct}/{len(labels)}\n")


def export_text(model, path):
    """Write configuration + weights as plain text so you can open it and read it."""
    with open(path, "w") as f:
        f.write("# line_model.txt -- written by layer_02_pytorch.py\n")
        f.write("# A model file = configuration (which layers, in what order) + weights.\n")
        f.write("# Task: 3x3 image (9 pixels) -> 2 scores (vertical line, horizontal line)\n\n")
        f.write(f"layers {len(model)}\n")
        for name, layer in model.named_children():
            if isinstance(layer, nn.Linear):
                f.write(f"\nlinear {name} {layer.in_features} {layer.out_features}\n")
                f.write("weight   # one row per output, one column per input\n")
                for row in layer.weight.tolist():
                    f.write("  " + " ".join(f"{v: .6f}" for v in row) + "\n")
                f.write("bias\n")
                f.write("  " + " ".join(f"{v: .6f}" for v in layer.bias.tolist()) + "\n")
            elif isinstance(layer, nn.ReLU):
                f.write(f"\nrelu {name}\n")
            else:
                raise TypeError(f"cannot export layer type {type(layer).__name__}")


def main():
    torch.manual_seed(0)

    # (a) ONE layer, weights typed by hand so the arithmetic is visible.
    #     Linear computes: y[o] = sum_i weight[o][i] * x[i] + bias[o]
    print("(a) One layer: Linear(3 -> 2), hand-typed weights")
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, -1.0],
                                         [0.5, 0.5, 0.5]]))
        layer.bias.copy_(torch.tensor([0.0, 1.0]))
    x = torch.tensor([2.0, 3.0, 4.0])
    y = layer(x)
    print(f"  weight shape={tuple(layer.weight.shape)}  bias shape={tuple(layer.bias.shape)}")
    print(f"  x = {x.tolist()}  shape={tuple(x.shape)}")
    print(f"  y = {y.tolist()}  shape={tuple(y.shape)}   (expected [-2.0, 5.5])")
    print(f"  ReLU(y) = {torch.relu(y).tolist()}   (negatives become 0)\n")

    # (b) A network = a chain of layers. The configuration is these 3 lines.
    print("(b) A network: 3 layers in a chain")
    model = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(9, 6)),
        ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(6, 2)),
    ]))
    print_summary(model)
    print("  PyTorch keeps the weights in named tensors:")
    for name, p in model.named_parameters():
        print(f"    {name:<11} shape={tuple(p.shape)}")
    print()

    # (c) Freshly created = random weights. The network knows nothing yet.
    images, labels = line_images()
    print("(c) Untrained (random weights)")
    evaluate(model, images, labels)

    # (d) Training: show examples, measure the error (loss), nudge the weights.
    #     The details are section 5. The point here: the weights come from DATA.
    print("(d) Training: the weights get learned from data")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(301):
        # 16 noisy copies of each image, so it learns "line", not 6 exact pictures
        batch = images.repeat(16, 1)
        batch = batch + 0.2 * torch.randn_like(batch)
        loss = loss_fn(model(batch), labels.repeat(16))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"  step {step:3d}  loss = {loss.item():.4f}")
    print()

    print("(e) Trained: same layers, same code, different numbers")
    evaluate(model, images, labels)

    # (f) The model file. state_dict() is what torch.save() would put in a .pt file;
    #     we write the same content as text so C++ and plain Python can read it.
    print("(f) Saving the model file")
    for name, tensor in model.state_dict().items():
        print(f"  state_dict['{name}']  shape={tuple(tensor.shape)}")
    export_text(model, MODEL_FILE)
    print(f"  wrote {MODEL_FILE}  ({MODEL_FILE.stat().st_size} bytes)\n")

    # (g) Real models are the same idea, just far more layers and weights.
    try:
        from torchvision.models import resnet18
    except ImportError:
        return
    net = resnet18(weights=None)  # architecture only, nothing is downloaded
    n_params = sum(p.numel() for p in net.parameters())
    n_layers = sum(1 for m in net.modules() if not list(m.children()))
    print("(g) A real model: ResNet-18")
    print(f"  layers = {n_layers}  params = {n_params:,}  "
          f"weights as float32 = {n_params * 4 / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
