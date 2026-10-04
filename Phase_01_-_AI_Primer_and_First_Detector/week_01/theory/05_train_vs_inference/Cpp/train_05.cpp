// train_05.cpp
// Train and inference are two different jobs.
//
//   Train     : show labeled examples, measure how wrong the network is (the LOSS),
//               then nudge every weight so the loss goes down. Which way to nudge is
//               the GRADIENT, computed by BACKPROPAGATION. Repeat many times.
//   Inference : the weights are frozen. Just run the input through the network.
//
// This file trains the 9 -> 6 -> 2 "vertical or horizontal line" network of
// 02_layer_weight from random weights, with no library at all.
//
// Build:  g++ -std=c++17 -O2 train_05.cpp -o train_05
// Run:    ./train_05

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// Tiny deterministic random generator, so C++ and Python produce the same numbers.
struct Lcg {
    uint32_t state;
    float next() {   // uniform in [-0.5, 0.5)
        state = state * 1664525u + 1013904223u;
        return (state >> 8) / 16777216.0f - 0.5f;
    }
};

// ---------------------------------------------------------------------------
// 1. The simplest possible training: ONE weight
// ---------------------------------------------------------------------------
// Model: y = w * x.  The data was made with w = 3, the model has to find that out.
const std::array<float, 4> XS = {1, 2, 3, 4};
const std::array<float, 4> YS = {3, 6, 9, 12};

// Loss = how wrong we are = mean of (prediction - target)^2
float loss_one_weight(float w) {
    float sum = 0.0f;
    for (size_t i = 0; i < XS.size(); ++i) sum += (w * XS[i] - YS[i]) * (w * XS[i] - YS[i]);
    return sum / XS.size();
}

// Gradient = d(loss)/d(w) = how much the loss changes when w grows a little.
// Here calculus gives it directly: mean of 2 * (w*x - y) * x
float grad_one_weight(float w) {
    float sum = 0.0f;
    for (size_t i = 0; i < XS.size(); ++i) sum += 2.0f * (w * XS[i] - YS[i]) * XS[i];
    return sum / XS.size();
}

// ---------------------------------------------------------------------------
// 2. The network: Linear(9 -> 6) -> ReLU -> Linear(6 -> 2)
// ---------------------------------------------------------------------------
constexpr size_t IN = 9, HID = 6, OUT = 2;

struct Net {
    std::vector<float> w1 = std::vector<float>(HID * IN), b1 = std::vector<float>(HID);
    std::vector<float> w2 = std::vector<float>(OUT * HID), b2 = std::vector<float>(OUT);

    size_t num_params() const { return w1.size() + b1.size() + w2.size() + b2.size(); }
};

struct Sample {
    std::array<float, IN> image;
    size_t label;   // 0 = vertical, 1 = horizontal
};

// What one forward pass produces. Inference only needs `logits`;
// training must also keep `hidden`, because the backward pass reads it.
struct Activations {
    std::array<float, HID> hidden;
    std::array<float, OUT> logits;
};

// Forward pass. `const Net&`: running the network never changes the weights.
Activations forward(const Net& net, const std::array<float, IN>& x) {
    Activations a;
    for (size_t i = 0; i < HID; ++i) {
        float sum = net.b1[i];
        for (size_t j = 0; j < IN; ++j) sum += net.w1[i * IN + j] * x[j];
        a.hidden[i] = std::max(sum, 0.0f);   // ReLU
    }
    for (size_t o = 0; o < OUT; ++o) {
        float sum = net.b2[o];
        for (size_t i = 0; i < HID; ++i) sum += net.w2[o * HID + i] * a.hidden[i];
        a.logits[o] = sum;
    }
    return a;
}

// Softmax turns the scores into probabilities that add up to 1.
std::array<float, OUT> softmax(const std::array<float, OUT>& logits) {
    float max = *std::max_element(logits.begin(), logits.end());
    std::array<float, OUT> p;
    float sum = 0.0f;
    for (size_t o = 0; o < OUT; ++o) sum += p[o] = std::exp(logits[o] - max);
    for (float& v : p) v /= sum;
    return p;
}

// ---------------------------------------------------------------------------
// 3. One training step = forward, loss, backward (gradients), update
// ---------------------------------------------------------------------------
// `Net&` without const: this is the ONLY function that modifies the weights.
float train_step(Net& net, const std::vector<Sample>& batch, float lr) {
    Net grad;   // one gradient per weight, same shapes as the weights, starts at 0
    float loss = 0.0f;
    const float n = (float)batch.size();

    for (const Sample& s : batch) {
        // forward
        Activations a = forward(net, s.image);
        std::array<float, OUT> p = softmax(a.logits);

        // loss (cross-entropy): 0 when the correct class gets probability 1, large when near 0
        loss += -std::log(p[s.label]) / n;

        // backward = backpropagation: walk the layers in REVERSE order and ask each one
        // "how does the loss change if your output changes?" (the chain rule)
        std::array<float, OUT> d_logits = p;      // d(loss)/d(logits) = p - one_hot(label)
        d_logits[s.label] -= 1.0f;
        for (float& v : d_logits) v /= n;

        std::array<float, HID> d_hidden{};
        for (size_t o = 0; o < OUT; ++o) {        // through Linear(6 -> 2)
            grad.b2[o] += d_logits[o];
            for (size_t i = 0; i < HID; ++i) {
                grad.w2[o * HID + i] += d_logits[o] * a.hidden[i];
                d_hidden[i] += net.w2[o * HID + i] * d_logits[o];
            }
        }
        for (size_t i = 0; i < HID; ++i) {        // through ReLU, then Linear(9 -> 6)
            if (a.hidden[i] <= 0.0f) continue;    // ReLU was off: nothing flows back
            grad.b1[i] += d_hidden[i];
            for (size_t j = 0; j < IN; ++j) grad.w1[i * IN + j] += d_hidden[i] * s.image[j];
        }
    }

    // update: move every weight a small step AGAINST its gradient
    auto step = [lr](std::vector<float>& w, const std::vector<float>& g) {
        for (size_t i = 0; i < w.size(); ++i) w[i] -= lr * g[i];
    };
    step(net.w1, grad.w1); step(net.b1, grad.b1);
    step(net.w2, grad.w2); step(net.b2, grad.b2);
    return loss;
}

// ---------------------------------------------------------------------------
// 4. Data and evaluation
// ---------------------------------------------------------------------------
std::vector<Sample> line_images() {
    std::vector<Sample> samples;
    for (size_t k = 0; k < 3; ++k) {
        Sample v{{}, 0}, h{{}, 1};
        for (size_t j = 0; j < 3; ++j) {
            v.image[j * 3 + k] = 1.0f;   // column k is lit
            h.image[k * 3 + j] = 1.0f;   // row k is lit
        }
        samples.push_back(v);
        samples.push_back(h);
    }
    return samples;
}

size_t count_correct(const Net& net, const std::vector<Sample>& samples) {
    size_t correct = 0;
    for (const Sample& s : samples) {
        Activations a = forward(net, s.image);
        correct += (size_t)(a.logits[1] > a.logits[0]) == s.label;
    }
    return correct;
}

// ---------------------------------------------------------------------------
// 5. Demo
// ---------------------------------------------------------------------------
int main() {
    // (a) One weight: loss, gradient, update
    std::printf("(a) Training ONE weight: find w so that w * x matches y = 3 * x\n");
    float w = 0.0f;
    const float h = 0.01f;
    std::printf("  gradient at w=0: by formula = %.2f, by nudging w a little = %.2f\n",
                grad_one_weight(w), (loss_one_weight(w + h) - loss_one_weight(w - h)) / (2 * h));
    std::printf("  step      w     loss  gradient\n");
    for (int step = 0; step <= 6; ++step) {
        float grad = grad_one_weight(w);
        std::printf("  %4d  %.4f  %7.4f  %8.4f\n", step, w, loss_one_weight(w), grad);
        w -= 0.05f * grad;   // learning rate 0.05: negative gradient -> w must grow
    }
    std::printf("  w walked from 0 to 3 by itself: that is \"learning\"\n\n");

    // (b) The same loop on a network: 74 weights, gradients from backpropagation
    std::printf("(b) TRAIN: Linear(9->6) -> ReLU -> Linear(6->2), starting from random weights\n");
    const std::vector<Sample> samples = line_images();
    Net net;
    Lcg rng{5};
    for (auto* v : {&net.w1, &net.b1, &net.w2, &net.b2})
        for (float& x : *v) x = rng.next();

    const int steps = 200;
    const float lr = 0.5f;
    std::printf("  before training: %zu/%zu correct\n", count_correct(net, samples), samples.size());
    for (int step = 0; step <= steps; ++step) {
        size_t correct = count_correct(net, samples);
        float loss = train_step(net, samples, lr);
        if (step % 40 == 0)
            std::printf("  step %3d  loss = %.4f  correct = %zu/%zu\n", step, loss, correct,
                        samples.size());
    }
    std::printf("\n");

    // (c) Inference: weights frozen, forward only
    std::printf("(c) INFERENCE: weights frozen, forward pass only\n");
    const Net& frozen = net;
    for (size_t i = 0; i < 2; ++i) {
        Activations a = forward(frozen, samples[i].image);
        std::array<float, OUT> p = softmax(a.logits);
        std::printf("  image %zu (%s): logits = [%+.3f %+.3f]  probabilities = [%.3f %.3f]\n", i,
                    samples[i].label == 0 ? "vertical" : "horizontal", a.logits[0], a.logits[1],
                    p[0], p[1]);
    }
    std::printf("\n");

    std::printf("(d) What each job needed\n");
    std::printf("  train    : %d steps x %zu images = %zu forward + %zu backward passes, "
                "%zu weights changed %d times\n",
                steps + 1, samples.size(), (steps + 1) * samples.size(),
                (steps + 1) * samples.size(), net.num_params(), steps + 1);
    std::printf("             memory: %zu weights + %zu gradients + activations kept for backward\n",
                net.num_params(), net.num_params());
    std::printf("  inference: 1 forward pass per image, 0 weights changed\n");
    std::printf("             memory: %zu weights\n", net.num_params());
    return 0;
}
