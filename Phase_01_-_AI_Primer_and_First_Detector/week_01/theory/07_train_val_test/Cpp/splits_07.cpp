// splits_07.cpp
// Three datasets, three jobs:
//
//   train : used to adjust the weights
//   val   : used to choose the configuration (here: how many epochs to train)
//   test  : used ONCE at the very end, to report the result
//
// Overfitting: the train loss keeps going down while the val loss goes UP.
// The network is memorizing the training images instead of learning the rule.
//
// The data: noisy 3x3 "vertical or horizontal line" images, and 1 label in 8 is
// wrong on purpose (like real annotation mistakes). The network is the one from
// 05_train_vs_inference, with a bigger hidden layer so it has room to memorize.
//
// Build:  g++ -std=c++17 -O2 splits_07.cpp -o splits_07
// Run:    ./splits_07

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

// Tiny deterministic random generator, so C++ and Python produce the same numbers.
struct Lcg {
    uint32_t state;
    uint32_t next_bits() {   // 24 random bits
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    }
    float next() { return next_bits() / 16777216.0f - 0.5f; }   // uniform in [-0.5, 0.5)
    uint32_t next_int(uint32_t n) { return next_bits() % n; }   // integer in [0, n)
};

// ---------------------------------------------------------------------------
// 1. Data: make it, shuffle it, split it
// ---------------------------------------------------------------------------
constexpr size_t IN = 9, HID = 24, OUT = 2;

struct Sample {
    std::array<float, IN> image;
    size_t label;   // 0 = vertical, 1 = horizontal
};

std::vector<Sample> make_dataset(Lcg& rng, size_t count) {
    std::vector<Sample> data;
    for (size_t i = 0; i < count; ++i) {
        Sample s{{}, i % 2};
        size_t k = (i / 2) % 3;                      // which column / row is lit
        for (size_t j = 0; j < 3; ++j) s.image[s.label == 0 ? j * 3 + k : k * 3 + j] = 1.0f;
        for (float& pixel : s.image) pixel += rng.next();   // noise: no two images are equal
        if (rng.next_int(8) == 0) s.label = 1 - s.label;    // 1 label in 8 is wrong
        data.push_back(s);
    }
    return data;
}

// Shuffle BEFORE splitting, otherwise the three sets could hold different kinds of images.
// Shuffle with a fixed seed, so the split is the same on every run.
std::vector<size_t> shuffled_indices(Lcg& rng, size_t count) {
    std::vector<size_t> idx(count);
    std::iota(idx.begin(), idx.end(), 0);
    for (size_t i = count - 1; i > 0; --i) std::swap(idx[i], idx[rng.next_int((uint32_t)i + 1)]);
    return idx;
}

// ---------------------------------------------------------------------------
// 2. The network and its training step (explained in 05_train_vs_inference)
// ---------------------------------------------------------------------------
struct Net {
    std::vector<float> w1 = std::vector<float>(HID * IN), b1 = std::vector<float>(HID);
    std::vector<float> w2 = std::vector<float>(OUT * HID), b2 = std::vector<float>(OUT);
};

struct Activations {
    std::array<float, HID> hidden;
    std::array<float, OUT> logits;
};

Activations forward(const Net& net, const std::array<float, IN>& x) {
    Activations a;
    for (size_t i = 0; i < HID; ++i) {
        float sum = net.b1[i];
        for (size_t j = 0; j < IN; ++j) sum += net.w1[i * IN + j] * x[j];
        a.hidden[i] = std::max(sum, 0.0f);
    }
    for (size_t o = 0; o < OUT; ++o) {
        float sum = net.b2[o];
        for (size_t i = 0; i < HID; ++i) sum += net.w2[o * HID + i] * a.hidden[i];
        a.logits[o] = sum;
    }
    return a;
}

std::array<float, OUT> softmax(const std::array<float, OUT>& logits) {
    float max = *std::max_element(logits.begin(), logits.end());
    std::array<float, OUT> p;
    float sum = 0.0f;
    for (size_t o = 0; o < OUT; ++o) sum += p[o] = std::exp(logits[o] - max);
    for (float& v : p) v /= sum;
    return p;
}

struct Metrics {
    float loss, accuracy;   // accuracy in percent
};

// Forward only: measures, never changes the weights. Used for val and test.
Metrics evaluate(const Net& net, const std::vector<Sample>& set) {
    float loss = 0.0f;
    size_t correct = 0;
    for (const Sample& s : set) {
        Activations a = forward(net, s.image);
        loss += -std::log(softmax(a.logits)[s.label]);
        correct += (size_t)(a.logits[1] > a.logits[0]) == s.label;
    }
    return {loss / set.size(), 100.0f * correct / set.size()};
}

// One epoch = one pass over the whole train set, then one weight update.
// Returns the train loss / accuracy measured during that pass.
Metrics train_epoch(Net& net, const std::vector<Sample>& set, float lr) {
    Net grad;
    float loss = 0.0f;
    size_t correct = 0;
    const float n = (float)set.size();

    for (const Sample& s : set) {
        Activations a = forward(net, s.image);
        std::array<float, OUT> p = softmax(a.logits);
        loss += -std::log(p[s.label]) / n;
        correct += (size_t)(a.logits[1] > a.logits[0]) == s.label;

        std::array<float, OUT> d_logits = p;
        d_logits[s.label] -= 1.0f;
        for (float& v : d_logits) v /= n;

        std::array<float, HID> d_hidden{};
        for (size_t o = 0; o < OUT; ++o) {
            grad.b2[o] += d_logits[o];
            for (size_t i = 0; i < HID; ++i) {
                grad.w2[o * HID + i] += d_logits[o] * a.hidden[i];
                d_hidden[i] += net.w2[o * HID + i] * d_logits[o];
            }
        }
        for (size_t i = 0; i < HID; ++i) {
            if (a.hidden[i] <= 0.0f) continue;
            grad.b1[i] += d_hidden[i];
            for (size_t j = 0; j < IN; ++j) grad.w1[i * IN + j] += d_hidden[i] * s.image[j];
        }
    }

    auto step = [lr](std::vector<float>& w, const std::vector<float>& g) {
        for (size_t i = 0; i < w.size(); ++i) w[i] -= lr * g[i];
    };
    step(net.w1, grad.w1); step(net.b1, grad.b1);
    step(net.w2, grad.w2); step(net.b2, grad.b2);
    return {loss, 100.0f * correct / n};
}

// ---------------------------------------------------------------------------
// 3. Demo
// ---------------------------------------------------------------------------
int main() {
    // (a) One dataset -> three sets that never share an image
    std::printf("(a) Split: shuffle once, then cut into three sets\n");
    Lcg rng{7};
    const size_t n_train = 80, n_val = 100, n_test = 100;
    const std::vector<Sample> data = make_dataset(rng, n_train + n_val + n_test);
    const std::vector<size_t> idx = shuffled_indices(rng, data.size());

    std::vector<Sample> train, val, test;
    for (size_t i = 0; i < idx.size(); ++i)
        (i < n_train ? train : i < n_train + n_val ? val : test).push_back(data[idx[i]]);

    std::printf("  %zu images -> train %zu / val %zu / test %zu\n", data.size(), train.size(),
                val.size(), test.size());
    std::printf("  first train indices:");
    for (size_t i = 0; i < 8; ++i) std::printf(" %zu", idx[i]);
    std::printf(" ...\n");
    std::printf("  (train is tiny here on purpose, so overfitting shows up fast. Real projects\n"
                "   keep most data for train: CIFAR-10 this week is 45000 / 5000 / 10000)\n\n");

    // (b) Train on TRAIN, watch VAL after every epoch
    std::printf("(b) Train on train, measure on val after every epoch\n");
    Net net;
    for (auto* v : {&net.w1, &net.b1, &net.w2, &net.b2})
        for (float& x : *v) x = rng.next();

    const int epochs = 300;
    Net best_net = net;              // the "checkpoint": a copy of the weights
    Metrics best_val{1e9f, 0.0f}, last_val{};
    int best_epoch = 0;

    std::printf("  epoch  train loss  train acc  val loss  val acc\n");
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        Metrics t = train_epoch(net, train, 0.5f);
        last_val = evaluate(net, val);
        if (last_val.loss < best_val.loss) {   // val decides which weights we keep
            best_val = last_val;
            best_epoch = epoch;
            best_net = net;
        }
        if (epoch == 1 || epoch % 30 == 0)
            std::printf("  %5d  %10.3f  %8.1f%%  %8.3f  %6.1f%%\n", epoch, t.loss, t.accuracy,
                        last_val.loss, last_val.accuracy);
    }
    std::printf("  train loss only goes down; val loss goes down, then UP: overfitting\n");
    std::printf("  train acc reaches 100%% although 1 label in 8 is wrong: it memorized them\n\n");

    // (c) val chooses the configuration
    std::printf("(c) Val chooses how many epochs to train\n");
    std::printf("  lowest val loss at epoch %d: val loss %.3f, val acc %.1f%%   <- keep these weights\n",
                best_epoch, best_val.loss, best_val.accuracy);
    std::printf("  after all %d epochs      : val loss %.3f, val acc %.1f%%\n\n", epochs,
                last_val.loss, last_val.accuracy);

    // (d) test is touched once, with the chosen weights
    std::printf("(d) Test: used once, at the very end, to report\n");
    Metrics t = evaluate(best_net, test);
    std::printf("  test loss %.3f, test acc %.1f%%  (weights of epoch %d)\n", t.loss, t.accuracy,
                best_epoch);
    std::printf("  val was used to choose, so it is no longer a fair judge. test never influenced anything.\n");
    return 0;
}
