// layers_04.cpp
// The layers that go with convolution, and how they stack into a CNN.
//
//   ReLU      : negatives become 0
//   MaxPool   : shrink the image, keep the largest value of every 2x2 block
//   BatchNorm : rescale every channel so values stay in a sane range
//   Linear    : (fully-connected) turn the features into one score per class
//   CNN       : blocks of "conv -> BatchNorm -> ReLU -> pool", then Linear
//
// Build:  g++ -std=c++17 -O2 layers_04.cpp -o layers_04
// Run:    ./layers_04

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 1. Tensor (CHW) and the weights of each layer type
// ---------------------------------------------------------------------------
struct Tensor {
    size_t C, H, W;
    std::vector<float> data;

    Tensor(size_t c, size_t h, size_t w) : C(c), H(h), W(w), data(c * h * w, 0.0f) {}

    float& at(size_t c, size_t h, size_t w)       { return data[(c * H + h) * W + w]; }
    float  at(size_t c, size_t h, size_t w) const { return data[(c * H + h) * W + w]; }
};

// Tiny deterministic random generator, so C++ and Python produce the same numbers.
struct Lcg {
    uint32_t state;
    float next() {   // uniform in [-0.5, 0.5)
        state = state * 1664525u + 1013904223u;
        return (state >> 8) / 16777216.0f - 0.5f;
    }
    void fill(std::vector<float>& v, float offset = 0.0f) {
        for (float& x : v) x = offset + next();
    }
};

struct Conv {
    size_t c_out, c_in, k;
    std::vector<float> weight, bias;   // (C_out, C_in, k, k) and (C_out)

    Conv(size_t co, size_t ci, size_t k_)
        : c_out(co), c_in(ci), k(k_), weight(co * ci * k_ * k_), bias(co) {}
    size_t num_params() const { return weight.size() + bias.size(); }
};

// Four numbers per channel. gamma/beta are learned like any weight;
// mean/var are statistics measured on the training data.
struct BatchNorm {
    std::vector<float> gamma, beta, mean, var;

    explicit BatchNorm(size_t channels)
        : gamma(channels, 1.0f), beta(channels, 0.0f), mean(channels, 0.0f), var(channels, 1.0f) {}
    size_t num_params() const { return gamma.size() + beta.size(); }
};

struct Linear {
    size_t in, out;
    std::vector<float> weight, bias;   // (out, in) and (out)

    Linear(size_t in_, size_t out_) : in(in_), out(out_), weight(in_ * out_), bias(out_) {}
    size_t num_params() const { return weight.size() + bias.size(); }
};

// ---------------------------------------------------------------------------
// 2. The layers
// ---------------------------------------------------------------------------
// ReLU: the only non-linear step. Without it, stacked layers collapse into one straight line.
Tensor relu(Tensor x) {
    for (float& v : x.data) v = std::max(v, 0.0f);
    return x;
}

// MaxPool: one output per k x k block = the largest value in that block. No weights.
Tensor maxpool2d(const Tensor& x, size_t k = 2, size_t stride = 2) {
    Tensor y(x.C, (x.H - k) / stride + 1, (x.W - k) / stride + 1);
    for (size_t c = 0; c < y.C; ++c)
        for (size_t oh = 0; oh < y.H; ++oh)
            for (size_t ow = 0; ow < y.W; ++ow) {
                float best = x.at(c, oh * stride, ow * stride);
                for (size_t kh = 0; kh < k; ++kh)
                    for (size_t kw = 0; kw < k; ++kw)
                        best = std::max(best, x.at(c, oh * stride + kh, ow * stride + kw));
                y.at(c, oh, ow) = best;
            }
    return y;
}

// BatchNorm at inference: y = gamma * (x - mean) / sqrt(var + eps) + beta, per channel.
// All four numbers are constants here, so it is just "x * scale + shift".
// (That is why inference engines can fold it into the conv in front of it.)
Tensor batchnorm(Tensor x, const BatchNorm& bn, float eps = 1e-5f) {
    for (size_t c = 0; c < x.C; ++c) {
        float scale = bn.gamma[c] / std::sqrt(bn.var[c] + eps);
        float shift = bn.beta[c] - bn.mean[c] * scale;
        for (size_t i = 0; i < x.H * x.W; ++i) {
            float& v = x.data[c * x.H * x.W + i];
            v = v * scale + shift;
        }
    }
    return x;
}

// Linear: y[o] = sum_i weight[o][i] * x[i] + bias[o]   (same as 02_layer_weight)
std::vector<float> linear(const std::vector<float>& x, const Linear& fc) {
    std::vector<float> y(fc.out);
    for (size_t o = 0; o < fc.out; ++o) {
        float sum = fc.bias[o];
        for (size_t i = 0; i < fc.in; ++i) sum += fc.weight[o * fc.in + i] * x[i];
        y[o] = sum;
    }
    return y;
}

// Convolution with zero padding (explained step by step in 03_convolution)
Tensor conv2d(const Tensor& x, const Conv& conv, size_t stride, size_t pad) {
    const size_t k = conv.k;
    Tensor y(conv.c_out, (x.H + 2 * pad - k) / stride + 1, (x.W + 2 * pad - k) / stride + 1);
    for (size_t co = 0; co < conv.c_out; ++co)
        for (size_t oh = 0; oh < y.H; ++oh)
            for (size_t ow = 0; ow < y.W; ++ow) {
                float sum = conv.bias[co];
                for (size_t ci = 0; ci < conv.c_in; ++ci)
                    for (size_t kh = 0; kh < k; ++kh)
                        for (size_t kw = 0; kw < k; ++kw) {
                            long ih = (long)(oh * stride + kh) - (long)pad;
                            long iw = (long)(ow * stride + kw) - (long)pad;
                            if (ih < 0 || ih >= (long)x.H || iw < 0 || iw >= (long)x.W) continue;
                            sum += x.at(ci, ih, iw) *
                                   conv.weight[((co * conv.c_in + ci) * k + kh) * k + kw];
                        }
                y.at(co, oh, ow) = sum;
            }
    return y;
}

// ---------------------------------------------------------------------------
// 3. Demo
// ---------------------------------------------------------------------------
void print_values(const char* name, const std::vector<float>& v, const char* fmt = " %+.3f") {
    std::printf("  %s [", name);
    for (float x : v) std::printf(fmt, x);
    std::printf(" ]\n");
}

void print_channel(const char* title, const Tensor& t) {
    std::printf("  %s  (%zu x %zu)\n", title, t.H, t.W);
    for (size_t h = 0; h < t.H; ++h) {
        std::printf("   ");
        for (size_t w = 0; w < t.W; ++w) std::printf("%3.0f", t.at(0, h, w));
        std::printf("\n");
    }
}

void print_row(const char* name, const Tensor& t, size_t params, const char* note = "") {
    std::string shape = "(" + std::to_string(t.C) + ", " + std::to_string(t.H) + ", " +
                        std::to_string(t.W) + ")";
    std::printf("  %-16s %-12s %6zu%s\n", name, shape.c_str(), params, note);
}

int main() {
    // (a) ReLU
    std::printf("(a) ReLU: negatives become 0\n");
    Tensor r(1, 1, 5);
    r.data = {-2.0f, -0.5f, 0.0f, 1.5f, 3.0f};
    print_values("in ", r.data, " %+.1f");
    print_values("out", relu(r).data, " %+.1f");
    std::printf("\n");

    // (b) MaxPool
    std::printf("(b) MaxPool 2x2: keep the largest value of every 2x2 block\n");
    Tensor p(1, 4, 4);
    p.data = {1, 3, 2, 0,
              4, 2, 1, 1,
              0, 1, 5, 6,
              2, 0, 7, 8};
    print_channel("in ", p);
    print_channel("out", maxpool2d(p));
    std::printf("  16 numbers -> 4 numbers: the next layer has 4x less work\n\n");

    // (c) BatchNorm: two channels on wildly different scales
    std::printf("(c) BatchNorm: put every channel on the same scale\n");
    Tensor b(2, 2, 2);
    b.data = {1000, 2000, 3000, 4000,    // channel 0
              1, 2, 3, 4};               // channel 1
    BatchNorm bn(2);
    bn.mean = {2500.0f, 2.5f};           // mean of each channel
    bn.var  = {1250000.0f, 1.25f};       // variance of each channel
    Tensor normed = batchnorm(b, bn);
    for (size_t c = 0; c < 2; ++c) {
        std::printf("  channel %zu  before [", c);
        for (size_t i = 0; i < 4; ++i) std::printf(" %.0f", b.data[c * 4 + i]);
        std::printf(" ]  after [");
        for (size_t i = 0; i < 4; ++i) std::printf(" %+.3f", normed.data[c * 4 + i]);
        std::printf(" ]\n");
    }
    std::printf("  per channel: 2 learned weights (gamma, beta) + 2 statistics (mean, var)\n\n");

    // (d) A CNN: two "conv -> BatchNorm -> ReLU -> pool" blocks, then Linear
    std::printf("(d) A CNN: 1x8x8 image -> 3 class scores\n");
    Lcg rng{7};
    Tensor image(1, 8, 8);
    rng.fill(image.data);

    Conv conv1(4, 1, 3);  BatchNorm bn1(4);
    Conv conv2(8, 4, 3);  BatchNorm bn2(8);
    Linear fc(8 * 2 * 2, 3);
    rng.fill(conv1.weight); rng.fill(conv1.bias);
    rng.fill(bn1.gamma, 1.0f); rng.fill(bn1.beta); rng.fill(bn1.mean); rng.fill(bn1.var, 1.0f);
    rng.fill(conv2.weight); rng.fill(conv2.bias);
    rng.fill(bn2.gamma, 1.0f); rng.fill(bn2.beta); rng.fill(bn2.mean); rng.fill(bn2.var, 1.0f);
    rng.fill(fc.weight); rng.fill(fc.bias);

    std::printf("  layer            output shape params\n");
    Tensor x = image;                      print_row("input", x, 0);
    x = conv2d(x, conv1, 1, 1);            print_row("conv1 1->4 3x3", x, conv1.num_params());
    x = batchnorm(x, bn1);                 print_row("bn1", x, bn1.num_params(), "  (+8 statistics)");
    x = relu(x);                           print_row("relu1", x, 0);
    x = maxpool2d(x);                      print_row("pool1 2x2", x, 0, "  (image halves)");
    x = conv2d(x, conv2, 1, 1);            print_row("conv2 4->8 3x3", x, conv2.num_params());
    x = batchnorm(x, bn2);                 print_row("bn2", x, bn2.num_params(), "  (+16 statistics)");
    x = relu(x);                           print_row("relu2", x, 0);
    x = maxpool2d(x);                      print_row("pool2 2x2", x, 0, "  (image halves)");

    // Flatten: (8, 2, 2) -> 32 numbers. Nothing is computed, the buffer is just read as 1D.
    std::vector<float> scores = linear(x.data, fc);
    std::printf("  %-16s %-12s %6d\n", "flatten", "(32)", 0);
    std::printf("  %-16s %-12s %6zu\n", "fc 32->3", "(3)", fc.num_params());
    std::printf("  total params = %zu\n",
                conv1.num_params() + bn1.num_params() + conv2.num_params() +
                bn2.num_params() + fc.num_params());
    print_values("scores", scores, " %+.4f");
    return 0;
}
