// conv_03.cpp
// Convolution = slide a small window (the kernel) over the image. At every position,
// multiply each pixel by the matching weight and add everything up -> ONE number.
//
//   kernel  k : how wide the window is
//   stride    : how many pixels the window moves per step
//   padding   : how many zero pixels are added around the border
//   H_out = (H + 2*pad - k) / stride + 1          (integer division)
//   params = C_out * C_in * k * k + C_out
//
// Build:  g++ -std=c++17 -O2 conv_03.cpp -o conv_03
// Run:    ./conv_03

#include <cstdint>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// 1. Tensor in CHW layout (one image, no batch dimension)
// ---------------------------------------------------------------------------
struct Tensor {
    size_t C, H, W;
    std::vector<float> data;

    Tensor(size_t c, size_t h, size_t w) : C(c), H(h), W(w), data(c * h * w, 0.0f) {}

    float& at(size_t c, size_t h, size_t w)       { return data[(c * H + h) * W + w]; }
    float  at(size_t c, size_t h, size_t w) const { return data[(c * H + h) * W + w]; }
};

// The weights of one conv layer: C_out kernels, each of size C_in x k x k, plus one bias each
struct Conv {
    size_t c_out, c_in, k;
    std::vector<float> weight;   // layout (C_out, C_in, k, k)
    std::vector<float> bias;     // C_out

    Conv(size_t co, size_t ci, size_t k_)
        : c_out(co), c_in(ci), k(k_), weight(co * ci * k_ * k_, 0.0f), bias(co, 0.0f) {}

    float w(size_t co, size_t ci, size_t kh, size_t kw) const {
        return weight[((co * c_in + ci) * k + kh) * k + kw];
    }
    size_t num_params() const { return weight.size() + bias.size(); }
};

// Tiny deterministic random generator, so C++ and Python produce the same numbers.
struct Lcg {
    uint32_t state;
    float next() {   // uniform in [-0.5, 0.5)
        state = state * 1664525u + 1013904223u;
        return (state >> 8) / 16777216.0f - 0.5f;
    }
};

// ---------------------------------------------------------------------------
// 2. The convolution itself
// ---------------------------------------------------------------------------
size_t conv_out_size(size_t in, size_t k, size_t stride, size_t pad) {
    return (in + 2 * pad - k) / stride + 1;
}

// Note: like PyTorch, the kernel is NOT flipped (signal-processing books flip it).
Tensor conv2d(const Tensor& x, const Conv& conv, size_t stride, size_t pad) {
    const size_t k = conv.k;
    Tensor y(conv.c_out, conv_out_size(x.H, k, stride, pad), conv_out_size(x.W, k, stride, pad));

    for (size_t co = 0; co < conv.c_out; ++co)           // every output channel has its own kernel
        for (size_t oh = 0; oh < y.H; ++oh)              // slide down
            for (size_t ow = 0; ow < y.W; ++ow) {        // slide right
                float sum = conv.bias[co];
                for (size_t ci = 0; ci < conv.c_in; ++ci)       // the window covers ALL input channels
                    for (size_t kh = 0; kh < k; ++kh)
                        for (size_t kw = 0; kw < k; ++kw) {
                            // which input pixel sits under this kernel cell
                            long ih = (long)(oh * stride + kh) - (long)pad;
                            long iw = (long)(ow * stride + kw) - (long)pad;
                            if (ih < 0 || ih >= (long)x.H || iw < 0 || iw >= (long)x.W)
                                continue;                       // in the padding: pixel is 0
                            sum += x.at(ci, ih, iw) * conv.w(co, ci, kh, kw);
                        }
                y.at(co, oh, ow) = sum;
            }
    return y;
}

// ---------------------------------------------------------------------------
// 3. Demo
// ---------------------------------------------------------------------------
void print_channel(const char* title, const Tensor& t) {
    std::printf("  %s  (%zu x %zu)\n", title, t.H, t.W);
    for (size_t h = 0; h < t.H; ++h) {
        std::printf("   ");
        for (size_t w = 0; w < t.W; ++w) std::printf("%3.0f", t.at(0, h, w));
        std::printf("\n");
    }
}

int main() {
    // (a) One kernel sliding over a 6x6 image: dark on the left, bright on the right
    std::printf("(a) Slide a 3x3 kernel over a 6x6 image\n");
    Tensor image(1, 6, 6);
    for (size_t h = 0; h < 6; ++h)
        for (size_t w = 3; w < 6; ++w) image.at(0, h, w) = 1.0f;
    print_channel("image", image);

    Conv vertical_edge(1, 1, 3);
    vertical_edge.weight = {-1, 0, 1,
                            -1, 0, 1,
                            -1, 0, 1};
    Conv horizontal_edge(1, 1, 3);
    horizontal_edge.weight = {-1, -1, -1,
                               0,  0,  0,
                               1,  1,  1};

    // The arithmetic at ONE window position: output[0][1]
    std::printf("  window at (row 0, col 1)   kernel       products\n");
    float sum = 0.0f;
    for (size_t kh = 0; kh < 3; ++kh) {
        std::printf("   ");
        for (size_t kw = 0; kw < 3; ++kw) std::printf("%3.0f", image.at(0, kh, 1 + kw));
        std::printf("                ");
        for (size_t kw = 0; kw < 3; ++kw) std::printf("%3.0f", vertical_edge.w(0, 0, kh, kw));
        std::printf("    ");
        for (size_t kw = 0; kw < 3; ++kw) {
            float p = image.at(0, kh, 1 + kw) * vertical_edge.w(0, 0, kh, kw);
            sum += p;
            std::printf("%3.0f", p + 0.0f);   // + 0 turns "-0" into "0"
        }
        std::printf("\n");
    }
    std::printf("  sum of the 9 products + bias = %.0f  -> output[0][1]\n", sum);

    print_channel("output of the vertical-edge kernel: non-zero exactly where dark meets bright",
                  conv2d(image, vertical_edge, 1, 0));
    print_channel("output of the horizontal-edge kernel: this image has none",
                  conv2d(image, horizontal_edge, 1, 0));
    std::printf("\n");

    // (b) kernel, stride, padding decide the output size
    std::printf("(b) Output size: H_out = (H + 2*pad - k) / stride + 1\n");
    std::printf("    H    k  stride  pad   formula   actual\n");
    const size_t cases[][2] = {{1, 0}, {1, 1}, {2, 0}, {2, 1}};   // {stride, pad}
    for (const auto& c : cases) {
        size_t formula = conv_out_size(6, 3, c[0], c[1]);
        size_t actual  = conv2d(image, vertical_edge, c[0], c[1]).H;
        std::printf("  %3d  %3d  %6zu  %3zu  %8zu  %7zu\n", 6, 3, c[0], c[1], formula, actual);
    }
    std::printf("  %3d  %3d  %6d  %3d  %8zu        -   (stride 2 halves the image)\n",
                224, 3, 2, 1, conv_out_size(224, 3, 2, 1));
    std::printf("  %3d  %3d  %6d  %3d  %8zu        -   (first conv of ResNet)\n\n",
                224, 7, 2, 3, conv_out_size(224, 7, 2, 3));

    // (c) A real-sized layer: 3 input channels (RGB), 16 output channels, 3x3 kernel
    std::printf("(c) Conv 3 -> 16 channels, kernel 3x3, stride 1, pad 1, on a 3x32x32 image\n");
    Lcg rng{2024};
    Tensor rgb(3, 32, 32);
    for (float& v : rgb.data) v = rng.next();
    Conv conv(16, 3, 3);
    for (float& v : conv.weight) v = rng.next();
    for (float& v : conv.bias)   v = rng.next();

    Tensor out = conv2d(rgb, conv, 1, 1);
    std::printf("  weight shape = (%zu, %zu, %zu, %zu)   bias shape = (%zu)\n",
                conv.c_out, conv.c_in, conv.k, conv.k, conv.bias.size());
    std::printf("  params = 16*3*3*3 + 16 = %zu\n", conv.num_params());
    std::printf("  input  shape = (%zu, %zu, %zu)\n", rgb.C, rgb.H, rgb.W);
    std::printf("  output shape = (%zu, %zu, %zu)\n", out.C, out.H, out.W);
    std::printf("  multiply-adds = 16*32*32 outputs x 3*3*3 each = %zu\n",
                out.data.size() * conv.c_in * conv.k * conv.k);
    std::printf("  out[0][0][0] = %+.4f   out[7][16][16] = %+.4f   out[15][31][31] = %+.4f\n",
                out.at(0, 0, 0), out.at(7, 16, 16), out.at(15, 31, 31));
    return 0;
}
