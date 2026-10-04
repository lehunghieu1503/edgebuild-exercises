// tensor_basics.cpp
// A tensor = a flat 1D buffer + metadata describing how to read it.
// Build:  g++ -std=c++17 -O2 tensor_basics.cpp -o tensor_basics
// Run:    ./tensor_basics

#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 1. Metadata: dtype and device
// ---------------------------------------------------------------------------
enum class DType  { Float32 };
enum class Device { CPU, GPU };   // GPU is only a label here; data stays in RAM

std::string to_string(DType)    { return "float32"; }
std::string to_string(Device d) { return d == Device::CPU ? "CPU" : "GPU"; }

// ---------------------------------------------------------------------------
// 2. The Tensor itself
// ---------------------------------------------------------------------------
struct Tensor {
    std::vector<float>  data;     // the "float buffer[N]" from the picture
    std::vector<size_t> shape;    // e.g. {1, 3, 224, 224}
    std::vector<size_t> strides;  // how many elements to jump per dimension
    DType  dtype  = DType::Float32;
    Device device = Device::CPU;

    explicit Tensor(std::vector<size_t> s) : shape(std::move(s)) {
        // N = product of all dimensions
        size_t n = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                   std::multiplies<size_t>());
        data.assign(n, 0.0f);

        // Row-major strides: last dimension is contiguous (stride 1).
        // For shape (1,3,224,224): strides = (3*224*224, 224*224, 224, 1)
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    size_t numel() const { return data.size(); }

    // Turn a multi-dimensional index into a position in the 1D buffer.
    size_t offset(const std::vector<size_t>& idx) const {
        if (idx.size() != shape.size()) throw std::out_of_range("rank mismatch");
        size_t off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] >= shape[i]) throw std::out_of_range("index out of range");
            off += idx[i] * strides[i];
        }
        return off;
    }

    // Convenience accessor for 4D NCHW tensors
    float& at(size_t n, size_t c, size_t h, size_t w) {
        return data[offset({n, c, h, w})];
    }

    void print_meta(const std::string& name) const {
        std::cout << name << ": shape=(";
        for (size_t i = 0; i < shape.size(); ++i)
            std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
        std::cout << ")  strides=(";
        for (size_t i = 0; i < strides.size(); ++i)
            std::cout << strides[i] << (i + 1 < strides.size() ? ", " : "");
        std::cout << ")  dtype=" << to_string(dtype)
                  << "  device=" << to_string(device)
                  << "  numel=" << numel() << "\n";
    }
};

// ---------------------------------------------------------------------------
// 3. HWC (OpenCV style, uint8, interleaved RGBRGB...) -> NCHW float tensor
// ---------------------------------------------------------------------------
// OpenCV stores pixel (h, w, c) at:  hwc[(h * W + w) * C + c]
// The network wants it at:           nchw[((n * C + c) * H + h) * W + w]
Tensor hwc_to_nchw(const std::vector<uint8_t>& hwc, size_t H, size_t W, size_t C) {
    Tensor t({1, C, H, W});
    for (size_t h = 0; h < H; ++h)
        for (size_t w = 0; w < W; ++w)
            for (size_t c = 0; c < C; ++c) {
                uint8_t px = hwc[(h * W + w) * C + c];
                t.at(0, c, h, w) = px / 255.0f;   // also normalize to [0, 1]
            }
    return t;
}

// ---------------------------------------------------------------------------
// 4. Demo
// ---------------------------------------------------------------------------
int main() {
    // (a) The tensor from the slide: one 224x224 RGB image
    Tensor img({1, 3, 224, 224});
    img.print_meta("img");
    std::cout << "  bytes in buffer = " << img.numel() * sizeof(float) << "\n\n";

    // (b) A tiny 2x3 image so we can actually see the memory layout
    const size_t H = 2, W = 3, C = 3;
    // Interleaved HWC: each pixel is R,G,B in a row
    std::vector<uint8_t> hwc = {
        // row 0                         
        10, 110, 210,   11, 111, 211,   12, 112, 212,
        // row 1
        13, 113, 213,   14, 114, 214,   15, 115, 215,
    };
    // R values are 10..15, G are 110..115, B are 210..215

    std::cout << "HWC buffer (interleaved, like OpenCV):\n  ";
    for (auto v : hwc) std::cout << (int)v << ' ';
    std::cout << "\n\n";

    Tensor x = hwc_to_nchw(hwc, H, W, C);
    x.print_meta("x");

    std::cout << "NCHW buffer (planar: R plane, then G plane, then B plane):\n  ";
    for (float v : x.data) std::cout << (int)(v * 255.0f + 0.5f) << ' ';
    std::cout << "\n\n";

    // (c) Same element, two views: index math vs. raw buffer position
    size_t n = 0, c = 1, h = 1, w = 2;   // green channel, row 1, col 2
    size_t off = x.offset({n, c, h, w});
    std::cout << "x[0][G][1][2] lives at buffer[" << off << "] = "
              << (int)(x.data[off] * 255.0f + 0.5f) << "  (expected 115)\n";
    return 0;
}