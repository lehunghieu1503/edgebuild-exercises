// Micro-benchmark của noise test: y = a * x + y trên dữ liệu vừa L2.
//
// Hai mảng float 128K phần tử = 2 x 512 KiB = 1 MiB, nằm gọn trong L2 của cả P-core (3 MiB)
// lẫn E-core (4 MiB dùng chung). Dữ liệu trong L2 thì thời gian phụ thuộc CPU,
// không phụ thuộc băng thông RAM — đúng thứ noise test cần soi.
//
// Không tự pin: chạy dưới `taskset -c <core>`. Đặt BENCH_CORE=<core> nếu muốn harness tự pin.
#include <cstdlib>
#include <vector>

#include "bench/bench.h"

namespace {

constexpr int kN = 128 * 1024;
constexpr int kPasses = 64;  // số lượt quét mảng trong một lần đo, để một mẫu dài cỡ 1 ms

void saxpy(float a, const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

}  // namespace

int main() {
    std::vector<float> x(kN, 1.0f), y(kN, 0.0f);

    bench::Options opt;
    opt.warmup = 50;
    opt.iters = 300;
    if (const char* core = std::getenv("BENCH_CORE")) opt.core = std::atoi(core);

    auto r = bench::run("saxpy", [&] {
        for (int p = 0; p < kPasses; ++p) {
            saxpy(1e-6f, x.data(), y.data(), kN);
            bench::do_not_optimize(y.data());
        }
    }, opt);

    bench::print(r);
    return 0;
}
