// bench.h — harness đo tối thiểu (tuần 00). Tuần 01 bổ sung p99, peak RSS, xuất JSON.
//
//   auto r = bench::run("saxpy", [&] { ... }, {.warmup = 50, .iters = 300, .core = 2});
//   r.p50_ns  // trung vị thời gian một lần gọi fn, tính bằng nano giây
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <sched.h>

namespace bench {

struct Options {
    int warmup = 20;  // số lần gọi fn bỏ đi trước khi đo (làm nóng cache, branch predictor, tần số)
    int iters = 100;  // số lần gọi fn được đo
    int core = -1;    // >= 0: pin thread vào core này; -1: giữ nguyên affinity (ví dụ do taskset đặt)
};

struct Result {
    std::string name;
    double p50_ns = 0;
    double min_ns = 0;
    double max_ns = 0;
    int iters = 0;
    int cpu = -1;  // CPU thread đang chạy lúc đo xong (sched_getcpu)
};

// Chặn compiler xóa một phép tính vì "không ai dùng kết quả".
template <class T>
inline void do_not_optimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

inline bool pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

template <class Fn>
Result run(std::string name, Fn&& fn, Options opt = {}) {
    if (opt.core >= 0 && !pin_to_core(opt.core)) {
        std::perror("bench: sched_setaffinity");
        std::exit(1);
    }

    for (int i = 0; i < opt.warmup; ++i) fn();

    using clock = std::chrono::steady_clock;
    std::vector<double> ns(static_cast<size_t>(opt.iters));
    for (int i = 0; i < opt.iters; ++i) {
        auto t0 = clock::now();
        fn();
        auto t1 = clock::now();
        ns[static_cast<size_t>(i)] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }

    std::sort(ns.begin(), ns.end());
    size_t n = ns.size();
    Result r;
    r.name = std::move(name);
    r.p50_ns = (n % 2) ? ns[n / 2] : 0.5 * (ns[n / 2 - 1] + ns[n / 2]);
    r.min_ns = ns.front();
    r.max_ns = ns.back();
    r.iters = opt.iters;
    r.cpu = sched_getcpu();
    return r;
}

inline void print(const Result& r) {
    std::printf("%s p50_ns=%.0f min_ns=%.0f max_ns=%.0f iters=%d cpu=%d\n", r.name.c_str(),
                r.p50_ns, r.min_ns, r.max_ns, r.iters, r.cpu);
}

}  // namespace bench
