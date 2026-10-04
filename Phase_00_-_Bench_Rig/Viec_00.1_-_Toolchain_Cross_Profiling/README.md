# bench-kit

Giàn đo dùng chung cho các dự án EdgeBuild: harness C++, `bench_mode.sh`, `env_report.sh`, noise test.

## Toolchain (tuần 00, việc 00.1)

Máy: Ubuntu 22.04.5, kernel 6.8.0-138-generic. Cập nhật 2026-10-03.

| Nhóm | Công cụ | Phiên bản | Trạng thái | Nguồn |
|---|---|---|---|---|
| Compiler mặc định | `gcc` / `g++` | 11.4.0 | đã có | Ubuntu — không đổi, không `update-alternatives` |
| Compiler mặc định | `clang` | 14.0.0 | đã có | Ubuntu |
| Compiler mới | `clang-18` / `clang++-18` | 18.1.8 | mới cài | apt.llvm.org (`/etc/apt/sources.list.d/llvm-18.list`) |
| Build | `cmake` | 3.31.12 | mới cài | apt.kitware.com, ghim 3.31.x (`/etc/apt/preferences.d/cmake-3x`) |
| Build | `ninja` | 1.10.1 | đã có | Ubuntu |
| Build | `ccache` | 4.5.1 | đã có | Ubuntu |
| Cross | `aarch64-linux-gnu-g++` | 11.4.0 | mới cài | Ubuntu |
| Cross | `qemu-aarch64-static` | 6.2.0 | mới cài | Ubuntu (`qemu-user-static`) |
| Thiết bị | `adb` | 1.0.41 | mới cài | Ubuntu |
| Profiling | `perf` | 6.8.12 | đã có | `kernel.perf_event_paranoid = 1` (`/etc/sysctl.d/99-perf-bench.conf`) |
| Profiling | `valgrind` | 3.18.1 | đã có | Ubuntu |
| Profiling | `heaptrack` | 1.3.0 | mới cài | Ubuntu |
| Profiling | Tracy | **v0.14.1** | mới build | source, `~/workspace/tools/src/tracy` |
| Profiling | Nsight Systems | 2026.3.2 | đã có | NVIDIA |

Ghi chú:

- **Chọn Clang 18 thay cho GCC 13.** PPA `ubuntu-toolchain-r/test` kéo theo `libstdc++6` và `libgcc-s1` bản snapshot GCC 16 cho cả hệ thống (kể cả i386), ảnh hưởng mọi chương trình C++ đang cài (ROS 2, CUDA). Clang 18 từ apt.llvm.org không đụng runtime hệ thống. Clang 18 dùng header libstdc++ 11.
- **CMake ghim ở 3.31.x.** Kitware mặc định cho 4.x, bản này bỏ tương thích `cmake_minimum_required` < 3.5 và có thể làm hỏng build các package cũ (ROS 2). Bỏ ghim: xóa `/etc/apt/preferences.d/cmake-3x`.
- **Tracy ghim ở tag v0.14.1.** Client (header nhúng vào project) và server (`tracy-profiler`, `tracy-capture`) phải cùng tag, lệch phiên bản protocol thì không kết nối được. Project nào dùng Tracy thì lấy đúng tag này.

## Build

Compiler chọn theo preset, không đổi compiler mặc định của hệ thống.

```sh
cmake --preset x86_64-clang18 && cmake --build --preset x86_64-clang18
cmake --preset x86_64-gcc11   && cmake --build --preset x86_64-gcc11
cmake --preset aarch64        && cmake --build --preset aarch64

./build/x86_64-clang18/hello                                         # hello from x86_64 (clang 18.1.8)
qemu-aarch64-static -L /usr/aarch64-linux-gnu ./build/aarch64/hello  # hello from aarch64 (gcc 11.4.0)
```

Tracy:

```sh
~/workspace/tools/src/tracy/profiler/build/tracy-profiler   # GUI
~/workspace/tools/src/tracy/capture/build/tracy-capture -o out.tracy
```
