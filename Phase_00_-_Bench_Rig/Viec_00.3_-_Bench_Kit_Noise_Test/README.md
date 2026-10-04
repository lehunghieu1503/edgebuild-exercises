# Việc 00.3 — `bench-kit`: bench mode, env report, P/E core, noise test

Máy: Core Ultra 9 275HX (8 P-core + 16 E-core, không Hyper-Threading), Ubuntu 22.04.5, kernel 6.8.0-138-generic. Số đo ngày 2026-10-04, cắm điện, Clang 18.1.8, `-O3 -march=x86-64-v3`.

## Chạy lại

```sh
./scripts/bench_mode.sh on            # governor performance, tắt turbo (cần sudo)
cmake --preset default                # cấu hình một lần
cmake --build --preset noise          # build saxpy rồi chạy noise test (thay cho `make noise`)
cmake --build build --target env-report | grep -v '^\[' | python3 -m json.tool
cmake --build build --target cores-check
./scripts/bench_mode.sh off           # trả máy về trạng thái cũ
```

## Thành phần

| File | Vai trò |
|---|---|
| `scripts/bench_mode.sh on\|off\|status` | bật/tắt chế độ đo; lưu trạng thái cũ ở `~/.local/state/bench-kit/bench_mode.state`; từ chối khi chạy pin |
| `scripts/env_report.sh` | môi trường đo ra JSON |
| `scripts/cores_check.sh` | xác định P/E core và đối chiếu `cores.env` |
| `scripts/noise.sh` | chạy `saxpy` 10 process liên tiếp, tính độ nhiễu, exit ≠ 0 khi > 2 % |
| `cores.env` | `P_CORES=0-7`, `E_CORES=8-23`, `BENCH_CORE=2`, `BENCH_E_CORE=12` |
| `include/bench/bench.h` | `bench::run(name, fn, {warmup, iters, core})` trả p50 (ns) |
| `noise/saxpy.cpp` | micro-benchmark, dữ liệu 1 MiB vừa L2 |

Biến môi trường: `TURBO=0|1` (mặc định 0), `ALLOW_BATTERY=1`, `RUNS` (10), `THRESHOLD_PCT` (2), `BENCH_CORE`.

## Kết quả nghiệm thu

**Noise test** — `bench_mode.sh on` (governor `performance`, turbo tắt), cắm điện, pin CPU 2:

| Lần | min p50 (ns) | median p50 (ns) | max p50 (ns) | Noise |
|---|---|---|---|---|
| 1 | 1 168 344 | 1 168 872 | 1 170 139 | 0,15 % |
| 2 | 1 168 277 | 1 168 767 | 1 169 476 | 0,10 % |
| 3 | 1 167 560 | 1 168 257 | 1 170 134 | 0,22 % |

Đạt ngưỡng ≤ 2 % ba lần liên tiếp.

**P-core ≠ E-core** — cùng binary, turbo tắt:

| Core | p50 (ns) | Tần số | IPC |
|---|---|---|---|
| P-core, `taskset -c 2` | 1 168 800 | 2,69 GHz | 1,13 |
| E-core, `taskset -c 12` | 1 380 300 | 2,09 GHz | 1,23 |

P-core nhanh hơn 1,18 lần. Số instructions bằng nhau (1,249 tỷ); chênh lệch đến từ tần số, không phải IPC — E-core thậm chí IPC cao hơn một chút ở bài này.

**Env report** — `python3 -m json.tool` chấp nhận; có model CPU, P/E core, governor, `no_turbo`, nguồn, nhiệt từng zone, kernel, compiler + flags, commit, load average.

## Chọn chế độ turbo: tắt

| Chế độ | p50 trên CPU 2 | Noise 10 lần |
|---|---|---|
| governor `performance`, turbo tắt | 1,169 ms | 0,10–0,22 % |
| governor `performance`, turbo bật | 0,600 ms | 9,22 % |
| ngoài chế độ đo (`powersave`, turbo bật) | — | 3,57 % |

Turbo bật nhanh gần gấp đôi nhưng không đạt ngưỡng 2 %, nên mọi số đo dùng `TURBO=0`. Số tuyệt đối vì vậy thấp hơn máy chạy thực tế; so sánh tương đối thì ổn định.

## Bẫy gặp thật trên máy này

- **Không có `/sys/devices/cpu_core` và `cpu_atom`.** Kernel 6.8 chạy PMU của Arrow Lake ở chế độ "generic architected perfmon", không tách hybrid. `cores_check.sh` suy ra từ cache L2: P-core có L2 riêng 3 MiB, E-core dùng chung L2 4 MiB theo cụm 4. Hệ quả cho mục "Sự cố của tuần": `perf stat` không có dòng `cpu_atom/cycles/`, phải nhìn `cpu-migrations` và tần số thay thế.
- **Cột `MAXMHZ` của `lscpu` không dùng được** để phân biệt: nó báo 6600–6800 cho P và 6500 cho E, không phải tần số thật.
- **`cat ... | grep -q` dưới `set -o pipefail` cho kết quả ngẫu nhiên.** `grep -q` thoát sớm, `cat` nhận SIGPIPE, cả pipeline thành lỗi. Bản đầu của `bench_mode.sh status` vì thế có lúc báo "đang ở chế độ đo" dù governor là `powersave`. Đã đổi sang `grep` đọc thẳng file.
- **`power-profiles-daemon` đang chạy.** Nó để governor `powersave` kèm EPP `performance`. Đổi profile trong GNOME lúc đang đo có thể ghi đè governor; `bench_mode.sh status` sẽ phát hiện.
- **Chưa thử được nhánh "từ chối khi chạy pin"** vì máy cắm điện suốt lúc làm; cần rút sạc rồi chạy `bench_mode.sh on` để xác nhận exit ≠ 0.
