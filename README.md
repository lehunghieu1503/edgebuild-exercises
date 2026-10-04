# EdgeBuild — code minh họa lý thuyết và bài giải bài tập

Code đi kèm loạt bài **Edge AI** trên [tinker log](https://tinkerlog.io.vn/topics/edge-ai/): chạy và tối ưu mô hình AI trên thiết bị edge, nhìn từ góc một lập trình viên C++ nhúng.

Mỗi bài viết có phần lý thuyết và phần bài tập. Repo này chứa code cho cả hai:

- **`theory/`** — code minh họa từng mục lý thuyết, để thấy khái niệm chạy ra con số thật.
- **`exercises/`** — bài giải phần bài tập.

Phần project của mỗi bài nằm ở repo riêng của từng dự án. Riêng bài 00 chưa thuộc dự án nào, nên phần dựng giàn đo của nó đặt ngay trong repo này.

## Bài viết và trạng thái

| Bài | Bài viết | Thư mục | Code lý thuyết | Bài giải |
|---|---|---|---|---|
| 00 | [Dựng máy, giàn đo benchmark và khung CI](https://tinkerlog.io.vn/2026/10/04/edge-ai-00-bench-rig-toolchain-ci/) | `Phase_00_-_Bench_Rig/` | chưa có | chưa có (00.A, 00.B, 00.C) |
| 01 | [Mạng nơ-ron từ số 0: tensor, CNN, train và inference](https://tinkerlog.io.vn/2026/10/04/edge-ai-01-neural-networks-from-zero/) | `Phase_01_-_AI_Primer_and_First_Detector/week_01/` | đủ 7 mục | chưa có (01.A, 01.B, 01.C) |

## Cấu trúc

```
Phase_00_-_Bench_Rig/                          bài 00: dựng giàn đo
├── Viec_00.1_-_Toolchain_Cross_Profiling/     toolchain host, cross-compile, profiling
├── Viec_00.2_-_GPU_NPU_Venv_Export/           CUDA, TensorRT, OpenVINO, venv export
└── Viec_00.3_-_Bench_Kit_Noise_Test/          bench-kit: bench mode, env report, noise test
Phase_01_-_AI_Primer_and_First_Detector/
└── week_01/                                   bài 01
    ├── theory/
    │   ├── 01_tensor/{Cpp,Python}
    │   ├── 02_layer_weight/{Cpp,Python}
    │   ├── ...
    │   └── 07_train_val_test/{Cpp,Python}
    └── exercises/
```

Mỗi thư mục trong `Phase_00_-_Bench_Rig/` có README riêng: cách chạy lại, phiên bản đã cài và số đo trên máy của tác giả.

## Code minh họa lý thuyết

Mỗi mục lý thuyết có ba bản cùng làm một việc:

| Bản | File | Dùng gì |
|---|---|---|
| C++ | `Cpp/<tên>.cpp` | C++17 thuần, không thư viện ngoài |
| Python thuần | `Python/<tên>_non_pytorch.py` | chỉ thư viện chuẩn, cấu trúc giống bản C++ |
| PyTorch | `Python/<tên>_pytorch.py` | cùng việc đó viết bằng PyTorch |

Riêng `01_tensor` đặt tên hai file Python là `non_pytorch.py` và `test_pytorch.py`.

Từ mục 03 trở đi, bản C++ và bản Python thuần in ra giống nhau từng ký tự, và bản PyTorch ra cùng các con số. Nhờ vậy có thể đối chiếu vòng lặp tự viết với thư viện thật.

### Bài 01 — Mạng nơ-ron từ số 0

Số thứ tự thư mục trùng với số mục lý thuyết và số hình trong bài viết: `03_convolution` là mục 3, Hình 1.3.

| Thư mục | Khái niệm | Chạy ra cho thấy |
|---|---|---|
| `01_tensor` | tensor, shape, strides, NCHW | một tensor là buffer 1 chiều kèm metadata; đổi HWC sang NCHW |
| `02_layer_weight` | lớp, trọng số, file model | PyTorch train và ghi file model, C++ đọc lại và chạy đúng kết quả |
| `03_convolution` | kernel, stride, padding | kernel 3×3 trượt trên ảnh; công thức kích thước đầu ra và số trọng số |
| `04_companion_layers` | ReLU, pooling, BatchNorm, Linear | từng lớp trên số nhỏ, rồi ghép thành một CNN |
| `05_train_vs_inference` | loss, gradient, backpropagation | backpropagation viết tay ra đúng loss như `loss.backward()` |
| `06_logits_top1_top5` | logits, softmax, top-1, top-5 | đọc logits và tính accuracy trên một batch |
| `07_train_val_test` | ba tập dữ liệu, overfitting | loss train giảm trong khi loss val tăng; val chọn epoch, test đo một lần |

## Build và chạy

Cần: `g++` hỗ trợ C++17, CMake ≥ 3.15, Python ≥ 3.10. Các file `*_pytorch.py` cần thêm `torch` và `torchvision`.

Ví dụ với mục 05, chạy từ `Phase_01_-_AI_Primer_and_First_Detector/week_01/theory/`:

```sh
cmake -S 05_train_vs_inference/Cpp -B 05_train_vs_inference/Cpp/build
cmake --build 05_train_vs_inference/Cpp/build
./05_train_vs_inference/Cpp/build/train_05

python3 05_train_vs_inference/Python/train_05_non_pytorch.py
python3 05_train_vs_inference/Python/train_05_pytorch.py
```

Các mục khác chạy cùng cách: đổi tên thư mục và tên file tương ứng.

## Giấy phép

MIT — xem [LICENSE](LICENSE).
