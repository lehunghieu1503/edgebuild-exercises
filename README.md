# EdgeBuild — bài tập và code minh họa lý thuyết

Code đi kèm lộ trình tự học **EdgeBuild**: chạy và tối ưu mô hình AI trên thiết bị edge, nhìn từ góc một lập trình viên C++ nhúng. Lộ trình chia theo phase, mỗi phase gồm vài tuần, mỗi tuần có ba phần: lý thuyết, bài tập, project.

Repo này chứa hai phần đầu:

- **`theory/`** — code minh họa từng mục lý thuyết, để thấy khái niệm chạy ra con số thật.
- **`exercises/`** — bài giải phần bài tập của tuần.

Phần project của mỗi tuần nằm ở repo riêng của từng dự án.

## Cấu trúc

```
Phase_00_-_Bench_Rig/                       dựng giàn đo: toolchain, GPU/NPU, noise test
Phase_01_-_AI_Primer_and_First_Detector/
└── week_01/
    ├── theory/
    │   ├── 01_tensor/{Cpp,Python}
    │   ├── 02_layer_weight/{Cpp,Python}
    │   └── ...
    └── exercises/
```

## Code minh họa lý thuyết

Mỗi mục lý thuyết có ba bản cùng làm một việc:

| Bản | File | Dùng gì |
|---|---|---|
| C++ | `Cpp/<tên>.cpp` | C++17 thuần, không thư viện ngoài |
| Python thuần | `Python/<tên>_non_pytorch.py` | chỉ thư viện chuẩn, cấu trúc giống bản C++ |
| PyTorch | `Python/<tên>_pytorch.py` | cùng việc đó viết bằng PyTorch |

Từ mục 03 trở đi, bản C++ và bản Python thuần in ra giống nhau từng ký tự, và bản PyTorch ra cùng các con số. Nhờ vậy bạn đối chiếu được vòng lặp tự viết với thư viện thật.

### Tuần 01 — Mạng nơ-ron từ số 0

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

Các thư mục trong `Phase_00_-_Bench_Rig/` có README riêng mô tả cách chạy.

## Giấy phép

MIT — xem [LICENSE](LICENSE).
