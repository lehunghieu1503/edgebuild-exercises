# Việc 00.2 — GPU, NPU và venv export

Máy: Ubuntu 22.04.5, kernel 6.8.0-138-generic, Core Ultra 9 275HX, RTX 5060 Laptop (8 GB, `sm_120`). Cập nhật 2026-10-03.

## Chạy lại

```sh
cmake --preset default            # cấu hình một lần (Ninja, thư mục build/)
cmake --build --preset check      # chạy hết các mục nghiệm thu bên dưới

cmake --build build --target cuda       # hello_cuda: in tên GPU và PASS
cmake --build build --target ort        # export resnet18 -> build/sample.onnx, chạy bằng ONNX Runtime, in top-1
cmake --build build --target trt        # trtexec build engine từ sample.onnx
cmake --build build --target openvino   # liệt kê thiết bị OpenVINO
cmake --build build --target ov-run     # chạy sample.onnx thật trên từng thiết bị OpenVINO
cmake --build build --target torch      # một phép tính thật trên GPU bằng PyTorch
cmake --build build --target lock       # ghi lại requirements.lock
```

Mọi file sinh ra (`hello_cuda`, `sample.onnx`, `sample.engine`) nằm trong `build/`.

Venv: `~/venvs/edge-export` (Python 3.10.12). Dựng lại: `python3 -m venv ~/venvs/edge-export && ~/venvs/edge-export/bin/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130`.

## Bảng trạng thái cài đặt

| Nhóm | Thành phần | Phiên bản | Trạng thái | Ghi chú |
|---|---|---|---|---|
| GPU | driver NVIDIA | 595.91.07 (open) | đã có | hỗ trợ CUDA tới 13.2 (theo `nvidia-smi`) |
| GPU | CUDA Toolkit | 13.4.2 (`nvcc` 13.4.92) | đã có | `/usr/local/cuda` -> `cuda-13.4`; bản 13.3 cũng có sẵn |
| GPU | cuDNN | 9.27.0.42 (cuda 13) | mới cài | apt `cudnn9-cuda-13-4` |
| GPU | TensorRT | 11.3.0.99 (+cuda13.4) | mới cài | apt `tensorrt`; `trtexec` ở `/usr/bin/trtexec` |
| Intel | OpenVINO | 2026.4.1 | mới cài | pip, trong venv |
| Intel | NPU driver user-space | 1.26.0 | mới cài | `intel-level-zero-npu`, `intel-driver-compiler-npu` |
| Intel | Level Zero loader | 1.24.2 | mới cài | deb từ GitHub oneapi-src/level-zero |
| Intel | firmware NPU | 20251016 (ud202544) | đã có | theo kernel; không cài gói `intel-fw-npu` vì trùng phiên bản |
| Intel | iGPU (Arrow Lake) cho OpenVINO | — | hoãn | chưa cài `intel-opencl-icd`; tuần 10 mới cần |
| Python | PyTorch | 2.14.1+cu130 | mới cài | wheel có kernel `sm_120` |
| Python | torchvision | 0.29.1+cu130 | mới cài | |
| Python | onnx / onnxruntime / onnxsim | 1.23.1 / 1.23.2 / 0.7.3 | mới cài | onnxruntime bản CPU |

## Kết quả nghiệm thu

| Lệnh | Kết quả |
|---|---|
| `./build/hello_cuda` | `GPU: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120, 7683 MiB)`, `PASS` |
| `trtexec --onnx=sample.onnx` | engine build xong trong 7,4 s, `PASSED`, exit 0 |
| `ls /dev/accel` | `accel0` (`root:render`, `0660`) |
| `ov.Core().available_devices` | `['CPU', 'GPU', 'NPU']` |
| `torch.ones(4, device='cuda').sum().item()` | `4.0` |
| `python tools/run_ort_sample.py` | top-1 `258 'Samoyed'` p=0,8846; khớp PyTorch, lệch logit tối đa 1,7e-05 |

## Bẫy gặp thật trên máy này

- **Toolkit mới hơn driver.** `nvcc` 13.4 mặc định sinh PTX để driver JIT lúc chạy; driver 595.91.07 chỉ hiểu PTX tới CUDA 13.2 nên kernel lỗi `the provided PTX was compiled with an unsupported toolchain` (bản 13.3 cũng lỗi). Cách xử lý: build thẳng mã máy cho GPU đích bằng `-arch=sm_120` (`CMAKE_CUDA_ARCHITECTURES=120-real` trong `CMakeLists.txt`), không nâng driver.
- **`GPU` của OpenVINO ở đây là card NVIDIA, không phải iGPU Intel.** Plugin GPU đi qua OpenCL, mà máy chỉ có `nvidia.icd`. Số đo "OpenVINO GPU" hiện tại là RTX 5060; muốn đo iGPU phải cài `intel-opencl-icd`.
- **Driver NPU cho Ubuntu 22.04 dừng ở v1.26.0** (11/2025, kiểm chứng với OpenVINO 2025.3). Các bản sau chỉ build cho 24.04/26.04. OpenVINO 2026.4.1 vẫn compile và chạy được resnet18 trên NPU (top-1 khớp CPU, lệch logit 4,3e-03), nhưng model phức tạp hơn có thể vấp vì lệch phiên bản.
- **Quyền trên `/dev/accel/accel0`.** Ban đầu là `root:root 0600`. Gói driver thêm udev rule chuyển sang nhóm `render`; user đã được thêm vào `render` nhưng phải đăng nhập lại mới có hiệu lực. Trước khi đăng nhập lại: `sg render -c "cmake --build build --target ov-run"`.
- **ROS 2 lọt vào venv qua `PYTHONPATH`.** Shell đã source ROS nên `pip` trong venv nhìn thấy gói ROS và cảnh báo thiếu dependency. target `lock` bỏ `PYTHONPATH` trước khi `pip freeze`.
