cmake_minimum_required(VERSION 3.28)

# nvcc không nằm trong PATH mặc định; phải chỉ trước khi project() bật ngôn ngữ CUDA
if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

# Driver 595.91.07 chỉ JIT được PTX tới CUDA 13.2; nvcc 13.4 phải sinh SASS thẳng cho GPU đích.
# "120-real" = chỉ mã máy sm_120, không kèm PTX.
set(CMAKE_CUDA_ARCHITECTURES "120-real" CACHE STRING "GPU đích của nvcc")

project(viec_00_2 LANGUAGES CXX CUDA)

set(VENV "$ENV{HOME}/venvs/edge-export" CACHE PATH "venv export")
set(PY "${VENV}/bin/python")
set(SAMPLE_ONNX "${CMAKE_BINARY_DIR}/sample.onnx")
find_program(TRTEXEC trtexec HINTS /usr/src/tensorrt/bin REQUIRED)

add_executable(hello_cuda hello/hello_cuda.cu)

add_custom_command(
  OUTPUT "${SAMPLE_ONNX}"
  COMMAND "${PY}" "${CMAKE_SOURCE_DIR}/tools/run_ort_sample.py" --onnx "${SAMPLE_ONNX}"
  DEPENDS tools/run_ort_sample.py
  USES_TERMINAL VERBATIM)

# USES_TERMINAL: in output trực tiếp và buộc Ninja chạy lần lượt từng target
add_custom_target(cuda COMMAND hello_cuda USES_TERMINAL VERBATIM)
# có COMMAND riêng nên luôn chạy lại ORT; --reuse: không export lại nếu sample.onnx đã có
add_custom_target(ort
  COMMAND "${PY}" "${CMAKE_SOURCE_DIR}/tools/run_ort_sample.py" --onnx "${SAMPLE_ONNX}" --reuse
  DEPENDS "${SAMPLE_ONNX}" USES_TERMINAL VERBATIM)
add_custom_target(trt
  COMMAND "${TRTEXEC}" "--onnx=${SAMPLE_ONNX}" "--saveEngine=${CMAKE_BINARY_DIR}/sample.engine"
  DEPENDS "${SAMPLE_ONNX}" USES_TERMINAL VERBATIM)
add_custom_target(openvino
  COMMAND "${PY}" -c "import openvino as ov; print(ov.Core().available_devices)"
  USES_TERMINAL VERBATIM)
# chạy sample.onnx thật trên từng thiết bị, không chỉ liệt kê
add_custom_target(ov-run
  COMMAND "${PY}" "${CMAKE_SOURCE_DIR}/tools/run_ov_devices.py" "${SAMPLE_ONNX}"
  DEPENDS "${SAMPLE_ONNX}" USES_TERMINAL VERBATIM)
add_custom_target(torch
  COMMAND "${PY}" -c "import torch; print(torch.ones(4, device='cuda').sum().item())"
  USES_TERMINAL VERBATIM)
# bỏ PYTHONPATH để gói ROS 2 của hệ thống không lọt vào file lock
add_custom_target(lock
  COMMAND "${CMAKE_COMMAND}" -E env --unset=PYTHONPATH "${VENV}/bin/pip" freeze > "${CMAKE_SOURCE_DIR}/requirements.lock"
  VERBATIM)

add_custom_target(check)
add_dependencies(check cuda ort trt openvino ov-run torch)
