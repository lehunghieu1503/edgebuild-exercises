// hello_cuda.cu — chương trình CUDA đầu tiên: cộng hai mảng trên GPU rồi so với CPU.
//
// Đuôi .cu báo cho nvcc biết file này có code chạy trên GPU. nvcc tách file làm hai phần:
//   - phần "host"   (chạy trên CPU): hàm main() và mọi thứ C++ bình thường -> giao cho g++
//   - phần "device" (chạy trên GPU): các hàm có __global__                 -> nvcc tự biên dịch
//
// Hai từ cần nhớ suốt file:
//   host   = CPU + RAM thường
//   device = GPU + VRAM (bộ nhớ riêng của card)
// Hai bên có bộ nhớ TÁCH BIỆT, nên muốn GPU tính trên dữ liệu nào thì phải copy dữ liệu đó sang.

#include <cmath>    // std::fabs (trị tuyệt đối của float)
#include <cstdio>   // std::printf, std::fprintf
#include <vector>   // std::vector — mảng động phía CPU

#include <cuda_runtime.h>  // API của CUDA: cudaMalloc, cudaMemcpy, cudaGetDeviceProperties...

// Gần như mọi hàm CUDA đều trả về một mã lỗi kiểu cudaError_t thay vì ném exception.
// Nếu không kiểm tra, lỗi sẽ bị nuốt mất và chương trình chạy tiếp với kết quả rác.
// Macro này bọc một lời gọi CUDA: gọi xong, nếu không phải cudaSuccess thì in ra
// lời gọi nào lỗi, ở file/dòng nào, lỗi gì — rồi thoát main() với mã 1.
//
// Chi tiết cú pháp:
//   #call        -> biến đoạn code truyền vào thành chuỗi, để in ra đúng tên lời gọi
//   __FILE__, __LINE__ -> tên file và số dòng nơi macro được dùng
//   do { ... } while (0) -> mẹo để macro nhiều dòng hoạt động như MỘT câu lệnh
//                           (dùng được an toàn sau if/else, và bắt buộc có dấu ; ở cuối)
//   dấu \ cuối dòng -> nối dòng, vì một macro phải nằm trên một dòng logic
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                         __LINE__, cudaGetErrorString(err_));                     \
            return 1;                                                             \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------------------
// KERNEL: hàm chạy trên GPU.
//
// __global__ nghĩa là: hàm này chạy trên device, nhưng được gọi từ host. Bắt buộc trả về void.
//
// Điểm khác biệt lớn nhất so với code CPU: ở đây KHÔNG có vòng for.
// GPU chạy cùng một hàm này trên hàng trăm nghìn "thread" song song; mỗi thread xử lý
// đúng MỘT phần tử. Việc của mỗi thread là tự hỏi: "mình là thread số mấy?" rồi xử lý
// phần tử có chỉ số tương ứng.
//
// Các thread được xếp thành nhiều "block", mỗi block có số thread bằng nhau:
//
//   block 0            block 1            block 2
//   [t0 t1 ... t255]   [t0 t1 ... t255]   [t0 t1 ... t255]  ...
//
//   blockIdx.x  = mình ở block thứ mấy
//   blockDim.x  = mỗi block có bao nhiêu thread (ở đây là 256)
//   threadIdx.x = mình là thread thứ mấy TRONG block đó (0..255)
//
// Suy ra chỉ số toàn cục: i = blockIdx.x * blockDim.x + threadIdx.x
// Ví dụ block 2, thread 5 -> i = 2 * 256 + 5 = 517 -> thread này tính out[517].
//
// Ba con trỏ a, b, out đều trỏ vào VRAM (bộ nhớ GPU), không phải RAM.
// ---------------------------------------------------------------------------------------
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Số thread được tạo ra là bội của 256 nên có thể nhiều hơn n một chút.
    // Các thread "thừa" phải bỏ qua, nếu không sẽ ghi ra ngoài mảng.
    if (i < n) out[i] = a[i] + b[i];
}

int main() {
    // --- 1. Hỏi GPU đang dùng là card nào -------------------------------------------------
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));  // lấy số thứ tự GPU hiện tại (máy một card rời thì là 0)

    cudaDeviceProp prop{};            // struct chứa thông tin card; {} = khởi tạo toàn số 0
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // prop.major/minor là "compute capability" — thế hệ kiến trúc của GPU.
    // RTX 5060 (Blackwell) là 12.0, viết gọn là sm_120. Con số này quyết định
    // code phải biên dịch cho đích nào (xem CMAKE_CUDA_ARCHITECTURES trong CMakeLists.txt).
    // totalGlobalMem tính bằng byte, chia 1024*1024 ra MiB.
    std::printf("GPU: %s (sm_%d%d, %.0f MiB)\n", prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / (1024.0 * 1024.0));

    // --- 2. Chuẩn bị dữ liệu trên CPU -----------------------------------------------------
    const int n = 1 << 20;  // 1 dịch trái 20 bit = 2^20 = 1.048.576 phần tử

    // a, b: hai mảng đầu vào. cpu: đáp án tính bằng CPU. gpu: nơi nhận kết quả từ GPU.
    std::vector<float> a(n), b(n), cpu(n), gpu(n);
    for (int i = 0; i < n; ++i) {
        a[i] = 0.5f * static_cast<float>(i);
        b[i] = 1.0f / static_cast<float>(i + 1);  // i + 1 để tránh chia cho 0
        cpu[i] = a[i] + b[i];                     // đáp án chuẩn để lát nữa đối chiếu
    }

    // --- 3. Cấp phát VRAM và copy dữ liệu sang GPU ----------------------------------------
    const size_t bytes = n * sizeof(float);  // 1.048.576 * 4 byte = khoảng 4 MiB mỗi mảng

    // Quy ước đặt tên: tiền tố d_ = con trỏ phía device (GPU).
    // Các con trỏ này KHÔNG dùng được trên CPU — viết d_a[0] trong main() sẽ crash,
    // vì địa chỉ đó nằm trong VRAM chứ không phải RAM.
    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));    // giống malloc, nhưng xin bộ nhớ trên GPU
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // cudaMemcpy(đích, nguồn, số byte, chiều copy).
    // a.data() là con trỏ tới vùng nhớ bên trong vector (phía CPU).
    // HostToDevice = từ RAM sang VRAM.
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    // --- 4. Chạy kernel trên GPU ----------------------------------------------------------
    const int threads = 256;  // số thread mỗi block (blockDim.x trong kernel)

    // Số block cần để phủ hết n phần tử, làm tròn LÊN.
    // (n + 255) / 256 là mẹo chia nguyên làm tròn lên: n = 1000 -> 4 block (1024 thread).
    const int blocks = (n + threads - 1) / threads;

    // Cú pháp riêng của CUDA: tên_kernel<<<số block, số thread mỗi block>>>(tham số).
    // Dòng này tạo ra blocks * threads = 1.048.576 thread GPU, mỗi thread chạy add_kernel một lần.
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);

    // Lời gọi kernel không trả về mã lỗi, nên phải hỏi riêng: lần gọi vừa rồi có lỗi không?
    CUDA_CHECK(cudaGetLastError());

    // QUAN TRỌNG: gọi kernel là BẤT ĐỒNG BỘ — CPU chỉ "gửi lệnh" rồi chạy tiếp ngay,
    // trong khi GPU có thể chưa tính xong. cudaDeviceSynchronize() bắt CPU đứng chờ
    // tới khi GPU hoàn tất. Khi đo thời gian mà quên dòng này thì số đo sẽ sai (nhanh ảo).
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 5. Copy kết quả về CPU -----------------------------------------------------------
    // DeviceToHost = từ VRAM về RAM. gpu.data() là đích phía CPU.
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // Trả VRAM lại cho GPU. Mỗi cudaMalloc phải đi với một cudaFree, giống malloc/free.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    // --- 6. So kết quả GPU với CPU --------------------------------------------------------
    int mismatches = 0;      // số phần tử lệch
    float max_diff = 0.0f;   // độ lệch lớn nhất
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(cpu[i] - gpu[i]);
        if (diff > max_diff) max_diff = diff;
        // So bằng tuyệt đối (diff != 0) được ở đây vì phép cộng một float theo chuẩn IEEE-754
        // cho kết quả giống hệt nhau trên CPU và GPU. Với phép tính phức tạp hơn (nhân rồi cộng,
        // tổng nhiều phần tử...) hai bên có thể lệch vài bit cuối, khi đó phải so theo sai số cho phép.
        if (diff != 0.0f) ++mismatches;
    }
    std::printf("n=%d mismatches=%d max_abs_diff=%g\n", n, mismatches, max_diff);
    std::printf("%s\n", mismatches == 0 ? "PASS" : "FAIL");

    // Mã thoát 0 = thành công, khác 0 = thất bại — để CMake và CI biết kết quả.
    return mismatches == 0 ? 0 : 1;
}
