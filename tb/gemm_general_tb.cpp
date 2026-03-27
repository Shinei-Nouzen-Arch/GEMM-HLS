//=============================================================================
// gemm_general_tb.cpp - GEMM_General 测试激励文件 (优化版)
//
// 功能: 验证通用矩阵乘法 IP 核心的正确性
// 测试: 随机矩阵测试、边界测试、不同维度测试
//
// 更新: 适配 512-bit AXI 接口
//=============================================================================

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <chrono>
#include <vector>
#include "types.h"

//=============================================================================
// 性能统计结构
//=============================================================================
struct PerformanceStats {
    std::string test_name;
    int M, N, K;
    double hw_time_ms;
    double gflops;
    
    PerformanceStats(const std::string& name, int m, int n, int k, double time_ms)
        : test_name(name), M(m), N(n), K(k), hw_time_ms(time_ms) {
        // 计算 GFLOPS: 矩阵乘法需要 2*M*N*K 次浮点运算
        double flops = 2.0 * M * N * K;
        gflops = (flops / 1e9) / (time_ms / 1000.0);
    }
};

//=============================================================================
// 对齐内存分配 - 512-bit AXI 接口需要 64 字节对齐
//=============================================================================
constexpr size_t AXI_ALIGNMENT = 64;  // 512-bit = 64 bytes

// HLS 接口 depth 设置 (来自 gemm_general.cpp 中的 pragma)
// depth=4096 表示 4096 个 512-bit 字 = 4096 * 16 floats = 65536 floats
// Co-simulation wrapper 会尝试从 testbench 缓冲区复制 depth 个元素
// 因此 testbench 必须分配至少 depth * 16 个 float 的内存
constexpr size_t HLS_INTERFACE_DEPTH = 4096;  // 512-bit words
constexpr size_t FLOATS_PER_512BIT = 16;      // 512 / 32 = 16 floats
constexpr size_t MIN_BUFFER_FLOATS = HLS_INTERFACE_DEPTH * FLOATS_PER_512BIT;  // 65536 floats

// 分配对齐内存 - 确保满足 HLS co-simulation 的 depth 要求
// 关键: 分配的内存必须 >= HLS 接口 pragma 中指定的 depth
// 否则 co-simulation wrapper 复制数据时会发生 SIGSEGV
inline float* aligned_alloc_float(size_t num_elements) {
    // 确保分配的元素数量至少满足 HLS depth 要求
    size_t actual_elements = (num_elements > MIN_BUFFER_FLOATS) ? num_elements : MIN_BUFFER_FLOATS;
    size_t num_bytes = actual_elements * sizeof(float);
    // 向上对齐到64字节边界
    size_t aligned_bytes = ((num_bytes + AXI_ALIGNMENT - 1) / AXI_ALIGNMENT) * AXI_ALIGNMENT;
    
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(aligned_bytes, AXI_ALIGNMENT);
#else
    if (posix_memalign(&ptr, AXI_ALIGNMENT, aligned_bytes) != 0) {
        ptr = nullptr;
    }
#endif
    // 初始化为零，避免未初始化内存问题
    if (ptr) {
        memset(ptr, 0, aligned_bytes);
    }
    return static_cast<float*>(ptr);
}

// 释放对齐内存
inline void aligned_free_float(float* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

//=============================================================================
// 被测函数声明 (512-bit 接口)
//=============================================================================
extern "C" {
    void GEMM_General(
        uint512_t* A,
        uint512_t* B,
        uint512_t* C,
        int M,
        int N_mat,
        int K
    );
}

//=============================================================================
// 软件参考实现 - 通用矩阵乘法
//=============================================================================
void gemm_reference(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

//=============================================================================
// 辅助函数
//=============================================================================

// 生成随机浮点数
float random_float(float min_val = -1.0f, float max_val = 1.0f) {
    return min_val + static_cast<float>(rand()) / RAND_MAX * (max_val - min_val);
}

// 初始化随机矩阵
void init_random_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = random_float();
    }
}

// 初始化零矩阵
void init_zero_matrix(float* mat, int rows, int cols) {
    memset(mat, 0, rows * cols * sizeof(float));
}

// 初始化单位矩阵
void init_identity_matrix(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mat[i * size + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

// 初始化常数矩阵
void init_constant_matrix(float* mat, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = val;
    }
}

// 比较两个矩阵
bool compare_matrices(
    const float* C_ref,
    const float* C_dut,
    int M,
    int N,
    float tolerance = 1e-3f
) {
    bool match = true;
    float max_error = 0;
    float max_rel_error = 0;
    int error_count = 0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            float diff = std::abs(C_ref[idx] - C_dut[idx]);
            float rel_error = diff / (std::abs(C_ref[idx]) + 1e-10f);

            if (diff > tolerance && rel_error > tolerance) {
                if (error_count < 10) {
                    std::cout << "Mismatch at [" << i << "][" << j << "]: "
                              << "ref=" << C_ref[idx] << ", dut=" << C_dut[idx]
                              << ", diff=" << diff << ", rel=" << rel_error << std::endl;
                }
                match = false;
                error_count++;
            }

            if (diff > max_error) max_error = diff;
            if (rel_error > max_rel_error) max_rel_error = rel_error;
        }
    }

    std::cout << "Max absolute error: " << max_error << std::endl;
    std::cout << "Max relative error: " << max_rel_error << std::endl;
    if (error_count > 0) {
        std::cout << "Total errors: " << error_count << " / " << M * N << std::endl;
    }

    return match;
}

// 打印矩阵 (部分)
void print_matrix_partial(const char* name, const float* mat, int M, int N, int max_rows = 4, int max_cols = 4) {
    std::cout << "Matrix " << name << " (" << M << "x" << N << "):" << std::endl;
    int rows_to_print = (M < max_rows) ? M : max_rows;
    int cols_to_print = (N < max_cols) ? N : max_cols;

    for (int i = 0; i < rows_to_print; i++) {
        for (int j = 0; j < cols_to_print; j++) {
            printf("%10.4f ", mat[i * N + j]);
        }
        if (cols_to_print < N) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows_to_print < M) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

//=============================================================================
// 测试用例
//=============================================================================

// 测试 1: 32x32 基本测试 (与脉动阵列相同维度)
bool test_32x32_basic(PerformanceStats& stats) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test 1: 32x32 Basic Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // 每个测试用例开始时重置随机种子，确保 C TB testing 和 post check 使用相同数据
    srand(42 + 1);  // 测试 1 使用种子 43

    const int M = 32, N = 32, K = 32;

    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(K * N);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_dut = aligned_alloc_float(M * N);

    init_random_matrix(A, M, K);
    init_random_matrix(B, K, N);
    init_zero_matrix(C_ref, M, N);
    init_zero_matrix(C_dut, M, N);

    gemm_reference(A, B, C_ref, M, N, K);
    
    // 测量硬件执行时间
    auto start = std::chrono::high_resolution_clock::now();
    GEMM_General(reinterpret_cast<uint512_t*>(A),
                 reinterpret_cast<uint512_t*>(B),
                 reinterpret_cast<uint512_t*>(C_dut),
                 M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double hw_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats = PerformanceStats("Test 1 (32x32)", M, N, K, hw_time_ms);

    print_matrix_partial("C_ref", C_ref, M, N);
    print_matrix_partial("C_dut", C_dut, M, N);

    bool pass = compare_matrices(C_ref, C_dut, M, N);
    std::cout << "HW Time: " << hw_time_ms << " ms, GFLOPS: " << stats.gflops << std::endl;
    std::cout << "Test 1: " << (pass ? "PASS" : "FAIL") << std::endl << std::endl;

    aligned_free_float(A); aligned_free_float(B); aligned_free_float(C_ref); aligned_free_float(C_dut);
    return pass;
}

// 测试 2: 64x64 测试 (2x2 分块)
bool test_64x64(PerformanceStats& stats) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test 2: 64x64 Test (2x2 tiles)" << std::endl;
    std::cout << "========================================" << std::endl;

    // 每个测试用例开始时重置随机种子，确保 C TB testing 和 post check 使用相同数据
    srand(42 + 2);  // 测试 2 使用种子 44

    const int M = 64, N = 64, K = 64;

    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(K * N);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_dut = aligned_alloc_float(M * N);

    init_random_matrix(A, M, K);
    init_random_matrix(B, K, N);
    init_zero_matrix(C_ref, M, N);
    init_zero_matrix(C_dut, M, N);

    gemm_reference(A, B, C_ref, M, N, K);
    
    auto start = std::chrono::high_resolution_clock::now();
    GEMM_General(reinterpret_cast<uint512_t*>(A),
                 reinterpret_cast<uint512_t*>(B),
                 reinterpret_cast<uint512_t*>(C_dut),
                 M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double hw_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats = PerformanceStats("Test 2 (64x64)", M, N, K, hw_time_ms);

    print_matrix_partial("C_ref", C_ref, M, N);
    print_matrix_partial("C_dut", C_dut, M, N);

    bool pass = compare_matrices(C_ref, C_dut, M, N);
    std::cout << "HW Time: " << hw_time_ms << " ms, GFLOPS: " << stats.gflops << std::endl;
    std::cout << "Test 2: " << (pass ? "PASS" : "FAIL") << std::endl << std::endl;

    aligned_free_float(A); aligned_free_float(B); aligned_free_float(C_ref); aligned_free_float(C_dut);
    return pass;
}

// 测试 3: 非方阵测试 (M != N != K)
bool test_non_square(PerformanceStats& stats) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test 3: Non-square Matrix Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // 每个测试用例开始时重置随机种子，确保 C TB testing 和 post check 使用相同数据
    srand(42 + 3);  // 测试 3 使用种子 45

    const int M = 64, N = 96, K = 128;

    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(K * N);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_dut = aligned_alloc_float(M * N);

    init_random_matrix(A, M, K);
    init_random_matrix(B, K, N);
    init_zero_matrix(C_ref, M, N);
    init_zero_matrix(C_dut, M, N);

    gemm_reference(A, B, C_ref, M, N, K);
    
    auto start = std::chrono::high_resolution_clock::now();
    GEMM_General(reinterpret_cast<uint512_t*>(A),
                 reinterpret_cast<uint512_t*>(B),
                 reinterpret_cast<uint512_t*>(C_dut),
                 M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double hw_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats = PerformanceStats("Test 3 (64x96x128)", M, N, K, hw_time_ms);

    print_matrix_partial("C_ref", C_ref, M, N);
    print_matrix_partial("C_dut", C_dut, M, N);

    bool pass = compare_matrices(C_ref, C_dut, M, N);
    std::cout << "HW Time: " << hw_time_ms << " ms, GFLOPS: " << stats.gflops << std::endl;
    std::cout << "Test 3: " << (pass ? "PASS" : "FAIL") << std::endl << std::endl;

    aligned_free_float(A); aligned_free_float(B); aligned_free_float(C_ref); aligned_free_float(C_dut);
    return pass;
}

// 测试 4: 单位矩阵测试
bool test_identity(PerformanceStats& stats) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test 4: Identity Matrix Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // 每个测试用例开始时重置随机种子，确保 C TB testing 和 post check 使用相同数据
    srand(42 + 4);  // 测试 4 使用种子 46

    const int M = 64, N = 64, K = 64;

    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(K * N);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_dut = aligned_alloc_float(M * N);

    init_random_matrix(A, M, K);
    init_identity_matrix(B, N);
    init_zero_matrix(C_ref, M, N);
    init_zero_matrix(C_dut, M, N);

    // A × I = A
    gemm_reference(A, B, C_ref, M, N, K);
    
    auto start = std::chrono::high_resolution_clock::now();
    GEMM_General(reinterpret_cast<uint512_t*>(A),
                 reinterpret_cast<uint512_t*>(B),
                 reinterpret_cast<uint512_t*>(C_dut),
                 M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double hw_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats = PerformanceStats("Test 4 (Identity)", M, N, K, hw_time_ms);

    bool pass = compare_matrices(C_ref, C_dut, M, N);
    std::cout << "HW Time: " << hw_time_ms << " ms, GFLOPS: " << stats.gflops << std::endl;
    std::cout << "Test 4: " << (pass ? "PASS" : "FAIL") << std::endl << std::endl;

    aligned_free_float(A); aligned_free_float(B); aligned_free_float(C_ref); aligned_free_float(C_dut);
    return pass;
}

// 测试 5: 常数矩阵测试
bool test_constant(PerformanceStats& stats) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test 5: Constant Matrix Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // 测试 5 使用常数矩阵，不需要随机种子，但为一致性仍设置
    srand(42 + 5);  // 测试 5 使用种子 47

    const int M = 64, N = 64, K = 64;

    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(K * N);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_dut = aligned_alloc_float(M * N);

    init_constant_matrix(A, M, K, 1.0f);
    init_constant_matrix(B, K, N, 1.0f);
    init_zero_matrix(C_ref, M, N);
    init_zero_matrix(C_dut, M, N);

    gemm_reference(A, B, C_ref, M, N, K);
    
    auto start = std::chrono::high_resolution_clock::now();
    GEMM_General(reinterpret_cast<uint512_t*>(A),
                 reinterpret_cast<uint512_t*>(B),
                 reinterpret_cast<uint512_t*>(C_dut),
                 M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double hw_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats = PerformanceStats("Test 5 (Constant)", M, N, K, hw_time_ms);

    // 验证结果: 每个元素应为 K
    bool pass = true;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(C_dut[i] - (float)K) > 1e-3f) {
            pass = false;
            break;
        }
    }

    std::cout << "Expected value: " << K << std::endl;
    std::cout << "Sample C_dut[0]: " << C_dut[0] << std::endl;
    std::cout << "HW Time: " << hw_time_ms << " ms, GFLOPS: " << stats.gflops << std::endl;
    std::cout << "Test 5: " << (pass ? "PASS" : "FAIL") << std::endl << std::endl;

    aligned_free_float(A); aligned_free_float(B); aligned_free_float(C_ref); aligned_free_float(C_dut);
    return pass;
}

// 测试 6: 大矩阵测试
bool test_large(PerformanceStats& stats) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test 6: Large Matrix Test (128x128)" << std::endl;
    std::cout << "========================================" << std::endl;

    // 每个测试用例开始时重置随机种子，确保 C TB testing 和 post check 使用相同数据
    srand(42 + 6);  // 测试 6 使用种子 48

    const int M = 128, N = 128, K = 128;

    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(K * N);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_dut = aligned_alloc_float(M * N);

    init_random_matrix(A, M, K);
    init_random_matrix(B, K, N);
    init_zero_matrix(C_ref, M, N);
    init_zero_matrix(C_dut, M, N);

    gemm_reference(A, B, C_ref, M, N, K);
    
    auto start = std::chrono::high_resolution_clock::now();
    GEMM_General(reinterpret_cast<uint512_t*>(A),
                 reinterpret_cast<uint512_t*>(B),
                 reinterpret_cast<uint512_t*>(C_dut),
                 M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double hw_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats = PerformanceStats("Test 6 (128x128)", M, N, K, hw_time_ms);

    bool pass = compare_matrices(C_ref, C_dut, M, N);
    std::cout << "HW Time: " << hw_time_ms << " ms, GFLOPS: " << stats.gflops << std::endl;
    std::cout << "Test 6: " << (pass ? "PASS" : "FAIL") << std::endl << std::endl;

    aligned_free_float(A); aligned_free_float(B); aligned_free_float(C_ref); aligned_free_float(C_dut);
    return pass;
}

// 测试 7: 边界值测试
bool test_boundary(PerformanceStats& stats) {
    std::cout << "========================================" << std::endl;
    std::cout << "Test 7: Boundary Values Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // 测试 7 使用常数矩阵，不需要随机种子，但为一致性仍设置
    srand(42 + 7);  // 测试 7 使用种子 49

    const int M = 64, N = 64, K = 64;

    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(K * N);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_dut = aligned_alloc_float(M * N);

    // 测试大数值
    init_constant_matrix(A, M, K, 100.0f);
    init_constant_matrix(B, K, N, 100.0f);
    init_zero_matrix(C_ref, M, N);
    init_zero_matrix(C_dut, M, N);

    gemm_reference(A, B, C_ref, M, N, K);
    
    auto start = std::chrono::high_resolution_clock::now();
    GEMM_General(reinterpret_cast<uint512_t*>(A),
                 reinterpret_cast<uint512_t*>(B),
                 reinterpret_cast<uint512_t*>(C_dut),
                 M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    double hw_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    bool pass1 = compare_matrices(C_ref, C_dut, M, N, 10.0f);

    // 测试小数值
    init_constant_matrix(A, M, K, 0.01f);
    init_constant_matrix(B, K, N, 0.01f);
    init_zero_matrix(C_ref, M, N);
    init_zero_matrix(C_dut, M, N);

    gemm_reference(A, B, C_ref, M, N, K);
    
    start = std::chrono::high_resolution_clock::now();
    GEMM_General(reinterpret_cast<uint512_t*>(A),
                 reinterpret_cast<uint512_t*>(B),
                 reinterpret_cast<uint512_t*>(C_dut),
                 M, N, K);
    end = std::chrono::high_resolution_clock::now();
    hw_time_ms += std::chrono::duration<double, std::milli>(end - start).count();

    bool pass2 = compare_matrices(C_ref, C_dut, M, N, 1e-6f);
    
    stats = PerformanceStats("Test 7 (Boundary)", M, N, K, hw_time_ms / 2.0);

    bool pass = pass1 && pass2;
    std::cout << "HW Time (avg): " << hw_time_ms / 2.0 << " ms, GFLOPS: " << stats.gflops << std::endl;
    std::cout << "Test 7: " << (pass ? "PASS" : "FAIL") << std::endl << std::endl;

    aligned_free_float(A); aligned_free_float(B); aligned_free_float(C_ref); aligned_free_float(C_dut);
    return pass;
}

//=============================================================================
// 主函数
//=============================================================================
int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "  GEMM_General Testbench (Optimized)" << std::endl;
    std::cout << "  Tiled Matrix Multiplication with" << std::endl;
    std::cout << "  32x32 Systolic Array + 512-bit AXI" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Tile size: 32 x 32" << std::endl;
    std::cout << "Data type: FP32" << std::endl;
    std::cout << "AXI width: 512-bit" << std::endl;
    std::cout << std::endl;

    // 固定随机种子
    srand(42);

    // 运行所有测试并收集性能统计
    int pass_count = 0;
    int total_tests = 7;
    std::vector<PerformanceStats> all_stats;
    all_stats.reserve(total_tests);

    PerformanceStats stats("", 0, 0, 0, 0.0);
    
    if (test_32x32_basic(stats)) pass_count++;
    all_stats.push_back(stats);
    
    if (test_64x64(stats)) pass_count++;
    all_stats.push_back(stats);
    
    if (test_non_square(stats)) pass_count++;
    all_stats.push_back(stats);
    
    if (test_identity(stats)) pass_count++;
    all_stats.push_back(stats);
    
    if (test_constant(stats)) pass_count++;
    all_stats.push_back(stats);
    
    if (test_large(stats)) pass_count++;
    all_stats.push_back(stats);
    
    if (test_boundary(stats)) pass_count++;
    all_stats.push_back(stats);

    // 打印总结
    std::cout << "============================================" << std::endl;
    std::cout << "               TEST SUMMARY                 " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Passed: " << pass_count << " / " << total_tests << std::endl;
    std::cout << std::endl;
    
    // 打印性能统计
    std::cout << "============================================" << std::endl;
    std::cout << "           PERFORMANCE SUMMARY              " << std::endl;
    std::cout << "============================================" << std::endl;
    printf("%-25s %10s %10s %12s %10s\n", "Test", "M×N×K", "Time(ms)", "GFLOPS", "Efficiency");
    std::cout << "------------------------------------------------------------" << std::endl;
    
    double total_gflops = 0.0;
    for (const auto& s : all_stats) {
        printf("%-25s %3dx%3dx%3d %10.3f %10.3f\n",
               s.test_name.c_str(), s.M, s.N, s.K,
               s.hw_time_ms, s.gflops);
        total_gflops += s.gflops;
    }
    
    std::cout << "------------------------------------------------------------" << std::endl;
    printf("Average GFLOPS: %.3f\n", total_gflops / all_stats.size());
    std::cout << "============================================" << std::endl;

    if (pass_count == total_tests) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
