//=============================================================================
// gemm_general.cpp - 通用矩阵乘法 IP 核心实现 (Stream + DATAFLOW 优化版)
//
// 功能: C[M×N] = A[M×K] × B[K×N]
// 约束: M, N, K 必须是 32 的倍数
// 架构: 分块矩阵乘法 + 32×32 脉动阵列 + Stream 流水线
//
// 优化策略:
// 1. 512-bit AXI 接口: 单周期读取 16 个 float (32个float需要2次传输)
// 2. Stream + DATAFLOW: 任务级流水线，数据加载与计算重叠
// 3. 单一计算实例: 整个 C 分块计算封装在一个函数中，确保只有一个脉动阵列
//=============================================================================

#include "gemm_general.h"
#include <cstring>

//=============================================================================
// Task 1: 读取 A 和 B 矩阵分块到 Stream
//=============================================================================
static void Task_Read_AB(
    const uint512_t* A,
    const uint512_t* B,
    hls::stream<uint512_t>& A_stream,
    hls::stream<uint512_t>& B_stream,
    int tile_row,
    int tile_col,
    int num_tiles_k,
    int K,
    int N_mat
) {
    #pragma HLS INLINE off
    
    int K_512 = K / FLOATS_PER_AXI;
    int N_512 = N_mat / FLOATS_PER_AXI;
    
    for (int tk = 0; tk < num_tiles_k; tk++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        
        // 读取 A 分块 (32x32，每行需要2个512-bit字)
        int base_row_A = tile_row * TILE_SIZE;
        int base_col_A = tk * AXI_WORDS_PER_ROW;  // 每个tile占用2个AXI字
        
        for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=A inter false
            int r = i / AXI_WORDS_PER_ROW;
            int w = i % AXI_WORDS_PER_ROW;
            int row = base_row_A + r;
            int addr = row * K_512 + base_col_A + w;
            uint512_t data = A[addr];
            A_stream.write(data);
        }
        
        // 读取 B 分块 (32x32，每行需要2个512-bit字)
        int base_row_B = tk * TILE_SIZE;
        int base_col_B = tile_col * AXI_WORDS_PER_ROW;  // 每个tile占用2个AXI字
        
        for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=B inter false
            int r = i / AXI_WORDS_PER_ROW;
            int w = i % AXI_WORDS_PER_ROW;
            int row = base_row_B + r;
            int addr = row * N_512 + base_col_B + w;
            uint512_t data = B[addr];
            B_stream.write(data);
        }
    }
}

//=============================================================================
// 延迟优化: Task_Read_A_Row - 将 A 矩阵的一行读入 URAM 缓存
// 功能: 一次性读取 A[tile_row*32 : (tile_row+1)*32-1][0:K-1] 到片上 URAM
// 优化: 避免 A 矩阵的重复 DRAM 访问（从 N/32 次降低到 1 次）
//=============================================================================
static void Task_Read_A_Row(
    const uint512_t* A,
    data_t A_row_cache[TILE_SIZE][MAX_K],
    int tile_row,
    int K
) {
    #pragma HLS INLINE off
    #pragma HLS BIND_STORAGE variable=A_row_cache type=ram_2p impl=uram
    // 移除 dim=1 的完全分区，让 HLS 自动选择更优的 URAM 映射
    // 这将把 256 个小 URAM 合并为少数几个大 URAM，释放布线资源
    
    int K_512 = K / FLOATS_PER_AXI;
    int base_row = tile_row * TILE_SIZE;
    
    // 读取整行 A (32 x K)
    READ_A_ROW: for (int i = 0; i < TILE_SIZE; i++) {
        READ_A_COL: for (int k = 0; k < K_512; k++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=A inter false
            #pragma HLS DEPENDENCE variable=A_row_cache inter false
            int row = base_row + i;
            int addr = row * K_512 + k;
            uint512_t wide_data = A[addr];
            
            // 解包到 URAM
            for (int j = 0; j < FLOATS_PER_AXI; j++) {
                #pragma HLS UNROLL
                unsigned int bits = wide_data.range((j + 1) * 32 - 1, j * 32);
                A_row_cache[i][k * FLOATS_PER_AXI + j] = uint32_to_float(bits);
            }
        }
    }
}

//=============================================================================
// 延迟优化: Task_Read_AB_Optimized - 从 URAM 读取 A，从 DRAM 流式读取 B
// 功能: A 从片上 URAM 缓存读取，B 从 DRAM 读取，推入 Stream
// 优化: A 的访问延迟极低（URAM），无需重复访问 DRAM
//=============================================================================
static void Task_Read_AB_Optimized(
    data_t A_row_cache[TILE_SIZE][MAX_K],
    const uint512_t* B,
    hls::stream<uint512_t>& A_stream,
    hls::stream<uint512_t>& B_stream,
    int tile_col,
    int num_tiles_k,
    int N_mat
) {
    #pragma HLS INLINE off
    
    int N_512 = N_mat / FLOATS_PER_AXI;
    
    for (int tk = 0; tk < num_tiles_k; tk++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        
        // 从 URAM 读取 A 分块 (32x32，每行需要2个512-bit字)
        for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=A_row_cache inter false
            int r = i / AXI_WORDS_PER_ROW;
            int w = i % AXI_WORDS_PER_ROW;
            uint512_t wide_data;
            for (int j = 0; j < FLOATS_PER_AXI; j++) {
                #pragma HLS UNROLL
                int k_idx = tk * TILE_SIZE + w * FLOATS_PER_AXI + j;
                unsigned int bits = float_to_uint32(A_row_cache[r][k_idx]);
                wide_data.range((j + 1) * 32 - 1, j * 32) = bits;
            }
            A_stream.write(wide_data);
        }
        
        // 从 DRAM 读取 B 分块 (32x32，每行需要2个512-bit字)
        int base_row_B = tk * TILE_SIZE;
        int base_col_B = tile_col * AXI_WORDS_PER_ROW;
        
        for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=B inter false
            int r = i / AXI_WORDS_PER_ROW;
            int w = i % AXI_WORDS_PER_ROW;
            int row = base_row_B + r;
            int addr = row * N_512 + base_col_B + w;
            uint512_t data = B[addr];
            B_stream.write(data);
        }
    }
}

//=============================================================================
// 延迟优化: Task_Read_B_All - 连续读取所有 B 矩阵列块
// 功能: 在内部包含 tn 循环，连续读取所有需要的 B 矩阵分块并推入 Stream
// 优化: DMA 引擎可以全速运行，不受计算任务的打断
//=============================================================================
static void Task_Read_B_All(
    data_t A_row_cache[TILE_SIZE][MAX_K],
    const uint512_t* B,
    hls::stream<uint512_t>& A_stream,
    hls::stream<uint512_t>& B_stream,
    int num_tiles_n,
    int num_tiles_k,
    int N_mat
) {
    #pragma HLS INLINE off
    
    int N_512 = N_mat / FLOATS_PER_AXI;
    
    // 外层循环：遍历所有 B 的列块
    for (int tn = 0; tn < num_tiles_n; tn++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        
        // 内层循环：遍历 K 维度的分块
        for (int tk = 0; tk < num_tiles_k; tk++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=256
            
            // 从 URAM 读取 A 分块（每个 tn 都需要复用相同的 A）(32x32，每行需要2个512-bit字)
            for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=A_row_cache inter false
                int r = i / AXI_WORDS_PER_ROW;
                int w = i % AXI_WORDS_PER_ROW;
                uint512_t wide_data;
                for (int j = 0; j < FLOATS_PER_AXI; j++) {
                    #pragma HLS UNROLL
                    int k_idx = tk * TILE_SIZE + w * FLOATS_PER_AXI + j;
                    unsigned int bits = float_to_uint32(A_row_cache[r][k_idx]);
                    wide_data.range((j + 1) * 32 - 1, j * 32) = bits;
                }
                A_stream.write(wide_data);
            }
            
            // 从 DRAM 读取 B 分块 (32x32，每行需要2个512-bit字)
            int base_row_B = tk * TILE_SIZE;
            int base_col_B = tn * AXI_WORDS_PER_ROW;
            
            for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=B inter false
                int r = i / AXI_WORDS_PER_ROW;
                int w = i % AXI_WORDS_PER_ROW;
                int row = base_row_B + r;
                int addr = row * N_512 + base_col_B + w;
                uint512_t data = B[addr];
                B_stream.write(data);
            }
        }
    }
}

//=============================================================================
// Task 2: 计算任务（原版，保留用于兼容）
//=============================================================================
static void Task_Compute(
    SystolicEngine& engine,
    hls::stream<uint512_t>& A_stream,
    hls::stream<uint512_t>& B_stream,
    hls::stream<uint512_t>& C_stream,
    int num_tiles_k
) {
    #pragma HLS INLINE off
    
    data_t A_buf[TILE_SIZE][TILE_SIZE];
    data_t B_buf[TILE_SIZE][TILE_SIZE];
    data_t C_buf[TILE_SIZE][TILE_SIZE];

    #pragma HLS DEPENDENCE variable=A_buf inter false
    #pragma HLS DEPENDENCE variable=B_buf inter false
    #pragma HLS DEPENDENCE variable=C_buf inter false
    
    #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=C_buf complete dim=2
    #pragma HLS BIND_STORAGE variable=A_buf type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=B_buf type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=C_buf type=ram_2p impl=bram
    
    engine.clear_all();
    
    for (int tk = 0; tk < num_tiles_k; tk++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        
        // 读取 A 分块 (32x32，每行2个512-bit字)
        for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=A_buf inter false
            int r = i / AXI_WORDS_PER_ROW;
            int w = i % AXI_WORDS_PER_ROW;
            uint512_t wide_data = A_stream.read();
            for (int j = 0; j < FLOATS_PER_AXI; j++) {
                #pragma HLS UNROLL
                unsigned int bits = wide_data.range((j + 1) * 32 - 1, j * 32);
                A_buf[r][w * FLOATS_PER_AXI + j] = uint32_to_float(bits);
            }
        }
        
        // 读取 B 分块 (32x32，每行2个512-bit字)
        for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=B_buf inter false
            int r = i / AXI_WORDS_PER_ROW;
            int w = i % AXI_WORDS_PER_ROW;
            uint512_t wide_data = B_stream.read();
            for (int j = 0; j < FLOATS_PER_AXI; j++) {
                #pragma HLS UNROLL
                unsigned int bits = wide_data.range((j + 1) * 32 - 1, j * 32);
                B_buf[r][w * FLOATS_PER_AXI + j] = uint32_to_float(bits);
            }
        }
        
        engine.compute_tile(A_buf, B_buf);
    }
    
    engine.drain_results(C_buf);
    
    // 写入 C 分块 (32x32，每行2个512-bit字)
    for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=C_buf inter false
        int r = i / AXI_WORDS_PER_ROW;
        int w = i % AXI_WORDS_PER_ROW;
        uint512_t wide_data;
        for (int j = 0; j < FLOATS_PER_AXI; j++) {
            #pragma HLS UNROLL
            unsigned int bits = float_to_uint32(C_buf[r][w * FLOATS_PER_AXI + j]);
            wide_data.range((j + 1) * 32 - 1, j * 32) = bits;
        }
        C_stream.write(wide_data);
    }
}

//=============================================================================
// 延迟优化: Task_Compute_All - 连续处理多个 C 分块
// 功能: 接收连续的 B 数据流，复用 URAM 中已加载的 A 矩阵行
// 优化: 在处理完一个 tn 分块后，立即开始处理下一个，无需复位流水线
//=============================================================================
static void Task_Compute_All(
    SystolicEngine& engine,
    hls::stream<uint512_t>& A_stream,
    hls::stream<uint512_t>& B_stream,
    hls::stream<uint512_t>& C_stream,
    int num_tiles_n,
    int num_tiles_k
) {
    #pragma HLS INLINE off
    
    data_t A_buf[TILE_SIZE][TILE_SIZE];
    data_t B_buf[TILE_SIZE][TILE_SIZE];
    data_t C_buf[TILE_SIZE][TILE_SIZE];
    
    #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=C_buf complete dim=2
    #pragma HLS BIND_STORAGE variable=A_buf type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=B_buf type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=C_buf type=ram_2p impl=bram
    
    // 外层循环：遍历所有 B 的列块
    for (int tn = 0; tn < num_tiles_n; tn++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        
        // 清除累加器，为新的 C 分块做准备
        engine.clear_all();
        
        // 内层循环：遍历 K 维度的分块
        for (int tk = 0; tk < num_tiles_k; tk++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=256
            
            // 从 Stream 读取 A 分块 (32x32，每行2个512-bit字)
            for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=A_buf inter false
                int r = i / AXI_WORDS_PER_ROW;
                int w = i % AXI_WORDS_PER_ROW;
                uint512_t wide_data = A_stream.read();
                for (int j = 0; j < FLOATS_PER_AXI; j++) {
                    #pragma HLS UNROLL
                    unsigned int bits = wide_data.range((j + 1) * 32 - 1, j * 32);
                    A_buf[r][w * FLOATS_PER_AXI + j] = uint32_to_float(bits);
                }
            }
            
            // 从 Stream 读取 B 分块 (32x32，每行2个512-bit字)
            for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=B_buf inter false
                int r = i / AXI_WORDS_PER_ROW;
                int w = i % AXI_WORDS_PER_ROW;
                uint512_t wide_data = B_stream.read();
                for (int j = 0; j < FLOATS_PER_AXI; j++) {
                    #pragma HLS UNROLL
                    unsigned int bits = wide_data.range((j + 1) * 32 - 1, j * 32);
                    B_buf[r][w * FLOATS_PER_AXI + j] = uint32_to_float(bits);
                }
            }
            
            // 执行计算
            engine.compute_tile(A_buf, B_buf);
        }
        
        // 收集结果并写入 Stream
        engine.drain_results(C_buf);
        
        // 写入 C 分块 (32x32，每行2个512-bit字)
        for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=C_buf inter false
            int r = i / AXI_WORDS_PER_ROW;
            int w = i % AXI_WORDS_PER_ROW;
            uint512_t wide_data;
            for (int j = 0; j < FLOATS_PER_AXI; j++) {
                #pragma HLS UNROLL
                unsigned int bits = float_to_uint32(C_buf[r][w * FLOATS_PER_AXI + j]);
                wide_data.range((j + 1) * 32 - 1, j * 32) = bits;
            }
            C_stream.write(wide_data);
        }
    }
}

//=============================================================================
// Task 3: 写回任务
//=============================================================================
static void Task_Write_C(
    uint512_t* C,
    hls::stream<uint512_t>& C_stream,
    int tile_row,
    int tile_col,
    int N_mat
) {
    #pragma HLS INLINE off
    
    int N_512 = N_mat / FLOATS_PER_AXI;
    int base_row = tile_row * TILE_SIZE;
    int base_col = tile_col * AXI_WORDS_PER_ROW;  // 修复：每个tile占用2个AXI字
    
    // 写回 C 分块 (32x32，每行2个512-bit字)
    for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=C inter false
        int r = i / AXI_WORDS_PER_ROW;
        int w = i % AXI_WORDS_PER_ROW;
        int row = base_row + r;
        uint512_t wide_data = C_stream.read();
        int addr = row * N_512 + base_col + w;
        C[addr] = wide_data;
    }
}

//=============================================================================
// 延迟优化: Task_Write_C_All - 连续写回所有 C 分块
// 功能: 在内部包含 tn 循环，连续接收计算结果并写回 DRAM
// 优化: DMA 引擎可以全速运行，不受计算任务的打断
//=============================================================================
static void Task_Write_C_All(
    uint512_t* C,
    hls::stream<uint512_t>& C_stream,
    int tile_row,
    int num_tiles_n,
    int N_mat
) {
    #pragma HLS INLINE off
    
    int N_512 = N_mat / FLOATS_PER_AXI;
    int base_row = tile_row * TILE_SIZE;
    
    // 外层循环：遍历所有 C 的列块
    for (int tn = 0; tn < num_tiles_n; tn++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        
        int base_col = tn * AXI_WORDS_PER_ROW;  // 修复：每个tile占用2个AXI字
        
        // 写回一个 C 分块 (32x32，每行2个512-bit字)
        for (int i = 0; i < TILE_SIZE * AXI_WORDS_PER_ROW; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=C inter false
            int r = i / AXI_WORDS_PER_ROW;
            int w = i % AXI_WORDS_PER_ROW;
            int row = base_row + r;
            uint512_t wide_data = C_stream.read();
            int addr = row * N_512 + base_col + w;
            C[addr] = wide_data;
        }
    }
}

//=============================================================================
// DATAFLOW 封装函数: 处理单个 C 分块
// 关键: engine 从顶层传入，确保只有一个脉动阵列实例
//=============================================================================
static void process_one_C_tile(
    SystolicEngine& engine,
    const uint512_t* A,
    const uint512_t* B,
    uint512_t* C,
    int tile_row,
    int tile_col,
    int num_tiles_k,
    int K,
    int N_mat
) {
    #pragma HLS INLINE off
    #pragma HLS STABLE variable=A
    #pragma HLS STABLE variable=B
    #pragma HLS STABLE variable=C
    
    // 清除累加器状态，确保每个 C 分块独立计算
    engine.clear_all();
    
    #pragma HLS DATAFLOW
    
    hls::stream<uint512_t> A_stream("A_stream");
    hls::stream<uint512_t> B_stream("B_stream");
    hls::stream<uint512_t> C_stream("C_stream");
    
    #pragma HLS STREAM variable=A_stream depth=32
    #pragma HLS STREAM variable=B_stream depth=32
    #pragma HLS STREAM variable=C_stream depth=32
    
    Task_Read_AB(A, B, A_stream, B_stream, tile_row, tile_col, num_tiles_k, K, N_mat);
    Task_Compute(engine, A_stream, B_stream, C_stream, num_tiles_k);
    Task_Write_C(C, C_stream, tile_row, tile_col, N_mat);
}

//=============================================================================
// 延迟优化: process_one_C_tile_optimized - 使用 URAM 缓存的 A 矩阵
// 功能: 处理单个 C 分块，A 从 URAM 读取，B 从 DRAM 读取
// 优化: 避免 A 矩阵的重复 DRAM 访问
//=============================================================================
static void process_one_C_tile_optimized(
    SystolicEngine& engine,
    data_t A_row_cache[TILE_SIZE][MAX_K],
    const uint512_t* B,
    uint512_t* C,
    int tile_row,
    int tile_col,
    int num_tiles_k,
    int N_mat
) {
    #pragma HLS INLINE off
    #pragma HLS STABLE variable=B
    #pragma HLS STABLE variable=C
    
    // 清除累加器状态，确保每个 C 分块独立计算
    engine.clear_all();
    
    #pragma HLS DATAFLOW
    
    hls::stream<uint512_t> A_stream("A_stream");
    hls::stream<uint512_t> B_stream("B_stream");
    hls::stream<uint512_t> C_stream("C_stream");
    
    #pragma HLS STREAM variable=A_stream depth=32
    #pragma HLS STREAM variable=B_stream depth=32
    #pragma HLS STREAM variable=C_stream depth=32
    
    Task_Read_AB_Optimized(A_row_cache, B, A_stream, B_stream, tile_col, num_tiles_k, N_mat);
    Task_Compute(engine, A_stream, B_stream, C_stream, num_tiles_k);
    Task_Write_C(C, C_stream, tile_row, tile_col, N_mat);
}

//=============================================================================
// 延迟优化: process_all_C_tiles_optimized - 处理一行的所有 C 分块
// 功能: 将 tn 循环下推到各个任务内部，实现真正的流水线并行
// 优化: 数据读取、计算和写回操作完全重叠，最大化性能
//=============================================================================
static void process_all_C_tiles_optimized(
    SystolicEngine& engine,
    data_t A_row_cache[TILE_SIZE][MAX_K],
    const uint512_t* B,
    uint512_t* C,
    int tile_row,
    int num_tiles_n,
    int num_tiles_k,
    int N_mat
) {
    #pragma HLS INLINE off
    #pragma HLS STABLE variable=B
    #pragma HLS STABLE variable=C
    #pragma HLS DATAFLOW
    
    hls::stream<uint512_t> A_stream("A_stream");
    hls::stream<uint512_t> B_stream("B_stream");
    hls::stream<uint512_t> C_stream("C_stream");
    
    #pragma HLS STREAM variable=A_stream depth=32
    #pragma HLS STREAM variable=B_stream depth=32
    #pragma HLS STREAM variable=C_stream depth=32
    
    // 三个任务并行执行，每个任务内部包含 tn 循环
    Task_Read_B_All(A_row_cache, B, A_stream, B_stream, num_tiles_n, num_tiles_k, N_mat);
    Task_Compute_All(engine, A_stream, B_stream, C_stream, num_tiles_n, num_tiles_k);
    Task_Write_C_All(C, C_stream, tile_row, num_tiles_n, N_mat);
}

//=============================================================================
// 顶层函数实现
//=============================================================================
extern "C" {

void GEMM_General(
    uint512_t* A,
    uint512_t* B,
    uint512_t* C,
    int M,
    int N_mat,
    int K
) {

    //=========================================================================
    // 接口配置
    //=========================================================================
    // depth=4096 支持最大 256x256 矩阵 (256*256/16=4096 个 512-bit 字)
    // 这确保 Co-simulation 适配器缓冲区足够大，避免 SIGSEGV
    #pragma HLS INTERFACE m_axi port=A bundle=gmem0 depth=4096 \
        max_read_burst_length=64 num_read_outstanding=16 \
        latency=64 offset=slave

    #pragma HLS INTERFACE m_axi port=B bundle=gmem1 depth=4096 \
        max_read_burst_length=64 num_read_outstanding=16 \
        latency=64 offset=slave

    #pragma HLS INTERFACE m_axi port=C bundle=gmem2 depth=4096 \
        max_read_burst_length=64 max_write_burst_length=64 \
        num_read_outstanding=16 num_write_outstanding=16 \
        latency=64 offset=slave

    #pragma HLS INTERFACE s_axilite port=A bundle=control
    #pragma HLS INTERFACE s_axilite port=B bundle=control
    #pragma HLS INTERFACE s_axilite port=C bundle=control
    #pragma HLS INTERFACE s_axilite port=M bundle=control
    #pragma HLS INTERFACE s_axilite port=N_mat bundle=control
    #pragma HLS INTERFACE s_axilite port=K bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    //=========================================================================
    // 硬件资源实例化 - 单一脉动阵列实例 (模块化行设计)
    //=========================================================================
    static SystolicEngine engine;
    #pragma HLS RESET variable=engine
    #pragma HLS ARRAY_PARTITION variable=engine.rows complete dim=0
    #pragma HLS ARRAY_PARTITION variable=engine.b_wire complete dim=0
    // 关键修复: 显式应用累加器存储指令，确保 BRAM 映射生效
    // 这是解决 244% FF 使用率的核心修复
    // dim=1 (行): 完全分区，32 行并行访问
    // dim=2 (块): 完全分区，4 块并行访问
    // dim=3 (深度): 绑定到 BRAM，32 深度的流水线累加器
    #pragma HLS ARRAY_PARTITION variable=engine.accumulators complete dim=1
    #pragma HLS ARRAY_PARTITION variable=engine.accumulators complete dim=2
    #pragma HLS BIND_STORAGE variable=engine.accumulators type=ram_2p impl=bram
    #pragma HLS DEPENDENCE variable=engine.accumulators inter false

    //=========================================================================
    // 计算分块数量
    //=========================================================================
    int num_tiles_m = M / TILE_SIZE;
    int num_tiles_n = N_mat / TILE_SIZE;
    int num_tiles_k = K / TILE_SIZE;

    //=========================================================================
    // 延迟优化: A 矩阵行缓存 (URAM)
    // 大小: 32 x MAX_K (最大支持 K=4096)
    // 优化: 移除 dim=1 分区，减少 URAM 实例数量，释放布线资源
    //=========================================================================
    static data_t A_row_cache[TILE_SIZE][MAX_K];
    #pragma HLS BIND_STORAGE variable=A_row_cache type=ram_2p impl=uram
    // 不再对 dim=1 进行完全分区，让 HLS 自动选择更优的存储映射

    //=========================================================================
    // 主循环: Row-Stationary 调度策略
    // 外层循环 (tm): 固定 A 的行块，缓存到 URAM
    // 内层循环 (tn): 遍历 B 的列块，复用 URAM 中的 A
    // 优化: A 的 DRAM 读取次数从 N/16 次降低到 1 次
    //=========================================================================
    for (int tm = 0; tm < num_tiles_m; tm++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=256
        
        // 步骤 1: 读取 A 的整行到 URAM（只读一次）
        Task_Read_A_Row(A, A_row_cache, tm, K);
        
        // 步骤 2: 遍历所有列，复用 URAM 中的 A
        for (int tn = 0; tn < num_tiles_n; tn++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=256
            
            process_one_C_tile_optimized(engine, A_row_cache, B, C, tm, tn, num_tiles_k, N_mat);
        }
    }
    
    } // extern "C"
}

