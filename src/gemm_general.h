//=============================================================================
// gemm_general.h - 通用矩阵乘法 IP 核心头文件
//
// 功能: 支持任意 M×K × K×N 矩阵乘法 (M, N, K 为 32 的倍数)
// 架构: 分块矩阵乘法 (Tiled GEMM) + 32×32 脉动阵列
// 优化:
//   1. Ping-Pong 缓冲 + Dataflow 流水线
//   2. 分块累加器设计 - 将 32 个 float 分成 4 块，每块 8 个 float (256-bit)
//   3. 累加器存储在 SystolicEngine 层级 - 确保 BIND_STORAGE pragma 生效
//   4. 降低 FF 使用率 - 正确映射到 BRAM
//=============================================================================

#ifndef GEMM_GENERAL_H
#define GEMM_GENERAL_H

#include "systolic_row.h"
#include <hls_stream.h>

//=============================================================================
// 配置参数
//=============================================================================

// 最大矩阵维度 (用于 depth 计算)
constexpr int MAX_DIM = 4096;
constexpr int MAX_TILES = MAX_DIM / TILE_SIZE;  // 256

// AXI 接口深度
constexpr int AXI_DEPTH = MAX_DIM * MAX_DIM;

//=============================================================================
// 脉动阵列计算引擎 (模块化行设计 + 累加器集中管理)
//=============================================================================
// 重构后的脉动阵列，使用 SystolicRow 模块化设计
// 累加器存储在 SystolicEngine 层级，确保 BIND_STORAGE pragma 生效
// 优势: 降低 FF 使用率，正确映射到 BRAM
//=============================================================================

class SystolicEngine {
public:
    //=========================================================================
    // 硬件资源声明 (模块化设计 + 累加器集中管理)
    //=========================================================================
    
    // 32 个脉动阵列行模块 - 核心计算单元
    SystolicRow rows[N];
    
    // 累加器存储 - 集中管理所有行的累加器
    // 维度: [行索引][块索引][深度], 每个元素包含 8 个 float (256 bits)
    // 关键: 在顶层类中声明，确保 BIND_STORAGE pragma 生效
    acc_chunk_t accumulators[N][ACC_NUM_CHUNKS][ACC_PIPELINE_DEPTH];
    
    // 互联线网 - B 数据垂直传递 (行间连接)
    // b_wire[0] 是顶部输入，b_wire[i+1] 是第 i 行的输出
    data_t b_wire[N + 1][N];
    
    //=========================================================================
    // 初始化 - 复位所有行模块和累加器
    //=========================================================================
    void init() {
        #pragma HLS INLINE
        
        // 累加器存储配置 - 关键: 在顶层类中应用 pragma
        // 按行完全分区 (dim=1)，每行独立访问
        // 按块完全分区 (dim=2)，4 块并行访问
        // 深度维度绑定到 BRAM (dim=3)
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=2
        #pragma HLS BIND_STORAGE variable=accumulators type=ram_2p impl=bram
        #pragma HLS DEPENDENCE variable=accumulators inter false
        
        // 初始化所有行模块
        INIT_ROW: for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL
            rows[i].init();
        }
        
        // 初始化所有累加器
        INIT_ACC_ROW: for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL
            INIT_ACC_CHUNK: for (int c = 0; c < ACC_NUM_CHUNKS; c++) {
                #pragma HLS UNROLL
                INIT_ACC_DEPTH: for (int d = 0; d < ACC_PIPELINE_DEPTH; d++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS DEPENDENCE variable=accumulators inter false
                    accumulators[i][c][d] = 0;  // 写入 256-bit 零
                }
            }
        }
    }
    
    //=========================================================================
    // 数据馈送 - 从外部缓冲区生成脉动阵列输入
    //=========================================================================
    void feed_data(
        int cycle,
        const data_t A_buf[N][N],
        const data_t B_buf[N][N],
        data_t a_input[N],
        data_t b_input[N]
    ) {
        #pragma HLS INLINE
        
        // 生成每行的 A 输入 (错位输入实现对角线馈送)
        FEED_A: for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL
            int k = cycle - i;
            if (k >= 0 && k < N) {
                a_input[i] = A_buf[i][k];
            } else {
                a_input[i] = 0;
            }
        }
        
        // 生成每列的 B 输入 (错位输入实现对角线馈送)
        FEED_B: for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL
            int k = cycle - j;
            if (k >= 0 && k < N) {
                b_input[j] = B_buf[k][j];
            } else {
                b_input[j] = 0;
            }
        }
    }
    
    //=========================================================================
    // 单周期计算步进 (模块化行设计 + 累加器传递)
    //=========================================================================
    void compute_cycle(
        data_t a_input[N],
        data_t b_input[N],
        int acc_idx
    ) {
        #pragma HLS INLINE
        
        #pragma HLS ARRAY_PARTITION variable=b_wire complete dim=0
        #pragma HLS ARRAY_PARTITION variable=rows complete
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=2
        #pragma HLS DEPENDENCE variable=accumulators inter false
        
        // 顶部边界: B 数据输入
        BOUNDARY_B: for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL
            b_wire[0][j] = b_input[j];
        }
        
        // 逐行计算，连接行间 B 数据传递
        COMPUTE_ROWS: for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL
            // 传递该行的累加器切片
            rows[i].compute_step(a_input[i], b_wire[i], acc_idx, accumulators[i]);
            
            // 收集当前行的 B 输出，传递到下一行
            COLLECT_B: for (int j = 0; j < N; j++) {
                #pragma HLS UNROLL
                b_wire[i + 1][j] = rows[i].b_out[j];
            }
        }
    }
    
    //=========================================================================
    // 结果收集 - 从行模块读取累加结果到外部缓冲区
    //=========================================================================
    void drain_results(data_t C_buf[N][N]) {
        #pragma HLS INLINE
        
        // C_buf 按列分区 (dim=2)，因此我们需要并行写入一行中的不同列
        #pragma HLS ARRAY_PARTITION variable=C_buf complete dim=2
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=2
        #pragma HLS DEPENDENCE variable=accumulators inter false
        
        DRAIN_ROW: for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=C_buf inter false
            
            // 传递该行的累加器切片
            rows[i].drain_results(C_buf[i], accumulators[i]);
        }
    }
    
    //=========================================================================
    // 清除累加器
    //=========================================================================
    void clear_accumulators() {
        #pragma HLS INLINE
        
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=2
        #pragma HLS DEPENDENCE variable=accumulators inter false
        
        CLEAR_ROW: for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL
            // 传递该行的累加器切片
            rows[i].clear_accumulators(accumulators[i]);
        }
    }
    
    //=========================================================================
    // 清除所有状态
    //=========================================================================
    void clear_all() {
        #pragma HLS INLINE off
        
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=2
        #pragma HLS DEPENDENCE variable=accumulators inter false
        
        CLEAR_ROW: for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL
            // 传递该行的累加器切片
            rows[i].clear_all(accumulators[i]);
        }
    }
    
    //=========================================================================
    // 执行单个分块的矩阵乘法 (不清除累加器，支持 K 维度累加)
    //=========================================================================
    void compute_tile(
        const data_t A_buf[N][N],
        const data_t B_buf[N][N]
    ) {
        #pragma HLS INLINE
        
        // 配合 gemm_general.cpp 中的 BRAM 映射，统一使用 dim=2 分区
        #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=2
        #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=2
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=2
        #pragma HLS DEPENDENCE variable=accumulators inter false
        
        data_t a_input[N];
        data_t b_input[N];
        #pragma HLS ARRAY_PARTITION variable=a_input complete
        #pragma HLS ARRAY_PARTITION variable=b_input complete
        
        COMPUTE_LOOP: for (int cycle = 0; cycle < TOTAL_CYCLES; cycle++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=rows inter false
            #pragma HLS DEPENDENCE variable=b_wire inter false
            #pragma HLS DEPENDENCE variable=A_buf inter false
            #pragma HLS DEPENDENCE variable=B_buf inter false
            #pragma HLS DEPENDENCE variable=accumulators inter false
            
            feed_data(cycle, A_buf, B_buf, a_input, b_input);
            // 计算累加器索引 (简单的模运算，HLS 会优化为计数器)
            int acc_idx = cycle % ACC_PIPELINE_DEPTH;
            compute_cycle(a_input, b_input, acc_idx);
        }
    }
};

#endif // GEMM_GENERAL_H
