//=============================================================================
// systolic_row.h - Systolic Array Row Module (模块化脉动阵列行)
//
// 功能: 封装一行的寄存器，降低综合复杂度
// 优化: 累加器存储移至 SystolicEngine 层级，确保 BIND_STORAGE pragma 生效
// 架构: 保持脉动阵列的数据流特性，每行 32 个 MAC 单元
//=============================================================================

#ifndef SYSTOLIC_ROW_H
#define SYSTOLIC_ROW_H

#include "types.h"

//=============================================================================
// SystolicRow - 脉动阵列单行模块 (累加器外置设计)
//=============================================================================
// 累加器存储移至 SystolicEngine 类，通过参数传递访问
// 优势:
// 1. BIND_STORAGE pragma 在顶层类中更容易被 HLS 识别
// 2. 避免类成员变量的 pragma 被忽略问题
// 3. 统一管理所有行的累加器，便于 BRAM 映射优化
//=============================================================================

class SystolicRow {
public:
    //=========================================================================
    // 硬件资源声明 (仅寄存器，累加器外置)
    //=========================================================================
    
    // 每列的 A 寄存器 (替代 PE::a_reg)
    data_t a_regs[N];
    
    // 每列的 B 寄存器 (替代 PE::b_reg)
    data_t b_regs[N];
    
    // 行内 A 数据传递线网 (水平传递)
    // a_wire[0] 是输入，a_wire[N] 是输出
    data_t a_wire[N + 1];
    
    // B 数据输出缓存 (垂直传递到下一行)
    data_t b_out[N];
    
    //=========================================================================
    // 初始化 - 复位所有寄存器 (累加器由外部初始化)
    //=========================================================================
    void init() {
        #pragma HLS INLINE
        
        // 寄存器数组完全分区 - 每个 PE 独立访问
        #pragma HLS ARRAY_PARTITION variable=a_regs complete
        #pragma HLS ARRAY_PARTITION variable=b_regs complete
        
        // 初始化所有寄存器
        INIT_REGS: for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL
            // Note: DEPENDENCE pragmas removed - redundant when loop is fully unrolled
            a_regs[j] = 0;
            b_regs[j] = 0;
        }
    }
    
    //=========================================================================
    // 单周期计算步进 - 行级接口 (累加器通过参数传递)
    //=========================================================================
    // 输入:
    //   a_in: 从左侧输入的 A 数据
    //   b_in_vec: 从上方输入的 B 数据向量 (每列一个)
    //   acc_idx: 当前累加器索引
    //   accumulators: 外部累加器存储 [ACC_NUM_CHUNKS][ACC_PIPELINE_DEPTH]
    // 输出:
    //   a_wire[N]: 传递到右侧的 A 数据
    //   b_out: 传递到下方的 B 数据向量
    //=========================================================================
    void compute_step(
        data_t a_in,
        const data_t b_in_vec[N],
        int acc_idx,
        acc_chunk_t accumulators[ACC_NUM_CHUNKS][ACC_PIPELINE_DEPTH]
    ) {
        #pragma HLS INLINE
        
        #pragma HLS ARRAY_PARTITION variable=a_wire complete
        #pragma HLS ARRAY_PARTITION variable=b_out complete
        #pragma HLS ARRAY_PARTITION variable=a_regs complete
        #pragma HLS ARRAY_PARTITION variable=b_regs complete
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
        #pragma HLS DEPENDENCE variable=accumulators inter false
        
        // 边界输入: A 从左侧进入
        a_wire[0] = a_in;
        
        // 新累加值临时数组
        data_t new_acc[N];
        #pragma HLS ARRAY_PARTITION variable=new_acc complete
        
        // 分块处理累加器 (4 块并行)
        COMPUTE_CHUNK: for (int c = 0; c < ACC_NUM_CHUNKS; c++) {
            #pragma HLS UNROLL
            
            // 读取当前块 (256 bits = 8 floats)
            acc_chunk_t chunk = accumulators[c][acc_idx];
            
            // 解包并计算块内 8 个 float
            COMPUTE_IN_CHUNK: for (int k = 0; k < FLOATS_PER_CHUNK; k++) {
                #pragma HLS UNROLL
                int col_idx = c * FLOATS_PER_CHUNK + k;
                
                // 解包: 从 256-bit 中提取 32-bit float
                unsigned int bits = chunk.range((k + 1) * FLOAT_WIDTH - 1, k * FLOAT_WIDTH);
                data_t current_acc = uint32_to_float(bits);
                
                // ===== MAC 运算 =====
                // 乘法运算 - 4 周期延迟
                data_t product = a_regs[col_idx] * b_regs[col_idx];
                #pragma HLS BIND_OP variable=product op=fmul impl=fulldsp latency=4
                
                // 累加 - 5 周期延迟
                new_acc[col_idx] = current_acc + product;
                #pragma HLS BIND_OP variable=new_acc op=fadd impl=fulldsp latency=5
            }
            
            // 打包新累加值到块 (纯布线，无逻辑延迟)
            acc_chunk_t new_chunk;
            PACK_CHUNK: for (int k = 0; k < FLOATS_PER_CHUNK; k++) {
                #pragma HLS UNROLL
                int col_idx = c * FLOATS_PER_CHUNK + k;
                unsigned int bits = float_to_uint32(new_acc[col_idx]);
                new_chunk.range((k + 1) * FLOAT_WIDTH - 1, k * FLOAT_WIDTH) = bits;
            }
            
            // 写回累加器块
            accumulators[c][acc_idx] = new_chunk;
        }
        
        // 数据传递 (独立于累加器处理)
        DATA_PASS: for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL
            // 水平传递 A 数据 (传给右邻)
            a_wire[j + 1] = a_regs[j];
            // 垂直传递 B 数据 (传给下邻)
            b_out[j] = b_regs[j];
        }
        
        // 更新寄存器 (为下一周期准备)
        UPDATE_COL: for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL
            a_regs[j] = a_wire[j];
            b_regs[j] = b_in_vec[j];
        }
    }
    
    //=========================================================================
    // 结果收集 - 从累加器读取结果 (累加器通过参数传递)
    //=========================================================================
    void drain_results(
        data_t c_out_vec[N],
        acc_chunk_t accumulators[ACC_NUM_CHUNKS][ACC_PIPELINE_DEPTH]
    ) {
        #pragma HLS INLINE
        
        #pragma HLS ARRAY_PARTITION variable=c_out_vec complete
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1
        #pragma HLS DEPENDENCE variable=accumulators inter false
        
        // 累加和临时数组
        data_t sum[N];
        #pragma HLS ARRAY_PARTITION variable=sum complete
        
        // 初始化累加和
        INIT_SUM: for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL
            sum[j] = 0;
        }
        
        // 遍历深度，逐步累加所有列
        REDUCE_DEPTH: for (int d = 0; d < ACC_PIPELINE_DEPTH; d++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS DEPENDENCE variable=accumulators inter false
            
            // 分块读取并累加
            REDUCE_CHUNK: for (int c = 0; c < ACC_NUM_CHUNKS; c++) {
                #pragma HLS UNROLL
                
                // 读取当前块
                acc_chunk_t chunk = accumulators[c][d];
                
                // 解包并累加到 sum
                REDUCE_IN_CHUNK: for (int k = 0; k < FLOATS_PER_CHUNK; k++) {
                    #pragma HLS UNROLL
                    int col_idx = c * FLOATS_PER_CHUNK + k;
                    unsigned int bits = chunk.range((k + 1) * FLOAT_WIDTH - 1, k * FLOAT_WIDTH);
                    data_t val = uint32_to_float(bits);
                    sum[col_idx] += val;
                }
            }
        }
        
        // 输出结果
        OUTPUT: for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL
            c_out_vec[j] = sum[j];
        }
    }
    
    //=========================================================================
    // 清除累加器 (累加器通过参数传递)
    //=========================================================================
    void clear_accumulators(
        acc_chunk_t accumulators[ACC_NUM_CHUNKS][ACC_PIPELINE_DEPTH]
    ) {
        #pragma HLS INLINE
        
        CLEAR_CHUNK: for (int c = 0; c < ACC_NUM_CHUNKS; c++) {
            #pragma HLS UNROLL
            CLEAR_DEPTH: for (int d = 0; d < ACC_PIPELINE_DEPTH; d++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=accumulators inter false
                accumulators[c][d] = 0;  // 写入 256-bit 零
            }
        }
    }
    
    //=========================================================================
    // 清除所有状态 (累加器通过参数传递)
    //=========================================================================
    void clear_all(
        acc_chunk_t accumulators[ACC_NUM_CHUNKS][ACC_PIPELINE_DEPTH]
    ) {
        #pragma HLS INLINE off
        
        // 清除寄存器
        CLEAR_REGS: for (int j = 0; j < N; j++) {
            #pragma HLS UNROLL
            // Note: DEPENDENCE pragmas removed - redundant when loop is fully unrolled
            a_regs[j] = 0;
            b_regs[j] = 0;
        }
        
        // 清除累加器 (分块清除)
        CLEAR_ACC_CHUNK: for (int c = 0; c < ACC_NUM_CHUNKS; c++) {
            #pragma HLS UNROLL
            CLEAR_ACC_DEPTH: for (int d = 0; d < ACC_PIPELINE_DEPTH; d++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=accumulators inter false
                accumulators[c][d] = 0;  // 写入 256-bit 零
            }
        }
    }
};

#endif // SYSTOLIC_ROW_H
