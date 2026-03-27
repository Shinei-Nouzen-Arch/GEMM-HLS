//=============================================================================
// types.h - 类型定义与硬件配置常量
//
// 设计目标: Alveo U50, 300MHz+, 32x32 FP32矩阵乘法
// 架构风格: 硬件描述风格 - 脉动阵列
//=============================================================================

#ifndef TYPES_H
#define TYPES_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

//=============================================================================
// 硬件配置参数
//=============================================================================

// 矩阵维度
constexpr int N = 32;                    // 矩阵大小 N×N
constexpr int N_ELEMENTS = N * N;        // 矩阵元素总数 = 1024

// 延迟优化配置 - 行缓存 (Row-Stationary)
constexpr int MAX_K = 4096;              // 最大 K 维度支持
constexpr int TILE_SIZE = N;             // 分块大小 = 32
constexpr int A_ROW_CACHE_SIZE = TILE_SIZE * MAX_K;  // A 行缓存大小 = 32 * 4096 = 131072

// AXI接口配置
constexpr int AXI_WIDTH = 512;           // AXI数据位宽
constexpr int FLOAT_WIDTH = 32;          // FP32位宽
constexpr int FLOATS_PER_AXI = AXI_WIDTH / FLOAT_WIDTH;  // = 16
constexpr int AXI_WORDS_PER_ROW = N / FLOATS_PER_AXI;    // = 2 (32个float需要2个512-bit字)

// 突发传输配置
constexpr int BURST_LEN = N;             // 突发长度 = 32
constexpr int MAX_OUTSTANDING = 16;           // 最大outstanding请求数

// 时序配置
constexpr int PIPELINE_DEPTH = 32;       // 脉动阵列流水线深度
constexpr int ACC_PIPELINE_DEPTH = 32;   // 累加器流水线深度
                                          // 设为 32 (与 TILE_SIZE 相同) 以完全消除
                                          // 同一累加器在分块计算中的 RAW 依赖
                                          // 每个 k 索引的乘积分配到独立累加器
constexpr int TOTAL_CYCLES = 3 * N - 1 + ACC_PIPELINE_DEPTH;  // 完成一次计算所需周期 = 127
                                          // 数据馈送: 2N-1 = 63 周期 (对角线馈送)
                                          // 数据传播: N-1 = 31 周期 (到最远PE)
                                          // 累加器排空: ACC_PIPELINE_DEPTH = 32 周期
                                          // 总计: 63 + 31 + 32 = 126, 取 127 保守

//=============================================================================
// 数据类型定义
//=============================================================================

// 主数据类型
typedef float data_t;                    // 计算数据类型: FP32

// 宽位AXI数据类型 - 用于512-bit接口
typedef ap_uint<512> uint512_t;

// 宽位数据类型 - 用于AXI突发传输
typedef struct {
    data_t data[FLOATS_PER_AXI];
} wide_bus_t;

// 控制信号结构
typedef struct {
    bool valid;                          // 数据有效信号
    bool last;                           // 最后一个数据标志
    bool clear;                          // 清除累加器信号
} ctrl_t;

// PE输入数据包
typedef struct {
    data_t a;                            // A矩阵元素
    data_t b;                            // B矩阵元素
    ctrl_t ctrl;                         // 控制信号
} pe_in_t;

// PE输出数据包
typedef struct {
    data_t a;                            // 传递给右邻PE的A数据
    data_t b;                            // 传递给下邻PE的B数据
    data_t c;                            // 累加结果(仅在drain时有效)
    ctrl_t ctrl;                         // 传递的控制信号
} pe_out_t;

//=============================================================================
// Stream类型定义
//=============================================================================

// AXI Stream类型
typedef hls::stream<data_t> data_stream_t;
typedef hls::stream<wide_bus_t> wide_stream_t;
typedef hls::stream<ctrl_t> ctrl_stream_t;

//=============================================================================
// 辅助宏定义
//=============================================================================

// 数组分区宏
#define PARTITION_COMPLETE(arr) \
    _Pragma("HLS ARRAY_PARTITION variable=arr complete dim=0")

#define PARTITION_CYCLIC(arr, factor) \
    _Pragma("HLS ARRAY_PARTITION variable=arr cyclic factor=factor")

// 流水线宏
#define PIPELINE_II(ii) \
    _Pragma("HLS PIPELINE II=ii")

// 循环展开宏
#define UNROLL_FULL \
    _Pragma("HLS UNROLL")

#define UNROLL_FACTOR(f) \
    _Pragma("HLS UNROLL factor=f")

//=============================================================================
// 辅助函数: float <-> ap_uint<32> 位级转换
//=============================================================================
// 用于累加器打包/解包，实现宽位宽存储优化
//=============================================================================

union float_uint32_converter {
    float f;
    unsigned int u;
};

static inline unsigned int float_to_uint32(float f) {
    #pragma HLS INLINE
    float_uint32_converter conv;
    conv.f = f;
    return conv.u;
}

static inline float uint32_to_float(unsigned int u) {
    #pragma HLS INLINE
    float_uint32_converter conv;
    conv.u = u;
    return conv.f;
}

//=============================================================================
// 累加器分块配置 (Chunked Accumulators)
//=============================================================================
// 将 32 个 float 分成 4 个块，每块 8 个 float (256 bits)
// 优势:
// 1. 256-bit 宽度更容易映射到 BRAM (比 1024-bit 高效)
// 2. 4 个数组比 32 个数组的综合复杂度低
// 3. 大幅减少流水线寄存器的 FF 消耗
//=============================================================================

constexpr int FLOATS_PER_CHUNK = 8;                              // 每块 8 个 float
constexpr int ACC_CHUNK_WIDTH = FLOATS_PER_CHUNK * FLOAT_WIDTH;  // = 8 * 32 = 256 bits
constexpr int ACC_NUM_CHUNKS = N / FLOATS_PER_CHUNK;             // = 32 / 8 = 4 块

// 分块累加器类型 (256-bit)
typedef ap_uint<ACC_CHUNK_WIDTH> acc_chunk_t;

// 保留旧定义以兼容 (可选，后续可删除)
constexpr int ACC_PACK_WIDTH = N * FLOAT_WIDTH;  // = 32 * 32 = 1024
typedef ap_uint<ACC_PACK_WIDTH> acc_wide_t;

#endif // TYPES_H
