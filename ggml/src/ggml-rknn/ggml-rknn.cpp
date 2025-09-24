#pragma GCC diagnostic ignored "-Woverlength-strings"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#endif

#include "ggml-rknn.h"
#include "ggml-backend.h"
#include "ggml-impl.h" // i
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "fp16/Float16.h"
#include "model_related_config.h"

#include <string.h>
#include <cstring>

#include <thread>
#include <vector>

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <fstream>
#include <limits>
#include <string>
#include <cmath>
#include <iostream>
#include <fcntl.h>
#include <sys/sysinfo.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <arm_neon.h>
// #include <nlohmann/json.hpp>
// using json = nlohmann::json;

#define GGML_COMMON_DECL_C
// #define RKNN_MATMUL_DEBUG

// #define RKNN_MATMUL_DEBUG_TIMING

#ifdef RKNN_MATMUL_DEBUG_TIMING
    int timing_debug_printf(const char * format, ...) {
        return printf(format);
    }
#else
    int timing_debug_printf(const char * format, ...) {
        return 0;
    }
#endif

#include "ggml-common.h"

using namespace rknpu2;
#define UNUSED(x) (void)(x)

#define GGML_RKNPU2_INPUT_SCALE 1.7f


#define DMA_HEAP_IOCTL_ALLOC	_IOWR(DMA_HEAP_IOC_MAGIC, 0x0,\
				      struct dma_heap_allocation_data)
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_BUF_SYNC_START  (0 << 2)
#define DMA_BUF_SYNC_END    (1 << 2)
#define DMA_BUF_SYNC_READ   (1 << 0)
#define DMA_BUF_SYNC_WRITE  (2 << 0)
#define DMA_BUF_SYNC_RW     (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_BASE        'b'
#define DMA_BUF_IOCTL_SYNC   _IOW(DMA_BUF_BASE, 0, uint64_t)
#define CMA_HEAP_SIZE       (1024 * 1024)

struct matrixShape{
    int64_t row;
    int64_t col;
};
struct matrixPair{
    matrixShape src0;
    matrixShape src1;
    std::string name;
};

std::vector<matrixPair> support_matrices;

struct pad_data{
    void * data;
    bool is_padded=false;
};

enum matrix_t{
    FLOAT16,
    INT8,
    INT4
};

// std::map<matrix_t, size_t> ELEMENT_SIZE = {
//     {FLOAT16, sizeof(float16)},
//     {INT8, sizeof(int8_t)},
//     {INT4, sizeof(int8_t)}
// };

inline size_t get_element_size(matrix_t type) {
    switch(type) {
        case FLOAT16: return sizeof(float16);
        case INT8:    return sizeof(int8_t);
        case INT4:    return sizeof(int8_t); // INT4 packed in int8_t
        default:      return 0;
    }
}

struct mat_info{
    int64_t row;
    int64_t col;
    int64_t pad_row;
    int64_t pad_col;
    matrix_t matrix_type;
    void * ori_data;
    void * pad_data;
    size_t ori_size;
    size_t pad_size;
    char * matrix_name;
    bool is_padded=false;
    bool is_A=false;

    mat_info(int64_t row, 
        int64_t col,
        int64_t pad_row, 
        int64_t pad_col, 
        matrix_t matrix_type, 
        void * ori_data, 
        void * pad_data, 
        size_t ori_size, 
        size_t pad_size, 
        char * matrix_name, 
        bool is_padded, 
        bool is_A): 
        row(row), 
        col(col), 
        pad_row(pad_row), 
        pad_col(pad_col), 
        matrix_type(matrix_type), 
        ori_data(ori_data), 
        pad_data(pad_data), 
        ori_size(ori_size), 
        pad_size(pad_size), 
        matrix_name(matrix_name), 
        is_padded(is_padded), 
        is_A(is_A)
        {}

    mat_info(int64_t row_, 
        int64_t col_,
        matrix_t matrix_type_, 
        void * origin_data_, 
        bool is_A_, 
        char* name_)
    : mat_info(
            row_,
            col_,
            row_,
            col_,
            matrix_type_,
            origin_data_,
            origin_data_,
            row_ * col_ * get_element_size(matrix_type_),
            row_ * col_ * get_element_size(matrix_type_),
            name_,
            false,
            is_A_
        )
    {
        // notes for padding: 
        // for A (M, K) (intermediate values), 
        //     performance layout [K1, M, subK] requires K to be aligned with subK (8 for F16) 
        // for B (K, N) (weights), 
        //     native layout [N1, K, subN, subK] requires 
        //     K to be aligned with subK (32 for F16), 
        //     N aligned with subN (16 for F16)
        // So at the end K will need to be aligned with 32, and N will need to be aligned with 16

        //TODO: also here K_align N_align is hardcoded 
        if(is_A_){
            // (M, K) 
            // this->pad_col = ((col_ - 1) / 32 + 1) * 32;
            // it's hard to pad K here, since A is now row-major and we're padding the end of rows 
            GGML_ASSERT(col_ % 32 == 0);
            
        }
        else {
            // no, B is already in native layout here
        }
    }

    mat_info(int64_t row_, int64_t col_, matrix_t matrix_type_, void * origin_data_, bool is_A_)
        : mat_info( row_, col_, matrix_type_, origin_data_, is_A_, NULL) 
    {}
};


struct matmul_ctx{
    mat_info mat_A;
    mat_info mat_B;
    rknn_matmul_type type;
    int thread_idx;
    bool matrix_B00_need_set_io = false;
    int64_t ori_n;
    const char * name;
    matmul_ctx(mat_info mat_A, mat_info mat_B, rknn_matmul_type type, int thread_idx, int64_t ori_n, const char* name_): mat_A(mat_A), mat_B(mat_B), type(type), thread_idx(thread_idx), ori_n(ori_n), name(name_) {}
};

void check_pad_float(const int64_t row, const int64_t col, void *pad_A01)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%2.f ", ((float*)pad_A01)[i * col + j]);
        }
        printf("\n");
    }
}
bool read_shape_pairs_from_json(
    const std::string& filename,
    std::vector<matrixPair>& out_pairs) {

    // std::ifstream ifs(filename);
    // if (!ifs.is_open()) {
    //     std::cerr << "Cannot open JSON file: " << filename << std::endl;
    //     return false;
    // }

    // json j;
    // try {
    //     ifs >> j;
    //     // 拿到 pairs 数组
    //     const auto& arr = j.at("pairs");
    //     for (const auto& item : arr) {
    //         matrixPair sp;
    //         sp.src0.row = item.at("src0").at("row").get<int64_t>();
    //         sp.src0.col = item.at("src0").at("col").get<int64_t>();
    //         sp.src1.row = item.at("src1").at("row").get<int64_t>();
    //         sp.src1.col = item.at("src1").at("col").get<int64_t>();
    //         sp.name = item.at("name").get<std::string>();
    //         out_pairs.push_back(sp);
    //     }
    // } catch (const std::exception &e) {
    //     std::cerr << "Get JSON failed: " << e.what() << std::endl;
    //     return false;
    // }
    return true;
}


struct ggml_rknpu2_matmul_kernel{
    void* workdata;
    size_t work_size = 0;
    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    std::atomic<bool> is_using = false;
    int thread_idx=0;
    const char * name;

    rknn_tensor_mem* A;
    rknn_tensor_mem* B;
    rknn_tensor_mem* C;

    void * A_data = NULL;
    void * B_data = NULL;
    size_t A_size = 0; //0 means not allocated
    size_t B_size = 0;
    bool B_is_copied = false;
};

static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}


static uint64_t rknpu2_allocated_bytes = 0;

struct matrix_ctx{
    int64_t row;
    int64_t col;
    void* data;
    const char* name;
};

struct in_kernel_time{
    double memcpy_to_kernel_time;
    double find_kernel_time;
    double set_io_time;
    double run_time;
    double memcpy_to_result_time;
    double sum_result_time;
};

void matrix_B_to_perf_layout_single_thread(int total_blocks_outer, int total_blocks_j, int32_t subN, int32_t subK, int32_t K, float16 *__restrict__ dst_ptr, const float16 *start_point, int thread_idx, int total_threads);
void A00xB00(const int64_t A_row_00, const int64_t B_col_00, ggml_tensor *dst, int64_t n, int offset_col, int offset_row);
void A00xB00(const int64_t A_row_00, const int64_t B_col_00, ggml_tensor *dst, int64_t n);
void side_matrix_multiplication(matrix_ctx A, matrix_ctx B, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time);
void side_matrix_multiplication(matrix_ctx A, matrix_ctx B, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time, bool matrix_B00_need_set_io);
void side_matrix_multiplication(const int64_t A_row_01, const int64_t A_col_01, const int64_t B_row_10, const int64_t B_col_10, void *pad_A01, void *pad_B10, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time);
void side_matrix_multiplication(const int64_t A_row_01, const int64_t A_col_01, const int64_t B_row_10, const int64_t B_col_10, void *pad_A01, void *pad_B10, rknn_matmul_type type, int thread_idx, int &C_tile, ggml_tensor *dst, int64_t n, int id, int offset_col, int offset_row, in_kernel_time &kernel_time, bool matrix_B00_need_set_io);

#define GGML_RKNPU2_MAX_MATMUL_KERNELS 256
static ggml_rknpu2_matmul_kernel matmul_kernels[GGML_RKNPU2_MAX_MATMUL_KERNELS];

static int matmul_kernels_count = 0;

const char* rknpu2_matmul_type_to_string(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return "FLOAT16_MM_FLOAT16_TO_FLOAT32";
        case RKNN_INT8_MM_INT8_TO_INT32:
            return "INT8_MM_INT8_TO_INT32";
        case RKNN_INT4_MM_INT4_TO_INT16:
            return "INT4_MM_INT4_TO_INT16";
        default:
            GGML_ASSERT(0);
    }
}

struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type, int thread_idx, const void * A_data, void * B_data, size_t A_size, size_t B_size, const char * name = NULL) {
  for (int i = 0; i < matmul_kernels_count; i++) {
    ggml_rknpu2_matmul_kernel *kernel = &matmul_kernels[i];
    bool flag = false;
    if (
            kernel->info.M == m &&
            kernel->info.K == k &&
            kernel->info.N == n &&
            kernel->is_using == false &&
            kernel->info.type == type 
            && kernel->thread_idx == thread_idx  // Add core affinity check
        ){
            flag = true;
    }
    if(flag && name != NULL && kernel->name != NULL){
        if(std::strcmp(name, kernel->name) == 0){
            timing_debug_printf("ggml-rknn: find a kernel at i: %d, kernel->name: %s:%d (%d,%d,%d)\n", i, kernel->name, kernel->thread_idx, m, k, n);
            return kernel;
        }
    }
  }
  return NULL;
}

// MARK: Kernel find

static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(matrix_ctx A, matrix_ctx B, rknn_matmul_type type, int thread_idx){
    return ggml_rknpu2_matmul_kernel_find(A.row, A.col, B.col, type, thread_idx, NULL, NULL, 0, 0);
}

static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(matmul_ctx ctx){
    return ggml_rknpu2_matmul_kernel_find(ctx.mat_A.row, ctx.mat_A.col, ctx.mat_B.col, ctx.type, ctx.thread_idx, ctx.mat_A.ori_data, ctx.mat_B.ori_data, ctx.mat_A.ori_size, ctx.mat_B.ori_size, ctx.name);
}

ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(const void* A_data, void* B_data, size_t A_size, size_t B_size, int m, int k, int n, rknn_matmul_type type, int thread_idx, int &initialized, bool is_init = false, const char * name = NULL){
    // find or create a new 
    ggml_rknpu2_matmul_kernel* kernel = NULL;
    rknn_core_mask core_mask = (rknn_core_mask)(1 << (thread_idx % 3));
    if(!is_init){
        kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type, thread_idx, A_data, B_data, A_size, B_size, name);
    }

    if(kernel != NULL && kernel -> C){
        //TODO: review the kernel logic
        return kernel;
    }
    else{
        if (m == 1) {
            // decode phase
            kernel = &matmul_kernels[matmul_kernels_count++];
        } else {
            // prefill phase 
            kernel = (ggml_rknpu2_matmul_kernel *)malloc(sizeof(ggml_rknpu2_matmul_kernel));
        }
        if(matmul_kernels_count % GGML_RKNPU2_MAX_MATMUL_KERNELS == 0)
            matmul_kernels_count = 0;
        memset(kernel, 0, sizeof(ggml_rknpu2_matmul_kernel));

        kernel->thread_idx = thread_idx;
        kernel->info.M = m;
        kernel->info.K = k;
        kernel->info.N = n;
        kernel->info.type = type;
        kernel->info.B_layout = 1; // B use native layout (weight)
        kernel->info.AC_layout = 1; // A and C use performance layout (intermediate)
        kernel->name = name;

        int ret = rknn_matmul_create(&(kernel->ctx), &(kernel->info), &(kernel->io_attr));
        GGML_ASSERT(ret == 0);


        // if(thread_idx == 0)
        //     rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_0);
        // else if(thread_idx == 1)
        //     rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_1);
        // else if(thread_idx == 2)
        //     rknn_matmul_set_core_mask(kernel->ctx, RKNN_NPU_CORE_2);

        rknn_matmul_set_core_mask(kernel->ctx, core_mask);

        {
            auto kernel_mem_create_time = std::chrono::high_resolution_clock::now();
            kernel->A = rknn_create_mem(kernel->ctx, kernel->io_attr.A.size);
            if (kernel->A == NULL) {
                fprintf(stderr, "ggml-rknn: rknn_create_mem failed for A node %s\n", name);
            }
            kernel->B = rknn_create_mem(kernel->ctx, kernel->io_attr.B.size);
            if (kernel->B == NULL) {
                fprintf(stderr, "ggml-rknn: rknn_create_mem failed for B node %s\n", name);
            }
            kernel->C = rknn_create_mem(kernel->ctx, kernel->io_attr.C.size);
            if (kernel->C == NULL) {
                fprintf(stderr, "ggml-rknn: rknn_create_mem failed for C node %s\n", name);
            }
            auto kernel_mem_create_time_end = std::chrono::high_resolution_clock::now();   
            auto kernel_mem_create_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_mem_create_time_end - kernel_mem_create_time).count();
            timing_debug_printf("ggml-rknn: kernel_mem_create_duration: %ld us %s\n", kernel_mem_create_duration, name);
            timing_debug_printf("ggml-rknn: matmul_kernel_create %s:%d, (%d,%d,%d)\n", name, thread_idx, m, k, n);
        }
    }
    {
        kernel->A_size = A_size;
        kernel->B_size = B_size;
    }

    return kernel;
}

static struct ggml_rknpu2_matmul_kernel * ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_matmul_type type, rknpu2::float16 * A_data, rknpu2::float16 * B_data, size_t A_size, size_t B_size, int &initialized) {
    return ggml_rknpu2_matmul_kernel_create(A_data, B_data, A_size, B_size, m,k,n,type,1, initialized);
}

void transposed_matrix_to_perf_layout(const void *src, void *dst, int32_t K, int32_t N, int32_t subK, int32_t subN)
{
    // subN = 16 subK = 32 on 3588
    const void * start_point = src;
    int dst_offset = 0;
    for(int outer = 0; outer < N / subN; outer++){
        for(int j = 0 ; j < K / subK; j++){
            int start_offset = outer * subN * K + j * subK;
            for(int i = 0 ; i < subN; i++){
                memcpy((float16 *)dst + dst_offset, (float16 *)start_point + start_offset, subK * sizeof(float16));
                dst_offset += subK;
                start_offset += K;
            }
        }
    }
}


static void process_range(
    const float16* src, float16* dst, int K,
    int start_outer, int end_outer,
    int subK, int subN, int total_j)
{
    const int block_size = subN * total_j * subK; // 每个outer块的大小
    
    for (int outer = start_outer; outer < end_outer; ++outer) {
        // 计算当前outer块的目标起始位置
        float16* block_dst = dst + outer * block_size;
        
        // 源矩阵的起始行
        const float16* outer_src = src + outer * subN * K;
        
        // 遍历j维度（K方向分块）
        for (int j = 0; j < total_j; ++j) {
            // 当前j块的起始位置
            const float16* j_src = outer_src + j * subK;
            
            // 遍历subN行
            for (int i = 0; i < subN; ++i) {
                // 目标位置：block + j块偏移 + 行内偏移
                float16* dst_pos = block_dst + (j * subN + i) * subK;
                
                // 源位置：当前outer块的i行，j列
                const float16* src_pos = j_src + i * K;
                
                // 32个float16=64字节，正好一个cacheline
                // static_assert(sizeof(float16)*32 == 64, "Cache line size mismatch");
                memcpy(dst_pos, src_pos, subK * sizeof(float16));
            }
        }
    }
}

//TODO: 
void transposed_matrix_to_perf_layout_multi_threads(
    const void* src, void* dst, 
    int32_t K, int32_t N,
    int32_t subK, int32_t subN) 
{
    const float16* src_ptr = static_cast<const float16*>(src);
    float16* dst_ptr = static_cast<float16*>(dst);

    const int total_outer = N / subN;  // 外层循环次数
    const int total_j = K / subK;      // j方向分块数
    
    // 根据物理核心数设置线程数（建议4-6）
    const int n_threads = 1; 
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    // 计算每个线程处理的outer范围
    const int min_blocks_per_thread = total_outer / n_threads;
    const int remainder = total_outer % n_threads;

    for (int t = 0; t < n_threads; ++t) {
        // 带余数的均衡分配
        const int start_outer = t * min_blocks_per_thread + std::min(t, remainder);
        const int end_outer = start_outer + min_blocks_per_thread + (t < remainder ? 1 : 0);

        threads.emplace_back([=]() {
            process_range(
                src_ptr, dst_ptr, K,
                start_outer, end_outer,
                subK, subN, total_j
            );
        });
    }

    for (auto& th : threads) {
        th.join();
    }
}

/**
    * @brief convert norm layout to perf layout
    * column major -> row major
    * norm layout (RKNN): [M,K] (calling [m * K + k])
    * perf layout (RKNN): [K/subK, M, subK] (calling [ksk * M * subK + m * subK + j])
    */
template <typename Ti, typename To>
void norm_layout_to_perf_layout(Ti * src, To * dst, int32_t M, int32_t K, int32_t subK, bool isInt4Type) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int outter_size = (int) std::ceil(K * 1.0f / subK);
    for (int i = 0; i < outter_size; i++) {
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < subK; j++) {
                int ki = i * subK + j;
                if (isInt4Type) {
                    int    input_index  = m * K + ki;
                    int    output_index = i * M * subK + m * subK + j;
                    int8_t int4         = src[input_index];
                    if (ki >= K) {
                        int4 = 0;
                    } else {
                        int4 = int4 & 0xf;
                    }
                    if (output_index % 2 == 0) {
                        dst[output_index / 2] = int4 << 4;
                    } else {
                        dst[output_index / 2] = (int4 | (int8_t) (dst[output_index / 2]));
                    }
                } else {
                    if (ki >= K) {
                        dst[i * M * subK + m * subK + j] = 0;
                    } else {
                        dst[i * M * subK + m * subK + j] = src[m * K + ki];
                    }
                }
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_debug_printf("ggml-rknn: norm_layout_to_perf_layout duration: %ld us\n", duration);
}

template void norm_layout_to_perf_layout<int8_t, int8_t>(int8_t * src, int8_t * dst, int32_t M, int32_t K, int32_t subK, bool isInt4Type);
template void norm_layout_to_perf_layout<float16, float16>(float16 * src, float16 * dst, int32_t M, int32_t K, int32_t subK, bool isInt4Type);

/**
     * @brief convert norm layout to native layout
     * norm layout:  [K,N]
     * native layout: [N1, K1, subN, subK]
     *
     */
template <typename Ti, typename To>
void norm_layout_to_native_layout(Ti * src, To * dst, int32_t K, int32_t N, int32_t subN, int32_t subK,
                                  bool isInt4Type) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int N_remain = (int) std::ceil(N * 1.0f / subN);
    int K_remain = (int) std::ceil(K * 1.0f / subK);
    for (int i = 0; i < N_remain; i++) {
        for (int j = 0; j < K_remain; j++) {
            for (int n = 0; n < subN; n++) {
                int ni = i * subN + n;
                for (int k = 0; k < subK; k++) {
                    int ki = j * subK + k;
                    if (isInt4Type) {
                        int    input_index  = ki * N + ni;
                        int    output_index = i * (K_remain * subN * subK) + j * (subN * subK) + n * subK + k;
                        int8_t int4         = src[input_index];
                        if (ki < K && ni < N) {
                            int4 = int4 & 0xf;
                        } else {
                            int4 = 0;
                        }
                        if (output_index % 2 == 0) {
                            dst[output_index / 2] = int4 << 4;
                        } else {
                            int8_t temp           = dst[output_index / 2];
                            int8_t result         = temp | int4;
                            dst[output_index / 2] = result;
                        }
                    } else {
                        if (ki < K && ni < N) {
                            dst[((i * K_remain + j) * subN + n) * subK + k] = src[ki * N + ni];
                        } else {
                            dst[((i * K_remain + j) * subN + n) * subK + k] = 0;
                        }
                    }
                }
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_debug_printf("ggml-rknn: norm_layout_to_native_layout duration: %ld us\n", duration);
}

template void norm_layout_to_native_layout<int8_t, int8_t>(int8_t * src, int8_t * dst, int32_t K, int32_t N,
                                                           int32_t subN, int32_t subK, bool isInt4Type);
template void norm_layout_to_native_layout<float16, float16>(float16 * src, float16 * dst, int32_t K, int32_t N,
                                                             int32_t subN, int32_t subK, bool isInt4Type);

/**
     * @brief convert ggml layout to native layout
     * ggml layout:  [N, K]
     * native layout: [N1, K1, subN, subK]
     *
     * note here ggml layout is column major, 

        -  normal layout: (K, N) (row major)
              [K1N1, K1N2, ..., K1Nn, --- continuous this direction ---
               K2N1, K2N2, ..., K2Nn,
               ...
               KkN1, KkN2, ..., KkNn]

              float16: for RK3588

        -   native layout: (N / 16, K / 32, 16, 32) (row major)
              [K1N1,  K2N1,  ..., K32N1, --- continuous this direction ---
               K1N2,  K2N2,  ..., K32N2,
               ...
               K1N16, K2N16, ..., K32N16,
               K33N1, K34N1, ..., K64N1,
               K33N2, K34N2, ..., K64N2,
               ...
               K(k-31)N16, K(k-30)N16, ..., KkN16,
               K1N17, K2N17, ..., K32N17,
               K1N18, K2N18, ..., K32N18,
               ...
               K(k-31)Nn, K(k-30)Nn, ..., KkNn]
        -   ggml layout: (N, K) (column major)
              [N1K1, N1K2, ..., N1Kn, 
               N2K1, N2K2, ..., N2Kn,
               ...
               NkK1, NkK2, ..., NkKn]
               |
               |
               continuous this direction 
     */
template <typename Ti, typename To>
void ggml_layout_to_native_layout(Ti * src, To * dst, int32_t K, int32_t N, int32_t subN, int32_t subK,
                                  bool isInt4Type) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int  N_remain   = (int) std::ceil(N * 1.0f / subN);
    int  K_remain   = (int) std::ceil(K * 1.0f / subK);
    for (int i = 0; i < N_remain; i++) {
        for (int j = 0; j < K_remain; j++) {
            for (int n = 0; n < subN; n++) {
                int ni = i * subN + n;
                for (int k = 0; k < subK; k++) {
                    int ki = j * subK + k;
                    if (isInt4Type) {
                        int    input_index  = ki * N + ni;
                        int    output_index = i * (K_remain * subN * subK) + j * (subN * subK) + n * subK + k;
                        int8_t int4         = src[input_index];
                        if (ki < K && ni < N) {
                            int4 = int4 & 0xf;
                        } else {
                            int4 = 0;
                        }
                        if (output_index % 2 == 0) {
                            dst[output_index / 2] = int4 << 4;
                        } else {
                            int8_t temp           = dst[output_index / 2];
                            int8_t result         = temp | int4;
                            dst[output_index / 2] = result;
                        }
                    } else {
                        //TODO: parallelize it 
                        memcpy(dst + ((i * K_remain + j) * subN + n) * subK, src + j * subK + K * ni, sizeof(To) * subK );
                        // if (ki < K && ni < N) {
                        //     dst[((i * K_remain + j) * subN + n) * subK + k] = src[ki + K * ni];
                        // } else {
                        //     dst[((i * K_remain + j) * subN + n) * subK + k] = 0;
                        // }
                    }
                }
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_debug_printf("ggml-rknn: ggml_layout_to_native_layout duration: %ld us\n", duration);
}

/**
     * @brief convert perf to norm layout
     * perf layout: [K1, M, subK]
     * norm layout: [M,K]
     *
     */
template <typename Ti, typename To>
void perf_layout_to_norm_layout(Ti * src, To * dst, int32_t M, int32_t K, int32_t K_remain, int32_t subK) {
    for (int i = 0; i < K_remain; i++) {
        for (int j = 0; j < subK; j++) {
            for (int m = 0; m < M; m++) {
                int ki = i * subK + j;
                if (ki < K) {
                    dst[m * K + ki] = src[i * M * subK + m * subK + j];
                }
            }
        }
    }
}

void matrix_B_to_perf_layout_single_thread(
    int total_outer, int total_j, int subN, int subK, int K,
    float16* dst, const float16* src,
    int thread_id, int num_threads)
{
    const int blocks_per_thread = (total_outer + num_threads - 1) / num_threads;
    const int start_outer = thread_id * blocks_per_thread;
    const int end_outer = std::min((thread_id + 1) * blocks_per_thread, total_outer);
    
    int dst_offset = start_outer * subN * (K / subK) * subK;
    
    for (int outer = start_outer; outer < end_outer; ++outer) {
        for (int j = 0; j < total_j; ++j) {
            const int src_offset = outer * subN * K + j * subK;
            for (int i = 0; i < subN; ++i) {
                const int src_pos = src_offset + i * K;
                const int dst_pos = dst_offset + (j * subN + i) * subK;
                memcpy(&dst[dst_pos], &src[src_pos], subK * sizeof(float16));
            }
        }
        dst_offset += subN * total_j * subK; // 移动到下一个连续区域
    }
}

void perf_matrixC_to_norm_layout(void *src, void *&dst, int32_t M, int32_t N){
    if(M == 1){
        dst = src;
        return;
    }

    const int mem_unit = 4;
    int dst_offset = 0;
    for(int i = 0; i < M; i++){
        for(int outer = 0; outer < N / mem_unit ; outer++){
            memcpy((float*)dst + dst_offset, (float*)src + outer * mem_unit * M + i * mem_unit, mem_unit * sizeof(float));
            dst_offset += mem_unit;
        }
    }
}
void matrixA_to_perf_layout(const void* src, void *&dst, int32_t M, int32_t K){
    dst = malloc(M * K * sizeof(float16));
    const int mem_unit = 8;
    int dst_offset = 0;
    for(int outer = 0; outer < K / mem_unit ; outer++){
        for(int i = 0; i < M; i++){
            int src_offset = outer * mem_unit + i *K;
            memcpy((float16 *)dst + dst_offset, (float16 *)src + src_offset, mem_unit * sizeof(float16));
            dst_offset += mem_unit;
        }
    }
}

void perf_matrixC_to_norm_transposed_layout(void *src, void *&dst, int32_t M, int32_t N){
    if(M == 1){
        dst = src;
        return;
    }
    const int mem_unit = 4;
    int dst_offset = 0;
    for(int outer = 0; outer < N / mem_unit ; outer++){
        for(int j = 0 ; j < mem_unit; j++){
            for(int i = 0; i < M; i++){
                ((float*)dst)[dst_offset] = ((float*) src)[outer * M * mem_unit + i * mem_unit + j];
                // printf("dst_offset: %d, src_offset: %d, dst: %5.f, src: %5.f\n", dst_offset, outer * M * mem_unit + i * mem_unit + j, ((float*)dst)[dst_offset], ((float*) src)[outer * M * mem_unit + i * mem_unit + j]);
                dst_offset++;
            }
        }
    }
}

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);
// prototypes
rknn_tensor_type ggml_type_to_rknn_type(ggml_type type);
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type);
rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type);
rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type);
const char* rknpu2_matmul_type_to_string(rknn_matmul_type type);
const char* rknpu2_tensor_type_to_string(rknn_tensor_type type);
size_t get_type_size(rknn_matmul_type type);
void dequantize_row_q8_0(const block_q8_0 * x, float * y, int64_t k) ;

void compute_submat_mul(int64_t m, int64_t k, const void * A_data, void * B_data, ggml_tensor * dst, int64_t row_start, int64_t row_end, int thread_idx, rknn_matmul_type type, const ggml_tensor * src0, const ggml_tensor * src1) ;


struct ggml_backend_rknn_context {
    int n_threads = 1;
};


// rknn tensor type -> rknn matmul type
rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_INT8_MM_INT8_TO_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_INT4_MM_INT4_TO_INT16;
        default:
            GGML_ASSERT(0);
    }
}

// rknn matmul type -> string


// rknn tensor type -> string
const char* rknpu2_tensor_type_to_string(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT32:
            return "FLOAT32";
        case RKNN_TENSOR_FLOAT16:
            return "FLOAT16";
        case RKNN_TENSOR_INT8:
            return "INT8";
        case RKNN_TENSOR_INT16:
            return "INT16";
        case RKNN_TENSOR_INT32:
            return "INT32";
        case RKNN_TENSOR_UINT8:
            return "UINT8";
        case RKNN_TENSOR_UINT16:
            return "UINT16";
        default:
            GGML_ASSERT(0);
    }
}

// ggml type -> rknn tensor type
rknn_tensor_type ggml_type_to_rknn_type(ggml_type type) {
    switch(type) {
        case GGML_TYPE_F32:
            return RKNN_TENSOR_FLOAT32;
        case GGML_TYPE_F16:
            return RKNN_TENSOR_FLOAT16;
        case GGML_TYPE_I8:
            return RKNN_TENSOR_INT8;
        case GGML_TYPE_I16:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

// rknn_matmul_type -> rknn_tensor_type
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return RKNN_TENSOR_FLOAT16;
        case RKNN_INT8_MM_INT8_TO_INT32:
            return RKNN_TENSOR_INT8;
        case RKNN_INT4_MM_INT4_TO_INT16:
            return RKNN_TENSOR_INT4;
        default:
            GGML_ASSERT(0);
    }
}

// rknn_tensor_type -> rknn_tensor_type
rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_TENSOR_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_TENSOR_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

static inline struct timespec * timespec_sub(const struct timespec *ts_a, const struct timespec *ts_b, struct timespec * ts_out){
    ts_out->tv_sec = ts_a->tv_sec - ts_b->tv_sec;
    ts_out->tv_nsec = ts_a->tv_nsec - ts_b->tv_nsec;
    if (ts_out->tv_nsec < 0) {
        ts_out->tv_sec--;
        ts_out->tv_nsec += 1000000000;
    }
    return ts_out;
}

static inline unsigned long long timespec_ns(const struct timespec * ts){
    return (unsigned long long)ts->tv_sec * 1000000000ull + (unsigned long long)ts->tv_nsec;
}

static ggml_status ggml_backend_rknn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // printf("rknn graph compute!!!!!!!!, cgraph->n_nodes: %d\n", cgraph->n_nodes);
    
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        struct timespec start_compute_forward;
        clock_gettime(CLOCK_MONOTONIC, &start_compute_forward);
        bool ok = ggml_rk_compute_forward(backend, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        struct timespec end_compute_forward;
        clock_gettime(CLOCK_MONOTONIC, &end_compute_forward);

        // printf("Node %d: %s (%s) start_time: %llu, end_time: %llu, compute time: %llu ns \n", 
        //     i, 
        //     node->name, 
        //     ggml_op_name(node->op), 
        //     timespec_ns(&start_compute_forward), 
        //     timespec_ns(&end_compute_forward),
        //     timespec_ns(timespec_sub(&end_compute_forward, &start_compute_forward, &end_compute_forward)) 
        // );


        GGML_ASSERT(ok);


    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_rknn2_free(ggml_backend_t backend) {
    //ggml_backend_rknn_context * ctx = (ggml_backend_rknn_context *) backend->context;
    // if(ctx != nullptr) delete ctx;
    for(int i = 0 ; i < matmul_kernels_count; i++){
        ggml_rknpu2_matmul_kernel *kernel = &matmul_kernels[i];
        rknn_destroy_mem(kernel->ctx, kernel->A);
        rknn_destroy_mem(kernel->ctx, kernel->B);
        rknn_destroy_mem(kernel->ctx, kernel->C);
        rknn_matmul_destroy(kernel->ctx);
    }

    delete backend;
}

static const char * ggml_backend_rknn_name(ggml_backend_t backend) {
    return "RKNN";

    UNUSED(backend);
}
static void ggml_backend_rknn_free(ggml_backend_t backend) {
    ggml_rknn2_free(backend);

    GGML_UNUSED(backend);
}

static bool has_init_kernel_from_file = false;

void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads){
    timing_debug_printf("ggml_backend_rknn_set_n_threads: sizeof backend %d\n", sizeof(typeof(backend_rknn)));
    timing_debug_printf("ggml_backend_rknn_set_n_threads start\n");
    timing_debug_printf("ggml_backend_rknn_set_n_threads: %d\n", n_threads);
    GGML_ASSERT(ggml_backend_is_rknn(backend_rknn));
    ggml_backend_rknn_context * ctx = (ggml_backend_rknn_context *) backend_rknn -> context;
    ctx->n_threads = n_threads;
   // printf("n_threads: %d\n", n_threads);
    if(!has_init_kernel_from_file && false){

        std::vector<matrixPair> matrix_pairs;
        bool status = read_shape_pairs_from_json(std::string(CONFIG_DIR) + "/mat_kernel_size.json", matrix_pairs);
        // bool status = true;
        if(!status){
            printf("read shape pairs from json failed!\n");
            exit(-1);
        }
    
        for(matrixPair &matrix_pair : matrix_pairs){
            // printf("matrix_pair: (%d, %d), (%d, %d)\n", matrix_pair.src0.row, matrix_pair.src0.col, matrix_pair.src1.row, matrix_pair.src1.col);
            matrix_ctx A = {matrix_pair.src0.row, matrix_pair.src0.col, NULL, "A"};
            matrix_ctx B = {matrix_pair.src1.row, matrix_pair.src1.col, NULL, "B"};
            size_t matrix_A_size = A.row * A.col * sizeof(float16);
            // size_t matrix_B_size = B.row * B.col * sizeof(float16);
            int initialized = 0;

            int mod_number = 16 * n_threads;
            // printf("matrix_pair.name.c_str(): %s\n", matrix_pair.name.c_str());
            for(int i = 0 ; i < n_threads;i++){
                    int split_B_col= (i + 1) * B.col / 16 / n_threads * 16 - i * B.col / 16 / n_threads * 16;
                    // if(i == n_threads - 1)
                    //     split_B_col = B.col - (n_threads - 1) * split_B_col;
                    size_t matrix_B_size = split_B_col * B.row * sizeof(float16);
                    char * op_name = (char*)malloc(sizeof(char) * matrix_pair.name.length());
                    for(int j = 0 ; j <matrix_pair.name.length();j++){
                        op_name[j] = matrix_pair.name[j];
                    }
                    ggml_rknpu2_matmul_kernel * kernel = ggml_rknpu2_matmul_kernel_create(
                    A.data, 
                    B.data, 
                    matrix_A_size, 
                    matrix_B_size, 
                    A.row, 
                    A.col, 
                    split_B_col, 
                    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, 
                    i, 
                    initialized,
                    true,
                    op_name
                );
            }
        }
        has_init_kernel_from_file = true;
        for(int i = 0 ; i < matmul_kernels_count; i++){
            printf("kernel %d:\n", i);
            printf("dims: %d, %d, %d, kernel->name: %s\n", matmul_kernels[i].info.M, matmul_kernels[i].info.K, matmul_kernels[i].info.N, matmul_kernels[i].name);
        }
    }
    // printf("set n threads done\n");
}

static ggml_backend_i ggml_backend_rknn_i = {
    /* .get_name                = */ ggml_backend_rknn_name,
    /* .free                    = */ ggml_backend_rknn_free,
    /* .set_tensor_async        = */ NULL,  /* ggml_backend_opencl_set_tensor_async */
    /* .get_tensor_async        = */ NULL,  /* ggml_backend_opencl_get_tensor_async */
    /* .cpy_tensor_async        = */ NULL,  /* ggml_backend_opencl_cpy_tensor_async */
    /* .synchronize             = */ NULL,  /* ggml_backend_opencl_synchronize */
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rknn_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};
static int ggml_backend_rknn_n_devices = 1;
static const char * ggml_backend_rknn_reg_get_name(ggml_backend_reg_t reg) {
    return "RKNN";

    GGML_UNUSED(reg);
}
static size_t ggml_backend_rknn_reg_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_rknn_n_devices;

    GGML_UNUSED(reg);
}
static const char * ggml_backend_rknn_device_get_name(ggml_backend_dev_t dev) {
    return "RKNN";
    GGML_UNUSED(dev);
}
static const char * ggml_backend_rknn_device_get_description(ggml_backend_dev_t dev) {
    #if defined(GGML_BLAS_USE_ACCELERATE)
        return "Accelerate";
    #elif defined(GGML_BLAS_USE_MKL)
        return "MKL";
    #elif defined(GGML_BLAS_USE_BLIS)
        return "BLIS";
    #elif defined(GGML_BLAS_USE_NVPL)
        return "NVPL";
    #elif defined(OPENBLAS_VERSION)
        return "OpenBLAS";
    #else
        return "RKNN";
    #endif

    GGML_UNUSED(dev);
}
static void ggml_backend_rknn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        *free = info.freeram;
        *total = info.totalram;
        *free = *free * info.mem_unit;
        *total = *total * info.mem_unit;
    } else {
        std::cout<< "sysinfo failed" << "\n";
    }

    GGML_UNUSED(dev);
}
static enum ggml_backend_dev_type ggml_backend_rknn_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}
static void ggml_backend_rknn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rknn_device_get_name(dev);
    props->description = ggml_backend_rknn_device_get_description(dev);
    props->type        = ggml_backend_rknn_device_get_type(dev);
    ggml_backend_rknn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { // check src/llama-context.cpp:276 
        /* .async                 = */ true,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ true,
    };
}
static ggml_backend_t ggml_backend_rknn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_rknn_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}
static ggml_backend_buffer_type_t ggml_backend_rknn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}
static ggml_backend_buffer_t ggml_backend_rknn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}
static bool ggml_backend_rknn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            const struct ggml_tensor * dst = op;
            const int64_t ne00 = src0->ne[0]; // k
            const int64_t ne01 = src0->ne[1]; // n
            const int64_t ne10 = src1->ne[0]; // k
            const int64_t ne11 = src1->ne[1]; // m
            const int64_t ne0 = dst->ne[0]; // n
            const int64_t ne1 = dst->ne[1]; // m


            bool result = true;

            if(dst->type != GGML_TYPE_F32){
                result = false;
            }
            result = false;

            // for(matrixPair &matrix_pair : support_matrices){
            //     matrix_ctx A = {matrix_pair.src0.row, matrix_pair.src0.col, NULL, matrix_pair.name.c_str()};
            //     matrix_ctx B = {matrix_pair.src1.row, matrix_pair.src1.col, NULL, matrix_pair.name.c_str()};
            //     if(A.row == ne11 && A.col == ne10 && B.row == ne00 && B.col == ne01\
            //         && std::strcmp(op->name, matrix_pair.name.c_str()) == 0    
            //     ){
            //         // printf("op->name: %s, src0->ne1: %d\n", op->name, ne01);
            //         result = true;
            //         break;
            //     }
            // }
            // return result;

            if (strstr(op->name, "ffn_") || strcmp(op->name, "result_output") == 0) {
                if (ne1 > 1) {
                    // no prefill
                    return false;
                }
                result = true;
            }
            // printf("ggml_backend_rknn_device_supports_op: %s, %d, %d, %d, %d\n", op->name, result, ne01, ne00, ne11); // n, k, m in rknn's notation
            return result;

        }

        default:
            return false;

    }

    GGML_UNUSED(dev);
}
static bool ggml_backend_rknn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}
static const struct ggml_backend_device_i ggml_backend_rknn_device_i = {
    /* .get_name             = */ ggml_backend_rknn_device_get_name,
    /* .get_description      = */ ggml_backend_rknn_device_get_description,
    /* .get_memory           = */ ggml_backend_rknn_device_get_memory,
    /* .get_type             = */ ggml_backend_rknn_device_get_type,
    /* .get_props            = */ ggml_backend_rknn_device_get_props,
    /* .init_backend         = */ ggml_backend_rknn_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rknn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_rknn_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_rknn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rknn_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
static ggml_backend_dev_t ggml_backend_rknn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_rknn_device = {
        /* .iface   = */ ggml_backend_rknn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,

    };

    // return &g_ggml_backend_rknn_device;
    return &ggml_backend_rknn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_rknn_reg_i = {
    /* .get_name         = */ ggml_backend_rknn_reg_get_name,
    /* .device_count     = */ ggml_backend_rknn_reg_device_count,
    /* .device_get       = */ ggml_backend_rknn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_rknn_get_proc_address,
};

static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_rknn_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return NULL;
}

ggml_backend_reg_t ggml_backend_rknn_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    if (!initialized) {
         reg = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface   = */ ggml_backend_rknn_reg_i,
            /* .context = */ NULL,
         };

        initialized = true;
    }

    return &reg;
}
static const char * ggml_backend_rknn_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    return "RKNN";
    GGML_UNUSED(buffer_type);
}

static ggml_guid_t ggml_backend_rknn_guid() {
    //c9bdb702-4936-4212-af35-a287d8c02920
    static ggml_guid guid = { 0xc9, 0xbd, 0xb7, 0x02, 0x49, 0x36, 0x42, 0x12, 0xaf, 0x35, 0xa2, 0x87, 0xd8, 0xc9, 0x29, 0x20 };
    return &guid;
}

bool ggml_backend_is_rknn(ggml_backend_t backend){
    return backend != NULL && ggml_guid_matches(backend -> guid, ggml_backend_rknn_guid());
}

// MARK: rknn INIT

ggml_backend_t ggml_backend_rknn_init(void) {
    // printf("@ggml-rknn.cpp\n");
    printf("ggml-rknn: start rknn init!\n");
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_rknn_reg(), 0);
    printf("ggml-rknn: register the rknn!\n");
    ggml_backend_rknn_context * context = (ggml_backend_rknn_context *) malloc(sizeof(ggml_backend_rknn_context));
    printf("ggml-rknn: creating the backend!\n");
    

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */  ggml_backend_rknn_guid(),
        /* .interface = */  ggml_backend_rknn_i,
        /* .device    = */  dev,
        /* .context   = */  context
    };
    printf("ggml-rknn: done creating rknn backend!\n");

    // bool status = read_shape_pairs_from_json(std::string(CONFIG_DIR) + "/mat_kernel_size.json", support_matrices);
    // // bool status = true;
    // if(!status){
    //     printf("ggml-rknn: read shape pairs from json failed!\n");
    //     return NULL;
    // }
    // printf("ggml-rknn: ne00: %d, ne01: %d, ne10: %d, ne11: %d, ne0: %d, ne1: %d\n", (int)ne00, (int)ne01, (int)ne10, (int)ne11, (int)ne0, (int)ne1);
    return backend;
}

struct ggml_rknn_data_pack{
    rknn_tensor_type type;
    void* ordered_data;
    int initialized;

    rknn_tensor_mem* B;
};


void copy_submatrix_A(bool is_last,
                    int A_block_row,
                    int A_block_column,
                    int A_row_ele_cnt,
                    int A_column_ele_cnt,
                    rknpu2::float16* sub_A_data,
                    rknpu2::float16* A_block_start,
                    int64_t ori_k,
                    int64_t A_row_cnt){
    if(is_last){
        for(int i = 0 ; i < A_block_row; i++){
            for(int j = 0 ; j < A_block_column; j++){
                if(i < A_row_ele_cnt && j < A_column_ele_cnt){
                    ((rknpu2::float16*)sub_A_data)[i * A_block_column + j] = ((rknpu2::float16*)A_block_start)[i * ori_k + j];
                }
                else{
                    ((rknpu2::float16*)sub_A_data)[i * A_block_column + j] = 0;
                }
            }
        }
    }
    else{
        for(int i = 0; i < A_row_cnt; i++){
            memcpy((rknpu2::float16*)sub_A_data + i * A_block_column, 
                    (rknpu2::float16*)A_block_start + i * ori_k , 
                    A_block_column * sizeof(rknpu2::float16));
        }
    }
}

void copy_submatrix_B(bool is_last,
                    int B_block_row,
                    int B_block_column,
                    int B_row_ele_cnt,
                    int B_column_ele_cnt,
                    rknpu2::float16* sub_B_data,
                    rknpu2::float16* B_block_start,
                    int64_t sub_n){
    if(is_last){
        for(int i = 0 ; i < B_block_row; i++){
            for(int j = 0 ; j < B_block_column; j++){
                if(i < B_row_ele_cnt && j < B_column_ele_cnt){
                    ((rknpu2::float16*)sub_B_data)[i * B_block_column + j] = ((rknpu2::float16*)B_block_start)[i * sub_n + j];
                }
                else{
                    ((rknpu2::float16*)sub_B_data)[i * B_block_column + j] = 0;
                }
            }
        }
    }
    else
    {
        for(int i = 0; i < B_block_row; i++){
            memcpy((rknpu2::float16*)sub_B_data + i * B_block_column , 
                    (rknpu2::float16*)B_block_start + i * sub_n, 
                    B_block_column * sizeof(rknpu2::float16));
        }
    }
}

size_t get_type_size(rknn_matmul_type type){
    switch(type){
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return sizeof(rknpu2::float16);
        case RKNN_INT8_MM_INT8_TO_INT32:
            return sizeof(int8_t);
        default:
            GGML_ASSERT(0);
    }
}

void dump_time_usage(double prepare_data_time, in_kernel_time &kernel_time, double total_run_time);
void side_matrix_mulmat_process(matmul_ctx &A00_B00, ggml_tensor *dst, in_kernel_time &kernel_time, int offset_col, int offset_row, int & C_tile);

// MARK: Matmul sub
void compute_submat_mul( // matrix A row
                        ggml_tensor *src_i,
                        ggml_tensor *src_w,
                        ggml_tensor *dst,
                        void *A_data,
                        void *B_data,
                        int64_t col_start,
                        int64_t col_end,
                        int thread_idx,
                        rknn_matmul_type type)
{
    // int64_t n = src_w->ne[1];
    int64_t dst_n = src_w->ne[1];
    int64_t m = src_i->ne[1];
    int64_t k = src_w->ne[0];
    int64_t ori_k = k;
    int64_t n = col_end - col_start;

    const int64_t A_row_00 = m; // m = 1
    const int64_t A_col_00 = k; // k = 2048

    const int64_t B_row_00 = k; 
    const int64_t B_col_00 = n; // n = 42752
    //TODO: subN parameterize instead of hardcode 16

    double prepare_data_time = 0; 
    double total_run_time = 0;
    in_kernel_time kernel_time;
    memset(&kernel_time, 0, sizeof(in_kernel_time));


    void * pad_A00 = nullptr; int fixed_A00 = 0;
    void * pad_B00 = nullptr; int fixed_B00 = 0;
    bool mat_A_mat_B_in_kernel = false;

    // void ** ptr_pad_A00 = &pad_A00;
    // void ** ptr_pad_B00 = &pad_B00;
    mat_info mat_A = mat_info(A_row_00, A_col_00, FLOAT16, A_data, true);
    mat_info mat_B = mat_info(B_row_00, B_col_00, FLOAT16, B_data, false);
    //TODO: size not padded
    
    matmul_ctx A00_B00 = matmul_ctx(mat_A, mat_B, type, thread_idx, dst_n, dst->name);

    ggml_rknpu2_matmul_kernel * tmp_kernel = ggml_rknpu2_matmul_kernel_find(A00_B00);

    if(tmp_kernel != NULL) mat_A_mat_B_in_kernel = true;

    // if(A_row_00 != 0 && A_col_00 != 0){
    //     matrixA_to_perf_layout(A_data, pad_A00, A_row_00, A_col_00);
    // }
    pad_A00 = mat_A.pad_data;
    bool matrix_B00_need_set_io = true;
    if(B_row_00 != 0 && B_col_00 != 0){
        // goals in this if condition:
        // 1. check if need set io
        // 2. pad_B00 is ready
        pad_B00 = mat_B.pad_data;
        // printf("pad_B00: %p\n", pad_B00);
        // check_pad(B_row_00, B_col_00, pad_B00);
        if(mat_A_mat_B_in_kernel){// make sure tmp_kernel != NULL
            fixed_B00 = 1; //B00's data should not be released after running
            if(tmp_kernel->B_data == pad_B00){
                // pad_B00 is already in kernel, do not need to set io
                matrix_B00_need_set_io = false;
                tmp_kernel->B_is_copied = true;
            }else{
                matrix_B00_need_set_io = true;
                tmp_kernel->B_is_copied = false;
            }
        }
    }
    A00_B00.matrix_B00_need_set_io = matrix_B00_need_set_io;

    int C_tile = 0;
    side_matrix_mulmat_process(A00_B00, dst, kernel_time, col_start, 0, C_tile);


    if(pad_A00!= nullptr && fixed_A00 == 0)
    {
        //TODO: 
        // free(pad_A00);
    }
    
    if(pad_B00!= nullptr && fixed_B00 == 0)
    {
        // free(pad_B00);
    }
}

//MARK: debug helper
// 一维矩阵乘法函数
template <typename Ti, typename To> std::vector<To> matrixMultiply(const Ti * A, const Ti * B, int M, int K, int N) {
    // A: [K, M] B: [K, N]
    std::vector<To> result(M * N, 0);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                rknpu2::float16 a_val = ((rknpu2::float16 *)A)[i * K + k];
                rknpu2::float16 b_val = ((rknpu2::float16 *)B)[k * N + j];
                float temp = (float) (a_val) * (float) (b_val);
                // rknpu2::float16 temp = a_val * b_val;
                // sum += (float) (a_val) * (float) (b_val);
                temp = temp + 0;
                if (!isnan(temp)){
                    sum += (float) (temp);
                }
            }
            result[i * N + j] = (To) sum;
        }
    }

    return result;
}

template <typename T>
bool arraysEqual(const std::vector<T> &arr1, const std::vector<T> &arr2, float eps = 0.0001f)
{
    if (arr1.size() != arr2.size())
    {
        return false;
    }

    for (size_t i = 0; i < arr1.size(); ++i)
    {
        if (std::abs(arr1[i] - arr2[i]) > eps)
        {
            return false;
        }
    }

    return true;
}

template bool arraysEqual<float>(const std::vector<float> &arr1, const std::vector<float> &arr2, float eps);
template bool arraysEqual<int32_t>(const std::vector<int32_t> &arr1, const std::vector<int32_t> &arr2, float eps);
template bool arraysEqual<int16_t>(const std::vector<int16_t> &arr1, const std::vector<int16_t> &arr2, float eps);
template bool arraysEqual<int8_t>(const std::vector<int8_t> &arr1, const std::vector<int8_t> &arr2, float eps);

template <typename T>
float arraysCosineSimilarity(const T *arr1, const T *arr2, size_t size, float eps=0.0001f)
{

    // 计算点积
    double dotProduct = 0.0;
    for (size_t i = 0; i < size; ++i)
    {
        dotProduct += arr1[i] * arr2[i];
    }

    // 计算向量范数
    double normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < size; ++i)
    {
        normA += std::pow(arr1[i], 2);
        normB += std::pow(arr2[i], 2);
    }

    return (dotProduct / (std::sqrt(normA) * std::sqrt(normB)));
}

template float arraysCosineSimilarity<float>(const float *arr1, const float *arr2, size_t size, float eps);
template float arraysCosineSimilarity<int32_t>(const int32_t *arr1, const int32_t *arr2, size_t size,
                                              float eps);
template float arraysCosineSimilarity<int16_t>(const int16_t *arr1, const int16_t *arr2, size_t size,
                                              float eps);

template <typename T>
void printMatrix(const T * matrix, int rows, int cols, const char * name = "Matrix", bool is_float = true) {
    printf("%s (%d x %d): \n[", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Adjust format specifier based on type T if needed, for now assume float/int compatible
            //   if constexpr (std::is_floating_point_v<T>) {
            if (is_float) {
                printf("%8.4f ", (T) matrix[i * cols + j]);
            } else {
                printf("%8d ", (T) matrix[i * cols + j]);
            }
        }
        printf("\n");
    }
    printf("]\n");
}

template void printMatrix<float>(const float* matrix, int rows, int cols, const char* name = "Matrix", bool is_float = true);
template void printMatrix<int8_t>(const int8_t* matrix, int rows, int cols, const char* name = "Matrix", bool is_float = true);
template void printMatrix<int16_t>(const int16_t* matrix, int rows, int cols, const char* name = "Matrix", bool is_float = true);
template void printMatrix<int32_t>(const int32_t* matrix, int rows, int cols, const char* name = "Matrix", bool is_float = true);

//MARK: DEMO HELPER
static const char *get_dims_string(rknn_matmul_tensor_attr *attr) {
    if (!attr->n_dims) {
        return "()";
    }
    static char dims_str[128];
    memset(&dims_str[0], 0, sizeof(dims_str));
    sprintf(&dims_str[0], "(%d", attr->dims[0]);
    for (uint32_t i = 1; i < attr->n_dims; ++i) {
        int idx = strlen(dims_str);
        sprintf(&dims_str[idx], ", %d", attr->dims[i]);
    }
    strcat(&dims_str[0], ")");
    return dims_str;
}

static int8_t get_virt_addr_int4(void *virt_addr, int index) {
    int8_t int4 = 0;
    if (index % 2 == 0) {
        int4 = (((int8_t *)virt_addr)[index / 2] >> 4) & 0xf;
    } else {
        int4 = (((int8_t *)virt_addr)[index / 2]) & 0xf;
    }
    if (int4 & 0x8) {
        int4 = int4 | 0xf0;
    }
    return int4;
}

static void dump_matmul_tensor(rknn_tensor_mem * tensor, rknn_matmul_tensor_attr * attr) {
    printf("  %s%s:\n", attr->name, get_dims_string(attr));
    // normal layout
    if (attr->n_dims == 2) {
        for (uint32_t i = 0; i < attr->dims[0]; ++i) {
            for (uint32_t j = 0; j < attr->dims[1]; ++j) {
                void * virt_addr = (void *) ((size_t) tensor->virt_addr + tensor->offset);
                if (attr->type == RKNN_TENSOR_INT8) {
                    printf(" %4d", ((int8_t *) virt_addr)[i * attr->dims[1] + j]);
                } else if (attr->type == RKNN_TENSOR_INT32) {
                    printf(" %6d", ((int32_t *) virt_addr)[i * attr->dims[1] + j]);
                } else if (attr->type == RKNN_TENSOR_FLOAT16) {
                    printf(" %5.2f", (float) (((float16 *) virt_addr)[i * attr->dims[1] + j]));
                } else if (attr->type == RKNN_TENSOR_FLOAT32) {
                    printf(" %5.2f", ((float *) virt_addr)[i * attr->dims[1] + j]);
                } else if (attr->type == RKNN_TENSOR_INT16) {
                    printf(" %d", ((int16_t *) virt_addr)[i * attr->dims[1] + j]);
                } else if (attr->type == RKNN_TENSOR_INT4) {
                    int    index = i * attr->dims[1] + j;
                    int8_t int4  = get_virt_addr_int4(virt_addr, index);
                    printf("%d ", int4);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    // perf layout
    else if (attr->n_dims == 3) {
        for (uint32_t i = 0; i < attr->dims[0]; ++i) {
            for (uint32_t j = 0; j < attr->dims[1]; ++j) {
                for (uint32_t k = 0; k < attr->dims[2]; ++k) {
                    void * virt_addr = (void *) ((size_t) tensor->virt_addr + tensor->offset);
                    if (attr->type == RKNN_TENSOR_INT4) {
                    int    index = (i * attr->dims[1] + j) * attr->dims[2] + k;
                    int8_t int4  = get_virt_addr_int4(virt_addr, index);
                    printf("%d ", int4);
                    } else if (attr->type == RKNN_TENSOR_INT8) {
                    printf(" %4d ", ((int8_t *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
                    } else if (attr->type == RKNN_TENSOR_INT16) {
                    printf(" %6d ", ((int16_t *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
                    } else if (attr->type == RKNN_TENSOR_INT32) {
                    printf(" %6d ", ((int32_t *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
                    } else if (attr->type == RKNN_TENSOR_FLOAT16) {
                    printf(" %5.2f ", (float) (((float16 *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]));
                    } else if (attr->type == RKNN_TENSOR_FLOAT32) {
                    printf(" %5.2f ", ((float *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    // native layout
    else if (attr->n_dims == 4) {
        // N / 16
        for (uint32_t n = 0; n < attr->dims[0]; ++n) {
            // K / 32
            for (uint32_t k = 0; k < attr->dims[1]; ++k) {
                // 16
                for (uint32_t nn = 0; nn < attr->dims[2]; ++nn) {
                    // 32
                    for (uint32_t kk = 0; kk < attr->dims[3]; kk++) {
                    void * virt_addr = (void *) ((size_t) tensor->virt_addr + tensor->offset);
                    if (attr->type == RKNN_TENSOR_INT4) {
                        int    index = ((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk;
                        int8_t int4  = get_virt_addr_int4(virt_addr, index);
                        printf("%d ", int4);
                    } else if (attr->type == RKNN_TENSOR_INT8) {
                        printf(" %4d ",
                               ((int8_t *)
                                    virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
                    } else if (attr->type == RKNN_TENSOR_INT32) {
                        printf(" %6d ",
                               ((int32_t *)
                                    virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
                    } else if (attr->type == RKNN_TENSOR_FLOAT16) {
                        printf(
                            " %5.2f ",
                            (float) ((
                                (float16 *)
                                    virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]));
                    } else if (attr->type == RKNN_TENSOR_FLOAT32) {
                        printf(
                            " %5.2f ",
                            ((float *) virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
                    }
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

//MARK: Matmul process
void side_matrix_mulmat_process(matmul_ctx &A00_B00, ggml_tensor *dst, in_kernel_time &kernel_time, int offset_col, int offset_row, int & C_tile){
    void *pad_A00 = (A00_B00.mat_A.pad_data);
    void *pad_B00 = (A00_B00.mat_B.pad_data);
    // printf("check pad_A00:\n");
    // check_pad(A00_B00.mat_A.row, A00_B00.mat_A.col, pad_A00);

    int thread_idx = A00_B00.thread_idx;

    mat_info mat_A = A00_B00.mat_A;
    mat_info mat_B = A00_B00.mat_B;

    int64_t A_row_00 = mat_A.row;
    int64_t A_col_00 = mat_A.col;
    int64_t B_row_00 = mat_B.row;
    int64_t B_col_00 = mat_B.col;

    #ifdef RKNN_MATMUL_DEBUG
    // A_row_00 = 1;
    // A_col_00 = 32;
    // B_row_00 = 32;
    // B_col_00 = 32;
    #endif

    int64_t A_pad_row_00 = mat_A.pad_row;
    int64_t A_pad_col_00 = mat_A.pad_col;
    int64_t B_pad_row_00 = mat_B.pad_row;
    int64_t B_pad_col_00 = mat_B.pad_col;

    int n = A00_B00.ori_n;

    size_t A_size = mat_A.pad_size;
    size_t B_size = mat_B.pad_size;

    rknn_matmul_type type = A00_B00.type;
    bool matrix_B00_need_set_io = A00_B00.matrix_B00_need_set_io;

    int initialized = 0;
    int ret = 0;

    // printf("start create kernel inside side_matrix_multiplication\n");
    ggml_rknpu2_matmul_kernel *sub_kernel = ggml_rknpu2_matmul_kernel_create(pad_A00, pad_B00, A_size, B_size, A_pad_row_00, A_pad_col_00, B_pad_col_00, type, thread_idx, initialized, false, A00_B00.name);
    sub_kernel->is_using = true;
    // printf("end create kernel inside side_matrix_multiplication\n");
    if (initialized == 0)
    {
        norm_layout_to_perf_layout<rknpu2::float16, rknpu2::float16>((rknpu2::float16 *)pad_A00, (rknpu2::float16 *)sub_kernel->A->virt_addr, A_pad_row_00, A_pad_col_00, 8, false);
        sub_kernel->A_data = pad_A00;
        // memcpy(sub_kernel->A->virt_addr, pad_A00, A_pad_row_00 * A_pad_col_00 * sizeof(rknpu2::float16));
        if(!sub_kernel->B_is_copied){
            // rknn_B_normal_layout_to_native_layout(pad_B00, sub_kernel->B->virt_addr, B_pad_row_00, B_pad_col_00, &sub_kernel->info);
            memcpy(sub_kernel->B->virt_addr, pad_B00, B_pad_row_00 * B_pad_col_00 * sizeof(rknpu2::float16));
            sub_kernel->B_is_copied = true;
            // norm_layout_to_native_layout<rknpu2::float16, rknpu2::float16>((rknpu2::float16 *)pad_B00, (rknpu2::float16 *)sub_kernel->B->virt_addr, B_pad_row_00, B_pad_col_00, 16, 32, false);
            sub_kernel->B_data = pad_B00;
        }
    }

    {
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->A, &sub_kernel->io_attr.A);
        if(matrix_B00_need_set_io)
        {
            rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->B, &sub_kernel->io_attr.B);
        }
        rknn_matmul_set_io_mem(sub_kernel->ctx, sub_kernel->C, &sub_kernel->io_attr.C);
    }

    
    {
        int64_t run_start = getCurrentTimeUs();
        ret = rknn_matmul_run(sub_kernel->ctx);
        int64_t run_end = getCurrentTimeUs();
        int64_t run_duration = run_end - run_start;
        if (ret != 0) {
            printf("ggml-rknn: rknn_matmul_run failed for %s:%d\n", sub_kernel->name, sub_kernel->thread_idx);
        }

        timing_debug_printf("ggml-rknn: rknn_matmul_run duration: %ld us\n", run_duration);
    }
    
    // #ifdef RKNN_MATMUL_DEBUG
    // {        
    //     printf("kernel %d dump: \n", thread_idx);
    //     dump_matmul_tensor(sub_kernel->A, &sub_kernel->io_attr.A);
    //     dump_matmul_tensor(sub_kernel->B, &sub_kernel->io_attr.B);
    //     dump_matmul_tensor(sub_kernel->C, &sub_kernel->io_attr.C);
    // }
    // #endif 

    {
        //TODO: if c layout = perf 
        float* norm_layout_C = (float *)malloc(A_row_00 * B_col_00 * sizeof(float));
        // perf_matrixC_to_norm_layout(sub_kernel->C->virt_addr, norm_layout_C, A_row_00, B_col_00);
        int32_t N_remain = sub_kernel->io_attr.C.dims[0];
        int32_t subN = sub_kernel->io_attr.C.dims[2];
        perf_layout_to_norm_layout<float, float>((float *)sub_kernel->C->virt_addr, (float *)norm_layout_C, A_row_00, B_col_00, N_remain, subN);

        std::vector<float> npu_res(norm_layout_C, norm_layout_C + A_row_00 * B_col_00);

        // printMatrix(npu_res.data(), A_row_00, B_col_00, "npu_res", true);

        auto start_time = std::chrono::high_resolution_clock::now();
        if (C_tile == 0)
        {
            // write back to column-major GGML dst
            // npu_res[i][j] = npu_res[i * B_pad_col_00 + j]
            // which should be dst_data[j][i]
            // dst_data[j][i] = dst[j + i * n]
            for (int i = 0; i < A_row_00; i++)
            {
                // what's the use of offset_row here? 
                float* dst_data = (float *)dst->data + i * n + offset_col; // + offset_row * n; 
                for (int j = 0; j < B_col_00; j++)
                {
                    //TODO: the offset is different in column-major 
                    dst_data[j] += npu_res[i * B_col_00 + j];
                    // dst_data[i * n + j] += ((float *)cpu_result.data())[i * B_pad_col_00 + j];
                }
            }
        }
        sub_kernel->is_using = false;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        timing_debug_printf("ggml-rknn: write back to column-major GGML dst duration: %ld us\n", duration);

        //TODO: refactor free prefill case 
        if (A_pad_row_00 > 1) {
            rknn_destroy_mem(sub_kernel->ctx, sub_kernel->A);
            rknn_destroy_mem(sub_kernel->ctx, sub_kernel->B);
            rknn_destroy_mem(sub_kernel->ctx, sub_kernel->C);
            rknn_matmul_destroy(sub_kernel->ctx);
            sub_kernel->A = nullptr;
            sub_kernel->B = nullptr;
            sub_kernel->C = nullptr;
            // sub_kernel->ctx = nullptr;

            timing_debug_printf("ggml-rknn: free prefill case %s:%d (%d,%d,%d)\n", sub_kernel->name, sub_kernel->thread_idx, A_pad_row_00, A_pad_col_00, B_pad_col_00);
        }

        #ifdef RKNN_MATMUL_DEBUG
        if (false) 
        {
            // compare CPU and NPU result 
            std::vector<float> cpu_result = matrixMultiply<float16, float>((float16 *)pad_A00, (float16 *)pad_B00, A_row_00, A_pad_col_00, B_pad_col_00);

            double cosine_similarity = arraysCosineSimilarity<float>(cpu_result.data(), (float *)norm_layout_C, A_row_00 * B_col_00);
            printf("arraysCosineSimilarity: %f\n", cosine_similarity);
            if (cosine_similarity < 0.99 || true)
            {
                printMatrix(cpu_result.data(), A_row_00, B_col_00, "cpu_result", true);
                printMatrix((float *)norm_layout_C, A_row_00, B_col_00, "norm_layout_C", true);

                printf("result is wrong\n");
            }
        }
        #endif 
    }

}


//TODO: 
void dump_time_usage(double prepare_data_time, in_kernel_time &kernel_time, double total_run_time)
{
    timing_debug_printf("ggml-rknn: prepare data time: %.f\n", prepare_data_time);
    timing_debug_printf("ggml-rknn: memcpy to kernel time: %.f\n", kernel_time.memcpy_to_kernel_time);
    timing_debug_printf("ggml-rknn: find kernel time: %.f\n", kernel_time.find_kernel_time);
    timing_debug_printf("ggml-rknn: set io time: %.f\n", kernel_time.set_io_time);
    timing_debug_printf("ggml-rknn: run time: %.f\n", kernel_time.run_time);
    timing_debug_printf("ggml-rknn: sum result time: %.f\n", kernel_time.sum_result_time);
    timing_debug_printf("ggml-rknn: total_run_time: %.f\n", total_run_time);
}


void check_pad(const int64_t row, const int64_t col, void *pad_A01)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%2.f ", (float)((rknpu2::float16 *)pad_A01)[i * col + j]);
        }
        printf("\n");
    }
}


void check_A00xB00_CPU(const int64_t A_row_00, const int64_t B_col_00, const int64_t A_col_00, void *sub_A_data, void *sub_B_data, float *dst, int64_t n)
{

    const float eps = 0.1f;
    for (int i = 0; i < A_row_00; i++)
    {
        for (int j = 0; j < B_col_00; j++)
        {
            float sum = 0;
            for (int k = 0; k < A_col_00; k++)
            {
                rknpu2::float16 a_val = ((rknpu2::float16 *)sub_A_data)[i * A_col_00 + k];
                rknpu2::float16 b_val = ((rknpu2::float16 *)sub_B_data)[k * B_col_00 + j];
                sum += (float)a_val * (float)b_val;
                // ugly code
                // sum += (float)(((rknpu2::float16 *)sub_A_data)[i * A_col_00 + k]) * (float)(((rknpu2::float16 *)sub_B_data)[k * B_col_00 + j]);
            }
            if (fabs((dst[i * n + j] - sum) > eps))
            {
                printf("result is wrong, i: %d, j: %d, dst: %f, cpu: %f\n", (int)i, (int)j, dst[i * n + j], sum);
            }
        }
    }
    printf("checked!\n");
}

void transpose_matrix_A(
    void * A_transposed_data,
    void * A_pad_data,
    int m,
    int k
){
    for(int i = 0; i < m; i++){
        for(int j = 0 ; j < k; j++){
            ((rknpu2::float16*)A_transposed_data)[j * m + i] = ((rknpu2::float16*)A_pad_data)[i * k + j];
        }
    }
}
void transpose_matrix_B(
    void * B_transposed_data,
    void * B_pad_data,
    int pad_n,
    int pad_k
){
    for(int i = 0; i < pad_n; i++){
        for(int j = 0 ; j < pad_k ; j++){
            ((rknpu2::float16*)B_transposed_data)[j * pad_n + i] = ((rknpu2::float16*)B_pad_data)[i * pad_k + j];
        }
    }
}

// MARK: ggml_rk_mul_mat
// TODO: Unify the meaning of variable names, m, k, n to rknn's definition

/*  
    in this function we do the transform between GGML (column major) and RKNN (row major)
    take ffn_up_0 as example of matrix length :
        (1, 2048)(intermediate) * (2048, 8192)(weight) -> (8192, 1)(output)

    GGML node: (column major)
    src0: weight (F16) (2048, 8192) (K, N)
    src1: intermediate (F32) (2048, 1) (K, M)
    dst:  output (F32) (8192, 1) (N, M)

    RKNN requires (row major) F16 * F16 -> F32 (M, K, N) = (1, 2048, 8192)
    in norm layout it should be:  
    A: intermediate ((F32) -> (F16)) (1, 2048) (M, K)
    B: weight (F16) (2048, 8192) (K, N)
    C: output (F32) (1, 8192) (M, N)
*/
static void ggml_rk_mul_mat(ggml_backend_t backend, ggml_tensor * src_w, ggml_tensor * src_i, ggml_tensor * dst, rknn_matmul_type inference_type) {
    std::vector<std::thread> threads;
    int n_threads = (int)((ggml_backend_rknn_context * )backend->context)->n_threads;
    threads.reserve(n_threads);
    
    const int64_t n = src_w->ne[1];
    int64_t k = src_w->ne[0];
    int64_t m = src_i->ne[1];

    void * A_data_f32 = src_i->data;
    // void * B_data = src_w->data;
    void * A_data = malloc(m * k * sizeof(rknpu2::float16));
    // void * B_data = malloc(k * n * sizeof(rknpu2::float16));

    // convert F32 to F16
    // also do the column major to row major
    /*
        accessing ggml matrix src_i[k0, m0] -> src_i[k0 + m0 * k]
        accessing rknn matrix A_data[m0, k0] -> A_data[m0 * k + k0]
    */
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int m0 = 0; m0 < m; m0++) {
        for (int k0 = 0; k0 < k; k0++) {
            // actually the same index
            ((rknpu2::float16 *) A_data)[m0 * k + k0] = (((float *) A_data_f32)[m0 * k + k0]);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_debug_printf("ggml-rknn: convert F32 to F16 duration: %ld us\n", duration);

    auto start_time2 = std::chrono::high_resolution_clock::now();
    /*
        accessing ggml matrix src_w[k0, n0] -> src_w[k0 + n0 * k]
        accessing rknn matrix B_data[k0, n0] -> B_data[k0 * n + n0]
    */
    // for (int k0 = 0; k0 < k; k0++) {
    //     for (int n0 = 0; n0 < n; n0++) {
    //         ((rknpu2::float16 *) B_data)[k0 * n + n0] = (((float16 *) src_w->data)[k0 + n0 * k]);
    //     }
    // }

    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end_time2 - start_time2).count();
    timing_debug_printf("ggml-rknn: copying B duration: %ld us\n", duration2);

    //TODO: should only save transformed weights 
    // void * A_perf_data;
    // if(src_i->extra == NULL || true){
    //     A_perf_data= malloc(m * k * sizeof(rknpu2::float16));
    //     // transposed_matrix_to_perf_layout_multi_threads(A_data, A_perf_data, k, m, 32, 16);
    //     norm_layout_to_perf_layout<rknpu2::float16, rknpu2::float16>((rknpu2::float16 *)A_data, (rknpu2::float16 *)A_perf_data, m, k, 8, false);
    //     src_i->extra = A_perf_data;
    // }else{
    //     A_perf_data = src_i->extra;
    // }

    void * B_native_data;
    if (src_w->extra == NULL) {
        B_native_data = malloc(k * n * sizeof(rknpu2::float16));
        // norm_layout_to_native_layout<rknpu2::float16, rknpu2::float16>((rknpu2::float16 *)B_data, (rknpu2::float16 *)B_native_data, k, n, 16, 32, false);
        ggml_layout_to_native_layout<rknpu2::float16, rknpu2::float16>((rknpu2::float16 *)src_w->data, (rknpu2::float16 *)B_native_data, k, n, 16, 32, false);


        // memcpy(B_native_data, B_data, k * n * sizeof(rknpu2::float16));
        // rknn_B_normal_layout_to_native_layout(B_data, B_native_data, k, n, &kernel->info);

        src_w->extra = B_native_data;
    }else{
        B_native_data = src_w->extra;
    }

    memset(dst->data, 0, dst->ne[0] * dst->ne[1] * sizeof(float));

    int threads_number = n_threads;

    // for(int i = n_threads; i >= 1; i--){
    //     if(m % (16 * i) != 0){
    //         threads_number = i;
    //     }else{
    //         break;
    //     }
    // }

    for(int t = 0; t < threads_number; t++){
        //TODO: assert sub_n is divisible by 16, consider the padding later 
        // devide b for multi thread
        GGML_ASSERT(n % 16 == 0);
        int n_quotient = n / 16;
        int64_t col_start = t * n_quotient / threads_number * 16;
        int64_t col_end = (t + 1) * n_quotient / threads_number * 16;
        if (col_end > n){
            col_end = n;
        }
        int64_t sub_n = col_end - col_start;

        // shape of B_native is [N/subN, K/subK, subN, subK]
        // so we need to offset the data
        // col_start is in normal layout view
        void * B_compute_data = (void *)((char *)B_native_data + col_start * k * sizeof(rknpu2::float16));
        void * A_compute_data = A_data;
        // void * B_compute_data = B_data;

        // run the thread;
        threads.emplace_back([n_quotient, A_compute_data, B_compute_data, dst, col_start, col_end, t, inference_type, k, src_i, src_w](){
            compute_submat_mul(src_i, src_w, dst, A_compute_data, B_compute_data, col_start, col_end, t, inference_type);
        });
    }
    // #ifdef RKNN_MATMUL_DEBUG
    // printf("layer_name: %s\n", dst->name);
    // #endif 
    for (auto & th : threads) {
        th.join();
    }
}

typedef void (*ggml_rk_func_t)(ggml_backend_t backend, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, rknn_matmul_type type);

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    ggml_rk_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    const bool any_on_device = tensor->extra
        || (src0 != nullptr && src0->extra)
        || (src1 != nullptr && src1->extra);

    if(tensor->op == GGML_OP_MUL_MAT){

        func = ggml_rk_mul_mat;
    }
    else{
        return false;
    }

    rknn_matmul_type matmul_type;
    matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    //TODO: 1. in ggml graph (column-major) src0 (F16 weight) and src1 (F32 intermediate)
    // in rknn (row-major) left A (F32 weight, perf) and right B(F16 intermediate, native)
    // 2. need to transpose
    if(src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16){
        matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    }else if(src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0){
        matmul_type = RKNN_INT8_MM_INT8_TO_INT32;
    }
    auto start = std::chrono::high_resolution_clock::now();
    func(backend, tensor->src[0], tensor->src[1], tensor, matmul_type);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    #ifdef RKNN_MATMUL_DEBUG
    printf("layer_name: %s\n", tensor->name);

    float * npu_result = (float *)malloc(tensor->ne[0] * tensor->ne[1] * sizeof(float));
    memcpy(npu_result, tensor->data, tensor->ne[0] * tensor->ne[1] * sizeof(float));

    std::vector<float> cpu_result = matrixMultiply<float16, float>((float16 *)tensor->src[0]->data, (float16 *)tensor->src[1]->data, tensor->src[0]->ne[1], tensor->src[0]->ne[0], tensor->src[1]->ne[1]);

    memcpy(tensor->data, cpu_result.data(), tensor->ne[0] * tensor->ne[1] * sizeof(float));

    // struct ggml_compute_params params = {
    //     /*.ith       =*/ 0,
    //     /*.nth       =*/ 1,
    //     /*.wsize     =*/ 0,
    //     /*.wdata     =*/ NULL,
    //     /*.threadpool=*/ NULL,
    // };

    // ggml_compute_forward_mul_mat(&params, tensor);

    // float* cpu_result = (float *)tensor->data;

    // printMatrix(cpu_result.data(), 32, 1, "cpu_result", true);
    // printMatrix(npu_result, 32, 1, "rknn_result", true);
    double cosine_similarity = arraysCosineSimilarity<float>(cpu_result.data(), npu_result, tensor->ne[0] * tensor->ne[1]);
    printf("cosine_similarity: %f\n", cosine_similarity);
    free(npu_result);
    #endif 

    return true;
}
