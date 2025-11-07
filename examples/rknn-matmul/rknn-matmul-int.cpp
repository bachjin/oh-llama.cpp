// https://github.com/likejazz/ggml-simple 
#include <cstdint>
#include "fp16/Float16.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "ggml-impl.h"
#include "ggml-common.h"
#include "ggml-quants.h"

// #include "_ggml.h"
// #include "_ggml-alloc.h"
// #include "_ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_RKNN
#include "ggml-rknn.h"
#endif

// disable this to use CPU
#define RKNN_MATMUL_DEBUG

// #define NO_CPU_COMPARE

#include "ggml-cpu.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <type_traits>


//MARK: HELPER
// 一维矩阵乘法函数
template <typename Ti, typename To>
std::vector<To> matrixMultiply_v(const Ti* A, const To* B, int M, int K, int N) {
    // A: [K, M] B: [K, N]
    // in row major order
    std::vector<To> result(M * N, 0);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (float) A[k * M + i] * (float) B[k * N + j];
            }
            result[i * N + j] = sum;
        }
    }

    return result;
}

template <typename Ti, typename To>
std::vector<To> matrixMultiply_r(const std::vector<Ti> & A, const std::vector<Ti> & B, int M, int K, int N) {
    // A: [M, K] B: [K, N]
    // in row major order
    std::vector<To> result(M * N, 0);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            long int sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (long int) A[m * K + k] * (long int) B[k * N + n];
            }
            result[m * N + n] = sum;
        }
    }

    return result;
}

template <typename T>
void col_to_row_transpose(T * matrix, T * dest_matrix, int rows, int cols) {
    // assume the input matrix is in column major order
    // access matrix[i][j] as matrix[i + j * rows]
    // access dest_matrix[i][j] as dest_matrix[i * cols + j]
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dest_matrix[i * cols + j] = matrix[i + j * rows];
        }
    }
}

template <typename T>
void row_to_col_transpose(T * matrix, T * dest_matrix, int rows, int cols) {
    // assume the input matrix is in row major order
    // access matrix[i][j] as matrix[i * cols + j]
    // access dest_matrix[i][j] as dest_matrix[i * cols + j]
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dest_matrix[i + j * rows] = matrix[i * cols + j];
        }
    }
}


// helper: copy our row-major (r x c) (float 32) into ggml's column-major buffer (float 16 or 32)
template <typename T>
static void copy_rowmajor_to_ggml(T * dst_cm, const T * src_rm, int rows, int cols) {
    // column-major index = r + rows*c
    // row-major    index = r*cols + c
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            dst_cm[r + rows * c] = src_rm[r * cols + c];
        }
    }
}

template <typename T>
float arraysCosineSimilarity(const T* arr1, const T* arr2, size_t size) {
    // 计算点积
    double dotProduct = 0.0;
    for (size_t i = 0; i < size; ++i) {
        dotProduct += arr1[i] * arr2[i];
    }

    // 计算向量范数
    double normA = 0.0;
    double normB = 0.0;
    for (size_t i = 0; i < size; ++i) {
        normA += std::pow(arr1[i], 2);
        normB += std::pow(arr2[i], 2);
    }

    if (normA == 0.0 || normB == 0.0) {
        return 0.0;
    }

    return dotProduct / (std::sqrt(normA) * std::sqrt(normB));
}

template float arraysCosineSimilarity<float>(const float* arr1, const float* arr2, size_t size);
template float arraysCosineSimilarity<ggml_fp16_t>(const ggml_fp16_t* arr1, const ggml_fp16_t* arr2, size_t size);

template <typename T>
float arraysNormalizedDifference(const T* arr1, const T* arr2, size_t size) {
    // Normalized absolute difference: sum(abs(a-b)) / sum(abs(a))
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        numerator += std::abs(arr1[i] - arr2[i]);
        denominator += std::abs(arr1[i]);
    }
    if (denominator == 0.0f) {
        return 0.0f;
    }
    return numerator / denominator;
}

template <typename T>
float arraysAbsoluteDifference(const T* arr1, const T* arr2, size_t size){
    float difference = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        difference += std::abs(arr1[i] - arr2[i]);
    }
    return difference;
}

template <typename T>
float arrayInfiniteNorm(const T* arr, size_t size){
    float norm = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        norm += std::abs(arr[i]);
    }
    return norm;
}

template <typename T>
float maxAbsDifference(const T* arr1, const T* arr2, size_t size){
    float max_difference = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        max_difference = std::max(max_difference, std::abs(arr1[i] - arr2[i]));
    }
    return max_difference;
}

template <typename T>
void printMatrix(const T * matrix, int rows, int cols, const char * name = "Matrix", bool is_float = true) {
    #ifdef NO_CPU_COMPARE
        return;
    #endif
    return;

    // assume the matrix is in row major order
    // access matrix[i][j] as matrix[i * cols + j]
    printf("%s (%d x %d): \n[", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Adjust format specifier based on type T if needed, for now assume float/int compatible
            //   if constexpr (std::is_floating_point_v<T>) {
            if (is_float) {
                printf("%4.2f ", (T) matrix[i * cols + j]);
            } else {
                printf("%4d ", (T) matrix[i * cols + j]);
            }
        }
        printf("\n");
    }
    printf("]\n");
}


// // Copy-pasted from ggml.c
// #define QK8_0 32
// typedef struct {
//     rknpu2::float16   d;          // delta
//     int8_t  qs[QK8_0];  // quants
// } block_q8_0;
// static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

// static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
//     ggml_fp16_internal_t tmp;
//     memcpy(&tmp, &h, sizeof(ggml_fp16_t));
//     return (float)tmp;
// }

// static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
//     ggml_fp16_t res;
//     ggml_fp16_internal_t tmp = f;
//     memcpy(&res, &tmp, sizeof(ggml_fp16_t));
//     return res;
// }

//MARK: QUANT HELPER

class custom_block_q8_0 : public block_q8_0
{
public:
    float d;
    int8_t * qs;
    int size;

    custom_block_q8_0() : d(0), qs(nullptr), size(0) {}

    custom_block_q8_0(float d, int8_t * qs, int size) : d(d), qs(qs), size(size) {}

    custom_block_q8_0(const float *src, int size) {
        
        float amax = 0.0f; // absolute max

        for (int j = 0; j < size; j++) {
            const float v = src[j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        this->d = d;
        this->size = size;
        this->qs = new int8_t[size];

        for (int j = 0; j < size; ++j) {
            const float x0 = src[j]*id;

            this->qs[j] = roundf(x0);
        }

    }

    ~custom_block_q8_0() {
        delete[] qs;
    }
};

void custom_quantize_row_q8_0(const float * GGML_RESTRICT x, custom_block_q8_0 * GGML_RESTRICT y, int64_t size) {

    float amax = 0.0f; // absolute max

    for (int j = 0; j < size; j++) {
        const float v = x[j];
        amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;
    y->size = size;
    y->qs = new int8_t[size];

    for (int j = 0; j < size; ++j) {
        const float x0 = x[j]*id;

        y->qs[j] = roundf(x0);
    }
}

void custom_quantize_row_q8_0(const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT y, int64_t size, float &d) {

    float amax = 0.0f; // absolute max

    #pragma omp parallel for reduction(max:amax)
    for (int j = 0; j < size; j++) {
        const float v = x[j];
        amax = MAX(amax, fabsf(v));
    }

    d = amax / ((1 << 7) - 1);
    float id = d ? 1.0f/d : 0.0f;

    for (int j = 0; j < size; ++j) {
        const float x0 = x[j]*id;

        y[j] = roundf(x0);
    }
}

void custom_dequantize_row_q8_0(const custom_block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y) {

    const float d = x->d;

    for (int j = 0; j < x->size; ++j) {
        y[j] = x->qs[j]*d;
    }
}

void custom_dequantize_row_q8_0(const int8_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int size, float d) {

    for (int j = 0; j < size; ++j) {
        y[j] = x[j]*d;
    }
}

void printQuantizedData(const block_q8_0* blocks, int K, int N) {
    // Print quantized data in model.a
    int num_blocks = (K * N) / QK8_0;

    float float_data[QK8_0];

    for (int i = 0; i < std::min(num_blocks, 5); i++) { // Print first few blocks
        printf("Block %d: d=%7.7lf, qs=[", i, ggml_compute_fp16_to_fp32(blocks[i].d));
        
        custom_dequantize_row_q8_0(blocks[i].qs, float_data, QK8_0, ggml_compute_fp16_to_fp32(blocks[i].d));

        for (int j = 0; j < QK8_0; j++) {
            printf("%d", blocks[i].qs[j]);
            if (j < QK8_0-1) printf(",");
        }
        printf("]\n");
        
        // Reconstruct and print float values for this block
        printf("Reconstructed floats: [");
        float d = ggml_compute_fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < QK8_0; j++) {
            // float reconstructed = d * blocks[i].qs[j];
            // printf("%.3f", reconstructed);

            printf("%.3f", float_data[j]);
            if (j < QK8_0-1) printf(",");
        }
        printf("]\n");
    }
}

/**
     * @brief generate random buffer
     *
     */
template <typename T> void generate_random_buffer(T * buffer, size_t size, std::vector<float> range) {
    if (buffer == nullptr || size == 0) {
        return;
    }
    // 设置随机种子
    srand((unsigned) time(NULL));

    float min = range[0], max = range[1];
    for (size_t i = 0; i < size; ++i) {
        buffer[i] = static_cast<T>(min + (max - min) * (static_cast<double>(rand()) / RAND_MAX));
    }
}

// static void ggml_log_callback_default(ggml_log_level level, const char* text, void* user_data) {
//     (void)level;
//     (void)user_data;
//     fputs(text, stderr);
//     fflush(stderr);
// }

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor* a{};
    struct ggml_tensor* b{};

    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t backend = nullptr;
    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer{};
    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context* ctx{};
};

//MARK: MAIN
int main() {
    ggml_time_init();

    /*NOTE: in order to simulate the ffn layer, 
            we want to use QK (int8) (weight) * F32 (intermediate) -> F32 (output/intermediate) for ffn layers

            and noted inside GGML because of column major, it's reversed when creating the graph 

            GGML shape for ffn_up & ffn_gate in llama3.2-1b-q8_0 is: 
            (2048, 8192) * (2048, 1) -> (8192, 1)
            (K, N) * (K, M) -> (N, M) 

            name A, B as vectorized matrices of row-major (human normal) 
            note r() c() as to-row / to-column operations 
            then it's c(C) = c(B)^T * c(A)
            and GGML reads c(B) c(A), wants to get c(C) in it's memory

            shape for ffn_down in llama3.2-1b is:
            (8192, 2048) * (2048, 1) -> (2048, 1)

            we can simulate the ffn_up layer by using the following shapes:
            (32, 64) * (32, 2) -> (64, 2)
            
            now I switch A and B to give correct inputs and aligns our definition of M,K,N in RKNN
            
    */

    int M = 2;
    int K = 32;
    int N = 64;

    // int M = 128;
    // int K = 128;
    // int N = 128;

    M = 1;
    K = 8192 + 32;
    N = 512;

    // M = 1;
    // K = 8960;
    // N = 1536;

    int num_threads = 1;

    // Check for environment variables to override M, K, N values
    const char* env_m = getenv("rknn-m");
    const char* env_k = getenv("rknn-k");
    const char* env_n = getenv("rknn-n");
    const char* env_threads = getenv("rknn-threads");
    
    if (env_m) {
        M = atoi(env_m);
        printf("Using M=%d from environment variable 'rknn-m'\n", M);
    }
    
    if (env_k) {
        K = atoi(env_k);
        printf("Using K=%d from environment variable 'rknn-k'\n", K);
    }
    
    if (env_n) {
        N = atoi(env_n);
        printf("Using N=%d from environment variable 'rknn-n'\n", N);
    }


    if (env_threads) {
        num_threads = atoi(env_threads);
        printf("Using num_threads=%d from environment variable 'rknn-threads'\n", num_threads);
    }

    // we send LEFT_OPERAND as A and RIGHT_OPERAND as B
    // so that A (weights, Qint8) * B (intermediate, F32) = C (output, F32)

    //MARK: INIT MATRIX
    // initialize data of matrices to perform matrix multiplication
    std::vector<float> matrix_l(K * N, 0);
    std::vector<float> matrix_r(K * M, 0);

    // here is row-major layout 
    // access A[k, n] as A[k * N + n]
    // access B[k, m] as B[k * M + m]

    // A: increasing matrix 
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            matrix_l[k * N + n] = k + n * 0.01f;
        }
    }

    // B: k + m * 0.01 (showing k.n)
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            matrix_r[k * M + m] = k + m * 100.0f;
        }
    }

    // // Q8 test: values that depend on both k and n/m for better pattern testing
    // for (int k = 0; k < K; k++) {
    //     for (int n = 0; n < N; n++) {
    //         matrix_l[k * N + n] = ((k + n) % 16) * 8.0f; // Values: 0, 8, 16, ..., 120 based on k+n
    //     }
    // }

    // for (int k = 0; k < K; k++) {
    //     for (int m = 0; m < M; m++) {
    //         matrix_r[k * M + m] = ((k * 2 + m) % 8) * 16.0f; // Values: 0, 16, 32, ..., 112 based on k*2+m
    //     }
    // }


    // // Q8 test with negative values: range from -128 to 127 (int8 range)
    // for (int k = 0; k < K; k++) {
    //     for (int n = 0; n < N; n++) {
    //         // Generate values in int8 range: -128 to 127
    //         int val = ((k + 2 * n) % 32) - 16; // Range: -16 to 15
    //         matrix_l[k * N + n] = val * (n + 1) * 8.0f; // Values: -128, -120, ..., -8, 0, 8, ..., 120
    //     }
    // }

    // for (int k = 0; k < K; k++) {
    //     for (int m = 0; m < M; m++) {
    //         // Generate values with both positive and negative numbers
    //         int val = ((k * 3 + m) % 16) - 8; // Range: -8 to 7
    //         matrix_r[k * M + m] = val * 16.0f; // Values: -128, -112, ..., -16, 0, 16, ..., 112
    //     }
    // }

    simple_model model;
    float* ldata = matrix_l.data();
    float* rdata = matrix_r.data();

    generate_random_buffer(ldata, K * N, {-200.0f, 200.0f});
    generate_random_buffer(rdata, K * M, {-100.0f, 100.0f});

    printMatrix(ldata, K, N, "LEFT", true);
    printMatrix(rdata, K, M, "RIGHT", true);

    // initialize the backend
// #ifdef GGML_USE_CUDA
//     fprintf(stderr, "%s: using CUDA backend\n", __func__);
//     model.backend = ggml_backend_cuda_init(0); // init device 0
//     if (!model.backend) {
//         fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
//     }
// #endif
//MARK: INIT RKNN 
#ifdef GGML_USE_RKNN
    #ifdef RKNN_MATMUL_DEBUG
    fprintf(stderr, "%s: using RKNN backend\n", __func__);
    // ggml_backend_eval_callback(ggml_log_callback_default, nullptr);
    model.backend = ggml_backend_rknn_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_rknn_init() failed\n", __func__);
    }
    #endif
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    int num_tensors = 2;

    ggml_init_params params{
        /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    //MARK: COPY TENSORS
    //TODO: we want INT8 (weight) * F32 (intermediate) -> F32 (output/intermediate) for ffn layers



    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_Q8_0, K, N);

    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, K, M);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    // ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    // ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));

    // copy_rowmajor_to_ggml<int8_t>((int8_t *) model.a->data, ldata, K, N);

    auto qfns = ggml_get_type_traits(GGML_TYPE_Q8_0);
    float* ldata_transposed = new float[K * N];
    row_to_col_transpose(ldata, ldata_transposed, K, N);
    qfns->from_float_ref(ldata_transposed, model.a->data, K * N);

    
    printQuantizedData((block_q8_0*)model.a->data, K, N);

    int8_t* ldata_quantized = new int8_t[K];
    float delta0;
    custom_quantize_row_q8_0(ldata_transposed, ldata_quantized, K, delta0);

    printMatrix(ldata_quantized, 1, K, "ldata_quantized", false);
    printf("delta0: %f\n", delta0);
    
    delete[] ldata_transposed;
    delete[] ldata_quantized;

    copy_rowmajor_to_ggml<float>((float *) model.b->data, rdata, K, M);
    

    // calculate the temporaly memory required to compute
    const ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    // create the worst case graph for memory usage estimation
    static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };
    // create a temporally context to build the graph
    ggml_context* ctx0 = ggml_init(params0);
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    // result = a^T*b
    ggml_tensor* result = ggml_mul_mat(ctx0, model.a, model.b);
    // build operations nodes
    ggml_build_forward_expand(gf, result);
    // delete the temporally context used to build the graph
    ggml_free(ctx0);

    ggml_gallocr_reserve(allocr, gf);
    size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
    fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size / 1024.0);

    //MARK: MATMUL
    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, 1);
    } else {
        printf("demo: set rknn to n threads: %d\n", num_threads);
        printf("demo: sizeof backend: %zu\n", sizeof(typeof(model.backend)));
        ggml_backend_rknn_set_n_threads(model.backend, num_threads);
    }
    ggml_backend_graph_compute(model.backend, gf);

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // printMatrix(out_data.data(), N, M, "result_col");
    printMatrix(out_data.data(), 1, N * M, "result_col");

    std::vector<float> out_data_transposed(N * M, 0);
    col_to_row_transpose(out_data.data(), out_data_transposed.data(), N, M);
    printMatrix(out_data_transposed.data(), N, M, "result_row");

    #ifdef NO_CPU_COMPARE
        std::vector<float> expected_result = std::vector<float>(N * M, 0);
    #else
        std::vector<float> expected_result = matrixMultiply_v<float, float>(matrix_l.data(), matrix_r.data(), N, K, M);
    #endif

    printMatrix(expected_result.data(), N, M, "expected result");

    printf("cosine similarity: %f\n", arraysCosineSimilarity(out_data_transposed.data(), expected_result.data(), N * M));
    printf("normalized difference: %f\n", arraysNormalizedDifference<float>(out_data_transposed.data(), expected_result.data(), N * M));
    printf("absolute difference sum: %f\n", arraysAbsoluteDifference<float>(out_data_transposed.data(), expected_result.data(), N * M));
    printf("infinite norm: %f %f\n", arrayInfiniteNorm<float>(out_data_transposed.data(), N * M), arrayInfiniteNorm<float>(expected_result.data(), N * M));
    printf("max element abs difference: %f\n", maxAbsDifference<float>(out_data_transposed.data(), expected_result.data(), N * M));
    printf("max element: %f %f\n", maxAbsDifference<float>(out_data_transposed.data(), std::vector<float>(N * M, 0).data(), N * M), maxAbsDifference<float>(expected_result.data(), std::vector<float>(N * M, 0).data(), N * M));

    // release backend memory used for computation
    ggml_gallocr_free(allocr);
    // free memory
    ggml_free(model.ctx);
    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    return 0;
}