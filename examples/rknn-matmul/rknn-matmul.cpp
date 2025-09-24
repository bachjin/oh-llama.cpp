// https://github.com/likejazz/ggml-simple 
#include "fp16/Float16.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// #include "_ggml.h"
// #include "_ggml-alloc.h"
// #include "_ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_RKNN
#include "ggml-rknn.h"
#endif

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
std::vector<To> matrixMultiply_v(const std::vector<Ti> & A, const std::vector<Ti> & B, int M, int K, int N) {
    // A: [K, M] B: [K, N]
    // in row major order
    std::vector<To> result(M * N, 0);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0;
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
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (float) A[m * K + k] * (float) B[k * N + n];
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


// helper: copy our row-major (r x c) (float 32) into ggml's column-major buffer (float 16 or 32)
template <typename T>
static void copy_rowmajor_to_ggml(T * dst_cm, const float * src_rm, int rows, int cols) {
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
                printf("%8d ", (T) matrix[i * cols + j]);
            }
        }
        printf("\n");
    }
    printf("]\n");
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
            we want to use F16 (weight) * F32 (intermediate) -> F32 (output/intermediate) for ffn layers

            and noted inside GGML because of column major, it's reversed when creating the graph 

            shape for ffn_up & ffn_gate in llama3.2-1b is: 
            (2048, 8192) * (2048, 1) -> (8192, 1)
            (K, N) * (K, M) -> (N, M) 

            name A,B as vectorized matrices of row-major (human normal) 
            call r() c() as to-row / to-column operations 
            then it's c(C) = c(B)^T * c(A)
            and GGML reads c(B) c(A) gets c(C) in it's memory

            shape for ffn_down in llama3.2-1b is:
            (8192, 2048) * (2048, 1) -> (2048, 1)

            we can simulate the ffn_up layer by using the following shapes:
            (32, 64) * (32, 2) -> (64, 2)
            
            now I switch A and B to give correct inputs and aligns our definition of M,K,N
            
    */

    // int M = 64;
    // int K = 64;
    // int N = 64;

    int N = 8192;
    int K = 2048;
    int M = 1;

    int num_threads = 3;

    // we send LEFT_OPERAND as A and RIGHT_OPERAND as B
    // so that A (weights, F16) * B (intermediate, F32) = C (output, F32)
    bool left_is_f16 = true;
    bool right_is_f16 = false;

    //MARK: INIT MATRIX
    // initialize data of matrices to perform matrix multiplication
    std::vector<float> matrix_l(K * N, 0);
    std::vector<float> matrix_r(K * M, 0);

    // here is row-major layout 
    // access A[k, n] as A[k * N + n]
    // access B[k, m] as B[k * M + m]
    
    // //A
    // for(int k = 0; k < K; k++){
    //     matrix_l[k * N + 0] = k + 1;
    // }

    // for(int k = 0; k < K; k++){
    //     for (int n = 0; n < M; n++){
    //         matrix_r[k * M + n] = k * 10 + n;
    //     }
    // }

    // A: increasing matrix 
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            // matrix_A[k * M + m] = k * 100.0f + m;
            matrix_l[k * N + n] = n * 0.01f + k;
        }
    }

    // // A: Diagonal matrix
    // for (int m = 0; m < N; m++) {
    //     for (int k = 0; k < K; k++) {
    //         matrix_l[k * N + m] = (k == m) ? float(k + 1) : 0.0f;
    //     }
    // }

    // // B: Diagonal matrix
    // for (int k = 0; k < K; k++) {
    //     for (int m = 0; m < M; m++) {
    //         matrix_r[k * M + m] = (k == m) ? float(k + 1) : 0.0f;
    //     }
    // }

    // B: k + m * 100 (showing k.n)
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            matrix_r[k * M + m] = float(k) + float((m) * 100.0f);
        }
    }

    simple_model model;
    float* ldata = matrix_l.data();
    float* rdata = matrix_r.data();

    // generate_random_buffer(ldata, K * N, {0.0f, 10.0f});
    // generate_random_buffer(rdata, K * M, {0.0f, 10.0f});

    printMatrix(ldata, K, N, "LEFT");
    printMatrix(rdata, K, M, "RIGHT");

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
    //TODO: we want F16 (weight) * F32 (intermediate) -> F32 (output/intermediate) for ffn layers



    // create tensors
    if (left_is_f16) {
        model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F16, K, N);
    } else {
        model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, K, N);
    }

    if (right_is_f16) {
        model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F16, K, M);
    } else {
        model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, K, M);
    }

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    // ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    // ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));

    if (left_is_f16) {
        copy_rowmajor_to_ggml<float16_t>((float16_t *) model.a->data, ldata, K, N);
    } else {
        copy_rowmajor_to_ggml<float>((float *) model.a->data, ldata, K, N);
    }
    if (right_is_f16) {
        copy_rowmajor_to_ggml<float16_t>((float16_t *) model.b->data, rdata, K, M);
    } else {
        copy_rowmajor_to_ggml<float>((float *) model.b->data, rdata, K, M);
    }

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
        printf("demo: sizeof backend: %d\n", sizeof(typeof(model.backend)));
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
        std::vector<float> expected_result = matrixMultiply_v<float, float>(matrix_l, matrix_r, N, K, M);
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