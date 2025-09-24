#ifndef GGML_RKNN_H
#define GGML_RKNN_H

#include "ggml.h"
#include "ggml-backend.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"



#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//

static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name) ;
GGML_BACKEND_API ggml_backend_t ggml_backend_rknn_init(void);
GGML_BACKEND_API bool ggml_backend_is_rknn(ggml_backend_t backend);
GGML_BACKEND_API void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rknn_reg(void);
#ifdef  __cplusplus
}
#endif

#endif // GGML_RKNN_H

