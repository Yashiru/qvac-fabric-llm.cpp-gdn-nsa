#include "common.cuh"

void ggml_cuda_op_nsa_cmp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_nsa_sel(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_nsa_supported(const ggml_tensor * op);
