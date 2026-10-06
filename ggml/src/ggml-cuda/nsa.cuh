#include "common.cuh"

void ggml_cuda_op_nsa_cmp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);  // lanceur GGML_OP_NSA_CMP : branche compressee + scores de blocs
void ggml_cuda_op_nsa_sel(ggml_backend_cuda_context & ctx, ggml_tensor * dst);  // lanceur GGML_OP_NSA_SEL : blocs choisis (plan 0) + fenetre (plan 1)

bool ggml_cuda_nsa_supported(const ggml_tensor * op);  // supports_op : D, groupe GQA, types K/V, SMEM tenus par les kernels
