#include "models.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

void llama_model_gdn_nsa::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // gated delta net layers
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // native sparse attention layers
    ml.get_key(LLM_KV_ATTENTION_NSA_BLOCK_SIZE,  hparams.nsa_block_size);
    ml.get_key(LLM_KV_ATTENTION_NSA_BLOCK_COUNT, hparams.nsa_block_count);
    ml.get_key(LLM_KV_ATTENTION_NSA_WINDOW_SIZE, hparams.nsa_window_size);

    uint32_t full_attn_interval = 2;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = (i + 1) % full_attn_interval != 0;
    }

    switch (hparams.n_layer()) {
        case 24: type = LLM_TYPE_350M; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gdn_nsa::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t head_dim  = hparams.ssm_d_state;
    const int64_t n_v_heads = hparams.ssm_dt_rank;
    const int64_t key_dim   = head_dim * hparams.ssm_n_group;
    const int64_t value_dim = hparams.ssm_d_inner;
    const int64_t conv_dim  = key_dim * 2 + value_dim;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        if (hparams.is_recr(i)) {
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", i), { n_embd, conv_dim }, 0);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", i), { n_embd, value_dim }, 0);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", i), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   i), { n_v_heads }, 0);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             i), { n_v_heads }, 0);
            layer.ssm_beta_alpha = create_tensor(tn(LLM_TENSOR_SSM_BETA_ALPHA, "weight", i), { n_embd, 2 * n_v_heads }, 0);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", i), { head_dim }, 0);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", i), { value_dim, n_embd }, 0);
        } else {
            layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
            layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), { n_embd, n_embd_k_gqa }, 0);
            layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), { n_embd, n_embd_v_gqa }, 0);
            layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), { n_embd_head_k * n_head, n_embd }, 0);
            // branch gates, row h*3 + j with j = compressed, selected, sliding window
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), { n_embd, 3 * n_head }, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gdn_nsa::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gdn_nsa::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recr(il)) {
            cur = build_layer_gdn(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_nsa(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_inp = cur;

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// same as the qwen3next linear attention layer, without grouped value heads
ggml_tensor * llama_model_gdn_nsa::graph::build_layer_gdn(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;
    const auto & layer    = model.layers[il];

    const int64_t head_dim     = hparams.ssm_d_state;
    const int64_t n_heads      = hparams.ssm_dt_rank;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(hparams.ssm_n_group == hparams.ssm_dt_rank);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur);
    qkv = ggml_reshape_3d(ctx0, qkv, qkv->ne[0], n_seq_tokens, n_seqs);
    cb(qkv, "linear_attn_qkv", il);

    ggml_tensor * z = build_lora_mm(layer.wqkv_gate, cur);
    cb(z, "linear_attn_z", il);

    // ssm_ba rows are b_0, a_0, b_1, a_1, ...
    ggml_tensor * ba = build_lora_mm(layer.ssm_beta_alpha, cur);
    ba = ggml_reshape_4d(ctx0, ba, 2, n_heads, n_seq_tokens, n_seqs);

    ggml_tensor * b = ggml_view_4d(ctx0, ba, 1, n_heads, n_seq_tokens, n_seqs, ba->nb[1], ba->nb[2], ba->nb[3], 0);
    ggml_tensor * a = ggml_view_4d(ctx0, ba, 1, n_heads, n_seq_tokens, n_seqs, ba->nb[1], ba->nb[2], ba->nb[3], ggml_element_size(ba));

    ggml_tensor * beta = ggml_sigmoid(ctx0, ggml_cont(ctx0, b));
    beta = ggml_reshape_4d(ctx0, beta, 1, n_heads, n_seq_tokens, n_seqs);

    ggml_tensor * alpha = ggml_cont_3d(ctx0, a, n_heads, n_seq_tokens, n_seqs);
    alpha = ggml_softplus(ctx0, ggml_add(ctx0, alpha, layer.ssm_dt));

    ggml_tensor * gate = ggml_mul(ctx0, alpha, layer.ssm_a); // -A_log.exp() * softplus
    gate = ggml_reshape_4d(ctx0, gate, 1, n_heads, n_seq_tokens, n_seqs);
    cb(gate, "gate", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    const int64_t conv_channels = qkv->ne[0];

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv, layer.ssm_conv1d->ne[0], conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_heads, n_seqs);

    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, layer.ssm_conv1d));
    cb(conv_out, "conv_output_silu", il);

    const size_t es  = ggml_element_size(conv_out);
    const size_t nb1 = ggml_row_size(conv_out->type, conv_channels);

    ggml_tensor * q = ggml_view_4d(ctx0, conv_out, head_dim, n_heads, n_seq_tokens, n_seqs, head_dim*es, nb1, nb1*n_seq_tokens, 0);
    ggml_tensor * k = ggml_view_4d(ctx0, conv_out, head_dim, n_heads, n_seq_tokens, n_seqs, head_dim*es, nb1, nb1*n_seq_tokens, 1*head_dim*n_heads*es);
    ggml_tensor * v = ggml_view_4d(ctx0, conv_out, head_dim, n_heads, n_seq_tokens, n_seqs, head_dim*es, nb1, nb1*n_seq_tokens, 2*head_dim*n_heads*es);

    q = ggml_l2_norm(ctx0, q, hparams.f_norm_rms_eps);
    k = ggml_l2_norm(ctx0, k, hparams.f_norm_rms_eps);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q, k, v, gate, beta, state, il);

    // gated rms norm: norm(o) * silu(z)
    z = ggml_reshape_4d(ctx0, z, head_dim, n_heads, n_seq_tokens, n_seqs);
    output = build_norm(output, layer.ssm_norm, nullptr, LLM_NORM_RMS, il);
    output = ggml_mul(ctx0, output, ggml_silu(ctx0, z));
    output = ggml_reshape_3d(ctx0, output, head_dim * n_heads, n_seq_tokens, n_seqs);

    cur = build_lora_mm(layer.ssm_out, output);
    cb(cur, "linear_attn_out", il);

    return ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
}

// native sparse attention: compressed + selected blocks + sliding window, mixed by per-head gates
ggml_tensor * llama_model_gdn_nsa::graph::build_layer_nsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const auto * mctx_cur = inp->mctx;
    const auto & layer    = model.layers[il];

    const int64_t n_head_l    = hparams.n_head(il);
    const int64_t n_head_kv_l = hparams.n_head_kv(il);
    const int64_t bs          = hparams.nsa_block_size;
    const float   kq_scale    = 1.0f/sqrtf(float(n_embd_head_k));

    // NSA needs key j at position j: one sequence, cells filled from 0
    GGML_ASSERT(ubatch.n_seqs_unq == 1);
    GGML_ASSERT(!inp->self_k_rot && !inp->self_v_rot);

    ggml_tensor * q = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, cur), n_embd_head_k, n_head_l,    n_tokens);
    ggml_tensor * k = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk, cur), n_embd_head_k, n_head_kv_l, n_tokens);
    ggml_tensor * v = ggml_reshape_3d(ctx0, build_lora_mm(layer.wv, cur), n_embd_head_k, n_head_kv_l, n_tokens);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "Qcur", il);
    cb(k, "Kcur", il);
    cb(v, "Vcur", il);

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, v);
    ggml_build_forward_expand(gf, k);

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k, inp->get_k_idxs(), il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v, inp->get_v_idxs(), il));

    // [n_embd_head, n_head_kv, n_kv], including the tokens of this ubatch
    ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);

    const int64_t n_kv  = k_all->ne[2];
    const int64_t n_blk = (n_kv + bs - 1)/bs;

    // mean of each block of bs keys; the last block is padded with zeros but is never used as a mean
    auto pool = [&](ggml_tensor * x) {
        x = ggml_cast(ctx0, x, GGML_TYPE_F32);
        x = ggml_reshape_2d(ctx0, x, n_embd_head_k*n_head_kv_l, n_kv);
        x = ggml_pad(ctx0, x, 0, n_blk*bs - n_kv, 0, 0);
        x = ggml_reshape_3d(ctx0, x, n_embd_head_k*n_head_kv_l, bs, n_blk);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
        x = ggml_mean(ctx0, x);
        return ggml_reshape_3d(ctx0, x, n_embd_head_k, n_head_kv_l, n_blk);
    };

    ggml_tensor * k_cmp = pool(k_all);
    ggml_tensor * v_cmp = pool(v_all);
    cb(k_cmp, "k_cmp", il);

    ggml_tensor * cmp = ggml_nsa_cmp(ctx0, q, k_cmp, v_cmp, inp_pos, bs, kq_scale);

    ggml_tensor * o_cmp  = ggml_view_3d(ctx0, cmp, n_embd_head_k, n_head_l, n_tokens,
            n_embd_head_k*sizeof(float), n_embd_head_k*n_head_l*sizeof(float), 0);
    ggml_tensor * scores = ggml_view_3d(ctx0, cmp, n_blk, n_head_kv_l, n_tokens,
            n_blk*sizeof(float), n_blk*n_head_kv_l*sizeof(float), n_embd_head_k*n_head_l*n_tokens*sizeof(float));
    cb(scores, "nsa_scores", il);

    ggml_tensor * idx = ggml_top_k(ctx0, scores, std::min<int64_t>(hparams.nsa_block_count, n_blk));
    cb(idx, "nsa_idx", il);

    ggml_tensor * sel = ggml_nsa_sel(ctx0, q, k_all, v_all, idx, inp_pos, bs, hparams.nsa_window_size, kq_scale);

    ggml_tensor * o_slc = ggml_view_3d(ctx0, sel, n_embd_head_k, n_head_l, n_tokens, sel->nb[1], sel->nb[2], 0);
    ggml_tensor * o_swa = ggml_view_3d(ctx0, sel, n_embd_head_k, n_head_l, n_tokens, sel->nb[1], sel->nb[2], sel->nb[3]);

    // gates: [3, n_head, n_tokens], j = compressed, selected, sliding window
    ggml_tensor * g = ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, cur));
    g = ggml_reshape_3d(ctx0, g, 3, n_head_l, n_tokens);

    auto gate = [&](int j) {
        return ggml_cont(ctx0, ggml_view_3d(ctx0, g, 1, n_head_l, n_tokens, g->nb[1], g->nb[2], j*ggml_element_size(g)));
    };

    ggml_tensor * o = ggml_mul(ctx0, o_slc, gate(1));
    o = ggml_add(ctx0, o, ggml_mul(ctx0, o_cmp, gate(0)));
    o = ggml_add(ctx0, o, ggml_mul(ctx0, o_swa, gate(2)));
    cb(o, "nsa_out", il);

    o = ggml_reshape_2d(ctx0, o, n_embd_head_k*n_head_l, n_tokens);

    return build_lora_mm(layer.wo, o);
}
