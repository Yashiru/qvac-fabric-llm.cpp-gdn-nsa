from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("HybridGatedDeltaNetForCausalLM")
@ModelBase.example("Idiap/gated-deltanet-nsa-0.4B-10B")
class GatedDeltaNetNSAModel(TextModel):
    """Gated DeltaNet + Native Sparse Attention hybrid (github.com/idiap/hybrid-linear-sparse-attention)"""
    model_arch = gguf.MODEL_ARCH.GDN_NSA

    # GDN layer tensors fused at conversion time into the qwen3next layout:
    #   attn_qkv       = cat(q_proj, k_proj, v_proj)
    #   ssm_conv1d     = cat(q_conv1d, k_conv1d, v_conv1d)   (depthwise conv: concatenating channels is exact)
    #   ssm_beta_alpha = rows interleaved per head: b_0, a_0, b_1, a_1, ...
    _FUSED = {
        "attn.q_proj.weight":   ("qkv",  0),
        "attn.k_proj.weight":   ("qkv",  1),
        "attn.v_proj.weight":   ("qkv",  2),
        "attn.q_conv1d.weight": ("conv", 0),
        "attn.k_conv1d.weight": ("conv", 1),
        "attn.v_conv1d.weight": ("conv", 2),
        "attn.b_proj.weight":   ("ba",   0),
        "attn.a_proj.weight":   ("ba",   1),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nsa_layers = set(self.hparams["attn"]["layers"])
        self._fuse_buf: dict[tuple[int, str], dict[int, Tensor]] = {}

    def set_vocab(self):
        # tokenizer.json only (Mistral-7B-v0.1 vocab), no tokenizer.model. LlamaHfVocab gives all tokens
        # the same score, so the SPM tokenizer would merge in arbitrary order: use score = -(merge rank)
        vocab = gguf.LlamaHfVocab(self.dir_model)

        with open(self.dir_model / "tokenizer.json", encoding="utf-8") as f:
            merges = json.load(f)["model"]["merges"]
        rank: dict[str, int] = {}
        for r, m in enumerate(merges):
            a, b = m.split(" ", 1) if isinstance(m, str) else m
            rank.setdefault(a + b, r)

        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []
        for text, _, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(-float(rank.get(text.decode("utf-8", errors="replace"), len(merges))))
            toktypes.append(toktype)
        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hp = self.hparams
        attn = hp["attn"]
        assert attn["attn_type"] == "nsa", "only the NSA variant is supported"
        assert attn["layers"] == list(range(1, self.block_count, 2)), "only the 1:1 GDN/NSA interleave is supported"
        assert hp["use_gate"] and hp["use_short_conv"] and hp["expand_v"] == 1

        n_embd = hp["hidden_size"]
        n_ff = hp.get("intermediate_size")
        if n_ff is None:
            # fla GatedMLP: multiple of 256 closest to 2/3 * hidden_size * hidden_ratio
            n_ff = int(n_embd * hp["hidden_ratio"] * 2 / 3)
            n_ff = 256 * ((n_ff + 256 - 1) // 256)

        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hp["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(n_ff)
        self.gguf_writer.add_layer_norm_rms_eps(hp["norm_eps"])

        # NSA layers
        self.gguf_writer.add_head_count(attn["num_heads"])
        self.gguf_writer.add_head_count_kv(attn["num_kv_heads"])
        self.gguf_writer.add_key_length(attn["head_dim"])
        self.gguf_writer.add_value_length(attn["head_dim"])
        self.gguf_writer.add_rope_dimension_count(attn["head_dim"])
        self.gguf_writer.add_rope_freq_base(attn["rope_theta"])
        self.gguf_writer.add_nsa_block_size(attn["block_size"])
        self.gguf_writer.add_nsa_block_count(attn["block_counts"])
        self.gguf_writer.add_nsa_window_size(attn["window_size"])

        # GDN layers (qwen3next keys), no grouped value heads in this model
        self.gguf_writer.add_ssm_conv_kernel(hp["conv_size"])
        self.gguf_writer.add_ssm_state_size(hp["head_dim"])
        self.gguf_writer.add_ssm_group_count(hp["num_heads"])
        self.gguf_writer.add_ssm_time_step_rank(hp["num_heads"])
        self.gguf_writer.add_ssm_inner_size(hp["head_dim"] * hp["num_heads"])

        # llama.cpp marks layer il as recurrent iff (il + 1) % interval != 0: even layers GDN, odd layers NSA
        self.gguf_writer.add_full_attention_interval(2)
        self.gguf_writer.add_file_type(self.ftype)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._fuse_buf:
            raise ValueError(f"Unfused tensors left: {list(self._fuse_buf.keys())}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR

        if bid is None:
            top = {
                "model.embeddings.weight": T.TOKEN_EMBD,
                "model.norm.weight":       T.OUTPUT_NORM,
            }
            if name not in top:
                raise ValueError(f"Can not map tensor {name!r}")
            yield self.format_tensor_name(top[name]), data_torch
            return

        sub = name[len(f"model.layers.{bid}."):]
        direct = {
            "attn_norm.weight":     T.ATTN_NORM,
            "mlp_norm.weight":      T.FFN_NORM,
            "mlp.gate_proj.weight": T.FFN_GATE,
            "mlp.up_proj.weight":   T.FFN_UP,
            "mlp.down_proj.weight": T.FFN_DOWN,
        }

        if bid in self._nsa_layers:
            direct.update({
                "attn.q_proj.weight": T.ATTN_Q,
                "attn.k_proj.weight": T.ATTN_K,
                "attn.v_proj.weight": T.ATTN_V,
                "attn.o_proj.weight": T.ATTN_OUT,
                "attn.g_proj.weight": T.ATTN_GATE,  # [3 * n_head, n_embd], row h*3 + j, j = cmp, slc, swa
            })
        else:
            direct.update({
                "attn.g_proj.weight": T.ATTN_GATE,  # output gate of the gated RMSNorm
                "attn.o_norm.weight": T.SSM_NORM,
                "attn.o_proj.weight": T.SSM_OUT,
            })
            if sub == "attn.A_log":
                yield self.format_tensor_name(T.SSM_A, bid, suffix=""), -torch.exp(data_torch)
                return
            if sub == "attn.dt_bias":
                yield self.format_tensor_name(T.SSM_DT, bid, suffix=".bias"), data_torch
                return
            if sub in self._FUSED:
                kind, idx = self._FUSED[sub]
                buf = self._fuse_buf.setdefault((bid, kind), {})
                buf[idx] = data_torch
                n_parts = 2 if kind == "ba" else 3
                if len(buf) < n_parts:
                    return
                del self._fuse_buf[(bid, kind)]
                parts = [buf[i] for i in range(n_parts)]
                if kind == "qkv":
                    yield self.format_tensor_name(T.ATTN_QKV, bid), torch.cat(parts, dim=0)
                elif kind == "conv":
                    # HF [channels, 1, d_conv] -> [channels, d_conv] -> ggml ne = [d_conv, channels]
                    yield self.format_tensor_name(T.SSM_CONV1D, bid), torch.cat([p.squeeze(1) for p in parts], dim=0)
                else:
                    b, a = parts
                    yield self.format_tensor_name(T.SSM_BETA_ALPHA, bid), torch.stack([b, a], dim=1).reshape(-1, b.shape[-1])
                return

        if sub not in direct:
            raise ValueError(f"Can not map tensor {name!r}")
        yield self.format_tensor_name(direct[sub], bid), data_torch
