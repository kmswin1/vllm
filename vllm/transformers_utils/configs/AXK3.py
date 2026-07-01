# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for A.X-K3 (AXK3).

AXK3 = A.X-K2 MLA/MoE backbone with a two-variant, per-layer interleaved
attention at an ``N:1`` ratio:

* **local** layers  -> MLA + sliding-window attention (SWA) + a per-head
  learnable attention sink (always on),
* **global** layers -> block-sparse MLA (MiniMax Sparse Attention, MSA style):
  block max-pool over kv-blocks -> top-k blocks -> token indices -> sparse
  ``flash_mla_with_kvcache``.

``N`` local layers are followed by ``1`` global layer, repeating. ``N`` is a
parameter (``local_per_global``); an explicit per-layer pattern
(``attn_variant_pattern``) overrides it.

NOTE: SKELETON. The real modeling weights/dims will be provided later; the
defaults below mirror the AXK2 (K2_DSA) checkpoint so the module graph can be
instantiated for import/wiring tests without a checkpoint.
"""

from transformers.configuration_utils import PretrainedConfig

# Default YaRN rope scaling for the K2 full 128K profile (matches AXK2).
_K3_FULL_YARN_ROPE_SCALING = {
    "type": "yarn",
    "factor": 1.0,
    "original_max_position_embeddings": 131072,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


class AXK3Config(PretrainedConfig):
    r"""Configuration for A.X-K3 (MLA-SWA + block-sparse MLA, N:1)."""

    model_type = "AXK3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=163840,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_nextn_predict_layers=0,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_shared_experts=1,
        n_routed_experts=256,
        ep_size=1,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method="noaux_tc",
        n_group=8,
        topk_group=4,
        num_experts_per_tok=8,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=163691,
        eos_token_id=163691,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # --- gating (shared by both attention variants) ---
        attention_output_gate=True,
        gated_norm=True,
        gated_norm_rank=16,
        # --- block-sparse MSA (global layers) ---
        index_n_heads=1,
        index_head_dim=128,
        index_topk=2048,
        msa_block_selection=True,
        msa_block_size=128,
        # --- MLA-SWA (local layers) ---
        sliding_window=4096,
        mla_swa_attention_sink=True,
        # --- per-layer N:1 interleave ---
        local_per_global=3,
        attn_variant_pattern=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # gating
        self.attention_output_gate = attention_output_gate
        self.gated_norm = gated_norm
        self.gated_norm_rank = gated_norm_rank

        # block-sparse MSA (global layers). index_topk is a TOKEN budget:
        # index_topk == topk_blocks * msa_block_size.
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.msa_block_selection = msa_block_selection
        self.msa_block_size = msa_block_size

        # MLA-SWA (local layers)
        self.sliding_window = sliding_window
        self.mla_swa_attention_sink = mla_swa_attention_sink

        # per-layer interleave: N local (MLA-SWA) : 1 global (block-sparse)
        self.local_per_global = local_per_global
        self.attn_variant_pattern = attn_variant_pattern

        if rope_scaling is None:
            rope_scaling = dict(_K3_FULL_YARN_ROPE_SCALING)
        self.rope_scaling = rope_scaling

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
