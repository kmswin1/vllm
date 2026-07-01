# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K3 (AXK3): two-variant interleaved MLA attention.

AXK3 keeps the A.X-K2 MLA + MoE backbone but interleaves two attention variants
per layer at an ``N:1`` ratio:

* **local** layers  -> MLA + sliding-window attention (SWA) with a per-head
  learnable **attention sink** (always on),
* **global** layers -> **block-sparse** MLA (MiniMax Sparse Attention style):
  the indexer scores are max-pooled over kv-blocks -> top-k blocks -> token
  indices, fed to the sparse ``flash_mla_with_kvcache``.

``N`` local layers precede each ``1`` global layer, repeating. ``N`` is the
config parameter ``local_per_global``; an explicit per-layer list
``attn_variant_pattern`` (entries ``"L"``/``"G"``) overrides it.

Both variants share the AXK2/AXK3 gating: an output gate
(``attention_output_gate``) and a low-rank gated RMSNorm (``gated_norm``).

STATUS: SKELETON. This wires the module graph, per-layer routing, config, and
registration so the structure is in place and importable. The real modeling
(the windowed + attention-sink FlashMLA call for local layers, the absorbed
sparse/dense decode dispatch, and the exact gate input) is marked ``TODO`` and
will be dropped in with the AXK3 checkpoint + modeling code. There is no AXK3
checkpoint yet, so this cannot be validated end-to-end here.
"""

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    DeepseekV2MoE,
    Indexer,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
)


def axk3_layer_variant(config, layer_idx: int) -> str:
    """Return ``"global"`` (block-sparse) or ``"local"`` (MLA-SWA) for a layer.

    Ratio is ``N`` local layers per ``1`` global layer (``local_per_global``):
    the global layer is the last of each ``N+1``-sized group. An explicit
    ``attn_variant_pattern`` (list of ``"L"``/``"G"``) overrides the ratio.
    MTP / next-n layers (idx >= num_hidden_layers) default to local.
    """
    pattern = getattr(config, "attn_variant_pattern", None)
    if pattern is not None and 0 <= layer_idx < len(pattern):
        return "global" if str(pattern[layer_idx]).upper() == "G" else "local"
    n = int(getattr(config, "local_per_global", 3))
    group = n + 1
    if group <= 1:
        return "global"
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is not None and layer_idx >= num_hidden_layers:
        return "local"
    return "global" if (layer_idx + 1) % group == 0 else "local"


class AXK3GatedRMSNorm(nn.Module):
    """Low-rank gated RMSNorm (AXK2/AXK3).

    ``y = RMSNorm(x); out = y * sigmoid(W_up(silu(W_down(y))))`` with a rank-r
    bottleneck. Drop-in for ``RMSNorm``: supports the fused ``(x, residual)``
    call used by the decoder layer.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.W_down = ReplicatedLinear(hidden_size, rank, bias=False)
        self.W_up = ReplicatedLinear(rank, hidden_size, bias=False)

    def _gate(self, y: torch.Tensor) -> torch.Tensor:
        raw = self.W_up(F.silu(self.W_down(y)[0]))[0]
        return (y * torch.sigmoid(raw.float())).to(y.dtype)

    def forward(self, x, residual=None):
        if residual is not None:
            y, residual = self.norm(x, residual)
            return self._gate(y), residual
        return self._gate(self.norm(x))


class AXK3Attention(nn.Module):
    """AXK3 MLA attention with per-layer variant routing.

    global -> block-sparse (indexer + MSA block selection + sparse decode)
    local  -> MLA-SWA (sliding window + per-head learnable attention sink)
    Both apply the shared output gate.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        input_size: int | None = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        proj_input_size = input_size if input_size is not None else hidden_size

        # --- MLA projections (mirrors DeepseekV2MLAAttention) ---
        if q_lora_rank is not None:
            self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
                proj_input_size,
                [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.kv_a_proj_with_mqa = None
            self.q_proj = None
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                kv_lora_rank + qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                proj_input_size,
                num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.fused_qkv_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None

        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # --- per-layer variant routing (N local : 1 global) ---
        layer_id = extract_layer_index(prefix)
        self.variant = axk3_layer_variant(config, layer_id)
        self.is_global = self.variant == "global"
        self.is_sparse = self.is_global

        # global -> block-sparse MSA indexer; local -> no indexer.
        if self.is_global:
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not getattr(
                    config, "indexer_rope_interleave", False
                ),
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
                is_inplace_rope=self.indexer_rope_emb.enabled(),
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        # local -> MLA-SWA: sliding window + per-head learnable attention sink
        # (always on). window_size is the causal left window.
        if not self.is_global:
            self.sliding_window = int(getattr(config, "sliding_window", 4096))
            # per-head sink logit (added to the softmax denominator by FlashMLA
            # sink_bias / attn_sink). float32, one per local head.
            self.attn_sink = nn.Parameter(
                torch.zeros(self.num_local_heads, dtype=torch.float32)
            )
        else:
            self.sliding_window = None
            self.attn_sink = None

        # --- shared output gate: attn_out * sigmoid(linear_gate(q_c)) ---
        # gate input is the compressed query (q_lora_rank) or hidden_size.
        self.use_output_gate = bool(
            getattr(config, "attention_output_gate", False)
        )
        if self.use_output_gate:
            gate_in = q_lora_rank if q_lora_rank is not None else hidden_size
            self.linear_gate = ColumnParallelLinear(
                gate_in,
                num_heads * v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_gate",
            )
        else:
            self.linear_gate = None

        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            # Output-gate input is the pre-layernorm compressed query (matches
            # the AXK2 modeling: gate = linear_gate(q_a_proj(hidden))).
            gate_in = q_c
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]
            gate_in = hidden_states

        kv_c, k_pe = kv_lora.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_c_normed = self.kv_a_layernorm(kv_c)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)
        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        # global (block-sparse): run the indexer to fill topk_indices_buffer.
        if self.indexer is not None and self.is_sparse:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        # Core attention.
        # TODO(modeling): the LOCAL (MLA-SWA) variant must route through the
        # windowed + attention-sink FlashMLA call
        # ``flash_mla_with_kvcache(window_size=self.sliding_window,
        #   attn_sink=self.attn_sink, ...)`` (dense within the window). The
        # GLOBAL variant uses the sparse path (indices from the indexer). This
        # skeleton sends both through MLAAttention; the sliding-window/sink
        # kernel wiring and absorbed decode dispatch come with the real
        # modeling code + AXK3 checkpoint.
        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(
                hidden_states.shape[0],
                self.num_local_heads * self.v_head_dim,
            ),
        )

        # Shared output gate.
        if self.linear_gate is not None:
            gate = self.linear_gate(gate_in)[0]
            attn_out = (attn_out * torch.sigmoid(gate.float())).to(
                attn_out.dtype
            )

        return self.o_proj(attn_out)[0]


class AXK3DecoderLayer(DeepseekV2DecoderLayer):
    """Decoder layer: AXK3Attention + (optional) gated RMSNorm.

    Reimplements ``__init__`` (swapping the attention class and the norms) and
    inherits the forward from ``DeepseekV2DecoderLayer``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.use_mha = False

        is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )

        self.self_attn = AXK3Attention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 0),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 0),
            v_head_dim=getattr(config, "v_head_dim", 0),
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank")
            else None,
            kv_lora_rank=getattr(config, "kv_lora_rank", 0),
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
        )

        if is_moe_layer:
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        eps = config.rms_norm_eps
        gated = bool(getattr(config, "gated_norm", False))
        rank = int(getattr(config, "gated_norm_rank", 16))
        # input_layernorm: gated in all layers when gated_norm is on.
        if gated:
            self.input_layernorm = AXK3GatedRMSNorm(
                config.hidden_size, eps=eps, rank=rank
            )
        else:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        # post_attention_layernorm: gated only on MoE layers (dense layers use
        # plain RMSNorm), matching the AXK2 modeling.
        if gated and is_moe_layer:
            self.post_attention_layernorm = AXK3GatedRMSNorm(
                config.hidden_size, eps=eps, rank=rank
            )
        else:
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


@support_torch_compile
class AXK3Model(DeepseekV2Model):
    """AXK3 backbone: same as DeepseekV2Model but builds AXK3DecoderLayer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = None
        from vllm.platforms import current_platform

        self.device = current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        # index_topk present -> allocate the shared top-k indices buffer used by
        # the block-sparse (global) layers' indexer.
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                self.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AXK3DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], self.hidden_size
            )
        )
        self.aux_hidden_state_layers = tuple[int, ...]()
        self.use_mha = False
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )


class AXK3ForCausalLM(DeepseekV2ForCausalLM):
    """A.X-K3 for causal LM (MLA-SWA + block-sparse MLA, N:1)."""

    model_cls = AXK3Model
