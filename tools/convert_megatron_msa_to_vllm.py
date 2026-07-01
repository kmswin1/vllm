# SPDX-License-Identifier: Apache-2.0
"""Megatron (block-sparse MSA + MLA-SWA, non-absorbed) -> vLLM (absorbed) checkpoint converter.

SKELETON / DRAFT (2026-06-23) — NOT runtime-validated. See MSA_MLASWA_INFERENCE_GAP_ANALYSIS.md.

Megatron TRAINS non-absorbed (per-head k/v from a combined ``linear_kv_up_proj``); vLLM SERVES
absorbed MLA (separate ``k_up`` / ``v_up``). This script (a) splits the combined kv-up-proj into
k-up / v-up, (b) renames Megatron param keys to vLLM keys, (c) passes the MSA index-branch weights
and the SWA attention sink through.

Target confirmed: A.X K2 / MSA serves on ``DeepseekV3ForCausalLM`` (deepseek_v2.py); registry maps
``AXK2ForCausalLM`` there. Names below are taken from that module's ``DeepseekV2MLAAttention`` +
``Indexer`` __init__ (NOT the deepseek_v4 compressor-based ``DeepseekV4Indexer``).

STATUS of each transform:
  - q-down + kv-down -> fused_qkv_a_proj .... IMPLEMENTED (concat q rows then kv rows).
  - q-up -> q_b_proj / layernorms / o_proj .. IMPLEMENTED (direct rename).
  - kv-up -> kv_b_proj (COMBINED, no split) . IMPLEMENTED (vLLM absorbs at runtime; split_kv_up_proj
                                              is kept only for an absorbed-at-convert target).
  - indexer wq_b / k_norm .................. IMPLEMENTED (direct rename).
  - indexer linear_wk -> wk_weights_proj .... IMPLEMENTED for the head_dim rows; n_head weight rows
                                              are PLACEHOLDER zeros — MSA has no weights_proj, so
                                              faithful serving needs an fp8_mqa_logits variant w/o
                                              relu+weights (gap-analysis open Q#2). WARNED at runtime.
  - attn_sink pass-through ................. IMPLEMENTED.
  - MoE / router / shared-expert / embed /   TODO — needs a real ckpt to confirm key strings
    final norm / lm_head ; dist-ckpt merge .. (tools/checkpoint/distributed_checkpoints_convertor ref).

Both checkpoints' exact key strings must be confirmed by dumping a real ckpt
(`python -c "import torch,safetensors; ..."`); the maps below are derived from the module code
(megatron multi_latent_attention.py / dsa.py / mla_swa.py; vllm models/deepseek_v4/attention.py).

Usage (skeleton):
    python tools/convert_megatron_msa_to_vllm.py --in <megatron_ckpt> --out <vllm_ckpt_dir> \
        --num-layers 48 --num-heads 64 --qk-head-dim 128 --v-head-dim 128 --kv-lora-rank 512
"""
from __future__ import annotations

import argparse
import re

import torch


# ---------------------------------------------------------------------------
# The one fully-specified transform: split Megatron's combined kv-up-proj.
# ---------------------------------------------------------------------------
def split_kv_up_proj(
    kv_up_weight: torch.Tensor,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split Megatron ``linear_kv_up_proj.weight`` into vLLM ``k_up`` / ``v_up`` weights.

    Megatron's combined kv-up produces, per head, ``[qk_head_dim (k_nope) | v_head_dim (v)]`` from
    the latent. The weight is ``[num_heads*(qk_head_dim+v_head_dim), kv_lora_rank]`` (out, in).
    Reshape to per-head, slice the out dim into the k-nope and v parts, flatten back.

    Returns (k_up_weight [num_heads*qk_head_dim, kv_lora_rank],
             v_up_weight [num_heads*v_head_dim,  kv_lora_rank]).
    """
    out_dim, kv_lora_rank = kv_up_weight.shape
    per_head = qk_head_dim + v_head_dim
    assert out_dim == num_heads * per_head, (
        f"kv_up out dim {out_dim} != num_heads*{per_head}={num_heads * per_head}; "
        f"check num_heads/qk_head_dim/v_head_dim."
    )
    w = kv_up_weight.view(num_heads, per_head, kv_lora_rank)
    k_up = w[:, :qk_head_dim, :].reshape(num_heads * qk_head_dim, kv_lora_rank).contiguous()
    v_up = w[:, qk_head_dim:, :].reshape(num_heads * v_head_dim, kv_lora_rank).contiguous()
    return k_up, v_up


# ---------------------------------------------------------------------------
# Name mapping. CONFIRMED against the vLLM-side target module (deepseek_v2.py,
# DeepseekV2MLAAttention + Indexer) — A.X K2 / MSA serves on DeepseekV3ForCausalLM (see
# registry.py: "AXK2ForCausalLM" -> deepseek_v2:DeepseekV3ForCausalLM).
# Megatron decoder param prefix: "decoder.layers.{i}.self_attention.<name>"
# vLLM   decoder param prefix:   "model.layers.{i}.self_attn.<name>"
#
# CRITICAL vLLM-target facts (deepseek_v2.py):
#   * q-down + kv-down are FUSED into ONE matrix `fused_qkv_a_proj` (DeepSeekV2FusedQkvAProjLinear),
#     out-dim = [q_lora_rank | kv_lora_rank + qk_rope_head_dim]. -> we must CONCAT Megatron's two
#     separate down-projs (q first, then kv), not rename them individually.   (handled in convert())
#   * KV up-proj is kept COMBINED as `kv_b_proj` (kv_lora_rank -> heads*(qk_nope+v)); vLLM absorbs at
#     runtime. -> DIRECT rename of Megatron `linear_kv_up_proj` (NO k/v split). `split_kv_up_proj`
#     above is only for an *absorbed-at-convert* target (e.g. the deepseek_v4 module) — NOT used here.
#   * Indexer fuses k-down + per-head weights into `wk_weights_proj` (MergedColumnParallelLinear,
#     out = [head_dim | n_head]). Megatron MSA `BlockSparseIndexer` has k-down (`linear_wk`) but NO
#     weights_proj (MSA = plain dot, max-over-heads, no per-head weighting). -> we place Megatron
#     `linear_wk` into the head_dim rows; the n_head weight rows are a PLACEHOLDER (see WARNING).
# ---------------------------------------------------------------------------
# Direct renames (weight passes through unchanged). LHS regex on the Megatron key.
DIRECT_RENAMES: list[tuple[str, str]] = [
    # MLA q path
    (r"self_attention\.linear_q_up_proj\.weight", "self_attn.q_b_proj.weight"),
    (r"self_attention\.q_layernorm\.weight", "self_attn.q_a_layernorm.weight"),
    # MLA kv path
    (r"self_attention\.kv_layernorm\.weight", "self_attn.kv_a_layernorm.weight"),
    (r"self_attention\.linear_kv_up_proj\.weight", "self_attn.kv_b_proj.weight"),  # COMBINED, no split
    # output
    (r"self_attention\.linear_proj\.weight", "self_attn.o_proj.weight"),
    # MSA index branch -> vLLM Indexer (deepseek_v2.py): wq_b / wk_weights_proj / k_norm.
    # `linear_wk` -> the head_dim rows of wk_weights_proj (the n_head weight rows are appended in
    # convert(); see WARNING about MSA scoring).
    (r"self_attention\.indexer\.linear_wq_b\.weight", "self_attn.indexer.wq_b.weight"),
    (r"self_attention\.indexer\.k_norm\.weight", "self_attn.indexer.k_norm.weight"),
    # SWA attention sink (per-head nn.Parameter) — for the local MLA-SWA layers.
    (r"self_attention\.attn_sink", "self_attn.attn_sink"),
]
# Keys handled specially in convert() (cannot be a 1:1 rename).
Q_DOWN_RE = re.compile(r"self_attention\.linear_q_down_proj\.weight")    # -> fused (q half)
KV_DOWN_RE = re.compile(r"self_attention\.linear_kv_down_proj\.weight")  # -> fused (kv half)
WK_RE = re.compile(r"self_attention\.indexer\.linear_wk\.weight")        # -> wk_weights_proj head rows
LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.")
# TODO (need a real ckpt to confirm key strings): MoE experts/router/shared-expert, embeddings, final
# norm, lm_head. The DSA dist-ckpt pattern in tools/checkpoint/distributed_checkpoints_convertor is
# the reference. TODO: TP/PP/EP shard merge must happen before this map (dist-ckpt is sharded).


def map_key(megatron_key: str) -> str | None:
    """Map a Megatron param key to a vLLM key via DIRECT_RENAMES. None if unhandled (skip + warn)."""
    for pat, repl in DIRECT_RENAMES:
        if re.search(pat, megatron_key):
            new = re.sub(pat, repl, megatron_key)
            new = new.replace("decoder.layers.", "model.layers.")
            return new
    return None


def _layer_of(key: str) -> str | None:
    m = LAYER_RE.search(key)
    return m.group(1) if m else None


def convert(state_dict: dict, args) -> dict:
    """Map Megatron (non-absorbed MSA) -> vLLM (DeepseekV3) attention/index weights.

    Transforms (attention only; MoE/embed/norm/lm_head are TODO — see module docstring):
      - fuse linear_q_down_proj + linear_kv_down_proj -> fused_qkv_a_proj (q rows then kv rows),
      - rename q_up/kv_up/layernorms/o_proj/indexer per DIRECT_RENAMES (kv_up kept combined),
      - build indexer.wk_weights_proj from linear_wk (head_dim rows) + placeholder weight rows.
    """
    out: dict[str, torch.Tensor] = {}
    unhandled: list[str] = []
    # Bucket the down-proj halves per layer so we can fuse them into one matrix.
    q_down: dict[str, torch.Tensor] = {}
    kv_down: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if Q_DOWN_RE.search(k):
            q_down[_layer_of(k)] = v
        elif KV_DOWN_RE.search(k):
            kv_down[_layer_of(k)] = v

    for k, v in state_dict.items():
        if Q_DOWN_RE.search(k) or KV_DOWN_RE.search(k):
            continue  # handled by the fusion pass below
        if WK_RE.search(k):
            # wk_weights_proj = [head_dim rows = linear_wk | n_head weight rows = PLACEHOLDER zeros].
            hidden = v.shape[1]
            n_head = int(args.index_n_heads)
            weight_rows = torch.zeros(n_head, hidden, dtype=v.dtype)
            fused = torch.cat([v, weight_rows], dim=0)
            base = re.sub(WK_RE, "", k).replace("decoder.layers.", "model.layers.")
            out[base + "self_attn.indexer.wk_weights_proj.weight"] = fused
            continue
        nk = map_key(k)
        if nk is None:
            unhandled.append(k)
            continue
        out[nk] = v

    # Fuse q-down + kv-down per layer -> fused_qkv_a_proj (q rows first, then kv rows).
    for layer in sorted(set(q_down) | set(kv_down), key=lambda x: int(x) if x else -1):
        if layer not in q_down or layer not in kv_down:
            print(f"[WARN] layer {layer}: missing q_down/kv_down half; cannot fuse fused_qkv_a_proj.")
            continue
        fused = torch.cat([q_down[layer], kv_down[layer]], dim=0)  # [q_lora + (kv_lora+rope), hidden]
        out[f"model.layers.{layer}.self_attn.fused_qkv_a_proj.weight"] = fused

    if any(WK_RE.search(k) for k in state_dict):
        print(
            "[WARNING] MSA index branch: vLLM scores via fp8_mqa_logits (DSA: relu + per-head "
            "weights). MSA is plain-dot / max-over-heads / NO weights. The n_head weight rows of "
            "wk_weights_proj are PLACEHOLDER zeros -> faithful MSA serving still needs an "
            "fp8_mqa_logits variant without relu/weights (gap-analysis open Q#2)."
        )
    if unhandled:
        print(f"[WARN] {len(unhandled)} unhandled keys (MoE/embed/norm/lm_head TODO). e.g.:")
        for k in unhandled[:20]:
            print("   ", k)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", required=True, help="Megatron checkpoint (state_dict / dir)")
    p.add_argument("--out", required=True, help="output vLLM checkpoint dir")
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--num-heads", type=int, required=True)
    p.add_argument("--qk-head-dim", type=int, default=128)
    p.add_argument("--v-head-dim", type=int, default=128)
    p.add_argument("--kv-lora-rank", type=int, default=512)
    p.add_argument("--index-n-heads", type=int, default=1, help="MSA index heads (1 = MQA default)")
    args = p.parse_args()

    # TODO: load the Megatron checkpoint (dist-ckpt merge across TP/PP/EP first). For a merged
    # single-file ckpt: sd = torch.load(args.inp, map_location="cpu")["model"] (adjust the key).
    raise SystemExit(
        "SKELETON: implement checkpoint load/save (dist-ckpt shard merge + safetensors write) and "
        "confirm the DIRECT_RENAMES / kv_up target names against a real ckpt + the vLLM MSA module. "
        "The split_kv_up_proj transform and the mapping rules above are ready to build on."
    )


if __name__ == "__main__":
    main()
