from typing import Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist

from jetengine.layers.layernorm import RMSNorm
from jetengine.layers.attention import BlockAttention
from jetengine.layers.linear import (
    RowParallelLinear,
    ReplicatedLinear,
    QKVParallelLinear,
)
from jetengine.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from jetengine.layers.rotary_embedding import get_rope
from jetengine.kernels import fused_moe
from jetengine.models.sdar_moe import SDARMoeMLP
# --------------------------------------------------------------------------- #
#                               LOW-LEVEL BLOCKS                              #
# --------------------------------------------------------------------------- #


class Qwen3NextAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        rotary_dim: int | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads * 2,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim or self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = BlockAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        q_by_head, gate = q_gate.view(-1, self.num_heads, self.head_dim * 2).chunk(2, dim=-1)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(-1, self.q_size)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        o = o * gate.reshape(o.shape).sigmoid()
        output = self.o_proj(o)
        return output


class Qwen3NextSparseMoeBlock(nn.Module):
    """Top-k sparse MoE block (Switch-Transformer routing)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Gating must see the full hidden dimension on every rank, so replicate instead of RowParallel
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SDARMoeMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        )
        self.shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False)
        self.shared_expert = SDARMoeMLP(hidden_size, shared_expert_intermediate_size)
        # Cache for fused weights - will be populated on first forward pass
        # self._w1 = None
        # self._w2 = None
        self._prepare_fused_weights = False


    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S, H = hidden_states.shape
        if not hasattr(self, '_w1') or self._w1 is None:
            raise RuntimeError("Fused MoE weights _w1 not initialized. Ensure load_model was called.")
        if not hasattr(self, '_w2') or self._w2 is None:
            raise RuntimeError("Fused MoE weights _w2 not initialized. Ensure load_model was called.")
        flat = hidden_states.view(-1, H)
        router_logits = self.gate(flat)
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)
        out = fused_moe(hidden_states=hidden_states, w1=self._w1, w2=self._w2, topk_weights=top_p, topk_ids=top_i, inplace=False)
        shared_out = self.shared_expert(flat)
        shared_out = self.shared_expert_gate(flat).sigmoid() * shared_out
        out += shared_out
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(out)
        return out.view(S, H), router_logits





# --------------------------------------------------------------------------- #
#                               DECODER LAYER                                 #
# --------------------------------------------------------------------------- #
class Qwen3NextDecoderLayer(nn.Module):
    """Decoder layer that can be either dense or MoE depending on config."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.self_attn = Qwen3NextAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            rotary_dim=rotary_dim,
        )

        is_moe_layer = (
            config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen3NextSparseMoeBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                shared_expert_intermediate_size=config.shared_expert_intermediate_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                rms_norm_eps=config.rms_norm_eps,
            )
        else:
            self.mlp = SDARMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # ---- Attention ----
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        # ---- FFN / MoE ----
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        mlp_out = self.mlp(hidden_states)

        if isinstance(mlp_out, tuple):   # MoE returns (H, logits)
            mlp_out, router_logits = mlp_out
        else:
            router_logits = None

        hidden_states = mlp_out          # residual is added inside layernorms
        return hidden_states, residual, router_logits


# --------------------------------------------------------------------------- #
#                                 FULL MODEL                                  #
# --------------------------------------------------------------------------- #
class Qwen3NextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3NextDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...] | None]:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        router_logits_accum: list[torch.Tensor] = []

        for layer in self.layers:
            hidden_states, residual, router_logits = layer(positions, hidden_states, residual)
            if router_logits is not None:
                router_logits_accum.append(router_logits)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, tuple(router_logits_accum) if router_logits_accum else None


class Qwen3NextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3NextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # --------------------------------------------------------------------- #
    #                        PUBLIC INFERENCE API                           #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden_states, _ = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits