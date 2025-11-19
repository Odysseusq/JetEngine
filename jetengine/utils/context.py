from dataclasses import dataclass, field
from typing import List
import torch

from jetengine.engine.sequence import RunType

@dataclass
class Context:
    run_type: RunType | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])
    block_length: int = 4
    seq_idx_denoising: List[int] = field(default_factory=list)
    seq_idx_saving: List[int] = field(default_factory=list)
    context_lens_denoising: torch.Tensor | None = None
    context_lens_saving: torch.Tensor | None = None
    block_tables_denoising: torch.Tensor | None = None
    block_tables_saving: torch.Tensor | None = None
    # For CUDA graph support (padding)
    seq_idx_denoising_out: torch.Tensor | None = None
    seq_idx_saving_out: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(run_type, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, is_last_denoise_step=[False], block_length=4, seq_idx_denoising=None, seq_idx_saving=None, context_lens_denoising=None, context_lens_saving=None, block_tables_denoising=None, block_tables_saving=None, seq_idx_denoising_out=None, seq_idx_saving_out=None):
    global _CONTEXT
    _CONTEXT = Context(
        run_type=run_type,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        is_last_denoise_step=is_last_denoise_step,
        block_length=block_length,
        seq_idx_denoising=seq_idx_denoising if seq_idx_denoising is not None else [],
        seq_idx_saving=seq_idx_saving if seq_idx_saving is not None else [],
        context_lens_denoising=context_lens_denoising,
        context_lens_saving=context_lens_saving,
        block_tables_denoising=block_tables_denoising,
        block_tables_saving=block_tables_saving,
        seq_idx_denoising_out=seq_idx_denoising_out,
        seq_idx_saving_out=seq_idx_saving_out
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
