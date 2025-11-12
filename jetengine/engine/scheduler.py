from collections import deque
import torch
from torch.nn import functional as F
import numpy as np

from jetengine.config import Config
from jetengine.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine.engine.block_manager import BlockManager
from jetengine.layers.sampler import sample_with_temperature_topk_topp
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK, Sample


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.sample_pipe = LogitsPipe([
                                Temperature(),      # Scale logits by temperature
                                TopK(),             # Apply top-k filtering
                                Softmax(),          # Convert logits to probabilities
                                TopP(),             # Apply top-p filtering
                            ])
        self.sample_pipe_topk0 = LogitsPipe([
                        Temperature(),      # Scale logits by temperature
                        Softmax(),          # Convert logits to probabilities
                        TopP(),             # Apply top-p filtering
                        ])

    def add(self, seq: Sequence):
        self.running.append(seq)

    def is_finished(self):
        return not self.running

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        # 1. Schedule new sequences for prefill
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            # Simple batching: take as many as fit
            for seq in prefill_candidates:
                # num_tokens for a waiting seq is its prefill length
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    prefill_batch.append(seq)
            if prefill_batch:
                return prefill_batch, RunType.PREFILL   
        # 2. If no prefilling, create a DENOISE batch.
        denoise_candidates = [s for s in self.running if s.status == SequenceStatus.DENOISING or s.status == SequenceStatus.SAVING]
        if denoise_candidates:
            denoise_batch = []
            for seq in denoise_candidates:
                num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                if len(denoise_batch) < self.max_num_seqs and self.block_manager.can_append_blocks(num_new_blocks):
                    self.block_manager.append_blocks(seq, num_new_blocks)
                    denoise_batch.append(seq)
            if denoise_batch:
                return denoise_batch, RunType.DENOISE

        return None, None     

    def _sample_probs(self, logits, seq):
        """Sample probabilities from logits using sequence parameters."""
        pipe = self.sample_pipe if seq.top_k > 0 else self.sample_pipe_topk0
        params = {'temperature': seq.temperature, 'top_p': seq.top_p}
        if seq.top_k > 0:
            params['top_k'] = seq.top_k
        return pipe(logits, **params)
    
    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        # Compute probabilities once if consistent sampling params
        batch_probs = None
        if self.consistent_sampling_params:
            batch_probs = self._sample_probs(logits, seqs[0])
        
        if run_type == RunType.PREFILL:
            for idx, seq in enumerate(seqs):
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
                probs = batch_probs[idx] if batch_probs is not None else self._sample_probs(logits[idx], seq)
                seq_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
                seq.intermediate_block_tokens[0] = seq_x0.item()
        
        elif run_type == RunType.DENOISE:
            start_idx = 0
            for seq in seqs:
                block_len = seq.block_length
                logits_slice = logits[start_idx : start_idx + block_len]
                probs = batch_probs[start_idx : start_idx + block_len] if batch_probs is not None else self._sample_probs(logits_slice, seq)
                seq_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
                seq_x0_p = torch.gather(probs, -1, seq_x0.unsqueeze(-1)).squeeze(-1)
                    
                if seq.status == SequenceStatus.DENOISING:
                    current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                    mask_index = (current_block_tensor == self.mask_token_id)
                    num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                    transfer_index = torch.zeros_like(seq_x0, dtype=torch.bool)
                    
                    if seq.remasking_strategy == 'sequential':
                        if mask_index.any():
                            first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                            end_pos = min(first_mask_pos + num_to_transfer, block_len)
                            transfer_index[first_mask_pos:end_pos] = True
                    else:
                        # Compute confidence for low_confidence strategies
                        seq_x0_p_input = torch.cat([seq_x0_p[:1], seq_x0_p[:-1]])
                        confidence = torch.where(mask_index, seq_x0_p_input, -np.inf) # confidence[0] is always -inf
                        
                        if 'low_confidence_dynamic' in seq.remasking_strategy:
                            transfer_index = confidence > seq.dynamic_threshold
                            transfer_index[-1] = False
                            if transfer_index.sum() < num_to_transfer:
                                _, top_indices = torch.topk(confidence, num_to_transfer)
                                transfer_index.fill_(False)
                                transfer_index[top_indices] = True
                                transfer_index[-1] = False
                            num_to_transfer = max(transfer_index.sum().item(), num_to_transfer)
                        elif 'low_confidence_static' in seq.remasking_strategy:
                            _, top_indices = torch.topk(confidence, num_to_transfer)
                            transfer_index[top_indices] = True
                        else:
                            raise ValueError(f"Unknown remasking strategy: {seq.remasking_strategy}")

                    # Update intermediate block tokens
                    seq_x0_input = torch.cat([current_block_tensor[:1], seq_x0[:-1]])
                    new_block_list = current_block_tensor.tolist()
                    for idx in transfer_index.nonzero(as_tuple=True)[0].tolist():
                        new_block_list[idx] = seq_x0_input[idx].item()
                    seq.intermediate_block_tokens = new_block_list
                    seq.current_denoising_step += 1
                    
                    # Check if block is fully denoised
                    is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                        (seq.current_denoising_step >= seq.denoising_steps)
                    if is_fully_denoised:
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = num_to_transfer
                    
                elif seq.status == SequenceStatus.SAVING:
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()
                        seq.intermediate_block_tokens[0] = seq_x0[-1].item()

                start_idx += block_len
                
        # Clean up finished sequences
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)