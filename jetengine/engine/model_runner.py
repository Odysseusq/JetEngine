import pickle
import os
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from jetengine.config import Config
from jetengine.engine.sequence import Sequence, RunType, SequenceStatus
from jetengine.models.qwen3_next import Qwen3NextForCausalLM
from jetengine.utils.context import set_context, get_context, reset_context
from jetengine.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, port: int, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        if "next" in hf_config.model_type:
            self.model = Qwen3NextForCausalLM(hf_config)
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")
        load_model(self.model, config.model)
        # Sampler is removed from here
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            shm_name = f"jetengineshm_{port}"
            if rank == 0:
                # Create (and clean up stale) shared memory
                try:
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**20)
                except FileExistsError:
                    try:
                        stale = SharedMemory(name=shm_name)
                        stale.close()
                        stale.unlink()
                    except FileNotFoundError:
                        pass
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len, self.config.mask_token_id) for _ in range(num_seqs)]
        self.run(seqs, RunType.PREFILL)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len == 0: return None
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32).cuda()

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, cu_seqlens_q, slot_mapping, is_last_step = [], [], [0], [], []
        max_seqlen_q = 0
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids)
            positions.extend(range(seqlen))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen)
            is_last_step.append(False)
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            for i in range(seqlen):
                block_idx = i // self.block_size 
                block_offset = i % self.block_size 
                physical_block_id = seq.block_table[block_idx]
                slot = physical_block_id * self.block_size + block_offset
                slot_mapping.append(slot)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(positions, dtype=torch.int64).cuda()
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).cuda()
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).cuda()
        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q, 
            cu_seqlens_k=cu_seqlens_q, 
            max_seqlen_q=max_seqlen_q, 
            max_seqlen_k=max_seqlen_q, 
            slot_mapping=slot_mapping, 
            is_last_denoise_step=is_last_step, # <-- Pass the new flag
            block_length=self.config.block_length
        )
        return input_ids, positions

    def prepare_denoise(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cached_lens_denoising, cached_lens_saving = [], []
        seq_idx_denoising, seq_idx_saving = [], []
        
        for idx, seq in enumerate(seqs):
            # The query is the current intermediate block
            q_tokens = seq.intermediate_block_tokens
            q_len = len(q_tokens)
            
            # The context (key/value) is the confirmed part of the sequence
            k_len = len(seq)
            
            input_ids.extend(q_tokens)
            # Positions are global
            positions.extend(range(k_len, k_len + q_len))
            if seq.status == SequenceStatus.DENOISING:
                seq_idx_denoising.append(idx)
                cached_lens_denoising.append(k_len)
            elif seq.status == SequenceStatus.SAVING:
                seq_idx_saving.append(idx)
                cached_lens_saving.append(k_len)
            else:
                raise ValueError(f"Sequence at index {idx} has invalid status for denoising: {seq.status}")

        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(positions, dtype=torch.int64).cuda()

        bs = len(seqs)
        
        # Pad and convert denoising indices
        num_denoising = len(seq_idx_denoising)
        pad_len_denoising = bs - num_denoising
        indices_denoising_read = torch.tensor(seq_idx_denoising + [0] * pad_len_denoising, dtype=torch.int64).cuda()
        indices_denoising_out = torch.tensor(seq_idx_denoising + [bs] * pad_len_denoising, dtype=torch.int64).cuda()
        cached_lens_denoising = torch.tensor(cached_lens_denoising + [0] * pad_len_denoising, dtype=torch.int32).cuda()
        
        seqs_denoising = [seqs[i] for i in seq_idx_denoising]
        if pad_len_denoising > 0 and seqs:
            seqs_denoising.extend([seqs[0]] * pad_len_denoising)
        block_tables_denoising = self.prepare_block_tables(seqs_denoising)

        # Pad and convert saving indices
        num_saving = len(seq_idx_saving)
        pad_len_saving = bs - num_saving
        indices_saving_read = torch.tensor(seq_idx_saving + [0] * pad_len_saving, dtype=torch.int64).cuda()
        indices_saving_out = torch.tensor(seq_idx_saving + [bs] * pad_len_saving, dtype=torch.int64).cuda()
        cached_lens_saving = torch.tensor(cached_lens_saving + [0] * pad_len_saving, dtype=torch.int32).cuda()

        seqs_saving = [seqs[i] for i in seq_idx_saving]
        if pad_len_saving > 0 and seqs:
            seqs_saving.extend([seqs[0]] * pad_len_saving)
        block_tables_saving = self.prepare_block_tables(seqs_saving)
        
        set_context(
            run_type=RunType.DENOISE,
            seq_idx_denoising=indices_denoising_read,
            seq_idx_saving=indices_saving_read,
            context_lens_denoising=cached_lens_denoising,
            context_lens_saving=cached_lens_saving,
            block_tables_denoising=block_tables_denoising,
            block_tables_saving=block_tables_saving,
            seq_idx_denoising_out=indices_denoising_out,
            seq_idx_saving_out=indices_saving_out,
            block_length=self.config.block_length
        )
        
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        context = get_context()
        run_type = context.run_type
        if run_type == RunType.PREFILL or self.enforce_eager or run_type is None:
            return self.model.compute_logits(self.model(input_ids, positions))
        
        bs = input_ids.size(0) // self.config.block_length
        graph_bs = next((x for x in self.graph_bs if x >= bs), None)
        if graph_bs is None:
            return self.model.compute_logits(self.model(input_ids, positions))
            
        graph = self.graphs[graph_bs]
        global_bs = bs * self.config.block_length
        self.static_input_ids[:global_bs].copy_(input_ids)
        self.static_positions[:global_bs].copy_(positions)
        self.static_seq_idx_denoising[:bs].copy_(context.seq_idx_denoising)
        self.static_seq_idx_saving[:bs].copy_(context.seq_idx_saving)
        self.static_seq_idx_denoising_out[:bs].copy_(context.seq_idx_denoising_out)
        self.static_seq_idx_saving_out[:bs].copy_(context.seq_idx_saving_out)
        self.static_context_lens_denoising[:bs].copy_(context.context_lens_denoising)
        self.static_context_lens_saving[:bs].copy_(context.context_lens_saving)
        if context.block_tables_denoising is not None:
            h, w = context.block_tables_denoising.shape
            self.static_block_tables_denoising[:h, :w].copy_(context.block_tables_denoising)
        if context.block_tables_saving is not None:
            h, w = context.block_tables_saving.shape
            self.static_block_tables_saving[:h, :w].copy_(context.block_tables_saving)
        if graph_bs > bs:
            self.static_input_ids[global_bs:graph_bs*self.config.block_length].fill_(0)
            self.static_positions[global_bs:graph_bs*self.config.block_length].fill_(0)
            self.static_seq_idx_denoising[bs:graph_bs].fill_(0)
            self.static_seq_idx_saving[bs:graph_bs].fill_(0)
            self.static_seq_idx_denoising_out[bs:graph_bs].fill_(graph_bs)
            self.static_seq_idx_saving_out[bs:graph_bs].fill_(graph_bs)
        graph.replay()
        return self.model.compute_logits(self.static_outputs[:global_bs])

    def run(self, seqs: list[Sequence], run_type: RunType) -> torch.Tensor:
        if run_type == RunType.PREFILL:
            input_ids, positions = self.prepare_prefill(seqs)
        elif run_type == RunType.DENOISE:
            input_ids, positions = self.prepare_denoise(seqs)
        else:
            return None

        logits = self.run_model(input_ids, positions)
        reset_context()
        return logits if self.rank == 0 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 256)
        max_global_bs = max_bs * self.config.block_length
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        self.static_input_ids = torch.zeros(max_global_bs, dtype=torch.int64)
        self.static_positions = torch.zeros(max_global_bs, dtype=torch.int64)
        self.static_outputs = torch.zeros(max_global_bs, hf_config.hidden_size)
        self.static_seq_idx_denoising = torch.zeros(max_bs, dtype=torch.int64)
        self.static_seq_idx_saving = torch.zeros(max_bs, dtype=torch.int64)
        self.static_seq_idx_denoising_out = torch.zeros(max_bs, dtype=torch.int64)
        self.static_seq_idx_saving_out = torch.zeros(max_bs, dtype=torch.int64)
        self.static_context_lens_denoising = torch.zeros(max_bs, dtype=torch.int32)
        self.static_context_lens_saving = torch.zeros(max_bs, dtype=torch.int32)
        self.static_block_tables_denoising = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        self.static_block_tables_saving = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            
            # Setup context with static buffers slices
            set_context(
                run_type=RunType.DENOISE,
                seq_idx_denoising=self.static_seq_idx_denoising[:bs],
                seq_idx_saving=self.static_seq_idx_saving[:bs],
                context_lens_denoising=self.static_context_lens_denoising[:bs],
                context_lens_saving=self.static_context_lens_saving[:bs],
                block_tables_denoising=self.static_block_tables_denoising[:bs],
                block_tables_saving=self.static_block_tables_saving[:bs],
                seq_idx_denoising_out=self.static_seq_idx_denoising_out[:bs],
                seq_idx_saving_out=self.static_seq_idx_saving_out[:bs],
                block_length=self.config.block_length
            )
            
            global_bs = bs * self.config.block_length
            self.model(self.static_input_ids[:global_bs], self.static_positions[:global_bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                self.static_outputs[:global_bs] = self.model(self.static_input_ids[:global_bs], self.static_positions[:global_bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
