import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
import math
from typing import Optional

# RWKV State Indices
PREV_X_TIME = 0
NUM_STATE = 1
DEN_STATE = 2
MAX_STATE = 3
PREV_X_CHANNEL = 4

@dataclass
class RWKVConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_embd: int = 256
    bias: bool = True
    intermediate_size: int = None
    use_customized_cuda_kernel: bool = False
    dtype: str = "float32"

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class ChannelMixing(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id
        
        n_embd = config.n_embd
        intermediate_size = (
            config.intermediate_size if config.intermediate_size is not None else 4 * n_embd
        )
        
        self.key_proj        = nn.Linear(n_embd, intermediate_size, bias=False)
        self.value_proj      = nn.Linear(intermediate_size, n_embd, bias=False)
        self.receptance_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.time_mix_key        = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))

    def forward(self, x, state=None):
        if state is not None:
             prev_x = state[self.layer_id, :, PREV_X_CHANNEL, :].unsqueeze(1)
             state[self.layer_id, :, PREV_X_CHANNEL, :] = x.squeeze(1)
        else:
            prev_x = self.time_shift(x)
            
        receptance = x * self.time_mix_receptance + prev_x * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)
        receptance = F.sigmoid(receptance)

        key = x * self.time_mix_key + prev_x * (1 - self.time_mix_key)
        key = self.key_proj(key)

        value = self.value_proj(torch.square(torch.relu(key)))

        out = receptance * value
        return out, state

class TimeMixing(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.layer_id = layer_id
        
        n_embd = config.n_embd
        attn_sz = n_embd

        self.key_proj        = nn.Linear(n_embd, attn_sz, bias=False)
        self.value_proj      = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_proj = nn.Linear(n_embd, attn_sz, bias=False)
        self.output_proj     = nn.Linear(attn_sz, n_embd, bias=False)

        self.time_decay          = nn.Parameter(torch.empty(attn_sz))
        self.time_first          = nn.Parameter(torch.empty(attn_sz))
        self.time_mix_key        = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_value      = nn.Parameter(torch.empty(1, 1, n_embd))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, n_embd))
    
    def forward(self, x, state=None):
        if state is not None:
            prev_x = state[self.layer_id, :, PREV_X_TIME, :].unsqueeze(1)
            state[self.layer_id, :, PREV_X_TIME, :] = x.squeeze(1)
        else:
            prev_x = self.time_shift(x)

        key = x * self.time_mix_key + prev_x * (1 - self.time_mix_key)
        key = self.key_proj(key)
        
        value = x * self.time_mix_value + prev_x * (1 - self.time_mix_value)
        value = self.value_proj(value)
        
        receptance = x * self.time_mix_receptance + prev_x * (1 - self.time_mix_receptance)
        receptance = self.receptance_proj(receptance)
        receptance = F.sigmoid(receptance)

        wkv, state  = self.wkv_function(key, value, use_customized_cuda_kernel=self.config.use_customized_cuda_kernel, state=state)
        
        rwkv = receptance * wkv
        rwkv = self.output_proj(rwkv)
        
        return rwkv, state

    def wkv_function(self, key, value, use_customized_cuda_kernel, state=None):
        if state is None and use_customized_cuda_kernel and WKVKernel is not None:
            B, T, C = key.size()
            return WKVKernel.apply(B, T, C, self.time_decay, self.time_first, key, value), None
        
        else:    
            return self.wkv_python_function(key, value, self.time_decay, self.time_first, self.layer_id, state)

    @staticmethod
    @torch.jit.script
    def wkv_python_function(key: torch.Tensor, value: torch.Tensor, time_decay: torch.Tensor, time_first: torch.Tensor, layer_id: int, state: Optional[torch.Tensor] = None):
        B, seq_length, C = key.size()
        output = torch.zeros_like(key)

        if state is None:
            num_state = torch.zeros((B, C), device=key.device, dtype=torch.float32)
            den_state = torch.zeros((B, C), device=key.device, dtype=torch.float32)
            max_state = torch.zeros((B, C), device=key.device, dtype=torch.float32) - 1e38
        else:
            # We must use a local tensor to satisfy TorchScript's type checker
            s = state
            num_state  = s[layer_id, :, 1, :] # NUM_STATE = 1
            den_state  = s[layer_id, :, 2, :] # DEN_STATE = 2
            max_state  = s[layer_id, :, 3, :] # MAX_STATE = 3

        decay = -torch.exp(time_decay)

        for current_index in range(seq_length):
            k = key[:, current_index].float()
            v = value[:, current_index]

            max_for_output = torch.maximum(max_state, k + time_first)
            e1 = torch.exp(max_state - max_for_output)
            e2 = torch.exp(k + time_first - max_for_output)
            numerator = e1 * num_state + e2 * v
            denominator = e1 * den_state + e2
            output[:, current_index] = (numerator / denominator).to(output.dtype)

            max_for_state = torch.maximum(max_state + decay, k)
            e1 = torch.exp(max_state + decay - max_for_state)
            e2 = torch.exp(k - max_for_state)
            num_state = e1 * num_state + e2 * v
            den_state = e1 * den_state + e2
            max_state = max_for_state
        
        if state is not None:
            s = state
            s[layer_id, :, 1, :] = num_state
            s[layer_id, :, 2, :] = den_state
            s[layer_id, :, 3, :] = max_state
            return output, s
        else:
            return output, None

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TimeMixing(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = ChannelMixing(config, layer_id)

    def forward(self, x, state=None):
        residual = x
        x, state = self.attn(self.ln_1(x), state=state)
        x = x + residual
        
        residual = x
        x, state = self.ffn(self.ln_2(x), state=state)
        x = x + residual
        return x, state

WKVKernel = None

class RWKVWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rwkv = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            ln_p = LayerNorm(config.n_embd, bias=config.bias),
            h = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # New: Reward and Termination heads
        self.reward_head = nn.Sequential(
            nn.Linear(config.n_embd, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.done_head = nn.Sequential(
            nn.Linear(config.n_embd, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)
        
        if self.config.use_customized_cuda_kernel:
            self.load_cuda_kernel(config.dtype)

    def _init_weights(self, module):
        if isinstance(module, TimeMixing):
            layer_id = module.layer_id
            n_layer = self.config.n_layer
            n_embd = self.config.n_embd
            attn_sz = n_embd
            
            with torch.no_grad():
                ratio_0_to_1 = layer_id / (n_layer - 1)
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
                ddd = torch.ones(1, 1, n_embd)
                for i in range(n_embd):
                    ddd[0, 0, i] = i / n_embd

                decay_speed = torch.ones(attn_sz)
                for h in range(attn_sz):
                    decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                module.time_decay = nn.Parameter(decay_speed)

                zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
                module.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
                module.time_mix_key = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                module.time_mix_value = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                module.time_mix_receptance = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
        
        elif isinstance(module, ChannelMixing):
            layer_id = module.layer_id
            n_layer = self.config.n_layer
            n_embd = self.config.n_embd
            
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
                ddd = torch.ones(1, 1, n_embd)
                for i in range(n_embd):
                    ddd[0, 0, i] = i / n_embd
                module.time_mix_key = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                module.time_mix_receptance = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

    def load_cuda_kernel(self, dtype):
        global WKVKernel
        from torch.utils.cpp_extension import load
        T_MAX = self.config.block_size
        base_path = os.path.join(os.path.dirname(__file__), "nanoRWKV/cuda")
        
        try:
            if dtype == "bfloat16":
                srcs = [os.path.join(base_path, "wkv_op_bf16.cpp"), os.path.join(base_path, "wkv_cuda_bf16.cu")]
                wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=srcs, verbose=False, 
                    extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
            else:
                srcs = [os.path.join(base_path, "wkv_op.cpp"), os.path.join(base_path, "wkv_cuda.cu")]
                wkv_cuda = load(name=f"wkv_{T_MAX}", sources=srcs, verbose=False, 
                    extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
                
            class WKV(torch.autograd.Function):
                @staticmethod
                def forward(ctx, B, T, C, w, u, k, v):
                    ctx.B = B; ctx.T = T; ctx.C = C
                    w = -torch.exp(w.float().contiguous())
                    u = u.contiguous().float(); k = k.contiguous().float(); v = v.contiguous().float()
                    y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
                    wkv_cuda.forward(B, T, C, w, u, k, v, y)
                    ctx.save_for_backward(w, u, k, v, y)
                    return y
                @staticmethod
                def backward(ctx, gy):
                    B = ctx.B; T = ctx.T; C = ctx.C
                    w, u, k, v, y = ctx.saved_tensors
                    gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
                    gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
                    gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
                    gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
                    wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)
                    gw = torch.sum(gw, dim=0); gu = torch.sum(gu, dim=0)
                    return (None, None, None, gw, gu, gk, gv)
            
            WKVKernel = WKV
            print("RWKV CUDA Kernel Loaded!")
        except Exception as e:
            print(f"Failed to load RWKV CUDA kernel: {e}. Using pure python implementation.")
            WKVKernel = None

    def forward(self, idx, inputs_embeds=None, state=None):
        if inputs_embeds is not None:
             x = inputs_embeds
        else:
             x = self.rwkv.wte(idx)
        
        x = self.rwkv.ln_p(x)
        # state must be passed
        for block in self.rwkv.h:
            x, state = block(x, state)
            
        x = self.rwkv.ln_f(x)
        logits = self.lm_head(x)
        
        # Predict reward and done for every step in the sequence
        reward_pred = self.reward_head(x)
        done_pred = self.done_head(x)
        
        return logits, x, (reward_pred, done_pred), state
