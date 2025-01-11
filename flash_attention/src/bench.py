import os 
import torch
import math
import time
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from flash_attn import flash_attn_func
from flashinfer import single_prefill_with_kv_cache

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
torch.manual_seed(0)
# torch.set_printoptions(profile="full")
# Load the CUDA kernel as a python module
myflash = load(name='myflash', 
                    sources=[
                        'flash_v3.cu', 
                    ], 
                    extra_cuda_cflags=[
                        '-O2', 
                        '-std=c++17', 
                        '-I/root/byf/opensource/cutlass/include', 
                    ], 
                )
batch_size = 1
head_num = 12
seq = 1024
dim = 64



q = torch.randn(batch_size, seq, head_num, dim).cuda().half()
k = torch.randn(batch_size, seq, head_num, dim).cuda().half()
v = torch.randn(batch_size, seq, head_num, dim).cuda().half()
q1 = q.transpose(1, 2).contiguous()
k1 = k.transpose(1, 2).contiguous()
v1 = v.transpose(1, 2).contiguous()
q2 = q.reshape(batch_size * seq, head_num, dim)
k2 = k.reshape(batch_size * seq, head_num, dim)
v2 = v.reshape(batch_size * seq, head_num, dim)

start_time = time.time()
r = myflash.flash(q1, k1, v1)
end_time = time.time()
print(f"myflash execution time: {(end_time - start_time)*1000} ms")

def manual_attn(q, k, v, attn_mask=None):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if attn_mask != None:
        att.masked_fill_(attn_mask, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y
start_time = time.time()
base = manual_attn(q1, k1, v1)
end_time = time.time()
print(f"base execution time: {(end_time - start_time)*1000} ms")

# start_time = time.time()
# flash_offical = flash_attn_func(q, k, v)
# end_time = time.time()
# print(f"flash_offical execution time: {(end_time - start_time)*1000} ms")

start_time = time.time()
flashinfer = single_prefill_with_kv_cache(q2, k2, v2)
end_time = time.time()
print(f"flashinfer execution time: {(end_time - start_time)*1000} ms")

# print(r)
# print(base)
print('attn values sanity check:', torch.allclose(r, base, rtol=1e-01, atol=1e-02))