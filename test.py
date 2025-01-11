# 4090
# l1 total 102400
shared_mem_per_block = 49152
shared_mem_per_sm = 101376 # max 
sm = 128
reg_per_sm = 65536
thread_per_sm = 1536

reg_per_thread = 106
shared_mem_per_thread = 1
thread_per_block = 32

while thread_per_block <= 2048:
    sm_limit_per_block = int(thread_per_sm / thread_per_block)
    reg_limit_per_block = int(reg_per_sm / (reg_per_thread * thread_per_block))
    shared_mem_limit_per_block = int(shared_mem_per_sm / (shared_mem_per_thread * thread_per_block))
    
    if shared_mem_per_thread * thread_per_block > shared_mem_per_block:
         shared_mem_limit_per_block = 0
    
    limit_per_sm = min(reg_limit_per_block,shared_mem_limit_per_block)
    limit_per_sm = min(limit_per_sm,sm_limit_per_block)

    print(sm_limit_per_block,reg_limit_per_block,shared_mem_limit_per_block)
    print("thread per block is",thread_per_block,"block per sm is",limit_per_sm,"occupancy is",thread_per_block*limit_per_sm/thread_per_sm,"block is",limit_per_sm*sm) 
    thread_per_block+=32
