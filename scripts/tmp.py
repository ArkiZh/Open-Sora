import torch.distributed as dist
import torch
# import os
# os.environ["OMP_NUM_THREADS"] = "2"

# Use address of one of the machines
dist.init_process_group()

rank = dist.get_rank()
# if rank==2:
#     raise Exception("222")

import time
time.sleep(rank)
# print(f"RANK {rank} !!!!!!!!!!!!!!!!!!!!")
# h = dist.isend(torch.tensor(rank), 2)


# All tensors below are of torch.int64 dtype.
# We have 2 process groups, 2 ranks.
device = torch.device(f'cuda:{rank}')
world = dist.get_world_size()
tensor_list = [torch.zeros(3, dtype=torch.int64, device=device) for _ in range(world)]
tensor = torch.tensor([rank], dtype=torch.int64, device=device)
dist.all_gather(tensor_list, tensor)
time.sleep(rank*0.1)
print(f"\nRANK: {rank}: {tensor_list}\n")
