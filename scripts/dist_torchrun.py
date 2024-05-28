
def print_env(verbose=False):
    import os

    print("============ ENV ===============")
    cared = ["SLURM_JOB_NODELIST", "SLURM_TASKS_PER_NODE",
            "SLURM_TOPOLOGY_ADDR", "SLURMD_NODENAME", "SLURM_LOCALID",
            "SLURM_NODEID", "SLURM_JOB_NUM_NODES",
            "SLURM_PROCID", "SLURM_NTASKS",
            "MASTER_ADDR","MASTER_PORT",
            "RANK","WORLD_SIZE", 
            "LOCAL_RANK","LOCAL_WORLD_SIZE"]

    for env_k in cared:
        env_v = os.environ.get(env_k, "MISSING")
        print(f"{env_k:-<25s}-> {env_v}")
    if verbose:
        print(f"---------- Others: ----------")

        for env_k, env_v in os.environ.items():
            if env_k not in cared:
                print(f"{env_k:-<25s}-> {env_v}")
    print("============ ENV END ===========")




def example(rank, world_size, local_rank):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP

    print(f"Start train: {rank}/{world_size}, local rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    print(f"current device: {torch.cuda.current_device()}, device count: {torch.cuda.device_count()}")
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    

    print(torch.tensor([1,2], device=local_rank))
    print(f"[{rank}] barrier...\n")
    dist.barrier()
    print(f"[{rank}] create local model...\n")
    # create local model
    model = nn.Linear(10, 10).to(local_rank)
    print(f"[{rank}] construct DDP model...\n")
    # construct DDP model
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # define loss function and optimizer
    print(f"[{rank}] constructed\n")
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    print(f"[{rank}] forward ...\n")
    outputs = ddp_model(torch.randn(20, 10).to(local_rank))
    print(f"[{rank}] forwarded\n")
    labels = torch.randn(20, 10).to(local_rank)
    # backward pass
    print(f"[{rank}] calc loss...\n")
    loss_fn(outputs, labels).backward()
    print(f"[{rank}] step on ...\n")
    # update parameters
    optimizer.step()
    print(f"[{rank}] barrier...\n")
    dist.barrier()
    print(f"[{rank}] Done\n")
    dist.destroy_process_group()



def train(rank=None, world_size=None, local_rank=None, master_addr=None, master_port=None):
    import os, time

    if rank is not None:
        os.environ["RANK"] = rank
    if world_size is not None:
        os.environ["WORLD_SIZE"] = world_size
    if local_rank is not None:
        os.environ["LOCAL_RANK"] = local_rank
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = master_port

    os.environ["NCCL_DEBUG"] = "INFO"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank==0:
        print_env()
    time.sleep(local_rank*0.1)
    example(rank, world_size, local_rank)


if __name__=="__main__":
    import argparse, os
    arg = argparse.ArgumentParser()
    arg.add_argument("-r", "--rank", type=int, default=None)
    arg.add_argument("-w", "--world-size", type=int, default=None)
    arg.add_argument("-l", "--local-rank", type=int, default=None)
    arg.add_argument("-a", "--master-addr", type=int, default=None)
    arg.add_argument("-p", "--master-port", type=int, default=None)

    params = arg.parse_args()
    print(params)
    train(params.rank, params.world_size, params.local_rank)


# torchrun standalone --nnodes=1 --nproc-per-node=3 dist_torchrun.py

# NCCL_DEBUG=INFO torchrun --nnodes=2 --nproc-per-node=1 --rdzv-id=demo --rdzv-backend=c10d --rdzv-endpoint=ks-gpu-1:19901 dist_torchrun.py
# NCCL_DEBUG=INFO torchrun --nnodes=2 --nproc-per-node=3 --rdzv-id=demo --rdzv-backend=c10d --rdzv-endpoint=ks-gpu-1:19901 dist_torchrun.py

# NCCL_DEBUG=INFO srun --ntasks=3 --gpus-per-task=1 --cpus-per-task=16 --nodelist=ks-gpu-7 --job-name=test --output=tmp/%x-%J-%t.log  -l python torch_dist.py
# NCCL_DEBUG=INFO srun --ntasks=3 --gpus-per-task=1 --cpus-per-task=16 --nodelist=ks-gpu-7 --job-name=test --output=tmp/%x-%J-%t.log  -l python torch_dist.py