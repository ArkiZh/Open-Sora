

def example(rank, world_size, local_rank):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP

    print(f"Start train. rank: {rank} world_size: {world_size}, local_rank: {local_rank}")
    print(f"Cuda count: {torch.cuda.device_count()}")
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    print(f"Attempt to use CUDA:{local_rank}: ", torch.tensor([6,6,6], device=local_rank))
    print("Waiting...")
    dist.barrier()
    # create local model
    model = nn.Linear(10, 10).to(local_rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[local_rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(local_rank))
    labels = torch.randn(20, 10).to(local_rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    dist.barrier()
    print("Done")


def train(rank, world_size, local_rank):
    import os
    os.environ["MASTER_ADDR"] = "ks-gpu-1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["NCCL_DEBUG"] = "INFO"
    example(rank, world_size, local_rank)


if __name__=="__main__":
    import argparse, os
    arg = argparse.ArgumentParser()
    arg.add_argument("-r", "--rank", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    arg.add_argument("-w", "--world-size", type=int, default=int(os.environ.get("SLURM_NTASKS", 0)))
    arg.add_argument("-l", "--local-rank", type=int, default=int(os.environ.get("SLURM_LOCALID", 0)))

    params = arg.parse_args()
    print(params)
    train(params.rank, params.world_size, params.local_rank)

# NCCL_DEBUG=INFO srun --ntasks=3 --gpus-per-task=1 --cpus-per-task=16 --nodelist=ks-gpu-7 --job-name=test --output=tmp/%x-%J-%t.log  -l python torch_dist.py
# NCCL_DEBUG=INFO srun --ntasks=2 --gpus=2 --nodelist=ks-gpu-1 --job-name=test --output=tmp/%x-%J-%t.log  -l python test_single_machine.py