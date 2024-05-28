
def example(rank, world_size):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP

    local_rank = 0

    print(f"Start train: {rank}/{world_size}, local rank: {local_rank}")
    print(f"Cuda count: {torch.cuda.device_count()}")
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    print(torch.tensor([1,2], device="cuda:0"))
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
    
def train1(rank, world_size):
    import os
    os.environ["MASTER_ADDR"] = "ks-gpu-1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["NCCL_DEBUG"] = "INFO"
    example(rank, world_size)

if __name__=="__main__":
    import argparse
    arg = argparse.ArgumentParser()
    arg.add_argument("-r", "--rank", type=int)
    arg.add_argument("-w", "--world-size", type=int)
    args = arg.parse_args()
    print(args)
    train1(args.rank, args.world_size)