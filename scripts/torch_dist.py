
def print_env():
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
    print(f"Cuda count: {torch.cuda.device_count()}")
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    print(torch.tensor([1,2], device="cuda:0"))
    print("Waiting...")
    dist.barrier()
    # create local model
    model = nn.Linear(10, 10).to(rank)
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


def parse_tasks_per_node(tasks_str):
    import re
    parts = tasks_str.split(',')
    result = []
    for part in parts:
        match = re.match(r'(\d+)\(x(\d+)\)', part)
        if match:
            element, times = match.groups()
            result.extend([int(element)] * int(times))
        else:
            result.append(int(part))
    return result

def parse_nodes(nodes_str):
    import re
    parts = nodes_str.split(',')
    result = []
    for part in parts:
        match = re.match(r'(\d+)\(x(\d+)\)', part)
        if match:
            element, times = match.groups()
            result.extend([int(element)] * int(times))
        else:
            result.append(int(part))
    return result


def set_env():
    import os
    os.environ["MASTER_ADDR"] = "10.20.0.2"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    # os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    # tasks_str = os.environ["SLURM_TASKS_PER_NODE"]
    # tasks = parse_tasks_per_node(tasks_str)
    # cur = tasks[int(os.environ["SLURM_NODEID"])]
    # os.environ["LOCAL_WORLD_SIZE"] = str(cur)


def train():
    import os
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # print(f"Start train: {rank}/{world_size}, local rank: {local_rank}")

    example(rank, world_size, 0)



def get_interfaces_and_ips():
    import subprocess
    import re

    output = subprocess.check_output("ip addr", shell=True).decode('utf-8')
    print(f"ip addr:\n{output}")

    lines = output.split('\n')

    interfaces = {}
    current_interface = None

    for line in lines:
        if not line.startswith(' ') and ":" in line:
            # This line contains the name of the interface
            current_interface = line.split(':')[1]
            print(current_interface)
            current_interface = current_interface.strip().split(' ')[0]
        elif 'inet ' in line:
            # This line contains the IP address for the current interface
            ip_address = re.search(r'inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line).group(1)
            interfaces[current_interface] = ip_address

    return interfaces

def print_interfaces_and_ips():
    interfaces = get_interfaces_and_ips()
    for iface, ip in interfaces.items():
        print(f"Current machine Interface: {iface}, IP: {ip}")


def get_ip_address_from_hostname(hostname):
    import socket
    try:
        ip_addr = socket.gethostbyname(hostname)
        print(f"Hostname: {hostname}, IP: {ip_addr}")
        return ip_addr
    except socket.gaierror:
        print(f"Hostname: {hostname}, IP CANNOT BE FOUND!")
        return None

get_ip_address_from_hostname("ks-gpu-7")

print_interfaces_and_ips()
set_env()
print_env()

# train()
# NCCL_DEBUG=INFO srun --ntasks=3 --gpus-per-task=1 --cpus-per-task=16 --nodelist=ks-gpu-7 --job-name=test --output=tmp/%x-%J-%t.log  -l python torch_dist.py
# NCCL_DEBUG=INFO srun --ntasks=3 --gpus-per-task=1 --cpus-per-task=16 --nodelist=ks-gpu-7 --job-name=test --output=tmp/%x-%J-%t.log  -l python torch_dist.py