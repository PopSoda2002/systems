import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"Rank {rank} has data: {data}")
    dist.all_reduce(data, async_op=False)
    print(f"Rank {rank} has reduced data: {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(distributed_demo, args=(world_size,), nprocs=world_size, join=True)