import os
import torch
import time
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, data_size):
    setup(rank, world_size)
    print(f"Rank {rank} has data: {data_size / 1024 / 1024} MB")
    length = data_size // 4
    data = torch.randint(0, 10, (length,))
    dist.all_reduce(data, async_op=False)
    print(f"Rank {rank} has reduced data: {data.shape}")

if __name__ == "__main__":
    processes_count = [2, 4, 6]
    communicated_data_size = [1024 * 1024, 1024 * 1024 * 1024]
    for num_processes in processes_count:
        for data_size in communicated_data_size:
            time_start = time.time()
            torch.cuda.synchronize()
            mp.spawn(distributed_demo, args=(num_processes, data_size), nprocs=num_processes, join=True)
            torch.cuda.synchronize()
            time_end = time.time()
            print(f"For {num_processes} processes, {data_size / 1024 / 1024} MB of data, Time taken: {time_end - time_start} seconds")