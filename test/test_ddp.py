# test_ddp.py
import os
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = os.uname()[1]
    print(f"[Rank {rank}] on {hostname} initialized. World size: {world_size}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
