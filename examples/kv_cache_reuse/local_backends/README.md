# Examples vLLM + LMCache w. local backends
LMCache should be able to reduce the generation time of the second and following calls.
## CPU offloading
- `python offload.py -v v0` - CPU offloading implementation for vLLM v0
- `python offload.py -v v1` - CPU offloading implementation for vLLM v1
## Disk offloading
- `python offload.py -v v0 --use-disk` - Disk offloading implementation for vLLM v0
- `python offload.py -v v1 --use-disk` - Disk offloading implementation for vLLM v1

## Multi-device disk offloading

Requires multiple GPUs (Tensor Parallelism). Multiple disks are optional —
any local directories work for testing the path routing logic.

> **Testing without SSDs:** Use plain directories to verify the routing logic:
> ```bash
> mkdir -p /tmp/disk{0,1}
> python offload.py --multi-disk /tmp/disk0 /tmp/disk1 --tensor-parallel-size 2
> ```

Two path sharding strategies are supported:

- **`by_gpu`** (default): each GPU selects one path based on its device index
  (`path[device_id % num_paths]`). Best when `#GPU == #SSD`.
- **`by_local_rank`**: each TP rank selects paths based on its local rank.
  Supports `#SSD > #GPU` by assigning a contiguous subset of paths per rank.
  Also ensures the same TP rank across DP replicas uses the same path,
  enabling KV cache reuse between DP replicas.

**`by_gpu` examples (TP=2, 2 SSD):**

```bash
python offload.py --multi-disk /mnt/disk0 /mnt/disk1 --tensor-parallel-size 2
```

- `cuda:0` → `/mnt/disk0`
- `cuda:1` → `/mnt/disk1`

**`by_local_rank` examples (TP=2, 4 SSD — 2 SSD per rank):**

```bash
python offload.py \
    --multi-disk /mnt/disk0 /mnt/disk1 /mnt/disk2 /mnt/disk3 \
    --path-sharding by_local_rank \
    --tensor-parallel-size 2
```

- `local_rank=0` → `/mnt/disk0`, `/mnt/disk1`
- `local_rank=1` → `/mnt/disk2`, `/mnt/disk3`

## RUST raw block based Disk offloading

   # WARNING: This will erase the content of target device.
- `python rust_backend_offload.py --disk_path=/dev/nvme0n1` - posix disk offloading
- `python rust_backend_offload.py --disk_path=/dev/nvme0n1 --use_uring` - io_uring disk offloading
