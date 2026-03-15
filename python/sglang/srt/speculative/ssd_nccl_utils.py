"""NCCL communication utilities for async SSD.

Fused int64 packing reduces NCCL call count by concatenating
multiple tensors into a single int64 payload.

Uses pg.send/pg.recv directly (not dist.send/dist.recv) because
the process group is created via ProcessGroupNCCL() constructor
rather than dist.new_group(), so it is not registered with
torch.distributed's group registry.

Protocol:
  Target → Draft:  cmd(1) → meta(N) → fused_int64(payload)
  Draft → Target:  fused_int64(response) → logits(float)
"""

import torch
import torch.distributed as dist


def _pg_send(pg, tensor: torch.Tensor, dst: int):
    """Send tensor via process group (blocking)."""
    work = pg.send([tensor.contiguous()], dst, 0)
    work.wait()


def _pg_recv(pg, tensor: torch.Tensor, src: int):
    """Receive into tensor via process group (blocking)."""
    work = pg.recv([tensor.contiguous()], src, 0)
    work.wait()


def concat_int64(*tensors: torch.Tensor) -> torch.Tensor:
    """Flatten and concatenate tensors into a single int64 payload."""
    parts = []
    for t in tensors:
        if t is None:
            continue
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        parts.append(t.reshape(-1))
    return torch.cat(parts, dim=0)


def send_int64(pg, dst: int, *tensors: torch.Tensor):
    """Send fused int64 payload via NCCL."""
    payload = concat_int64(*tensors)
    _pg_send(pg, payload, dst)


def recv_int64(pg, src: int, total_length: int, device: torch.device) -> torch.Tensor:
    """Receive fused int64 payload of known total length."""
    t = torch.empty((total_length,), dtype=torch.int64, device=device)
    _pg_recv(pg, t, src)
    return t


def send_cmd(pg, dst: int, cmd: int, device: torch.device):
    """Send a single int64 command."""
    t = torch.tensor([cmd], dtype=torch.int64, device=device)
    _pg_send(pg, t, dst)


def recv_cmd(pg, src: int, device: torch.device) -> int:
    """Blocking receive of a single int64 command."""
    t = torch.empty(1, dtype=torch.int64, device=device)
    _pg_recv(pg, t, src)
    return int(t.item())


def send_meta(pg, dst: int, *values: int, device: torch.device):
    """Send metadata as int64 tensor."""
    t = torch.tensor(list(values), dtype=torch.int64, device=device)
    _pg_send(pg, t, dst)


def recv_meta(pg, src: int, n: int, device: torch.device) -> list:
    """Receive N int64 metadata values."""
    t = torch.empty(n, dtype=torch.int64, device=device)
    _pg_recv(pg, t, src)
    return t.tolist()


def temps_to_int64(temps: torch.Tensor) -> torch.Tensor:
    """Encode float32 temperatures as int64 for fused transport."""
    return temps.to(torch.float32).view(torch.int32).to(torch.int64)


def int64_to_temps(t: torch.Tensor) -> torch.Tensor:
    """Decode int64-encoded temperatures back to float32."""
    return t.to(torch.int32).view(torch.float32)


# Command constants
CMD_SPECULATE = 0
CMD_PREFILL = 1
CMD_EXIT = 2
