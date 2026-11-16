"""Fixed-shape CUDA Graph replay for optimizer update callables."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch


class CudaGraphUpdate:
    """Warm up an update, capture its static tensor path, then replay it.

    The callable must return a fixed dict of detached device tensors and must
    not read tensor values on the host. Inputs are expected to live in persistent
    GPU buffers, such as Omni's replay buffer.
    """

    def __init__(self, fn: Callable[[], dict[str, torch.Tensor]], name: str, warmup: int = 3):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph updates require CUDA")
        self.fn, self.name, self.warmup = fn, name, warmup
        self.calls = 0
        self.graph = None
        self.outputs = None
        self.keys = None

    def __call__(self) -> dict[str, torch.Tensor]:
        if self.graph is not None:
            self.graph.replay()
            return {key: value.clone() for key, value in zip(self.keys, self.outputs)}
        self.calls += 1
        if self.calls <= self.warmup:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.fn()
            torch.cuda.current_stream().wait_stream(stream)
            return self.fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        start = time.perf_counter()
        with torch.cuda.graph(graph):
            result = self.fn()
        torch.cuda.synchronize()
        if not result or not all(isinstance(v, torch.Tensor) for v in result.values()):
            raise TypeError("CUDA Graph update must return a non-empty dict of tensors")
        self.keys = tuple(result)
        self.outputs = tuple(result[k] for k in self.keys)
        self.graph = graph
        print(f"[INFO] captured {self.name} CUDA Graph in {time.perf_counter() - start:.2f}s", flush=True)
        return {key: value.clone() for key, value in zip(self.keys, self.outputs)}
