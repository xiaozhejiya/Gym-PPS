"""Transactional capture of one DDPG update; graph buffers are never checkpointed."""
from copy import deepcopy
import time

import numpy as np
import torch


# cuBLAS caches a workspace per stream. Reuse the capture stream across agents
# and recaptures instead of retaining another workspace for every new shape.
_CAPTURE_STREAMS = {}


class CudaGraphPreparationError(RuntimeError):
    """Capture failed, but the training state was successfully restored."""


def _capture_stream(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _CAPTURE_STREAMS:
        _CAPTURE_STREAMS[index] = torch.cuda.Stream(device=device)
    return _CAPTURE_STREAMS[index]


class CudaGraphUpdate:
    def __init__(self, agent, batch, signature):
        started = time.perf_counter()
        self.signature = signature
        try:
            self.inputs = tuple(torch.as_tensor(value, dtype=torch.float32, device=agent.device).clone()
                                for value in batch)
            self.graph = torch.cuda.CUDAGraph()
            snapshot = deepcopy(agent.state_dict())
            numpy_rng = np.random.get_state()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            # Keep live tensor identities: captured kernels retain their addresses.
            parameter_values = [(parameter, parameter.detach().clone())
                                for network in (agent.actor, agent.actor_target, agent.critic, agent.critic_target)
                                for parameter in network.parameters()]
            optimizer_values = [(optimizer, {parameter: deepcopy(state)
                                            for parameter, state in optimizer.state.items()})
                                for optimizer in (agent.actor_opt, agent.critic_opt)]
        except Exception as error:
            # Preparation has not changed any training values or RNG yet. Allow
            # recoverable allocation failures to fall back, but stop on CUDA faults.
            torch.cuda.synchronize(agent.device)
            raise CudaGraphPreparationError(str(error)) from error

        def restore_values():
            with torch.no_grad():
                for parameter, saved in parameter_values:
                    parameter.copy_(saved)
                for optimizer, saved_states in optimizer_values:
                    for parameter, live in optimizer.state.items():
                        saved = saved_states.get(parameter)
                        for key, value in live.items():
                            if torch.is_tensor(value):
                                if saved is None:
                                    value.zero_()  # Adam's lazy state starts at zero.
                                else:
                                    value.copy_(saved[key])
                            elif saved is not None:
                                live[key] = saved[key]

        try:
            stream = _capture_stream(agent.device)
            stream.wait_stream(torch.cuda.current_stream(agent.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    agent._update_tensors(self.inputs)
            stream.synchronize()
            restore_values()
            stream.wait_stream(torch.cuda.current_stream(agent.device))
            with torch.cuda.graph(self.graph, stream=stream):
                self.losses = agent._update_tensors(self.inputs)
            stream.synchronize()
            restore_values()
            torch.cuda.synchronize(agent.device)
        except Exception as error:
            # Loading is only used on failure, when graph addresses will not be reused.
            # Drain partial warmup work on the capture stream before restoring;
            # an unrecoverable CUDA error must propagate instead of retrying.
            torch.cuda.synchronize(agent.device)
            agent.load_state_dict(snapshot)
            for network in (agent.actor, agent.critic):
                for parameter in network.parameters():
                    parameter.grad = None
            raise CudaGraphPreparationError(str(error)) from error
        finally:
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        self.capture_seconds = time.perf_counter() - started

    def replay(self, batch):
        for destination, source in zip(self.inputs, batch):
            destination.copy_(torch.as_tensor(source, dtype=torch.float32))
        self.graph.replay()
        return self.losses
