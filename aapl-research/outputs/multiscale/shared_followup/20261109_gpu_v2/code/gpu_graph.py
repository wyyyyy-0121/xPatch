"""Replay the unchanged fixed-size forward/backward kernel sequence.

Adam and finite-value checks stay outside the graph. Capture warmup is rolled
back, including BatchNorm buffers and random generators, before any real step.
"""
import torch


class GraphBackward:
    def __init__(self, net, optimizer, objective, args, truth):
        self.net = net
        self.parameters = tuple(net.parameters())
        self.args = tuple(torch.empty_like(value) for value in args)
        self.truth = torch.empty_like(truth)
        self.shapes = tuple(value.shape for value in args), truth.shape
        for dst, src in zip(self.args, args):
            dst.copy_(src)
        self.truth.copy_(truth)
        state = {key: value.detach().clone() for key, value in net.state_dict().items()}
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                objective(net(*self.args), self.truth).backward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self.value = objective(net(*self.args), self.truth)
            self.value.backward()
        self.gradients = tuple(parameter.grad for parameter in self.parameters)
        if any(gradient is None for gradient in self.gradients):
            raise RuntimeError('Graph capture omitted a gradient tensor')
        net.load_state_dict(state)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        torch.cuda.synchronize()

    def __call__(self, args, truth):
        if (tuple(value.shape for value in args), truth.shape) != self.shapes:
            raise ValueError('CUDA graph has a fixed batch shape')
        for parameter, gradient in zip(self.parameters, self.gradients):
            parameter.grad = gradient
        for dst, src in zip(self.args, args):
            dst.copy_(src)
        self.truth.copy_(truth)
        self.graph.replay()
        return self.value
