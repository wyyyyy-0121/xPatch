"""Capture the unchanged complete forward/loss/backward computation."""
import torch


class CapturedStep:
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = torch.empty_like(inputs)
        self.targets = torch.empty_like(targets)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.inputs.copy_(inputs)
        self.targets.copy_(targets)
        saved_buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        stream = torch.cuda.Stream(device=inputs.device)
        stream.wait_stream(torch.cuda.current_stream(inputs.device))
        with torch.cuda.stream(stream), torch.enable_grad():
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(self.inputs), self.targets)
                loss.backward()
            del loss
        torch.cuda.current_stream(inputs.device).wait_stream(stream)
        stream.synchronize()
        model.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.enable_grad():
            self.output = model(self.inputs)
            self.loss = torch.nn.functional.mse_loss(self.output, self.targets)
            self.loss.backward()
        self.gradients = [p.grad for p in self.parameters]
        with torch.no_grad():
            for k, v in model.named_buffers():
                v.copy_(saved_buffers[k])
        model.zero_grad(set_to_none=True)

    def run(self, inputs, targets):
        self.inputs.copy_(inputs)
        self.targets.copy_(targets)
        for p, g in zip(self.parameters, self.gradients):
            p.grad = g
        self.graph.replay()
        return self.output, self.loss


class StepCache:
    def __init__(self, model):
        self.model = model
        self.captures = {}

    def run(self, inputs, targets):
        key = (tuple(inputs.shape), inputs.stride(), tuple(targets.shape),
            targets.stride(), self.model.training)
        if key not in self.captures:
            self.captures[key] = CapturedStep(self.model, inputs, targets)
        return self.captures[key].run(inputs, targets)


class CapturedForward:
    def __init__(self, model, inputs):
        self.inputs = torch.empty_like(inputs)
        self.inputs.copy_(inputs)
        stream = torch.cuda.Stream(device=inputs.device)
        stream.wait_stream(torch.cuda.current_stream(inputs.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                model(self.inputs)
        torch.cuda.current_stream(inputs.device).wait_stream(stream)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.inference_mode():
            self.output = model(self.inputs)

    def run(self, inputs):
        self.inputs.copy_(inputs)
        self.graph.replay()
        return self.output


class ForwardCache:
    def __init__(self, model):
        self.model, self.captures = model, {}

    def run(self, inputs):
        if self.model.training:
            raise RuntimeError("Inference graph requires model.eval()")
        key = (tuple(inputs.shape), inputs.stride())
        if key not in self.captures:
            self.captures[key] = CapturedForward(self.model, inputs)
        return self.captures[key].run(inputs)
