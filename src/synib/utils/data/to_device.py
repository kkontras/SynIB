import torch

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out[k] = to_device(v, device)
        return out
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    if isinstance(x, set):
        return {to_device(v, device) for v in x}
    return x

def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out[k] = to_float(v)
        return out
    if isinstance(x, (list, tuple)):
        return type(x)(to_float(v) for v in x)
    if isinstance(x, set):
        return {to_float(v) for v in x}
    return x
