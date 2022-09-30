import torch
from torch import nn
from functools import partial

from .linear import forward as linear_forward
from .conv2d import forward as conv2d_forward

supports = {
    nn.Linear: linear_forward,
    nn.Conv2d: conv2d_forward,
}

class DropITer(object):
    def __init__(self, strategy, gamma):
        self.gamma = gamma
        self.reserve = 1 - gamma
        self.select = getattr(self, f"select_{strategy}")
        self.pad = getattr(self, f"pad_{strategy}")

    # --- VRANDOM ---    
    def select_vrandom(self, x: torch.Tensor):
        x = x.view(-1)
        return (torch.rand_like(x, device=x.device) <= self.reserve) * x
    
    def pad_vrandom(self, x):
        return x
    # --- VRANDOM ---  
    
    # --- MINK ---  
    def select_mink(self, x: torch.Tensor, ctx=None):
        numel = x.numel()
        x = x.view(-1)
        idxs = x.abs().topk(int(numel * self.reserve), sorted=False)[1]
        x = x[idxs]
        x.dropped = True # provide a flag for combing with others
        ctx.idxs = idxs.to(torch.int32)
        ctx.numel = numel
        return x
    
    def pad_mink(self, x, ctx=None):      
        idxs = ctx.idxs.to(torch.int64)
        del ctx.idxs
        return torch.zeros(
            ctx.numel, device=x.device, dtype=x.dtype
        ).scatter_(0, idxs, x)
    # --- MINK --- 
    
    # --- MINK (but move idx to cpu) ---  
    def select_mink_cpuidx(self, x: torch.Tensor, ctx=None):
        numel = x.numel()
        x = x.view(-1)
        idxs = x.abs().topk(int(numel * self.reserve), sorted=False)[1]
        x = x[idxs]
        x.dropped = True # provide a flag for act judges
        ctx.idxs = idxs.cpu()
        ctx.numel = numel
        return x
    
    def pad_mink_cpuidx(self, x, ctx=None):      
        idxs = ctx.idxs.to(x.device)
        del ctx.idxs
        return torch.zeros(
            ctx.numel, device=x.device, dtype=x.dtype
        ).scatter_(0, idxs, x)
    # --- MINK (but move idx to cpu) --- 

    @staticmethod
    def transfer(model, strategy, gamma, autocast):
        _type = type(model)
        if _type in supports and not hasattr(model, 'no_dropit'):
            dropiter = DropITer(strategy, gamma)
            dropiter.autocast = autocast # just for recording
            model.forward = partial(supports[_type], model)
            model.dropiter = dropiter
            print(f"{_type.__name__}.forward => dropit.{_type.__name__}.forward")
        for child in model.children():
            DropITer.transfer(child, strategy, gamma, autocast)
        return model
    
def to_dropit(model: nn.Module, strategy: str, gamma: float, autocast: bool):
    return DropITer.transfer(model, strategy, gamma, autocast)
    