# Copyright (c) 2021-present, Zhuang AI Group.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

if 'mesa' not in __name__:
    import custom_quant
else:
    from . import custom_quant

use_dropit = True

class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):

        custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)
        ctx.save_for_backward(weight, bias)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None

        weight, bias = ctx.saved_tensors
        input = custom_quant.Quant.restore(ctx)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(dtype=grad_output.dtype))

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

# --- MINK ---  
def select_mink(x: torch.Tensor, ctx=None):
    numel = x.numel()
    x = x.view(-1)
    idxs = x.abs().topk(int(numel * 0.1), sorted=False)[1]
    x = x[idxs]
    ctx.idxs = idxs.to(torch.int32)
    ctx.numel = numel
    return x

def pad_mink(x, ctx=None):      
    idxs = ctx.idxs.to(torch.int64)
    del ctx.idxs
    return torch.zeros(
        ctx.numel, device=x.device, dtype=x.dtype
    ).scatter_(0, idxs, x)
# --- MINK --- 

class _DropITLinearAutocast(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16) # to be compatible with amp
    def forward(
        ctx,
        x: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter
    ):
        ctx.x_shape = x.shape
        ctx.has_bias = bias is not None
        ctx.save_for_backward(select_mink(x, ctx), weight)
        return F.linear(x, weight, bias)
    
    @staticmethod 
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, list(range(grad_output.dim()-1))) if ctx.has_bias else None
        ic, oc = weight.shape
        x = pad_mink(x, ctx)
        grad_weight = grad_output.view(-1,ic).T.mm(x.view(-1,oc))
        grad_input = torch.matmul(grad_output, weight, out=x.view(ctx.x_shape))
        return grad_input, grad_weight, grad_bias, None

dropit_linear = _DropITLinearAutocast.apply

class Linear(nn.Linear, custom_quant.Quant):
    def __init__(self, in_features, out_features, bias=True, args=None, logger=None, quant_groups=1):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.tag = 'fc'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable and self.training:
            if use_dropit and not hasattr(self, 'no_dropit'):
                y = dropit_linear(x, self.weight, self.bias)
            else:
                y = linear.apply(x, self.weight, self.bias, self.clip_val, self.level,
                                self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.linear(x, self.weight, self.bias)
        return y

if __name__ == "__main__":
    model = Linear(100, 100)
    print(model)
    model.enable = True
    print(model)

