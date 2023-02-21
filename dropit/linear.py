import torch
from torch import nn
from torch.autograd import Function
from torch.functional import F
from torch.cuda.amp import custom_fwd, custom_bwd

class _DropITLinearAutocast(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        x: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter,
        dropiter
    ):
        ctx.dropiter = dropiter
        ctx.x_shape = x.shape
        ctx.has_bias = bias is not None
        ctx.save_for_backward(dropiter.select(x, ctx), weight)
        return F.linear(x, weight, bias)
    
    @staticmethod 
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, list(range(grad_output.dim()-1))) if ctx.has_bias else None
        ic, oc = weight.shape
        x = ctx.dropiter.pad(x, ctx)
        grad_weight = grad_output.view(-1,ic).T.mm(x.view(-1,oc))
        grad_input = torch.matmul(grad_output, weight, out=x.view(ctx.x_shape))
        return grad_input, grad_weight, grad_bias, None

linear_with_autocast = _DropITLinearAutocast.apply

class _DropITLinear(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter,
        dropiter
    ):
        ctx.dropiter = dropiter
        ctx.x_shape = x.shape
        ctx.has_bias = bias is not None
        ctx.save_for_backward(dropiter.select(x, ctx), weight)
        return F.linear(x, weight, bias)
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, list(range(grad_output.dim()-1))) if ctx.has_bias else None
        ic, oc = weight.shape
        x = ctx.dropiter.pad(x, ctx)
        grad_weight = grad_output.view(-1,ic).T.mm(x.view(-1,oc))
        grad_input = torch.matmul(grad_output, weight, out=x.view(ctx.x_shape))
        return grad_input, grad_weight, grad_bias, None

linear_without_autocast = _DropITLinear.apply

def forward(self, x):
    if self.training:
        x = linear_with_autocast(x, self.weight, self.bias, self.dropiter) \
            if self.dropiter.autocast else linear_without_autocast(x, self.weight, self.bias, self.dropiter)
    else:
        x = F.linear(x, self.weight, self.bias)
    return x