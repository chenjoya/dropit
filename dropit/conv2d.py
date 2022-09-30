import torch
from torch import nn
from torch.functional import F
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class _DropITConv2dAutocast(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        x: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter,
        stride, padding, dilation, groups,
        dropiter
    ):
        ctx.dropiter = dropiter
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.save_for_backward(dropiter.select(x, ctx), weight)
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    
    @staticmethod 
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, (0,2,3)) if ctx.has_bias else None
        x = ctx.dropiter.pad(x, ctx).view(ctx.x_shape)
        grad_input, grad_weight = torch.ops.aten.convolution_backward(
            grad_output, x, weight, None,
            ctx.stride, ctx.padding, ctx.dilation,
            False, [0], ctx.groups, (True, True, False)
        )[:2]
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

conv2d_with_autocast = _DropITConv2dAutocast.apply

class _DropITConv2d(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: nn.Parameter,
        bias: nn.Parameter,
        stride, padding, dilation, groups,
        dropiter
    ):
        ctx.dropiter = dropiter
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        ctx.save_for_backward(dropiter.select(x, ctx), weight)
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_bias = torch.sum(grad_output, (0,2,3)) if ctx.has_bias else None
        x = ctx.dropiter.pad(x, ctx).view(ctx.x_shape)
        grad_input, grad_weight = torch.ops.aten.convolution_backward(
            grad_output, x, weight, None,
            ctx.stride, ctx.padding, ctx.dilation,
            False, [0], ctx.groups, (True, True, False)
        )[:2]
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

conv2d_without_autocast = _DropITConv2d.apply

def forward(self, x):
    conv2d = conv2d_with_autocast if self.dropiter.autocast else conv2d_without_autocast
    padding = self.padding
    if self.padding_mode != 'zeros':
        x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        padding = (0,0)
    return conv2d(x, self.weight, self.bias, 
        self.stride, padding, self.dilation, self.groups,
        self.dropiter)