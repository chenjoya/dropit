import torch
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class _DropITAddmmAutocast(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx,
        bias: nn.Parameter,
        x: torch.Tensor,
        weight: nn.Parameter,
        dropiter
    ):
        ctx.dropiter = dropiter
        ctx.save_for_backward(dropiter.select(x, ctx), weight)
        return torch.addmm(bias, x, weight)
    
    @staticmethod 
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_bias = grad_output.sum(dim=0)
        x = ctx.dropiter.pad(x, ctx).view(-1, weight.size(0))
        grad_input = grad_output.mm(weight.T)
        grad_weight = x.T.mm(grad_output)
        return grad_bias, grad_input, grad_weight, None

addmm_with_autocast = _DropITAddmmAutocast.apply

class _DropITAddmm(Function):
    @staticmethod
    def forward(
        ctx,
        bias: nn.Parameter,
        x: torch.Tensor,
        weight: nn.Parameter,
        dropiter
    ):
        ctx.dropiter = dropiter
        ctx.save_for_backward(dropiter.select(x, ctx), weight)
        return torch.addmm(bias, x, weight)
    
    @staticmethod 
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        grad_bias = grad_output.sum(dim=0)
        x = ctx.dropiter.pad(x, ctx).view(-1, weight.size(0))
        grad_input = grad_output.mm(weight.T)
        grad_weight = x.T.mm(grad_output)
        return grad_bias, grad_input, grad_weight, None

addmm_without_autocast = _DropITAddmm.apply

# from transformers.pytorch_utils.Conv1D
def forward(self, x):
    size_out = x.size()[:-1] + (self.nf,)
    if self.training:
        addmm = addmm_with_autocast if self.dropiter.autocast else addmm_without_autocast
        x = addmm(self.bias, x.view(-1, x.size(-1)), self.weight, self.dropiter)
    else:
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
    return x.view(size_out)

if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.rand(8, 32, 128, requires_grad=True)
    from transformers.pytorch_utils import Conv1D
    from dropit.dropiter import DropITer
    layer = Conv1D(64, 128)
    layer.train()
    dropiter = DropITer('mink', 0.0)
    dropiter.autocast = False
    layer.dropiter = dropiter

    # layer(x).sum().backward()
    forward(layer, x).sum().backward()
    print(x.grad.mean(), layer.weight.grad.mean(), layer.bias.grad.mean())
    pass