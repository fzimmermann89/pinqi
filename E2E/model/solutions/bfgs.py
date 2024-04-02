import torch
from typing import Optional, Tuple, List, Callable
from torch import Tensor
from .cg import cg
from ...util.bind import bind, invbind


class BFGS(torch.autograd.Function):
    """
    Wrapper around optim.bfgs
    """

    @staticmethod
    def forward(ctx, f: Callable, y: Tensor, xreg: Tensor, l: Tensor, x0: Optional[Tensor] = None, bounds: Optional[Tensor] = None):
        """
        BFGS
        """
        ctx.f = f
        if x0 is None:
            x = invbind(xreg.detach().clone(), bounds).requires_grad_(True)
        else:
            x = invbind(x0.detach().clone(), bounds).requires_grad_(True)
        xreg = xreg.detach()
        y = y.detach()
        lscalar = l.item()
        optim = torch.optim.LBFGS([x], history_size=10, max_iter=200, tolerance_change=1e-15, tolerance_grad=1e-15, lr=1.0, line_search_fn="strong_wolfe")  # *x.flatten(0,-2)

        def closure():
            x.grad = None
            bound = bind(x, bounds)
            fx = f(bound)
            a = torch.nn.functional.mse_loss(fx, y, reduction="sum")
            b = lscalar * torch.nn.functional.mse_loss(bound, xreg, reduction="sum")
            output = a + b
            output.backward()
            return output

        optim.step(closure)
        optim.step(closure)
        x.grad = None
        x = bind(x, bounds)
        ctx.save_for_backward(x, y, xreg, l)
        return torch.clip(torch.nan_to_num(x), -1e6, 1e6)

    @staticmethod
    def backward(ctx, grad):
        xprime, y, xreg, l = ctx.saved_tensors
        xprime = xprime.detach().clone().requires_grad_(True)
        params = (y, xreg, l)
        params = [p.detach().clone().requires_grad_(True) if ctx.needs_input_grad[i + 1] else p.detach() for i, p in enumerate(params)]
        dparams = [p for p in params if p.requires_grad]
        (y, xreg, l) = params
        f = ctx.f
        objective = lambda x: (torch.nn.functional.mse_loss(f(x), y, reduction="sum") + l * torch.nn.functional.mse_loss(x, xreg, reduction="sum"))
        A = lambda v: torch.autograd.functional.vhp(objective, xprime, v=v)[1]
        iHv = cg(A, grad, maxiter=100, tol=1e-15, dims=tuple(range(-1, -xreg.ndim - 1, -1)))
        with torch.enable_grad():
            gt = torch.autograd.grad(objective(xprime), xprime, create_graph=True)[0]
            ggt = list(torch.autograd.grad(gt, dparams, -iHv))
        grad_output = [None]
        for need_grad in ctx.needs_input_grad[1:4]:
            if need_grad:
                w = ggt.pop(0)
                w = torch.clip(torch.nan_to_num(w), -1e6, 1e6)
                grad_output.append(w)
            else:
                grad_output.append(None)

        grad_output.extend([None, None])  # x0, bounds
        return tuple(grad_output)
