from typing import Callable, Optional, Tuple, Union, List
from torch import Tensor
import torch


def norm(x: Tensor, dims: Optional[List[int]] = None):
    # ||x||_2^2
    x = torch.abs((x * x.conj()))
    if dims is None:
        return x.sum()
    else:
        return x.sum(dims, keepdim=True)


def numel(x, dims):
    numel = 1
    for i in dims:
        numel *= x.size(i)
    return numel


def cg(A: Callable, b: Tensor, x0: Optional[Tensor] = None, tol: float = 1e-8, maxiter=50, dims=(-1,)):
    # conjugate gradient solver for linear problem
    x = b.clone().detach() if x0 is None else x0.clone()
    r = b - A(x)
    p = r
    rsold = norm(r, dims)
    bnorm = 1  # 1/(norm(b, dims)+1e-6)
    if torch.max(rsold * bnorm) < (tol or 1e-12):
        return x
    for i in range(maxiter):  # max iterations
        Ap = A(p)
        alpha = rsold * 1 / ((p.conj() * Ap).sum(dims, keepdims=True))
        alpha[torch.isnan(alpha)] = 0
        x.addcmul_(alpha, p)
        r.addcmul_(alpha, Ap, value=-1)
        rsnew = norm(r, dims)
        if tol is not None:
            if torch.max(rsnew * bnorm) < tol:
                break
        beta = torch.nan_to_num_(rsnew / rsold)  # rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    x[torch.isnan(x)] = 0
    return x


class CGO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, AHA: Callable, AHy: Tensor, xreg: Tensor, x0: Tensor, l: Tensor, dims, tol: Optional[float] = 1e-8, maxiter: int = 100) -> Tensor:
        H = lambda x: AHA(x) + l * x
        b = AHy + l * xreg
        x = cg(H, b, x0, dims=dims, tol=tol, maxiter=maxiter)
        ctx.save_for_backward(x, xreg, l)
        ctx.H = H
        ctx.dims = dims
        ctx.tol = tol
        ctx.maxiter = maxiter
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, Tensor, None, Optional[Tensor], None]:
        # Calculate gradients explicitly instead of autogradding through the solver by inverting Hessian with CG
        x, xreg, l, *_ = ctx.saved_tensors
        iHgrad = cg(ctx.H, grad_output, dims=ctx.dims, maxiter=ctx.maxiter, tol=ctx.tol)
        grad_xreg = l * iHgrad
        grad_l = ((xreg - x).conj() * iHgrad).sum().real
        return None, None, grad_xreg, None, grad_l, None, None, None


class CGO2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, AHA: Callable, AHy: Tensor, xreg1: Tensor, xreg2: Tensor, x0: Tensor, l1: Tensor, l2: Tensor, dims, tol: Optional[float] = 1e-8, maxiter: int = 100) -> Tensor:
        H = lambda x: AHA(x) + (l1 + l2) * x
        b = AHy + l1 * xreg1 + l2 * xreg2
        x = cg(H, b, x0, dims=dims, tol=tol, maxiter=maxiter)
        ctx.save_for_backward(x, xreg1, l1, xreg2, l2)
        ctx.H = H
        ctx.dims = dims
        ctx.tol = tol
        ctx.maxiter = maxiter
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, Tensor, None, Optional[Tensor], None]:
        # Calculate gradients explicitly instead of autogradding through the solver by inverting Hessian with CG
        x, xreg1, l1, xreg2, l2, *_ = ctx.saved_tensors
        iHgrad = cg(ctx.H, grad_output, dims=ctx.dims, maxiter=ctx.maxiter, tol=ctx.tol)
        grad_xreg1 = l1 * iHgrad
        grad_l1 = ((xreg1 - x).conj() * iHgrad).sum().real
        grad_xreg2 = l2 * iHgrad
        grad_l2 = ((xreg2 - x).conj() * iHgrad).sum().real
        return None, None, grad_xreg1, grad_xreg2, None, grad_l1, grad_l2, None, None, None


class NewCGO(torch.autograd.Function):
    """
    Solves min_x |FCx-y|^2 + l*|x-xreg|^2
    with
    C: (...,x,y)->(...,Nc,x,y)
    F: (...,Nc,x,y)->(...,Nc,kx,ky)
    y: (...,Nc,kx,ky)
    
    Params:
    FHF: Callable: (...,Nc,x,y)->(...,Nc,x,y)
    xreg: (...,x,y)
    l: scalar
    xreg: (...,x,y)
    C (optional): (...,x,y)->(...,Nc,x,y), defaults to identity
    x0 (optional): (...,x,y), defaults to xreg

    """

    @staticmethod
    def forward(ctx, FHF: Callable, FHy: Tensor, xreg: Tensor, l: Tensor, c: Optional[Tensor], x0: Optional[Tensor] = None, tol: Optional[float] = 1e-8, maxiter: int = 100) -> Tensor:

        dims = (-1, -2, -3)
        if x0 is None:
            x0 = xreg.detach().clone()
        if c is None:
            AHA = FHF
            AHy = FHy
        else:
            CH = lambda x: (x * c.conj()).sum(-3)
            C = lambda x: x.unsqueeze(-3) * c
            AHA = lambda x: CH(FHF(C(x)))
            AHy = CH(FHy)

        H = lambda x: AHA(x) + l * x
        b = AHy + l * xreg
        x = cg(H, b, x0, dims=dims, tol=tol, maxiter=maxiter)
        ctx.save_for_backward(x, xreg, l, c)
        ctx.FHF = FHF
        ctx.dims = dims
        ctx.tol = tol
        ctx.maxiter = maxiter
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, Tensor, None, Optional[Tensor], None]:
        """
        Calculate gradients explicitly instead of autogradding through the solver by inverting Hessian with CG
        """
        x, xreg, l, c, *_ = ctx.saved_tensors
        FHF = ctx.FHF
        if c is None:
            AHA = FHF
            C = lambda x: x
        else:
            CH = lambda x: (x * c.conj()).sum(-3)
            C = lambda x: x.unsqueeze(-3) * c
            AHA = lambda x: CH(FHF(C(x)))

        H = lambda x: AHA(x) + l * x
        g = cg(H, grad_output, dims=ctx.dims, maxiter=ctx.maxiter, tol=ctx.tol)

        grad_FHy = C(g) if ctx.needs_input_grad[1] else None
        grad_xreg = l * g if ctx.needs_input_grad[2] else None
        grad_l = ((xreg - x) * g.conj()).sum().real if ctx.needs_input_grad[3] else None
        if c is not None and ctx.needs_input_grad[4]:
            grad_c = (FHy - FHF(C(x))) * g.conj() - FHF(C(g)) * x.conj()  # ((C'*F'*(F*C*x-y)))'*g wrt C
        else:
            grad_c = None

        return None, grad_FHy, grad_xreg, grad_l, grad_c, None, None, None
