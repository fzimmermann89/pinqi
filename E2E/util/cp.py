from typing import Optional, Union, Callable
from torch import Tensor
import torch
from . import GradOperators


def chambolle_pock(
    b,
    A: Callable,
    AH: Callable,
    gradOp: GradOperators,
    iters: int = 100,
    LA: Optional[float] = None,
    x0: Optional[Tensor] = None,
    lam: Union[float, Tensor] = 0.01,
    taufudge: Union[float, Tensor] = 1.0,
    sigmafudge: Union[float, Tensor] = 1.0,
    thetafudge: Union[float, Tensor] = 1.0,
    verbose: bool = False,
):
    """
    Chambolle Pock L2 TV according to
        arxiv.org/abs/1111.5632
        https://blog.allardhendriksen.nl/cwi-ci-group/chambolle_pock_using_tomosipo/
        https://github.com/pierrepaleo/ChambollePock/

    solves |Ax-b|^2 + lam*|TV(x)|_1
    A, AH: Callable Linear Operator and its adjoint
    gradOp: Instance of GradOperators providing apply_G and apply_GH for correct dimenions
    iters: Number of iterations
    LA: Operator Norm of AHA. If None, try to estimate by iterating
    lam: lambda
    taufudge, sigmafudge, thetafudge: Multiplicative modifieres of the constants in the algorithm, default to 1.
    verbose: return list of objective values
    """
    if LA is None:
        LA = operator_norm(lambda x: AH(A(x)), torch.randn_like(AH(b)))
    L = (LA**2 + gradOp.normGHG**2) ** (1 / 2)
    tau = 1 / L * taufudge
    sigma = 1 / L * sigmafudge
    theta = 1 * thetafudge
    x = 0 * AH(b) if x0 is None else x0
    xbar = x.clone()
    p = torch.zeros_like(b)
    q = torch.zeros_like(gradOp.apply_G(x))
    obj = []
    for n in range(iters):
        p.add_(A(xbar) - b, alpha=sigma)
        p.mul_(1 / (1 + sigma))
        q = clip(q + sigma * gradOp.apply_G(xbar), lam, -gradOp.dim - 1)
        xbar = -theta * x
        x.add_((AH(p) + gradOp.apply_GH(q)), alpha=-tau)
        xbar.add_(x, alpha=theta + 1)
        if verbose:
            obj.append(((A(x) - b).abs().square().sum().item(), lam * mag(gradOp.apply_G(x), -gradOp.dim - 1).sum().item()))
    if verbose:
        return x, obj
    else:
        return x


def mag(x, dim: int):
    return torch.norm(x, p=2, dim=dim, keepdim=True)


def clip(z, lam: float, dim: int = -3):
    ret = z * torch.clamp_(lam / mag(z, dim), min=None, max=1.0)
    return ret


def operator_norm(AHA, x0, num_iter=10):
    def norm(x):
        return x.square().sum().sqrt()

    x = torch.view_as_real(x0 + 0j)
    for i in range(num_iter):
        xnew = torch.view_as_real(AHA(torch.view_as_complex(x)) + 0j)
        estimate = norm(xnew) / norm(x)
        x = xnew / norm(xnew)
    return estimate.sqrt().item()
