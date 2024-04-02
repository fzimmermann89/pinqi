from scipy.optimize import minpack2
import torch
from .tensorgrad import SmoothTV
import numpy as np


def linesearch(fg, old_fgp, c1=1e-4, c2=0.9, amax=50, amin=1e-10, xtol=1e-14, maxiter=100):
    phi1, derphi1 = old_fgp
    alpha1 = 1.0
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b"START"
    for i in range(maxiter):
        stp, phi1, derphi1, task = minpack2.dcsrch(alpha1, phi1, derphi1, c1, c2, xtol, task, amin, amax, isave, dsave)
        if task[:2] == b"FG":
            alpha1 = stp
            f, g, gp = fg(stp)
            phi1, derphi1 = f.item(), gp.item()
        else:
            break
    else:
        return None, (fg(0)[:2])
    if task[:5] == b"ERROR" or task[:4] == b"WARN":
        print(task)
        return None, (fg(0)[:2])
    return stp, (f, g)


def conjugate_gradient_TV(A, AH, b, lam, gradOp, mu=1e-4, iters=50, tol=1e-16, x0=None):
    AHb = AH(b)
    AHA = lambda x: AH(A(x))
    x = torch.zeros_like(AHb) if x0 is None else x0
    p = torch.zeros(
        [1,] * x.ndim,
        device=x.device,
        dtype=x.dtype,
    ).broadcast_to(x.shape)

    def fg(alpha):
        ## todo: factor out fg to be able to use it for different problems
        xt = x + alpha * p
        tv, gradTV = tv_smoothed(xt, mu, gradOp, True)
        err = 0.5 * torch.view_as_real(A(xt) - b).square().sum()
        f = err + lam * tv
        g = AHA(xt) - AHb + lam * gradTV
        gp = dot(g, p)
        return (f, g, gp)

    f, oldg, _ = fg(0.0)
    p = -oldg
    gp = -dot(oldg)
    _ = fg(0), fg(1)

    for k in range(0, iters):
        alpha, (f, newg) = linesearch(fg, (f, gp), xtol=tol)
        if alpha is None:
            break
        x.add_(p, alpha=alpha)
        beta = max(0, dot(newg, (newg - oldg)) / dot(oldg))
        oldg = newg
        p *= beta
        p -= newg
        gp = dot(newg, p)
        if gp > 0:
            p = -newg
            gp = -1
    return x


def tv_smoothed(x, mu, gradOp, returnGrad=False):
    xg = x.detach().view(x.shape).requires_grad_(returnGrad)
    tv = SmoothTV.apply(xg, mu, gradOp)
    if returnGrad:
        grad = torch.autograd.grad(tv, xg)[0]
        return tv, grad
    return tv


def dot(x1, x2=None):
    if x1.is_complex():
        x1 = torch.view_as_real(x1)
    if x2 is None:
        return x1.ravel().square().sum()
    if x2.is_complex():
        x2 = torch.view_as_real(x2)
    return torch.dot(x1.ravel(), x2.ravel())
