# Based on https://github.com/mikgroup/sigpy/blob/1817ff849d34d7cbbbcb503a1b310e7d8f95c242/sigpy/sim.py#L10
# which is part of sigpy and licensed under BSD 3 Clause license 
# with copyright 2016 Frank Ong and The Regents of the University of California 

from typing import Union, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ellipsoiddata:
    amplitude: float
    scale: Tuple[float, float, float]
    offset: Tuple[float, float, float]
    angle: Tuple[float, float]


def shepp_logan(shape: Union[Tuple[int, int, int], Tuple[int, int]]):
    """
    Generates a Shepp Logan phantom with a given shape
    shape: Tuple of integer dimensions of phantom. If len==3, first dimension is 'head-foot'
    returns ndarray

    """
    ellipsoids = [
        ellipsoiddata(1, [0.69, 0.92, 0.81], [0.0, 0.0, 0], [0, 0, 0]),
        ellipsoiddata(1, [0.69, 0.92, 0.81], [0.0, 0.0, 0], [0, 0, 0]),
        ellipsoiddata(-0.8, [0.6624, 0.874, 0.78], [0.0, -0.0184, 0], [0, 0, 0]),
        ellipsoiddata(-0.2, [0.11, 0.31, 0.22], [0.22, 0.0, 0], [-18, 0, 10]),
        ellipsoiddata(-0.2, [0.16, 0.41, 0.28], [-0.22, 0.0, 0], [18, 0, 10]),
        ellipsoiddata(0.1, [0.21, 0.25, 0.41], [0.0, 0.35, -0.15], [0, 0, 0]),
        ellipsoiddata(0.1, [0.046, 0.046, 0.05], [0.0, 0.1, 0.25], [0, 0, 0]),
        ellipsoiddata(0.1, [0.046, 0.046, 0.05], [0.0, -0.1, 0.25], [0, 0, 0]),
        ellipsoiddata(0.1, [0.046, 0.046, 0.05], [-0.08, -0.605, 0], [0, 0, 0]),
        ellipsoiddata(0.1, [0.023, 0.023, 0.02], [0.0, -0.606, 0], [0, 0, 0]),
        ellipsoiddata(0.1, [0.023, 0.023, 0.02], [0.06, -0.605, 0], [0, 0, 0]),
    ]
    return phantom(shape, ellipsoids)


def phantom(shape, ellipsoids):
    """
    Generate a cube of given shape using a list of ellipsoid
    parameters.
    """

    if len(shape) == 2:
        ndim = 2
        shape = (1, *shape)
    elif len(shape) == 3:
        ndim = 3
    else:
        raise ValueError("Incorrect dimension")
    out = np.zeros(shape)
    z, y, x = np.mgrid[
        -(shape[-3] // 2) : ((shape[-3] + 1) // 2),
        -(shape[-2] // 2) : ((shape[-2] + 1) // 2),
        -(shape[-1] // 2) : ((shape[-1] + 1) // 2),
    ]
    coords = np.stack((x.ravel() / shape[-1] * 2, y.ravel() / shape[-2] * 2, z.ravel() / shape[-3] * 2))

    for ellipsoid in ellipsoids:
        R = Rotation.from_euler("zxz", -np.array(ellipsoid.angle), degrees=True)
        r = (R.apply(np.array(coords.T)) - np.array(ellipsoid.offset)) / np.array(ellipsoid.scale)
        out.ravel()[(r**2).sum(-1) <= 1] += ellipsoid.amplitude
    if ndim == 2:
        return out[0, :, :]
    else:
        return out
