import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor, tensor


def cutnan(array: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    remove full-nan rows and columns (last two dimensions) of array
    """
    ind0 = ~np.all(np.isnan(np.asarray(array)), axis=tuple(s for s in range(array.ndim - 1)))
    ind1 = ~np.all(np.isnan(np.asarray(array)), axis=tuple(s for s in range(array.ndim - 2)) + (array.ndim - 1,))
    return array[..., ind1, :][..., ind0]


def AddFillMask(x):
    mask = torch.all(torch.isfinite(x), 0, True)
    return torch.cat((torch.nan_to_num(x, 0, 0, 0), mask), 0)


def RemoveFillMask(x):
    mask = x[-1:, ...] < 0.95
    x = x[:-1, ...]
    x[mask.expand(x.shape)] = np.nan
    return x
