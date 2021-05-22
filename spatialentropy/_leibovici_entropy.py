from typing import Union, List

import numpy as np
from neighborhood_analysis import get_point_neighbors

from ._utils import get_pair_count, types2int, get_pair


def leibovici(points: Union[List[float], np.ndarray],
              types: Union[List[str], np.ndarray],
              d: Union[int, float, None] = None,
              order: bool = True,
              base: Union[int, float, None] = None
              ) -> float:
    """Leibovici entropy

    Args:
        points: array, 2d array
        types: array, the length should correspond to points
        d: int or float, cut-off distance, default is 10
        order: bool, if True, (x1, x2) and (x2, x1) is not the same
        base: int or float, the log base, default is e

    Returns:
        float

    """
    if len(points) != len(types):
        raise ValueError("Array of points and types should have same length")

    if base is None:
        base = np.e

    if d is None:
        d = 10
    elif isinstance(d, (int, float)):
        pass
    else:
        raise TypeError("d should be a number.")

    points = [tuple(i) for i in points]
    if isinstance(types[0], str):
        types = types2int(types)

    neighbors = get_point_neighbors(points, r=d, labels=types)
    pair = get_pair(types, neighbors)
    pair_count = get_pair_count(pair, order)
    v = pair_count.values()
    # clean all elements that equal to zero to prevent divide by zero error
    v = np.array([i for i in v if i != 0])

    v = v / v.sum()
    v = v * np.log(1 / v) / np.log(base)

    return v.sum()


class leibovici_entropy(object):
    """Leibovici entropy

    Attributes:
        entropy: float, the value of leibovici entropy

    """

    def __init__(self, points, types, d=None, order=False, base=None):
        """

        Args:
            points: array, 2d array
            types: array, the length should correspond to points
            d: int or float, cut-off distance, default is 10
            order: bool, if True, (x1, x2) and (x2, x1) is not the same
            base: int or float, the log base, default is e

        """
        self.entropy = leibovici(points, types, d, order, base)
