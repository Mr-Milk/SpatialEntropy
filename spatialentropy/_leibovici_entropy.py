from typing import Sequence

import numpy as np
from sklearn.metrics import pairwise_distances

from ._utils import pairs_counter, type_adj_matrix


class leibovici_entropy(object):
    """Calculate teh leibovici entropy

    Attributes:
        adj_matrix: array, return the adjacency matrix
        pairs_counts: dict, the counts of each pairs
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
        if len(points) != len(types):
            raise ValueError("Array of points and types should have same length")

        if base is None:
            base = np.e

        if d is None:
            d1 = 0
            d2 = 10
        elif isinstance(d, (int, float)):
            d1 = 0
            d2 = d
        elif isinstance(d, Sequence):
            d1 = d[0]
            d2 = d[1]
        else:
            raise ValueError("d could either be a number or an interval.")

        self._points = points
        self._types = types
        self._order = order
        self._base = base
        self.adj_matrix = pairwise_distances(self._points)
        self._d1 = d1
        self._d2 = d2

        self._count()

    def _count(self):
        bool_matx = ((self.adj_matrix >= self._d1) & (self.adj_matrix <= self._d2)).astype(int)
        type_matx, utypes = type_adj_matrix(bool_matx, self._types)
        pairs_counts = pairs_counter(type_matx, utypes, self._order)

        v = pairs_counts.values()
        # clean all elements that equal to zero to prevent divide by zero error
        v = np.array([i for i in v if i != 0])

        v = v / v.sum()
        v = v * np.log(1 / v) / np.log(self._base)

        self.pairs_counts = pairs_counts
        self.entropy = v.sum()
