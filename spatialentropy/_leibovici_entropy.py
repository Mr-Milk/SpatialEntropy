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

    def __init__(self, points, types, d, order=False, base=np.e):
        """

        Args:
            points: array, 2d array
            types: array, the length should correspond to points
            d: int or float, cut-off distance
            order: bool, if True, (x1, x2) and (x2, x1) is not the same
            base: int or float, the log base

        """
        if len(points) != len(types):
            raise ValueError("Array of points and types should have same length")

        self._points = points
        self._types = types
        self._d = d
        self._order = order
        self._base = base
        self.adj_matrix = pairwise_distances(self._points)

        self._count()

    def _count(self):
        bool_matx = (self.adj_matrix <= self._d).astype(int)
        type_matx, utypes = type_adj_matrix(bool_matx, self._types)
        pairs_counts = pairs_counter(type_matx, utypes, self._order)

        v = np.array(list(pairs_counts.values()))
        v = v / v.sum()
        v = v * np.log(1 / v) / np.log(self._base)

        self.pairs_counts = pairs_counts
        self.entropy = v.sum()
