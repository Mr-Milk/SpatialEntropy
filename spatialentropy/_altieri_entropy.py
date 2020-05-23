from typing import Sequence

import numpy as np
from sklearn.metrics import pairwise_distances

from ._utils import interval_pairs, pairs_counter, type_adj_matrix


class altieri_entropy(object):
    """Calculate teh altieri entropy

    Attributes:
        adj_matrix: array, return the adjacency matrix
        mutual_info: float, the value of the spatial mutual information part of the entropy
        residue: float, the value of the spatial residue entropy
        entropy: float, the value of leibovici entropy, equal to (mutual_info + residue)

    """

    def __init__(self, points, types, cut=1, order=False):
        """

        Args:
            points: array, 2d array
            types: array, the length should correspond to points
            cut: int or array, number means how many cut to make from [0, max], array allow you to make your own cut
            order: bool, if True, (x1, x2) and (x2, x1) is not the same

        """
        self._points = points
        self._types = types
        self._order = order
        self.adj_matrix = pairwise_distances(self._points)

        if isinstance(cut, int):
            self._break = interval_pairs(np.linspace(0, self.adj_matrix.max(), cut + 2))

        elif isinstance(cut, Sequence):
            self._break = interval_pairs(cut)

        else:
            raise ValueError("'cut' must be an int or an array-like object")

        self._wrap()

    def _Z_W(self):

        zw = []
        for (p1, p2) in self._break:
            bool_matx = ((self.adj_matrix > p1) & (self.adj_matrix < p2)).astype(int)
            type_matx, utypes = type_adj_matrix(bool_matx, self._types)
            pairs_counts = pairs_counter(type_matx, utypes, self._order)
            zw.append(pairs_counts)

        return zw

    def _Z(self):

        bool_matx = (self.adj_matrix >= 0).astype(int)
        type_matx, utypes = type_adj_matrix(bool_matx, self._types)
        z = pairs_counter(type_matx, utypes, self._order)

        return z

    def _W(self):

        w = []
        for (p1, p2) in self._break:
            w.append(p2 - p1)

        w = np.asarray(w)

        w = w / w.sum()

        return w

    def _wrap(self):

        zw = np.asarray(self._Z_W())
        z = self._Z()
        w = np.asarray(self._W())

        pz = np.array(list(z.values()))
        pz = pz / pz.sum()

        H_Zwk = []  # H(Z|w_k)
        PI_Zwk = []  # PI(Z|w_k)

        for i in zw:
            v = np.array(list(i.values()))
            v = v / v.sum()
            H = v * np.log(1 / v)
            PI = v * np.log(pz / v)
            H_Zwk.append(H.sum())
            PI_Zwk.append(PI.sum())

        self.residue = (w * np.asarray(H_Zwk)).sum()
        self.mutual_info = (w * np.asarray(PI_Zwk)).sum()
        self.entropy = self.mutual_info + self.residue
