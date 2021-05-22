from typing import Sequence, Union, List

import numpy as np
from sklearn.metrics import pairwise_distances

from ._utils import interval_pairs, type_adj_matrix, pairs_counter


def altieri(points: Union[List[float], np.ndarray],
            types: Union[List[str], np.ndarray],
            cut: Union[int, list, None] = None,
            order: bool = True,
            base: Union[int, float, None] = None
            ) -> (float, float, float):
    """Altieri entropy

    Args:
        points: array, 2d array
        types: array, the length should correspond to points
        cut: int or array, number means how many cut to make from [0, max], array allow you to make your own cut
        order: bool, if True, (x1, x2) and (x2, x1) is not the same
        base: int or float, the log base, default is e

    Returns:
        float

    """
    if len(points) != len(types):
        raise ValueError("Array of points and types should have same length")

    if base is None:
        base = np.e

    points = np.asarray(points)
    dist_max = np.sqrt(sum([i * i for i in points.T.max(axis=1) - points.T.min(axis=1)]))
    adj_matrix = pairwise_distances(points)

    if isinstance(cut, int):
        break_interval = interval_pairs(np.linspace(0, dist_max, cut + 2))

    elif isinstance(cut, Sequence):
        break_interval = interval_pairs(cut)

    elif cut is None:
        break_interval = interval_pairs(np.linspace(0, dist_max, 3))

    else:
        raise ValueError("'cut' must be an int or an array-like object")

    w, zw = [], []
    for (p1, p2) in break_interval:
        bool_matx = ((adj_matrix > p1) & (adj_matrix <= p2)).astype(int)
        type_matx, utypes = type_adj_matrix(bool_matx, types)
        pairs_counts = pairs_counter(type_matx, utypes, order)
        zw.append(pairs_counts)
        w.append(p2 - p1)

    bool_matx = (adj_matrix > 0).astype(int)
    type_matx, utypes = type_adj_matrix(bool_matx, types)
    z = pairs_counter(type_matx, utypes, order)

    w = np.asarray(w)
    w = w / w.sum()

    zw = np.asarray(zw)

    pz = np.array(list(z.values()))
    pz = pz / pz.sum()

    H_Zwk = []  # H(Z|w_k)
    PI_Zwk = []  # PI(Z|w_k)

    for i in zw:
        vi = i.values()

        v, pz_ = [], []
        for ix, x in enumerate(vi):
            if x != 0:
                v.append(x)
                pz_.append(pz[ix])

        v = np.asarray(v)
        pz_ = np.asarray(pz_)

        v = v / v.sum()
        H = v * np.log(1 / v) / np.log(base)
        PI = v * np.log(v / pz_) / np.log(base)
        H_Zwk.append(H.sum())
        PI_Zwk.append(PI.sum())

    residue = (w * np.asarray(H_Zwk)).sum()
    mutual_info = (w * np.asarray(PI_Zwk)).sum()
    entropy = residue + mutual_info

    return entropy, mutual_info, residue


class altieri_entropy(object):
    """Altieri entropy

    Attributes:
        mutual_info: float, the value of the spatial mutual information part of the entropy
        residue: float, the value of the spatial residue entropy
        entropy: float, the value of leibovici entropy, equal to (mutual_info + residue)

    """

    def __init__(self,
                 points: Union[list, np.ndarray],
                 types: Union[list, np.ndarray],
                 cut: Union[int, list, None] = None,
                 order: bool = True,
                 base: Union[int, float, None] = None
                 ):
        """

        Args:
            points: array, 2d array
            types: array, the length should correspond to points
            cut: int or array, number means how many cut to make from [0, max], array allow you to make your own cut
            order: bool, if True, (x1, x2) and (x2, x1) is not the same
            base: int or float, the log base, default is e

        """
        self.entropy, self.mutual_info, self.residue = altieri(points, types, cut, order, base)
