from collections import OrderedDict
from itertools import combinations_with_replacement, product

import numpy as np


def reduce_matrix_row(matrix, types, storage):
    """reduce matrix size by merging the row that has same type

    Args:
        matrix: array, An N * M matrix
        types: array, length should equal to N
        storage: dict, storage object

    Returns:
        array, row-merged matrix based on types

    """
    for i, arr in enumerate(matrix):
        storage[types[i]].append(arr)

    for k, v in storage.items():
        storage[k] = np.asarray(v).sum(axis=0)

    new_types = []
    new_matx = []
    for k, v in storage.items():
        new_types.append(k)
        new_matx.append(v)

    return new_matx


def type_adj_matrix(matrix, types):
    """return an N * N matrix, N is the number of unique types

    Args:
        matrix: array
        types: array

    Returns:
         tuple, matrix and the unique types

    """
    unitypes = np.unique(types)

    storage = OrderedDict(zip(unitypes, [[] for _ in range(len(unitypes))]))
    new_matrix = reduce_matrix_row(matrix, types, storage)

    storage = OrderedDict(zip(unitypes, [[] for _ in range(len(unitypes))]))
    type_matrix = reduce_matrix_row(np.asarray(new_matrix).T, types, storage)

    return np.array(type_matrix), unitypes


def pairs_counter(matrix, types, order=False):
    """count how many pairs of types in the matrix

    Args:
    matrix: array
    types: array
    order: bool, if True, (x1, x2) and (x2, x1) is not the same

    Returns:
        dict, the count of each pairs

    """
    it = np.nditer(matrix, flags=["multi_index"])

    if order:
        combs = [i for i in product(types, repeat=2)]
        storage = OrderedDict(zip(combs, [0 for _ in range(len(combs))]))

        for x in it:
            (i1, i2) = it.multi_index
            storage[(types[i1], types[i2])] += x
    else:
        combs = [i for i in combinations_with_replacement(types, 2)]
        storage = OrderedDict(zip(combs, [0 for _ in range(len(combs))]))

        for x in it:
            (i1, i2) = it.multi_index
            if i1 <= i2:
                storage[(types[i1], types[i2])] += x
            else:
                storage[(types[i2], types[i1])] += x

    return storage


def interval_pairs(arr):
    new_arr = []
    for i, x in enumerate(arr):
        if i < len(arr) - 1:
            new_arr.append((x, arr[i + 1]))

    return new_arr
