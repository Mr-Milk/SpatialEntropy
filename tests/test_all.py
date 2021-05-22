import numpy as np

from spatialentropy import leibovici_entropy, altieri_entropy

# some fake data
points = 100 * np.random.randn(1000, 2) + 1000
types = np.random.choice(range(30), 1000)


def test_leibovici():
    e1 = leibovici_entropy(points, types, 5, order=False)
    e2 = leibovici_entropy(points, types, 5, order=True)


def test_altieri():
    e1 = altieri_entropy(points, types, cut=1, order=False)
    e2 = altieri_entropy(points, types, cut=[0, 4, 10], order=False)
    e3 = altieri_entropy(points, types, cut=1, order=True)

    assert e1.mutual_info + e1.residue == e1.entropy
    assert e2.mutual_info + e2.residue == e2.entropy
    assert e3.mutual_info + e3.residue == e3.entropy
