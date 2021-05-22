# SpatialEntropy

![Test](https://github.com/Mr-Milk/SpatialEntropy/workflows/Test/badge.svg) [![PyPI version](https://badge.fury.io/py/spatialentropy.svg)](https://badge.fury.io/py/spatialentropy)

This is a python implementation of spatial entropy, inspired by the R package *spatentropy*. For now, two spatial entropy methods have been implemented:

- Leibovici’s entropy
- Altieri's entropy


## Compare with shannon entropy

![Compare](https://github.com/Mr-Milk/SpatialEntropy/blob/master/img/example.png?raw=true)


## Installation

It's available on PyPI

```shell
pip install spatialentropy
```


## Usage

[Check out an example](https://nbviewer.jupyter.org/gist/Mr-Milk/af67ac0957201227723ed76f526487ea)

Let's generate some fake data first:

```python
import numpy as np

points = 100 * np.random.randn(10000, 2) + 1000
types = np.random.choice(range(30), 10000)
```

Here we have 10,000 points and then we assigned each point with a category from 30 categories.

### Quick start

```python
from spatialentropy import leibovici_entropy

e = leibovici_entropy(points, types)
e.entropy
```

### Leibovici entropy

To calculate the leibovici entropy, we need to set up a distance or an interval to define the co-occurrences.

```python
from spatialentropy import leibovici_entropy

# set the distance cut-off to 5
e = leibovici_entropy(points, types, d=5)
# if you want to change the base of log
e = leibovici_entropy(points, types, base=2)

e.entropy # to get the entropy value
e.adj_matrix # to get the adjacency matrix
e.pairs_counts # to get the counts for each pair of co-occurrences
```

### Altieri entropy

To calculate the altieri entropy, we need to set up intervals to define the co-occurrences.

```python
from spatialentropy import altieri_entropy

# set cut=2, it means we will create 3 intervals evenly from [0,max]
e = altieri_entropy(points, types, cut=2)

# or you want to define your own intervals
e = altieri_entropy(points, types, cut=[0,4,10])

e.entropy # to get the entropy value, e.entropy = e.mutual_info + e.residue
e.mutual_info # the spatial mutual information
e.residue # the spatial residue entropy
e.adj_matrix # to get the adjacency matrix
```
