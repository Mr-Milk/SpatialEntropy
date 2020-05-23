# SpatialEntropy

![Test](https://github.com/Mr-Milk/SpatialEntropy/workflows/Test/badge.svg)

This is a python implementation of spatial entropy, inspired by the R package *spatentropy*. For now, two spatial entropy
methods has been implemented:

- Leibovici’s entropy
- Altieri's entropy



## Usage

Let's generated some fake data first:

```python
import numpy as np

points = 100 * np.random.randn(10000, 2) + 1000
types = np.random.choice(range(30), 10000)
```

Here we have 10,000 points and then we assigned each point with a category from 30 categories.



To calculate the libovici entropy, we need to set up a distance to define the co-occurrences.

```python
from spatialentropy import leibovici_entropy

# here we set the distance d into 5
e = leibovici_entropy(points, types, 5)

e.entropy # to get the entropy value
e.adj_matrix # to get the adjacency matrix
e.pairs_counts # to get the counts for each pair of co-occurrences
```



To calculate the latieri entropy, we need to set up a distance to define the co-occurrences.

```python
from spatialentropy import altieri_entropy

# if the cut is set as a number, it means how many times to cut evenly from [0,max]
# there for it will generate cut + 1 intervals
# if the cut is an array, it lets you define your own intervals
# e = leibovici_entropy(points, types, cut=[0,4,10])
e = leibovici_entropy(points, types, cut=2)

e.entropy # to get the entropy value, e.entropy = e.mutual_info + e.residue
e.mutual_info # the spatial mutual information
e.residue # the spatial residue entropy
e.adj_matrix # to get the adjacency matrix
```
