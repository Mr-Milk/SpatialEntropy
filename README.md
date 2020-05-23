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



> If you are interested in the definition of spatial entropy, I summarize it myself in the below, but please do refer to the original articles for accurate explanation.

### Leibovici’s entropy

A new variable $Z$ is introduced. $Z$ is defined as co-occurrences across the space.

For example, we have $I$ types of cells. The combination of any two type of cells is $(x_i, x_{i'})$, the number of all combinations is denoted as $R$.

If order is preserved, $R = P_I^2 = I^2$; If the combinations are unordered, $R = C_I^2= (I^2+I)/2$

To a more general situation, consider the combination of $m$ types of cell, $R = P_I^m = I^m$ or $R=C_I^m$

At a user defined distance $d$, only co-occurrences with the distance $d$ will take into consideration.

$$H(Z|d) = \sum_{r=1}^{I^m}{p(z_r|d)}log(\frac{1}{p(z_r|d)})$$

> Leibovici, D. G., Claramunt, C., Le Guyader, D., & Brosset, D. (2014). Local and global spatio-temporal entropy indices based on distance-ratios and co-occurrences distributions. International Journal of Geographical Information Science, 28(5), 1061-1084. [link](https://www.tandfonline.com/doi/full/10.1080/13658816.2013.871284)



### Altieri's entropy

This introduce another new vairable $W$. $w_k$ represents a series of sample window, i.e. $[0,2][2,4][4,10],[10,...]$
while $k=1,...,K$

The purpose of this entropy is to decompose it into **Spatial mutual information** $MI(Z,W)$ and **Spatial residual entropy** $H(Z)_W$. 

$$H(Z)=\sum_{r=1}^Rp(z_r)log(\frac{1}{p(z_r)})=MI(Z,W)+H(Z)_W$$

$$H(Z)_W = \sum_{k=1}^Kp(w_k)H(Z|w_k)$$

$$H(Z|w_k) = \sum_{r=1}^Rp(z_r|w_k)log(\frac{1}{p(z_r|w_k)})$$

$$MI(Z,W)=\sum_{k=1}^Kp(w_k)PI(Z|w_k)$$

$$PI(Z|w_k)=\sum_{r=1}^Rp(z_r|w_k)log(\frac{p(z_r|w_k)}{p(z_r)})$$

> Altieri, L., Cocchi, D., & Roli, G. (2018). A new approach to spatial entropy measures. Environmental and ecological statistics, 25(1), 95-110. [link](https://link.springer.com/article/10.1007/s10651-017-0383-1)


