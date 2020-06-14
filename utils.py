from collections import Counter

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from pointpats import PoissonPointProcess, PoissonClusterPointProcess, Window

np.random.seed(0)


def runif_in_circle(n, radius=1.0, center=(0., 0.), burn=2):
    good = np.zeros((n, 2), float)
    c = 0
    r = radius
    r2 = r * r
    it = 0
    while c < n:
        x = np.random.uniform(-r, r, (burn*n, 1))
        y = np.random.uniform(-r, r, (burn*n, 1))
        ids = np.where(x*x + y*y <= r2)
        candidates = np.hstack((x, y))[ids[0]]
        nc = candidates.shape[0]
        need = n - c
        if nc > need:  # more than we need
            good[c:] = candidates[:need]
        else:  # use them all and keep going
            good[c:c+nc] = candidates
        c += nc
        it += 1
    return good + np.asarray(center)


def plot_points(points, types, ax=None, title=None):
    data = pd.DataFrame(points, columns=['x', 'y'])
    data['types'] = types
    
    p = sns.scatterplot(data=data, x='x', y='y', hue='types', ax=ax)
    p.legend(loc='upper right', ncol=1)
    if ax is None:
        plt.title(title)
    else:
        ax.title.set_text(title)
    return p


def random_data(window, n):
    window = Window(window)
    rpp = PoissonPointProcess(window, 200, 1, conditioning=False, asPP=False)
    return rpp.realizations[0]


def cluster_data(window, n, parents, d, types):
    window = Window(window)
    l, b, r, t = window.bbox
    children = int(n / parents)
    # get parent points
    pxs = np.random.uniform(l, r, (int(n/children), 1))
    pys = np.random.uniform(b, t, (int(n/children), 1))
    cents = np.hstack((pxs, pys))
    
    pnts = {tuple(center): runif_in_circle(children, d, center) for center in cents}
    
    types_counts = Counter(types)
    utypes = types_counts.keys()
    type_mapper = {}
    for k,v in pnts.items():
        points_count = len(v)
        new_arr = []
        for i in utypes:
            if len(new_arr) != points_count:
                c = types_counts[i]
                if c != 0:
                    current_len = points_count - len(new_arr)
                    if c >= current_len:
                        new_arr += [i for _ in range(current_len)]
                        types_counts[i] -= current_len
                    else:
                        new_arr += [i for _ in range(c)]
                        types_counts[i] = 0
            else:
                break
        type_mapper[k] = new_arr
     
    points = []
    ordered_types = []
    
    for k,v in pnts.items():
        points += list(v)
        ordered_types += type_mapper[k]
        
    return points, ordered_types