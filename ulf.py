"""Base for Ulf's personal useful functions."""

from __future__ import print_function

import sys, os
import subprocess
from functools import wraps
from IPython.display import HTML, display
from calendar import monthrange
import unicodedata

import numpy as np
from collections import defaultdict, Counter
from scipy.interpolate import interp1d
from scipy import stats, misc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial import ConvexHull
from difflib import SequenceMatcher

import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import gmplot
import pytz, shapefile

import networkx as nx
import community
import infomap

import math
from math import factorial
from random import shuffle
import re


def plt_log_hist(v, **kwargs):
    """Make logplot of histogram.

    Input
    -----
        v : array of values
        bins : int or array (use int)
    """
    # Handle input paramaters
    bins = kwargs.get('bins', 10)
    if 'bins' in kwargs:
        del kwargs['bins']
    
    # Construct log-bins
    if min(v) == 0:
        v = (np.array(v) + 0.01)
    logbins = np.logspace(np.log10(min(v)), np.log10(max(v)), bins)

    # Plot
    plt.hist(v, bins=logbins, **kwargs)
    plt.xscale("log")


def plt_cumulative_hist(v, bins=10):
    """Make cumulative histogram plot."""
    values, base = np.histogram(v, bins=bins)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c='blue')


def pareto_distribution(v, p=0.8):
    """Get the number of entries in v which accounts for p of its sum.

    v has to be sorted in descending order.
    """
    thr = np.sum(v)*p
    cumsum = 0
    for i, _v in enumerate(v, 1):
        cumsum += _v
        if cumsum >= thr:
            return i * 1.0 / len(v)


def approximate_entropy(U, m, r):
    """Compute approximate entropy of time series.

    Parameters
    ----------
    U : time-series list

        Example
        -------
        [85, 80, 89, 85, 80, 89]

    m : int
        Comparation period

    r : int/float
        Irregularity sensitivity. High sensitivity reduces entropy.

    Output
    ------
    approximate entropy : float
    
    Example
    -------
    >>> U = np.array([85, 80, 89] * 17)
    >>> approximate_entropy(U, 2, 3)
    1.0996541105257052e-05
    """
    import numpy as np

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m) - _phi(m + 1))


def smooth(y, box_pts):
    """Sliding box smoothening of noisy data.

    Parameters
    ----------
    y : list
        Noisy y-variable. Must be sorted wrt. time.

    box_pts : int
        Convolution box size. The greater the box the smoother the plot.

    Output
    ------
    y_smooth : list
        Smooth points to replace y. Same dimensions as y.

    Example
    -------
    >>> x = np.linspace(0, 2 * np.pi, 100)
    >>> y = np.sin(x) + np.random.random(100)
    >>>
    >>> plt.plot(x, y,'o')
    >>> plt.plot(x, smooth(y, 18))
    >>> plt.plot(x, smooth(y, 9))
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def smooth_median(y, window_size):
    """Sliding box smoothening of noisy data.

    Parameters
    ----------
    y : list
        Noisy y-variable. Must be sorted wrt. time.

    box_pts : int
        Convolution box size. The greater the box the smoother the plot.

    Output
    ------
    y_smooth : list
        Smooth points to replace y. Same dimensions as y.

    Example
    -------
    >>> x = np.linspace(0, 2 * np.pi, 100)
    >>> y = np.sin(x) + np.random.random(100)
    >>>
    >>> plt.plot(x, y,'o')
    >>> plt.plot(x, smooth_median(y, 18))
    >>> plt.plot(x, smooth_median(y, 9))
    """
    if window_size % 2 == 1:
        window_size -= 1
    y_smooth = []
    for j in range(len(y)):
        l = max(j - window_size/2, 0)
        u = min(j + window_size/2+1, len(y))
        y_smooth.append(np.median(y[l:u]))
    return y_smooth


def transpose_3d():
    """Transpose of 3d-array is 2d-array-wise transpose of each j'th slice.

    This is an example to show that the transpose of a 3d array is just the
    2d-array-wise transpose of every j'th slice.
    """
    tmp = np.random.random((10, 10, 10))

    a = tmp.T
    b = np.empty(tmp.shape)
    for j in range(tmp.shape[1]):
        b[:, j, :] = tmp[:, j, :].T

    print(np.all(a == b))


def to_frac(fval, tol=1e-3):
    """Take float with trailing decimals and reduce to closest exact form."""
    def simplify_fraction(numer, denom):
        def _gcd(a, b):
            """Calculate the Greatest Common Divisor of a and b.

            Unless b==0, the result will have the same sign as b (so that when
            b is divided by it, the result comes out positive).
            """
            while b:
                a, b = b, a % b
            return a

        if denom == 0:
            return "Division by 0 - result undefined"

        # Remove greatest common divisor:
        common_divisor = _gcd(numer, denom)
        (reduced_num, reduced_den) = (numer / common_divisor, denom / common_divisor)
        # Note that reduced_den > 0 as documented in the gcd function.

        if reduced_den == 1:
            return "%d / %d is simplified to %d" % (numer, denom, reduced_num)
        elif common_divisor == 1:
            return "%d / %d is most simple fraction" % (numer, denom)
        else:
            return "%d / %d is simplified to %d/%d" % (numer, denom, reduced_num, reduced_den)

    diff = 1
    div = 0.0
    while diff > tol:
        div += 1
        diff = (fval * div) % 1.0

    return simplify_fraction(int(fval*div), int(div))


def jsdiv(P, Q):
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """

    def _kldiv(A, B):
        return np.sum([v for v in A * np.log2(A * 1.0 / B) if not np.isnan(v)])

    P = np.array(P)
    Q = np.array(Q)

    M = 0.5 * (P + Q)

    return 0.5 * (_kldiv(P, M) + _kldiv(Q, M))

def jsdiv_sparse(P, Q, apply_norm=False):
    """Compute the Jensen-Shannon divergence between two normalized counters.

    Input
    -----
    P, Q : Counter
        Values in each must sum to 1
    """
    def _kldiv(A, B):
        return np.sum([
            A[k] * np.log2(A[k] * 1.0 / B[k])
            for k in set(A.keys()) | set(B.keys())
            if not 0 in [A[k], B[k]] and not np.nan in [A[k], B[k]]
        ])

    if apply_norm:
        P = normalize_counter(P)
        Q = normalize_counter(Q)
        
    M = normalize_counter(P + Q)

    return 0.5 * (_kldiv(P, M) + _kldiv(Q, M))

def cosine_sim_counters(a, b):
    """Get cosine similarity between two collection of values.

    Input
    -----
        a, b : Counter
            Represents an unnormalized sparse probability vector

    Output
    ------
        out : float
            Cosine similarity between vector representations of either collection
    """
    union_ab = sorted((a | b).keys())
    veca = np.array([a[element] if element in a else 0 for element in union_ab])
    vecb = np.array([b[element] if element in b else 0 for element in union_ab])
    return np.dot(veca, vecb) / (np.linalg.norm(veca) * np.linalg.norm(vecb))


def jssim_counters(a, b, weighted=False):
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Input
    -----
        a, b : Counter
            Represents an unnormalized sparse probability vector
        weighted : bool/str
            How to weight the KL divergences of each distribution to the mean. 'inversely',
            punishes deviation in vector length (or "neighborhood size").
    
    Output
    ------
        out : float
            Jensen Shannon similarity between vector representations of either collection
    """

    def _kldiv(R, S):
        return np.nansum([v for v in R * np.log2(R * 1.0 / S)])

    union_ab = sorted((a | b).keys())
    P = np.array([a[state] * 1. / sum(a.values()) if state in a else 0 for state in union_ab])
    Q = np.array([b[state] * 1. / sum(b.values()) if state in b else 0 for state in union_ab])

    M = 0.5 * (P + Q)

    if weighted is True:
        norm = sum(a.values()) + sum(b.values())
        wP = sum(a.values()) * 1. / norm
        wQ = sum(b.values()) * 1. / norm
    elif weighted == "inversely":
        norm = sum(a.values()) + sum(b.values())
        wQ = sum(a.values()) * 1. / norm
        wP = sum(b.values()) * 1. / norm
    else:
        wP = 0.5
        wQ = 0.5
    
    return 1 - (wP * _kldiv(P, M) + wQ * _kldiv(Q, M))

def randomize_by_edge_swaps(G, num_iterations):
    """Randomizes the graph by swapping edges or interchanging data when a
    u--x or v--y edge already exists.

    A double-edge swap removes two randomly chosen edges u-v and x-y
    and creates the new edges u-x and v-y::

    u--v            u  v
           becomes  |  |
    x--y            x  y
    """
    G_copy = G.copy()
    edge_list = list(G_copy.edges())
    num_edges = len(edge_list)
    total_iterations = num_edges * num_iterations

    for _ in range(total_iterations):
        i, j = np.random.choice(num_edges, 2, replace=False)
        u, v = edge_list[i]
        x, y = edge_list[j]

        if len(set([u, v, x, y])) < 4:
            continue

        # Save edge data
        i_data = G_copy[u][v]
        j_data = G_copy[x][y]

        if G_copy.has_edge(u, x) or G_copy.has_edge(v, y):
            # Interchange edge data
            G_copy.remove_edges_from(((u, v), (x, y)))
            G_copy.add_edges_from(((u, v, j_data), (x, y, i_data)))
        else:
            # Regular swap
            G_copy.remove_edges_from(((u, v), (x, y)))
            G_copy.add_edges_from(((u, x, i_data), (v, y, j_data)))

            edge_list[i] = (u, x)
            edge_list[j] = (v, y)

    assert len(G_copy.edges()) == num_edges
    return G_copy

def colormixer(colors, weights=None):
    """Take array of colors in hex format and return the average color.
    
    Input
    -----
        colors : array of hex values
    
    Example
    -------
        >>> colormixer(['#3E1F51', '#FEE824', '#1F908B'])
        '#4af134'
    """
    def _to_hex(v):
        v_hex = hex(v)[2:]
        if len(v_hex) == 1:
            v_hex = "0" + v_hex
        return v_hex

    # Compute mean intensities for red, green and blue
    if weights is None:
        r = int(np.mean([int(c[1:3], 16) for c in colors]))
        g = int(np.mean([int(c[3:5], 16) for c in colors]))
        b = int(np.mean([int(c[5:7], 16) for c in colors]))
    else:
        r = int(sum([int(c[1:3], 16) * w for c, w in zip(colors, weights)]) / sum(weights))
        g = int(sum([int(c[3:5], 16) * w for c, w in zip(colors, weights)]) / sum(weights))
        b = int(sum([int(c[5:7], 16) * w for c, w in zip(colors, weights)]) / sum(weights))
    
    # Take mean of each and convert back to hex
    return '#' + _to_hex(r) + _to_hex(g) + _to_hex(b)


def custom_cmap(colors):
    return mpl.colors.LinearSegmentedColormap.from_list('', colors)

def display_cmap_color_range(cmap_style='rainbow'):
    """Display the range of colors offered by a cmap.
    """
    cmap = plt.get_cmap(cmap_style)
    for c in range(256):
        plt.scatter([c], [0], s=500, c=cmap(c), lw=0)
    plt.show()

class cmap_in_range:
    """Create map to range of colors inside given domain.

    Example
    -------
    >>> cmap = cmap_in_range([0, 100])
    >>> cmap(10)
    (0.30392156862745101, 0.30315267411304353, 0.98816547208125938, 1.0)
    """
    def __init__(self, cmap_domain, cmap_range=[0, 1], cmap_style='rainbow'):
        self.cmap_domain = cmap_domain
        self.cmap_range = cmap_range
        self.m = interp1d(cmap_domain, cmap_range)
        self.cmap = plt.get_cmap(cmap_style)
        
    def __call__(self, value):
        if not self.cmap_domain[0] <= value <= self.cmap_domain[1]:
            raise Exception("Value must be inside cmap_domain.")
        return self.cmap(self.m(value))

class cmap_in_categories:
    """Create map to range of colors inside given categories.

    Example
    -------
    >>> cmap = cmap_in_categories(['cats', 'dogs', 'squirrels'])
    >>> cmap('squirrels')
    (0.30392156862745101, 0.30315267411304353, 0.98816547208125938, 1.0)
    """
    def __init__(self, cmap_categories, cmap_range=[0, 1], cmap_style='rainbow'):
        self.cmap_domain_map = dict(list(zip(cmap_categories, list(range(len(cmap_categories))))))
        self.cmap_domain = [min(self.cmap_domain_map.values()), max(self.cmap_domain_map.values())]
        self.cmap_categories = cmap_categories
        self.cmap_range = cmap_range
        self.m = interp1d(self.cmap_domain, self.cmap_range)
        self.cmap = plt.get_cmap(cmap_style)
        
    def __call__(self, category):
        if not category in self.cmap_categories:
            raise Exception("Category must be inside cmap_categories.")
        return self.cmap(self.m(self.cmap_domain_map[category]))

def default_to_regular(d):
    """Recursively convert nested defaultdicts to nested dicts.

    Source: http://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o
    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def n_choose_k(n, k):
    return misc.comb(n, k)


def get_x_y_steps(x, y, where="post"):
    """Plot step function from x and y coordinates."""
    if where == "post":
        x_step = [x[0]] + [_x for tup in zip(x, x)[1:] for _x in tup]
        y_step = [_y for tup in zip(y, y)[:-1] for _y in tup] + [y[-1]]
    elif where == "pre":
        x_step = [_x for tup in zip(x, x)[:-1] for _x in tup] + [x[-1]]
        y_step = [y[0]] + [_y for tup in zip(y, y)[1:] for _y in tup]
    return x_step, y_step

def chunks(l, n):
    """Yield successive n-sized chunks from a list l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def point_inside_polygon(xxx_todo_changeme,poly):
    """Determine if points x and y are inside poly.
    Source: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    (x,y) = xxx_todo_changeme
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def shuffle_list(l):
    """Non-inline list shuffle.

    Input
    -----
    l : list

    Output
    ------
    out : list
    """
    l_out = list(l)[:]
    shuffle(l_out)
    return l_out

def draw(G, partition=False, colormap='rainbow', labels=None):
    """Draw graph G in my standard style.

    Uses graphviz. Do `conda install graphviz` if not installed.

    Input
    -----
    G : networkx graph
    partition : bool
    colormap : matplotlib colormap
    labels : dict (Node labels in a dictionary keyed by node of text labels)
    """

    def shuffle_list(l):
        l_out = list(l)[:]
        shuffle(l_out)
        return l_out
    
    def _get_cols(partition):
        return dict(
            list(zip(
                shuffle_list(set(partition.values())),
                np.linspace(0, 256, len(set(partition.values()))).astype(int)
            ))
        )

    cmap = plt.get_cmap(colormap)
    if partition == True:
        partition = community.best_partition(G)
        cols = _get_cols(partition)
        colors = [cmap(cols[partition[n]]) for n in G.nodes()]
    elif type(partition) is dict and len(partition) >= len(G.nodes()):
        cols = _get_cols(partition)
        colors = [cmap(cols[partition[n]]) for n in G.nodes()]
    elif type(partition) in [list, tuple] and len(partition) == len(G.nodes()):
        colors = list(partition)
    else:
        try:
            colors = [n[1]['group'] for n in G.nodes(data=True)]
        except KeyError:
            # nodes do not have node_color attribute
            colors = "grey"
    
    pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    nx.draw_networkx_edges(G, pos=pos, width=2, alpha=.3, zorder=-10)
    nx.draw_networkx_nodes(G, pos=pos, node_size=120, alpha=1, linewidths=0, node_color=colors)
    
    if labels is not None:
        nx.draw_networkx_labels(G, pos=dict((k, (v[0]+15, v[1])) for k, v in list(pos.items())), labels=labels, font_size=16)

    #nx.draw_networkx_labels(G, pos=pos, font_color="red")
    plt.axis("off")

def unwrap(l, depth=1):
    """Unwrap a list of lists to a single list.

    Input
    -----
    l : list of lists
    depth : number of unwrap operations to perform (int)

    Output
    ------
    out : list
    """
    def _unwrap_one(arr):
        error_count = 0
        a0 = []
        for a1 in arr:
            try:
                for v in a1:
                    a0.append(v)
            except TypeError:
                a0.append(a1)
                error_count += 1
        if len(a0) == error_count:
            print("Max depth reached. Decrement depth to increase speed.")
        return a0
    if depth <= 0:
        return l
    if depth == 1:
        return _unwrap_one(l)
    return unwrap(_unwrap_one(l), depth=depth-1)

def hinton(matrix, max_weight=None, ax=None, facecolor='#ecf0f1', color_pos='#3498db', color_neg='#d35400'):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))
    
    ax.patch.set_facecolor(facecolor)
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = color_pos if w > 0 else color_neg
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

def jaccard_mutual_information(p1, p2):
    """Intersection weighted average over jaccard similarities between communities in two partitions.

    This is an alternative to NMI with a slightly simpler interpretation

    Input
    -----
    p1 : dict or list
        A partition formatted as {node: community, ...}
    p2 : dict or list

    Output
    ------
    out : float
        Value between 0 and 1
    """
    if type(p1) in [list, np.ndarray]:
        p1 = dict(list(zip(list(range(len(p1))), p1)))
    if type(p2) in [list, np.ndarray]:
        p2 = dict(list(zip(list(range(len(p2))), p2)))
    if set(p1.keys()) != set(p2.keys()):
        raise Exception("p1 and p2 does not have the same nodes")
    N = len(p1)
    
    # Invert partition datastructures to # {cluster: [node, node, ...], cluster: [node, node, ...]}
    p1_inv = defaultdict(list)
    p2_inv = defaultdict(list)
    for n, c in list(p1.items()):
        p1_inv[c].append(n)
    for n, c in list(p2.items()):
        p2_inv[c].append(n)

    # Compute average weighted jaccard similarity
    J = 0
    for ci, ni in list(p1_inv.items()):
        for cj, nj in list(p2_inv.items()):
            n_ij = len(set(ni) & set(nj))
            A_inter_B = len(set(ni) & set(nj))
            A_union_B = len(set(ni) | set(nj))
            J += (n_ij * 1.0 / N) * (A_inter_B * 1.0 / A_union_B)
            
    return J

def jaccard_sim(set1, set2):
    """Compute Jaccard similarity between two sets.

    Input
    -----
        set1 : list/set
        set2 : list/set

    Return
    ------
        out : float
    """
    set1, set2 = set(set1), set(set2)
    return len(set1 & set2) * 1.0 / len(set1 | set2)


def graph_list_to_pajek(G_list):
    """Convert list of graphs to multilayer pajek string
    
    Input
    -----
    G_list : list
        Networkx graphs
    
    Output
    ------
    out : str
        Pajek filestring in *Intra format
    """
    def _write_pajek(A, node_labels=None, index_from=0):
        """Return multiplex representation of multiplex network adjacency matrix A

        Providing an adjacency tensor where A[:, :, k] is adjacency matrix of temporal
        layer k, return a pajek format representation of the temporal network which weights interlayer
        edges by state node neighborhood similarity. 

        Parameters
        ----------
        A : numpy.3darray
            3d tensor where each A[:, :, k] is a layer adjacency matrix
        max_trans_prob : float/str
            Cap on interlayer edge weights. 'square' for square penalty.
        power_penalty : int/float
            Power to jaccard similarity betw. state nodes to penalize low similarity
        index_from : int
            From which number to index nodes and layers in pajek format from
        style : bool
            Either 'zigzag', 'vertical', or 'simple'. 'vertical' will give working results but is
            essentially wrong use of Infomap, 'simple' should be possible to use in Infomap but is not
            at this point, so 'zigzag' is preferred because it is an explicit representation of the way
            the network should be represented internally in Infomap.

        Returns
        -------
        out_file : string
            A network string in multiplex format
        intid_to_origid : dict
            Key-value pairs of node integer id and original id
        origid_to_intid : dict
            Reverse of intid_to_origid
        """

        def _write_outfile(A):
            """Write nodes and intra/inter-edges from A and J to string."""
            def __remove_symmetry_A(A):
                A_triu = defaultdict(int)
                for (i, j, k), w in list(A.items()):
                    if j > i:
                        A_triu[(i, j, k)] = w
                return A_triu
            def __write_nodes(outfile):
                outfile += "*Vertices %d" % Nn
                for nid, label in enumerate(nodes):
                    outfile += '\n%d "%s" 1.0' % (nid + index_from, str(label))
                return outfile
            def __write_intra_edges(outfile):
                outfile += "\n*Intra\n# layer node node [weight]"
                for (i, j, k), w in list(__remove_symmetry_A(A).items()):
                    outfile += '\n%d %d %d %f' % (
                        k + index_from,  # layer
                        nodemap[i] + index_from,  # node
                        nodemap[j] + index_from,  # node
                        w                # weight
                    )
                return outfile

            outfile = ""
            outfile = __write_nodes(outfile)
            outfile = __write_intra_edges(outfile)

            return outfile

        nodes = sorted(set([n for i, j, _ in list(A.keys()) for n in [i, j]]))
        Nn = len(nodes)
        Nl = len(set([k for i, j, k in list(A.keys())]))

        nodemap = dict(list(zip(nodes, list(range(Nn)))))

        return _write_outfile(A)

    def _create_adjacency_matrix(layer_edges):
        """Return 3d adjacency matrix of the temporal network.
        
        Input
        -----
        layer_edges : dict
        
        Output
        ------
        A : dict
        """
        A = defaultdict(int)
        for l, edges in list(layer_edges.items()):
            for edge in edges:
                    A[(edge[0], edge[1], l)] += 1
                    A[(edge[1], edge[0], l)] += 1    
        return A
    
    return _write_pajek(
        _create_adjacency_matrix(
            dict(list(zip(list(range(len(G_list))), [G.edges() for G in G_list])))
        )
    )

def invert_partition(partition):
    """Invert a dictionary representation of a graph partition.

    Inverts a dictionary representation of a graph partition from nodes -> communities
    to communities -> lists of nodes, or the other way around.
    """
    if partition == {}:
        return {}
    elif type(list(partition.items())[0][1]) is list:
        partition_inv = dict()
        for c, nodes in list(partition.items()):
            for n in nodes:
                partition_inv[n] = c
    else:
        partition_inv = defaultdict(list)
        for n, c in list(partition.items()):
            partition_inv[c].append(n)
    return default_to_regular(partition_inv)

def multigraph_to_weighted_graph(M):
    """Convert a nx.MultiGraph into a weighted nx.Graph."""
    G = nx.Graph()
    for u,v,data in M.edges_iter(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G

def shannon_entropy(counts):
    """Compute shannon entropy of counts"""
    freq = np.array(counts) * 1.0 / np.sum(counts)
    return -np.sum([f * np.log2(f) for f in freq if f != 0])

def pca_transform(X, n_components=None):
    """Simple PCA transform of data matrix X"""
    return PCA(n_components=n_components).fit_transform(X)

def tsne_transform(X, n_components=2, random_state=None):
    """Simple T-SNE transform of data matrix X"""
    return TSNE(n_components=n_components, random_state=random_state).fit_transform(X)

def size_of_variables(glob):
    """Return list of variable size consumption in megabytes.
    
    Input
    -----
    glob : `globals()` dictionary
    
    Return
    ------
    out : list of tuples, reverse sorted by values at index 1.
    """
    return sorted(
        [
            (k, sys.getsizeof(glob[k]) / 1e6)
            for k in list(glob.keys())
        ],
        key=lambda k_v: k_v[1],
        reverse=True
    )

def domain_range(domain, _range=[0, 1], return_transform=False):
    """Create domain-range object for mapping from one scale to another.

    Input
    -----
    domain : list (min, max of input domain)
    _range : list (min, max of output range)

    Return
    ------
    out : scipy.interpolate.interpolate.interp1d object

    Example
    -------
    >>> m = ulf.domain_range([5, 10], [0, 1])
    >>> m(6)
    0.2
    """

    if not return_transform:
        return interp1d([min(domain), max(domain)], [min(_range), max(_range)], bounds_error=False)
    else:
        m = interp1d([min(domain), max(domain)], [min(_range), max(_range)])
        return [float(m(v)) for v in domain]  # Take float, else returns weird numpy.ndarray element

class domain_range:
    def __init__(self, domain, _range=[0, 1]):
        self.transform = interp1d([min(domain), max(domain)], [min(_range), max(_range)], bounds_error=False)
        self.inverse_transform = interp1d([min(_range), max(_range)], [min(domain), max(domain)], bounds_error=False)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = savitzky_golay(y, window_size=31, order=4)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, y, label='Noisy signal')
    >>> plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> plt.legend()
    >>> plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def window_kernel_smoothener(x, y, kernel='normal', sigma=None, window=2., ticks=100, domain=None, use_x=False):
    """Smoothen a dependent variable with respect to a sliding window on the independent variable.
    
    Input
    -----
    x : array
        Independent variable
    y : array
        Dependent variable
    kernel : string/function
        Kernel used for smoothening. Pass PDF as function or string (only 'normal' supported)
    sigma : float
        If 'normal' is used this is the SD of the PDF. Default is 1/20 of the range of x.
    window : float
        If 'normal' is used this is the number of SDs that the window slides over in each step
    ticks : int
        Number of grid ticks in which to evaluate y
    domain : list of size 2
        The domain in which the kernels are evaluated
    use_x : bool
        Whether to use x array values as kernels (True) or create a new evenly spaced array with `ticks` values.
        
    Output
    ------
    x_smooth : np.array
    y_smooth : np.array
    """
    def _normal_PDF(x, mu=0, sigma=1):
        return 1.0 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2.0 * sigma**2))

    if len(x) != len(y):
        raise Exception("x must equal y")
        
    if sigma is None:
        sigma = (max(x) - min(x)) / 20.

    if kernel == "normal":
        kernel = _normal_PDF
    elif type(kernel) is not str:
        raise Exception("`kernel` must be 'normal' or PDF function which takes arguments `x`, `mu`, `sigma`")

    if domain is None:
        domain = [min(x), max(x)]
    elif type(domain) != list or len(domain) != 2:
        raise Exception("`domain` must be None, or array of size 2")
    
    x, y = np.array(x), np.array(y)
    
    window = window * sigma / 2

    x_smooth = x if use_x else np.linspace(domain[0], domain[-1], ticks)
    y_smooth = []
    for mu in x_smooth:
        mask = (x > mu - window) & (x < mu + window)
        x_ = x[mask]
        y_ = y[mask]
        w = kernel(x_, mu=mu, sigma=sigma)
        y_smooth.append(sum(w * y_ / np.sum(w)))
    
    return np.array(x_smooth), np.array(y_smooth)

def bootstrap(X):
    """Return `X.shape[0]` random 0th axis sample from 2d array.
    
    Input
    -----
    X : np.array (2d)
    
    Output
    ------
    out : np.array (2d)
    """
    return X[np.random.choice(list(range(X.shape[0])), size=X.shape[0]), :]

def mean_confidence_interval(data, confidence=0.95):
    """Compute confidence interval of series.

    Input
    -----
        data : list
        confidence : float. 0 < confidence < 1.

    Output
    ------
        m : float. mean
        m - h : float. lower bound of confidence interval
        m + h : float. upper bound of confidence interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1 + confidence) /2., n - 1)
    return m, m - h, m + h

def plot_confidence(x, Y, confidence=0.95, shade_color="#3498db", line_color="#34495e"):
    """Given x variable and y distribution, plot mean with confidence interval.
    
    Input
    -----
        x : list
        Y : 2d array
        confidence : float. 0 < confidence < 1.
    """
    y, lb, ub = zip(*[mean_confidence_interval(Y[:, j], confidence) for j in range(Y.shape[1])])
    plt.fill_between(
        range(Y.shape[1]),
        lb, ub,
        color=shade_color
    )
    plt.plot(range(Y.shape[1]), y, lw=2, color=line_color)

def plot_binned_confidence(x, y, bins=10, smooth=0, order=2, color="b", label=""):
    """Split x and y coordinates into bins and plot the series with confidence intervals in each bin.

    Input
    -----
        x, y : list
        bins : int
        smooth : int. smoothen y vals with savitzky_golay filter
        order : int. order of smoothening. only applies if smooth > 0
        color : str
        label : str
    """

    if len(x) != len(y):
        raise Exception("length of x and y does not match")

    # Sort x and y, and remove nans if there are any
    x, y = list(zip(*sorted([(i, j) for i, j in zip(x, y) if not np.isnan(i)], key=lambda i_j: i_j[0])))
    x, y = np.array(x), np.array(y)

    # Get domain
    domain = domain_range(list(range(bins+1)), [min(x), max(x)], return_transform=True)

    # Build arrays
    x_vals, y_vals, y_l_vals, y_u_vals = [], [], [], []
    for b in range(bins):
        x_b = x[(domain[b] <= x) & (x < domain[b+1])]
        y_b = y[(domain[b] <= x) & (x < domain[b+1])]

        x_b_p = np.mean([min(x_b), max(x_b)]) if x_b != [] else np.nan
        y_b_p, y_b_lc, y_b_uc = mean_confidence_interval(y_b)

        x_vals.append(x_b_p)
        y_vals.append(y_b_p)
        y_l_vals.append(y_b_lc)
        y_u_vals.append(y_b_uc)
    
    if smooth > 0:
        y_l_vals, y_u_vals = savitzky_golay(y_l_vals, smooth, order), savitzky_golay(y_u_vals, smooth, order)
        y_vals = savitzky_golay(y_vals, smooth, order)
        
    plt.fill_between(x_vals, y_l_vals, y_u_vals, alpha=0.5, color=color)
    plt.plot(x_vals, y_vals, color=color, label=label)

def CCDF(X, steps=100, log_steps=False, normed=False):
    """Calculate the survival distribution (CCDF) simply by counting the occurence. 

    Input:
    ------
    X : 1d numpy array of datapoints 
    steps : number of steps to evaluate the function in (int)
    log_steps : whether to make step evaluation log spaces or not (bool)

    Return: 
    -------
    args_CCDF : sorted version of X
    vals_CCDF : for each value x of X, counts normalized value of how many values are >= x
    """
    X = np.array(X)
    
    # Adjust parameters
    if normed:
        norm = float(np.size(X))
    else:
        norm = 1
    
    if log_steps:
        if X.min() <= 0:
            print("Can't use log steps for negative numbers. Translating to postive with 0.1 as minumum.")
            X += X.min() + 0.1
        args_CCDF = np.logspace(X.min(), X.max(), steps)
    else:
        args_CCDF = np.linspace(X.min(), X.max(), steps)

    # Calculate CCDF
    vals_CCDF = np.array([np.size(X[X >= i]) / norm for i in args_CCDF])
    plt.plot(args_CCDF, vals_CCDF)
    return args_CCDF, vals_CCDF

def keeptrying(error, verbose=False):
    """Decorator to keep trying to run some function given an error it may throw.

    Given an error that a certain function may throw occationally, keep trying to
    run the function.
    
    Input
    -----
    error : The type of error that may occur. E.g ValueError.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Test something with the args, kwargs and the wrapped function f
            while True:
                try:
                    return f(*args, **kwargs)
                except error as e:
                    if verbose:
                        print(e)
        return wrapper
    return decorator


import subprocess, re
from collections import defaultdict, Counter

def infomap_communities(G):
    """Partition network with the Infomap algorithm.

    Input
    -----
        G : nx.Graph

    Output
    ------
        out : dict (node-community pairs)
    """
    name_map = {}
    name_map_inverted = {}
    for n in G.nodes():
        id_ = hash(n) % 100000
        name_map_inverted[id_] = n
        name_map[n] = id_
        
    infomapSimple = infomap.Infomap("--two-level")
    network = infomapSimple.network()
    
    for n1, n2, data in G.edges(data=True):
        network.addLink(name_map[n1], name_map[n2], data['weight'] if 'weight' in data else 1)

    infomapSimple.run()

    return dict(
        (name_map_inverted[node.physicalId], node.moduleIndex())
        for node in infomapSimple.iterTree()
        if node.isLeaf()
    )

def Infomap(pajek_string, *args, **kwargs):
    """Function that pipes commands to subprocess and runs native Infomap implementation.
    
    Requires a root /tmp directory which can be written to, as well as an Infomap executable
    in the /usr/local/bin which must be sourced so `Infomap` is a global variable in the shell.
    
    Parameters
    ----------
    pajek_string : str
        Pajek representation of the network (str)
    *args : dict
        Infomap execution options. (http://www.mapequation.org/code.html#Options)
        
    Returns
    -------
    communities : dict of Counters (how many times each node participates in each community)
    layer_communities : dict of dicts (layer -> communities -> members)
    layer_node_flow : dict
    node_flow : dict
    community_flow : dict

    Example
    -------
    >>> network_pajek = ulf.write_pajek(multilayer_edge_list)
    >>>
    >>> communities, layer_communities, node_flow, community_flow = ulf.Infomap(
    >>>     network_pajek,
    >>>     '-i', 'multilayer',
    >>>     '--multilayer-js-relax-rate', '0.25',
    >>>     '--overlapping',
    >>>     '--expanded',  # required
    >>>     '--clu',       # required
    >>>     '-z',          # required if multilayer
    >>>     '--two-level'
    >>> )

    """
    
    def _default_to_regular(d):
        """Recursively convert nested defaultdicts to nested dicts.
        """
        if isinstance(d, defaultdict):
            d = {k: _default_to_regular(v) for k, v in d.items()}
        return d
    
    def _get_id_to_label(filename):
        def __int_if_int(val):
            try: return int(val)
            except ValueError: return val
        with open('/tmp/input_infomap/' + filename + ".net", 'r') as fp:
            parsed_network = fp.read()
        return dict(
            (int(n.split()[0]), __int_if_int(n.split('"')[1]))
            for n in re.split(r"\*.+", parsed_network)[1].split("\n")[1:-1]
        )
    
    def multilayer(id_to_label, filename):
        with open('/tmp/output_infomap/'+filename+"_expanded.clu", 'r') as infile:
            clusters = infile.read()

        # Get layers, nodes and clusters from _extended.clu file
        la_no_clu_flow = re.findall(r'\d+ \d+ \d+ \d.*\d*', clusters) # ["30 1 2 0.00800543",...]
        la_no_clu_flow = [tuple(i.split()) for i in la_no_clu_flow]

        layer_node_flow_json = defaultdict(float)  # {layer_node: flow, ...}
        node_flow_json = defaultdict(float)        # {node: flow, ...}
        community_flow_json = defaultdict(float)   # {community: flow, ...}
        communities_json = defaultdict(set)        # {layer: {(node, cluster), ...}, ...}
        for layer, node, cluster, flow in la_no_clu_flow:
            layer_node_flow_json["%s_%s" % (layer, id_to_label[int(node)])] += float(flow)
            node_flow_json["%s" % (id_to_label[int(node)])] += float(flow)
            community_flow_json[cluster] += float(flow)
            communities_json[int(layer)].add((id_to_label[int(node)], int(cluster)))

        return communities_json, layer_node_flow_json, node_flow_json, community_flow_json
    
    def _parse_communities_planar(id_to_label, filename):
        with open('/tmp/output_infomap/'+filename+".clu", 'r') as infile:
            clusters = infile.read()
        
        # Get nodes and clusters from .clu file
        no_clu = [tuple(i.split()[:-1]) for i in re.findall(r"\d+ \d+ \d.*\d*", clusters)]  # [(node, cluster), ...]
        return {0: set([(id_to_label[int(no)], int(clu)) for no, clu in no_clu])}
    
    def _clean_up(filename):
        subprocess.call(['rm', '/tmp/input_infomap/' + filename + '.net'])
        subprocess.call(['rm', '/tmp/output_infomap/' + filename + '_expanded.clu'])
        subprocess.call(['rm', '/tmp/output_infomap/' + filename + '.clu'])
    
    # Check for process id in args (for multiprocessing)
    if args[-1][:3] == "pid":
        pid = args[-1][3:]
        args = args[:-1]
    else:
        pid = ""

    # Try to make input_infomap and output_infomap folders in /tmp
    subprocess.call(['mkdir', '/tmp/input_infomap', '/tmp/output_infomap'])
    
    
    # Get network in multilayer string format and define filename
    filename = 'tmpnet' + pid

    # Store locally
    with open("/tmp/input_infomap/"+filename+".net", 'w') as outfile:
        outfile.write(pajek_string)
    
    # Run Infomap for multilayer network
    subprocess.call(
        ['Infomap', '/tmp/input_infomap/'+filename+".net", '/tmp/output_infomap'] + \
        list(args)
    )
    
    # Parse communities from Infomap output_infomap
    id_to_label = _get_id_to_label(filename)
    
    if 'multilayer' in list(args):
        parsed_communities, layer_node_flow, node_flow, community_flow = multilayer(id_to_label, filename)
    if 'pajek' in list(args):
        parsed_communities = _parse_communities_planar(id_to_label, filename)
        
    _clean_up(filename)

    # Produce layer communities
    layer_communities = {}
    for layer, group in list(parsed_communities.items()):
        communities = {}
        for no, clu in group: 
            try:
                communities[clu-1].append(no)
            except KeyError:
                communities[clu-1] = [no]
        layer_communities[layer] = communities
        
    # Produce community_members
    community_members = defaultdict(Counter)
    for _, communities in list(layer_communities.items()):
        for c, members in list(communities.items()):
            community_members[c].update(members)

    return [
        _default_to_regular(community_members),
        layer_communities,
        _default_to_regular(layer_node_flow),
        _default_to_regular(node_flow),
        _default_to_regular(community_flow)
    ]


def write_pajek(ml_edgelist, index_from=0):
    """Return multiplex representation of multiplex network adjacency matrix A.

    Providing an adjacency tensor where A[:, :, k] is adjacency matrix of layer k, 
    return a pajek format representation of the multilayer network which weights interlayer
    edges by state node neighborhood similarity. 

    Parameters
    ----------
    ml_edgelist : pd.DataFrame
        Must have the three columns `node1`, `node2` and `layer`
    index_from : int
        From which number to index nodes and layers in pajek format from (default=0)

    Returns
    -------
    out : string
        A network string in pajek format
    """
    def _build_adjacency_tensor(ml_edgelist, index="zero"):
        """Return adjacency tensor representation of multilayer edgelist."""
        layers = sorted(set(ml_edgelist['layer']))
        nodes = set(list(ml_edgelist['node1']) + list(ml_edgelist['node2']))
        ind = dict((n, i) for i, n in enumerate(nodes))

        A = defaultdict(int)
        for l in layers:
            for _, row in ml_edgelist.loc[ml_edgelist['layer'] == l].iterrows():
                # Must add both ways if undirected so A becomes symmetrical. If only added one-way
                # triu will only be connections from 'node1' and and tril from 'node2' or vice versa.
                if index == "zero":
                    A[(ind[row['node1']], ind[row['node2']], l)] += 1
                    A[(ind[row['node2']], ind[row['node1']], l)] += 1
                else:
                    A[(row['node1'], row['node2'], l)] += 1
                    A[(row['node2'], row['node1'], l)] += 1
        return A, dict((v, k) for k, v in list(ind.items()))

    def _write_outfile(A):
        """Write nodes and intra/inter-edges from A and J to string."""
        def __remove_symmetry_A(A):
            A_triu = defaultdict(int)
            for (i, j, k), w in list(A.items()):
                if j > i:
                    A_triu[(i, j, k)] = w
            return A_triu
        def __write_nodes(outfile):
            outfile += "*Vertices %d" % Nn
            for nid, label in enumerate(nodes):
                outfile += '\n%d "%s" 1.0' % (nid + index_from, labelmap[label])
            return outfile
        def __write_edges(outfile):
            outfile += "\n*Intra\n# layer node node [weight]"
            sorted_A_sparse = sorted(list(__remove_symmetry_A(A).items()), key=lambda ind__: ind__[0][2])
            for (i, j, k), w in sorted_A_sparse:
                outfile += '\n%d %d %d %f' % (
                    k + index_from,  # layer
                    nodemap[i] + index_from,  # node
                    nodemap[j] + index_from,  # node
                    w                         # weight
                )
            return outfile
        
        outfile = ""
        outfile = __write_nodes(outfile)
        outfile = __write_edges(outfile)
        
        return outfile

    A, labelmap = _build_adjacency_tensor(ml_edgelist)

    nodes = sorted(set([n for i, j, _ in list(A.keys()) for n in [i, j]]))
    Nn = len(nodes)
    Nl = len(set([k for i, j, k in list(A.keys())]))

    nodemap = dict(list(zip(nodes, list(range(Nn)))))

    return _write_outfile(A)


def local_dt_from_ts_and_geo(ts, lat, lon, shapefile_path="/tmp/ne_10m_time_zones/ne_10m_time_zones.shp"):
    """Take UNIX timestamp and coordinates and return location aware datetime object.
    
    If shapefiles do not exist, or are provided, they will automatically be downloaded
    to the /tmp folder. It takes about 10 seconds, but will only be done once.

    Input
    -----
    ts : int
    lat : float
    lon : float
  
    Output
    ------
    out : datetime.datetime
    """

    def _get_location_name(lat, lon):
        """Extract zone label like 'Europe/Copenhagen' from gps coordinates."""
        for shapeRecords in sf.iterShapeRecords():
            shape, record = shapeRecords.shape, shapeRecords.record
            bbox, points = shape.bbox, shape.points
            if point_inside_polygon((lon, lat), [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]):
                if point_inside_polygon((lon, lat), points):
                    return record[13]

    # If shapefiles are provided, use them. If not, download fresh ones from
    # naturalearthdata.com, and put them in the /tmp folder. If no shapefiles
    # are given, this will only happen once, since default path is where the
    # script downloads the shapefiles to. O_o
    try:
        sf = shapefile.Reader(shapefile_path)
    except shapefile.ShapefileException:
        print("Cannot find shapefile. Downloading to /tmp.")
        subprocess.call(['wget', '-O', '/tmp/ne_10m_time_zones.zip', 'http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_time_zones.zip'])
        subprocess.call(['unzip', '-d', '/tmp/ne_10m_time_zones', '/tmp/ne_10m_time_zones.zip'])
        subprocess.call(['rm', '/tmp/ne_10m_time_zones.zip'])
        sf = shapefile.Reader(shapefile_path)

    # Get zone label (e.g. 'Europe/Copenhagen')
    location = _get_location_name(lat, lon)

    # If this runs it is not good.
    if location is None:
        print("Warning: lat, lon is not associated with any location")

    # Do complicated datetime stuff and return the result
    local_tz = pytz.timezone(location)
    dt_utc0 = datetime.datetime.fromtimestamp(ts).replace(tzinfo=pytz.utc)
    return dt_utc0.astimezone(local_tz)

def print_json_tree(d, indent=0):
    """Print tree of keys in JSON object.

    Input
    -----
    d : dict
    """
    for key, value in d.items():
        print('    ' * indent + str(key), end=' ')
        if isinstance(value, dict):
            print(); print_json_tree(value, indent+1)
        else:
            print(":", str(type(d[key])).split("'")[1], "-", str(len(str(d[key]))))

def mutual_information_intersection(p1, p2, measure=normalized_mutual_info_score):
    """Take two partitions as dicts and returns MI of common nodes."""
    nodes = sorted(set(p1.keys()) & set(p2.keys()))
    if nodes == []: return 0
    return measure(
        [p1[n] for n in nodes],
        [p2[n] for n in nodes]
    )

def mutual_information_union(p1, p2, measure=normalized_mutual_info_score):
    """Take two partitions as dicts and returns MI of union of nodes."""
    nodes = sorted(set(p1.keys()) | set(p2.keys()))
    if nodes == []: return 0
    return measure(
        [p1[n] if n in p1 else np.random.randint(1e12) for n in nodes],
        [p2[n] if n in p2 else np.random.randint(1e12) for n in nodes]
    )

def softmax(p):
    """Softmax transformation (prob. dist) of array of values

    Input
    -----
    p : array
    """
    p_exp = np.exp(p)
    return p_exp / np.sum(p_exp)

def is_pos_semidef(X, positive_definite=False):
    """Check if matrix is positive semi-definite.

    Input
    -----
        X : numpy.array (square)
        positive_definite : bool
            If true, tests for positive definiteness, not semidefiniteness
    """
    if not positive_definite:
        return np.all(np.linalg.eigvals(X) >= 0)
    else:
        return np.all(np.linalg.eigvals(X) > 0)

def logistic_map(x, r=1):
    """Not really a utility function, just a cool implementation of the logistic map."""
    def _map(x):
        return r * x * (1 - x)
    output = _map(x)
    while True:
        yield output
        output = _map(output)

def log_progress(sequence, every=None, size=None, name='Items'):
    """Wrapper around list object in loop to log loop progress.

    Input
    -----
        sequence : list or iterator
        every : int, iteration step to evaluate progress at
        size : total number of iterations (good for iterators)
        name : who knows

    Example
    -------
    >>> for v in ulf.log_progress(range(1000), every=10):
    >>>     print v
    []
    """
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = '{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

def cartesian_encoder(coord, r_E=6371):
    """Convert lat/lon to cartesian points on Earth's surface.

    Input
    -----
        coord : numpy 2darray (size=(N, 2))
        r_E : radius of Earth

    Output
    ------
        out : numpy 2darray (size=(N, 3))
    """
    def _to_rad(deg):
        return deg * np.pi / 180.

    theta = _to_rad(coord[:, 0])  # lat [radians]
    phi = _to_rad(coord[:, 1])    # lon [radians]

    x = r_E * np.cos(phi) * np.cos(theta)
    y = r_E * np.sin(phi) * np.cos(theta)
    z = r_E * np.sin(theta)

    return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)

def cartesian_decoder(coord, r_E=6371):
    """Convert cartesian points on Earth's surface to lat/lon.

    Input
    -----
        coord : numpy 2darray (size=(N, 3))
        r_E : radius of Earth

    Output
    ------
        out : numpy 2darray (size=(N, 2))
    """
    def _to_deg(rad):
        return rad * 180. / np.pi

    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

    theta = np.arcsin(z / r_E)
    phi = np.arctan(y / x)

    # Convert to degrees. Longitudes, are bound between -90;90 in decode step, so correct in 3 and 4th quadrant of x-y plane (Asia)
    lat = _to_deg(theta)
    lon = _to_deg(phi) - 180 * ((x < 0) * (y < 0)) + 180 * ((x < 0) * (y > 0))

    return np.concatenate([lat.reshape(-1, 1), lon.reshape(-1, 1)], axis=1)

def haversine(coord1, coord2, r_E=6371):
    """Compute the haversine distances between two arrays of gps coordinates.

    Input
    -----
        coord1 : numpy 2darray (size=(N, 2))
        coord2 : numpy 2darray (size=(N, 2))

    Output
    ------
        out : float (haversine distance in km)
    """

    def _to_rad(deg):
        return deg * np.pi / 180.
    
    lat1, lat2 = _to_rad(coord1[:, 0]), _to_rad(coord2[:, 0])
    dlat = _to_rad(coord1[:, 0] - coord2[:, 0])
    dlon = _to_rad(coord1[:, 1] - coord2[:, 1])

    a = np.sin(dlat / 2.) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.) ** 2
    c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 
    return r_E * c

def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.

    Parameters
    ----------
        query : str
        corpus : str
        step : int
            Step size of first match-value scan through corpus. Can be thought of
            as a sort of "scan resolution". Should not exceed length of query.
        flex : int
            Max. left/right substring position adjustment value. Should not
            exceed length of query / 2.

    Outputs
    -------
        output0 : str
            Best matching substring.
        output1 : float
            Match ratio of best matching substring. 1 is perfect match.
    """

    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)

    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        bmv_l = match_values[p_l / step]
        bmv_r = match_values[p_l / step]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))

        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])

    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        print("Warning: flex exceeds length of query / 2. Setting to default.")
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step

    pos_left, pos_right, match_value = adjust_left_right_positions()

    return corpus[pos_left: pos_right].strip(), match_value

def cart_transform(data, precision=1e-2, eps=1e-12, cartogram=(None, None, None)):
    """Make cartogram transformation of a set of coordinates.
    
    First install all the `cart` software from http://www-personal.umich.edu/~mejn/cart/.
    
    Input
    -----
        data : numpy 2darray (N, 2)
        precision : float
            Grid coarseness. smaller -> more grid points -> slower)
        eps : float
            Zero density offset. Low value sets non-zero density values increasingly
            apart from zero density values.
        cartogram : (str, int [xdim], int [ydim])
            Path to precomputed cartogram, and cartogram lattice dimensions.
    
    Output
    ------
        data_transformed : numpy 2darray (N, 2)
    """

    # Compute the extent of the data
    min_lat, max_lat = min(data[:, 0]), max(data[:, 0])
    min_lon, max_lon = min(data[:, 1]), max(data[:, 1])
    
    # Compute cartogram
    if cartogram[0] is None:
        
        # Compute lattice point densities
        density_scores, (lat_points, lon_points) = np.histogramdd(
            data,
            bins=(
                int((max_lat + precision - min_lat) / precision),
                int((max_lon + precision - min_lon) / precision),
            )
        )
        lat_points, lon_points = lat_points[:-1], lon_points[:-1]
        
        path = "/tmp/output_hist.dat"
        xdim, ydim = len(lon_points), len(lat_points)
        print("xdim, ydim:", xdim, ydim)
        
        # Compute cartograms
        np.savetxt("/tmp/density_map_hist.dat", np.log(density_scores + eps) - np.log(eps) + 1, fmt="%.03f")
        subprocess.call([
            "cart", str(xdim), str(ydim),           # Size parameters
            "/tmp/density_map_hist.dat",  # Input
            path                          # Output
        ])
    else:
        path, xdim, ydim = cartogram

    # Create mapping functions between coordinate and lattice space
    lat_scaler = MinMaxScaler(feature_range=(0, ydim - 1))
    lat_scaler.fit(np.array([[min_lat], [max_lat]]))
    lon_scaler = MinMaxScaler(feature_range=(0, xdim - 1))
    lon_scaler.fit(np.array([[min_lon], [max_lon]]))
    
    # Translate coordiantes to cartogram indices
    data_cart_domain = np.concatenate([
        lon_scaler.transform(data[:, 1].reshape(-1, 1)),
        lat_scaler.transform(data[:, 0].reshape(-1, 1))
    ], axis=1)  # cast in x, y (lon, lat)
    np.savetxt("/tmp/coordinates.dat", data_cart_domain, fmt="%.03f")
    
    # Interpolate them onto the cartogram
    cat = subprocess.Popen(["cat", "/tmp/coordinates.dat"], stdout=subprocess.PIPE)
    with open("/tmp/coordinates_transformed.dat", 'w') as fp:
        subprocess.call(
            ["interp", str(xdim), str(ydim), path],
            stdin=cat.stdout, stdout=fp
        )

    # Load and transform the coordinates back into geo-space
    data_cart_domain_transformed = np.loadtxt("/tmp/coordinates_transformed.dat")
    data_transformed = np.concatenate([
        lat_scaler.inverse_transform(data_cart_domain_transformed[:, 1].reshape(-1, 1)),
        lon_scaler.inverse_transform(data_cart_domain_transformed[:, 0].reshape(-1, 1))
    ], axis=1)
    
    return data_transformed

def plot_html(lats, lons, zoom=11, heatmap=True, scatter=True):
    """Produce renderable html file with points on real map.

    Inputs
    ------
        lats : list
        lons : list

    Output
    ------
        out : `mymap.html`
    """
    gmap = gmplot.GoogleMapPlotter(np.median(lats), np.median(lons), zoom=zoom)
    
    if heatmap: gmap.heatmap(lats, lons)
    if scatter: gmap.scatter(lats, lons, 'k', size=6, marker=False)
    gmap.draw("/Users/ulfaslak/Desktop/mymap.html")

def clip_range(x, xlim):
    """Constraint x to the domain of xlim.

    Input
    -----
        x : number (int/float)
        xlim : list [min, max]

    Return
    ------
        out : number (int/float)
    """
    return min([max([x, xlim[0]]), xlim[1]])

def plot_polygon(points, **kwargs):
    """Plot set of points as polygon.

    Input
    -----
        points : array-like (N, 2)
        **kwargs : valid keyword-arguments
    """
    plt.gca().add_collection(
        PatchCollection(
            [Polygon(points, True)],
            **kwargs)
    )

def cosine_sim_collections(a, b):
    """Get cosine similarity between two collection of values.

    Input
    -----
        a/b : 1d list of values.
            Duplicates extrudes the vector more in the corresponding direction

    Output
    ------
        out : float (cos. distance between vector representations of either collection)
    """
    setab = sorted(set(a) | set(b))
    countera, counterb = Counter(a), Counter(b)
    veca = [countera[element] if element in a else 0 for element in setab]
    vecb = [counterb[element] if element in b else 0 for element in setab]
    return dot(veca, vecb) / (norm(veca) * norm(vecb))

def get_networks_from_multiplex_pajek(pajek):
    """Returns intra and interlayer networks from pajek network.

    NOT TESTED
    
    Input
    -----
        pajek : str
            Pajek formatted multiplex network. Links formatting must be:
                *Intra
                # layer node node [weight]
                0 102 109 1.0
                0 25 43 1.0
                ...
                *Inter
                # layer node layer [weight]
                0 1 1 1.0
                0 40 2 1.0
                ...
            Or:
                *multiplex
                # layer node layer node [weight]
                0 0 0 1 0.25
                0 0 0 2 0.25
                ...            
    """

    def _extract_intra_links(pajek):
        """Return dictionary of intralayer `nx.Graph`s from input formatted pajek"""
        string_links = re.findall(r"\d+ \d+ \d+.*", pajek.split("*Intra")[1].split("*Inter")[0])
        intra_links = [list(map(eval, link.split())) for link in string_links]
        G_arr_intra = defaultdict(lambda: nx.Graph)
        for l in intra_links:
            G_arr_intra[l[0]].add_edge(l[1], l[2])
        return G_arr_intra
    
    def _extract_intra_links_from_multiplex(pajek):
        """Return dictionary of intralayer `nx.Graph`s from output formatted pajek"""
        string_links = re.findall(r"\d+ \d+ \d+ \d+.*", pajek.split("*multiplex")[1])
        value_links = [list(map(eval, link.split())) for link in string_links]
        G_arr_intra = defaultdict(lambda: nx.Graph)
        for l in value_links:
            if l[0] == l[2]:
                G_arr_intra[l[0]].add_edge(l[1], l[3])
        return G_arr_intra
    
    if "*Intra" in pajek:
        return _extract_intra_links(pajek)
    if "*multiplex" in pajek:
        return _extract_intra_links_from_multiplex(pajek)
    
    raise ValueError("No intra links inside pajek")

def force_ascii(text):
    """Remove all non-ascii characters from a string.

    Input
    -----
        text : str

    Output
    ------
        out : str
    """
    return "".join([c for c in text if ord(c) < 128])

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def convert_to_pajek(G):
    """Takes a network in nx.Graph format and converts it to a pajek string.

    Input
    -----
        G : nx.Graph

    Output
    ------
        pajek : str (example below)

            *Vertices 2078
            1 "Gamora" 1.0
            2 "Gazelle (Marvel Comics)" 1.0
            3 "Hope Summers (comics)" 1.0
                ...
            *Args 533
            1 638 1.0
            3 181 1.0
            3 1431 1.0

    """
    # Add nodes
    pajek = "*Vertices %d" % len(G)
    name_to_index = dict()
    for i, n in enumerate(G.nodes(), 1):
        pajek += '\n%d "%s" 1.0' % (i, n)
        name_to_index[n] = i

    # Add edges
    pajek += "\n*Args %d" % len(G.edges())
    for n1, n2, d in G.edges(data=True):
        weight = d['weight'] if 'weight' in d else 1.0
        pajek += "\n%d %d %.03f" % (name_to_index[n1], name_to_index[n2], weight)

    return pajek

def connect_components_in_ring(G):
    """Create a connected network out of a multi-component network.

    Picks one node from each component and tailors them together with
    a ring graph.

    Input
    -----
        G : nx.Graph

    Output
    ------
        G_connected : nx.Graph
    """
    # Pick a random node from each component
    anchor_nodes = [
        np.random.choice(list(_G.nodes()))
        for _G in nx.components.connected_component_subgraphs(G)
    ]

    # Define new edges as a "ring" graph around these anchor nodes
    new_edges = list(zip(anchor_nodes, anchor_nodes[1:] + [anchor_nodes[0]]))
    
    # Add new edges
    G_connected = G.copy()
    G_connected.add_edges_from(new_edges)
    
    return G_connected

def normalize_counter(counter):
    """Return a defaultdict-version of a counter with values that sum to one.

    Input
    -----
        counter : collections.Counter

    Output
    ------
        normed_counter : collections.defaultdict(float)
    """
    norm = sum(counter.values()) * 1.
    normed_counter = defaultdict(float)
    for k, v in list(counter.items()):
        normed_counter[k] = v/norm
    return Counter(normed_counter)

def binrange(_min, _max, stepsize, include_upper=False):
    """Create a range between _min and _max rounded by stepsize.

    Input
    -----
        _min : int
        _max : int
        stepsize : int
        include_upper : bool
            Whether or not to include rounded max value as last value in list

    Output
    ------
        out : list

    Example
    -------
        >>> binrange(min(timestamps), max(timestamps), 86400)
    """
    _min = _min - _min % stepsize
    _max = _max - _max % stepsize + stepsize * (1 + include_upper)
    return np.arange(_min, _max, stepsize)

def create_monthly_bins(d0, d1, output_timestamps=False, buffer=0):
    """Return list of edgevalues for monthly bins between two datetime objects.
    
    This function is complicated because the duration of months varies.
    
    Input
    -----
        d0, d1: datetime.datetime
        output_datetime : bool
        buffer : int
            Add a buffer to extend the number of bins
        
    Output
    ------
        bins : list (of timestamps)
    """
    def _round_datetime_to_month_start(d):
        return d - datetime.timedelta(days=d.day-1, hours=d.hour, minutes=d.minute, seconds=d.second)

    def _round_datetime_to_month_end(d):
        return d + datetime.timedelta(days=monthrange(d.year, d.month)[1]-d.day, hours=23-d.hour, minutes=59-d.minute, seconds=59-d.second)

    _min = _round_datetime_to_month_start(d0)
    _max = _round_datetime_to_month_end(d1)

    n_months = int(round((_max - _min).days / (3.154e7/86400/12)))

    bins = [_min]
    for n in range(n_months + buffer):
        year = _min.year + (_min.month-1 + n) / 12
        month = (_min.month-1 + n) % 12 + 1
        bins.append(bins[-1] + datetime.timedelta(days=monthrange(year, month)[1]))

    if output_timestamps:
        return [int(d.strftime("%s")) for d in bins]
    return bins

def all_pairs(items, sort=False):
    """Yield combinations of values in list.

    Inputs
    ------
        items : list
        sort : bool

    Output
    ------
        out : iterator
    """
    if sort:
        items = sorted(items)
    for i, ni in enumerate(items):
        for j, nj in enumerate(items):
            if j > i: yield ni, nj

def index_of_max_change(vals):
    """Get the index where a list of numbers change the most.

    Input
    -----
        vals : list of numbers

    Output
    ------
        out : int

    Example
    -------
        >>> index_of_max_change([1, 3, 2, 3, -2, 1])
        3
    """
    i_vals = zip(range(len(vals)), vals)
    vals = [v for i, v in i_vals]
    vals_diff = [abs(v1 - v0) for v0, v1 in zip(vals[:-1], vals[1:])]
    return i_vals[vals_diff.index(max(vals_diff))][0]

def remove_consequetive_duplicates(your_list):
    """Return list of consequetively unique items in your_list.

    Input
    -----
        your_list : list

    Output
        out : list (np.array if input type is np.array)
    """
    out = [v for i, v in enumerate(your_list) if i == 0 or v != your_list[i-1]]
    if type(your_list) == np.ndarray:
        return np.array(out)
    return out

def networkx_graph_to_json(G, save=False):
    partition = best_partition(G)
    out = {
        "nodes": [
            {"id": n, "size": d, "group": partition[n]}
            for n, d in G.degree()
        ],
        "links": [
            {"source": n1, "target": n2, "weight": data["weight"]} if 'weight' in data else
            {"source": n1, "target": n2}
            for n1, n2, data in G.edges(data=True)
        ]
    }
    if save:
        with open("tmp.json", 'w') as fp:
            json.dump(out, fp)
    else:
        return out

def display_strptime_formatters():
    """Print table of valid datetime formatters for strptime."""
    data = [
        ["%a", "Weekday as locale's abbreviated name.", "Mon"],
        ["%A", "Weekday as locale's full name.", "Monday"],
        ["%w", "Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.", "1"],
        ["%d", "Day of the month as a zero-padded decimal number.", "30"],
        ["%-d", "Day of the month as a decimal number. (Platform specific)", "30"],
        ["%b", "Month as locale's abbreviated name.", "Sep"],
        ["%B", "Month as locale's full name.", "September"],
        ["%m", "Month as a zero-padded decimal number.", "09"],
        ["%-m", "Month as a decimal number. (Platform specific)", "9"],
        ["%y", "Year without century as a zero-padded decimal number.", "13"],
        ["%Y", "Year with century as a decimal number.", "2013"],
        ["%H", "Hour (24-hour clock) as a zero-padded decimal number.", "07"],
        ["%-H", "Hour (24-hour clock) as a decimal number. (Platform specific)", "7"],
        ["%I", "Hour (12-hour clock) as a zero-padded decimal number.", "07"],
        ["%-I", "Hour (12-hour clock) as a decimal number. (Platform specific)", "7"],
        ["%p", "Locale's equivalent of either AM or PM.", "AM"],
        ["%M", "Minute as a zero-padded decimal number.", "06"],
        ["%-M", "Minute as a decimal number. (Platform specific)", "6"],
        ["%S", "Second as a zero-padded decimal number.", "05"],
        ["%-S", "Second as a decimal number. (Platform specific)", "5"],
        ["%f", "Microsecond as a decimal number, zero-padded on the left.", "000000"],
        ["%z", "UTC offset in the form +HHMM or -HHMM (empty string if the the object is naive).", ""],
        ["%Z", "Time zone name (empty string if the object is naive).", ""],
        ["%j", "Day of the year as a zero-padded decimal number.", "273"],
        ["%-j", "Day of the year as a decimal number. (Platform specific)", "273"],
        ["%U", "Week number of the year (Sunday as the first day of the week) as a zero padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0.", "39"],
        ["%W", "Week number of the year (Monday as the first day of the week) as a decimal number. All days in a new year preceding the first Monday are considered to be in week 0.", "39"],
        ["%c", "Locale's appropriate date and time representation.", "Mon Sep 30 07:06:05 2013"],
        ["%x", "Locale's appropriate date representation.", "09/30/13"],
        ["%X", "Locale's appropriate time representation.", "07:06:05"],
        ["%%", "A literal '%' character.", "%"]
    ]

    display(HTML(
       '<table><tr>{}</tr></table>'.format(
           '</tr><tr>'.join(
               '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
           )
    ))

def convex_hull(points):
    """Get points on convex hull of nd point distribution.

    Input
    -----
        points : ndarray (N, D)

    Output
        out : ndarray (n, D)
    """
    points = np.array(points)
    hull = ConvexHull(points)
    return points[hull.vertices, :]

def benjamin_hochberg_correction(p_values, alpha=0.05):
    """Get largest p-value that passes the Benjamin-Hochberg correction.
    
    Input
    -----
        p_values : list
        alpha : float
            Significance level
    
    Returns
    -------
        out : float
            Largest significant p-value. Returns 0 if none are significant
    """
    m = len(p_values)
    p_values = sorted(p_values)
    for k, p in enumerate(p_values):
        if p > (k + 1) / m * alpha:
            index = k - 1
            if index < 0:
                return 0
            return p_values[index]

def shuffle_array_along(X, axis=0, inline=True):
    """Shuffle an array along a given axis.
    
    Parameters
    ----------
        X : nd.array
        axis : int
        inline : bool
    
    Return
    ------
        if inline : X
        else : None
    """
    if not inline:
        X = X.copy()
    np.apply_along_axis(np.random.shuffle, axis, X)
    if not inline:
        return X

    
# ------------------------- #
# Non importable (examples) #
# ------------------------- #

class how_to_unittest:
    def unittest(f):    
        from functools import wraps
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Test something with the args, kwargs and the wrapped function f
            return f(*args, **kwargs)
        return wrapper

    @unittest
    def f(arg1, arg2, kwarg1=1, kwarg2=2):
        # A function that does something
        return G_copy


# Python figure default layouts
def _just_a_box():
    """Templates for manipulating matplotlib layout styles.

    Example
    -------
        >>> import matplotlib as mpl
        >>> mpl.rc({
            "figure": {
                "figsize": (3.7, 2),  
                "dpi": 300, 
                "frameon": False, 
            }
        })
    """
    {
        "pgf": {
            "texsystem": "", 
            "debug": "", 
            "rcfonts": "", 
            "preamble": ""
        }, 
        "verbose": {
            "level": "", 
            "fileo": ""
        }, 
        "figure": {
            "facecolor": "", 
            "titlesize": "", 
            "titleweight": "", 
            "figsize": "", 
            "max_open_warning": "", 
            "edgecolor": "", 
            "dpi": "", 
            "frameon": "", 
            "autolayout": ""
        }, 
        "savefig": {
            "transparent": "", 
            "facecolor": "", 
            "pad_inches": "", 
            "orientation": "", 
            "format": "", 
            "jpeg_quality": "", 
            "directory": "", 
            "edgecolor": "", 
            "dpi": "", 
            "frameon": "", 
            "bbox": ""
        }, 
        "text": {
            "color": "", 
            "antialiased": "", 
            "hinting": "", 
            "hinting_factor": "", 
            "usetex": ""
        }, 
        "image": {
            "resample": "", 
            "cmap": "", 
            "composite_image": "", 
            "interpolation": "", 
            "lut": "", 
            "aspect": "", 
            "origin": ""
        }, 
        "examples": {
            "directory": ""
        }, 
        "axes3d": {
            "grid": ""
        }, 
        "font": {
            "fantasy": "", 
            "monospace": "", 
            "weight": "", 
            "serif": "", 
            "family": "", 
            "stretch": "", 
            "variant": "", 
            "cursive": "", 
            "style": "", 
            "size": ""
        }, 
        "contour": {
            "corner_mask": "", 
            "negative_linestyle": ""
        }, 
        "backend": {
            "qt4": "", 
            "qt5": ""
        }, 
        "ps": {
            "useafm": "", 
            "papersize": "", 
            "usedistiller": "", 
            "fonttype": ""
        }, 
        "axes": {
            "labelweight": "", 
            "facecolor": "", 
            "axisbelow": "", 
            "titlesize": "", 
            "titleweight": "", 
            "labelpad": "", 
            "prop_cycle": "", 
            "ymargin": "", 
            "labelcolor": "", 
            "unicode_minus": "", 
            "hold": "", 
            "autolimit_mode": "", 
            "linewidth": "", 
            "xmargin": "", 
            "edgecolor": "", 
            "titlepad": "", 
            "labelsize": "", 
            "grid": ""
        }, 
        "markers": {
            "fillstyle": ""
        }, 
        "hist": {
            "bins": ""
        }, 
        "polaraxes": {
            "grid": ""
        }, 
        "animation": {
            "convert_path": "", 
            "frame_format": "", 
            "embed_limit": "", 
            "html": "", 
            "html_args": "", 
            "avconv_args": "", 
            "codec": "", 
            "bitrate": "", 
            "ffmpeg_args": "", 
            "ffmpeg_path": "", 
            "convert_args": "", 
            "writer": "", 
            "avconv_path": ""
        }, 
        "tk": {
            "window_focus": ""
        }, 
        "hatch": {
            "color": "", 
            "linewidth": ""
        }, 
        "boxplot": {
            "bootstrap": "", 
            "patchartist": "", 
            "meanline": "", 
            "vertical": "", 
            "showfliers": "", 
            "showbox": "", 
            "notch": "", 
            "showmeans": "", 
            "whiskers": "", 
            "showcaps": ""
        }, 
        "docstring": {
            "hardcopy": ""
        }, 
        "errorbar": {
            "capsize": ""
        }, 
        "xtick": {
            "direction": "", 
            "labelbottom": "", 
            "alignment": "", 
            "labeltop": "", 
            "color": "", 
            "bottom": "", 
            "top": "", 
            "labelsize": ""
        }, 
        "ytick": {
            "direction": "", 
            "right": "", 
            "alignment": "", 
            "color": "", 
            "labelright": "", 
            "labelleft": "", 
            "left": "", 
            "labelsize": ""
        }, 
        "grid": {
            "alpha": "", 
            "color": "", 
            "linewidth": "", 
            "linestyle": ""
        }, 
        "mathtext": {
            "it": "", 
            "fontset": "", 
            "default": "", 
            "tt": "", 
            "cal": "", 
            "sf": "", 
            "bf": "", 
            "rm": "", 
            "fallback_to_cm": ""
        }, 
        "path": {
            "simplify": "", 
            "sketch": "", 
            "snap": "", 
            "effects": "", 
            "simplify_threshold": ""
        }, 
        "legend": {
            "shadow": "", 
            "facecolor": "", 
            "markerscale": "", 
            "loc": "", 
            "handleheight": "", 
            "borderaxespad": "", 
            "scatterpoints": "", 
            "numpoints": "", 
            "framealpha": "", 
            "columnspacing": "", 
            "handlelength": "", 
            "fontsize": "", 
            "edgecolor": "", 
            "labelspacing": "", 
            "frameon": "", 
            "fancybox": "", 
            "handletextpad": "", 
            "borderpad": ""
        }, 
        "svg": {
            "hashsalt": "", 
            "image_inline": "", 
            "fonttype": ""
        }, 
        "lines": {
            "solid_capstyle": "", 
            "markersize": "", 
            "antialiased": "", 
            "dotted_pattern": "", 
            "scale_dashes": "", 
            "solid_joinstyle": "", 
            "color": "", 
            "dashdot_pattern": "", 
            "markeredgewidth": "", 
            "dashed_pattern": "", 
            "linewidth": "", 
            "marker": "", 
            "dash_joinstyle": "", 
            "dash_capstyle": "", 
            "linestyle": ""
        }, 
        "patch": {
            "edgecolor": "", 
            "antialiased": "", 
            "facecolor": "", 
            "linewidth": "", 
            "force_edgecolor": ""
        }, 
        "keymap": {
            "fullscreen": "", 
            "quit": "", 
            "grid_minor": "", 
            "all_axes": "", 
            "yscale": "", 
            "quit_all": "", 
            "save": "", 
            "back": "", 
            "zoom": "", 
            "xscale": "", 
            "home": "", 
            "pan": "", 
            "forward": "", 
            "grid": ""
        }, 
        "webagg": {
            "port_retries": "", 
            "address": "", 
            "open_in_browser": "", 
            "port": ""
        }, 
        "pdf": {
            "use14corefonts": "", 
            "compression": "", 
            "inheritcolor": "", 
            "fonttype": ""
        }, 
        "scatter": {
            "marker": ""
        }
    }        
# Non-python stuff
"""rm build/Infomap/infomap/MultiplexNetwork.o; make; ./Infomap foursquares.net ../output/ -i multiplex --multiplex-js-relax-rate 0.05 --overlapping --expanded --pajek -z"""
