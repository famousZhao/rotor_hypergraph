"""Reference (unoptimised) implementation of the overlapping-community stage.

This module is not part of the method.  It is kept only so that experiment 6
can time the pipeline against the implementation it replaces, on the same
machine and the same signal: dictionary-of-lists clique graph construction
plus the pure-Python python-louvain modularity pass.  Nothing in the
manuscript's results is computed with these functions.
"""
import numpy as np
import networkx as nx

import community as community_louvain

from topology import sparsify


def ni_louvain_reference(A, remove_ratio=0.75, resolution=1.0, seed=42,
                         max_cliques=200000):
    """The original NI-Louvain: python-louvain on a clique graph."""
    n = A.shape[0]
    As = sparsify(A, remove_ratio=remove_ratio, seed=seed)
    G = nx.from_scipy_sparse_array(As)
    cliques = list(nx.find_cliques(G))
    if len(cliques) > max_cliques:
        cliques = sorted(cliques, key=len, reverse=True)[:max_cliques]
    if not cliques:
        return [{i} for i in range(n)]

    node2clq = {}
    for ci, cl in enumerate(cliques):
        for v in cl:
            node2clq.setdefault(v, []).append(ci)

    CG = nx.Graph()
    CG.add_nodes_from(range(len(cliques)))
    for _, cs in node2clq.items():
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                CG.add_edge(cs[a], cs[b])

    if CG.number_of_edges() == 0:
        part = {i: i for i in range(len(cliques))}
    else:
        part = community_louvain.best_partition(CG, resolution=resolution,
                                                random_state=seed)

    membership = {}
    for ci, cl in enumerate(cliques):
        c = part.get(ci, 0)
        for v in cl:
            membership.setdefault(v, set()).add(c)
    communities = {}
    for v, cs in membership.items():
        for c in cs:
            communities.setdefault(c, set()).add(v)
    return list(communities.values())
