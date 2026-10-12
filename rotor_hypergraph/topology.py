"""Recurrence network, NI-Louvain overlapping communities and hypergraph
feature extraction.

Pipeline
--------
time series -> delay embedding -> recurrence network (complex network)
            -> NI-Louvain overlapping communities -> hypergraph
            -> topology features
"""
import random

import numpy as np
import networkx as nx
import igraph as ig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from config import REC_PERCENTILE, REMOVE_RATIO
from psr import reconstruct


# ------------------------------------------------------------------ modularity
def _seed_igraph(seed):
    """Make igraph's randomised passes reproducible."""
    ig.set_random_number_generator(random.Random(int(seed)))


def _louvain(n_nodes, edges, resolution=1.0, seed=42):
    """Louvain partition of a graph given as (n_nodes, edge array).

    igraph's multilevel routine is the same Blondel greedy modularity
    optimisation as python-louvain but runs in compiled code.  On the
    clique graphs met here it is roughly an order of magnitude faster,
    which is what makes the overlapping-community stage affordable.
    """
    if n_nodes == 0:
        return np.zeros(0, dtype=int)
    g = ig.Graph(n=int(n_nodes), edges=edges, directed=False)
    if g.ecount() == 0:
        return np.arange(n_nodes)
    _seed_igraph(seed)
    part = g.community_multilevel(resolution=resolution)
    return np.asarray(part.membership, dtype=int)


def _edges_of(A):
    """Upper-triangular edge list of a symmetric sparse adjacency matrix."""
    coo = A.tocoo()
    m = coo.row < coo.col
    return np.column_stack([coo.row[m], coo.col[m]]).astype(np.int64)


# ------------------------------------------------------------------ network
def recurrence_network(traj, percentile=REC_PERCENTILE, max_nodes=1500,
                       seed=0):
    """Build the recurrence network of a phase-space trajectory.

    The recurrence threshold is a percentile of the pairwise distance
    distribution, estimated from a random subsample for speed.  Nodes are the
    trajectory points; an edge joins two points closer than the threshold.
    """
    n = len(traj)
    if n > max_nodes:
        stride = int(np.ceil(n / max_nodes))
        traj = traj[::stride]
        n = len(traj)
    if n < 20:
        return csr_matrix((n, n)), n

    rng = np.random.default_rng(seed)
    m = min(n, 600)
    sub = traj[rng.choice(n, m, replace=False)]
    d = np.linalg.norm(sub[:, None, :] - traj[None, :, :], axis=2)
    thr = np.percentile(d.ravel(), percentile)

    from scipy.spatial import cKDTree
    pairs = cKDTree(traj).query_pairs(r=thr, output_type="ndarray")
    if len(pairs) == 0:
        return csr_matrix((n, n)), n
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.int8)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A.setdiag(0)
    A.eliminate_zeros()
    return A, n


# ------------------------------------------------------------------ communities
def louvain_partition(A, resolution=1.0, seed=42):
    """Non-overlapping Louvain partition (ablation baseline)."""
    n = A.shape[0]
    lab = _louvain(n, _edges_of(A), resolution=resolution, seed=seed)
    groups = {}
    for node, c in enumerate(lab):
        groups.setdefault(int(c), set()).add(node)
    return list(groups.values())


def sparsify(A, remove_ratio=0.75, seed=42):
    """Randomly remove a fraction of edges, keeping the node set."""
    coo = A.tocoo()
    upper = coo.row < coo.col
    r, c = coo.row[upper], coo.col[upper]
    n_e = len(r)
    if n_e == 0:
        return A
    rng = np.random.default_rng(seed)
    keep = rng.random(n_e) >= remove_ratio
    r, c = r[keep], c[keep]
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=A.shape)


def ni_louvain(A, remove_ratio=0.75, resolution=1.0, seed=42,
               max_cliques=200000):
    """NI-Louvain overlapping community detection (the author's algorithm).

    1. randomly remove `remove_ratio` of the edges to break up the dense
       recurrence structure;
    2. enumerate the maximal cliques of the sparsified graph;
    3. build the clique graph (cliques are nodes, edges join cliques that
       share an original node) and run Louvain on it;
    4. map cliques back to vertices.  A vertex belonging to cliques from
       several communities receives several memberships, which is where the
       overlap comes from.

    The three costly steps are all done without Python-level loops over
    clique pairs: the enumeration is a capped generator, the clique graph is
    the sparsity pattern of an incidence product, and the partition is
    computed by igraph.
    """
    n = A.shape[0]
    As = sparsify(A, remove_ratio=remove_ratio, seed=seed)
    G = nx.from_scipy_sparse_array(As)

    cliques = []
    for cl in nx.find_cliques(G):
        cliques.append(cl)
        if len(cliques) >= max_cliques:
            break
    if not cliques:
        return [{i} for i in range(n)]

    # clique graph: cliques are nodes, an edge joins two cliques that share
    # at least one phase-space point.  S = B B^T holds the intersection sizes,
    # so its off-diagonal pattern is exactly the clique-graph adjacency.
    k = len(cliques)
    rows = np.repeat(np.arange(k, dtype=np.int32),
                     [len(cl) for cl in cliques])
    cols = np.concatenate(cliques).astype(np.int32)
    B = csr_matrix((np.ones(len(rows), np.int32), (rows, cols)),
                   shape=(k, n))
    S = (B @ B.T).tocoo()
    keep = S.row < S.col
    edges = np.column_stack([S.row[keep], S.col[keep]]).astype(np.int64)

    lab = _louvain(k, edges, resolution=resolution, seed=seed)

    membership = {}
    for ci, cl in enumerate(cliques):
        c = int(lab[ci])
        for v in cl:
            membership.setdefault(v, set()).add(c)

    communities = {}
    for v, cs in membership.items():
        for c in cs:
            communities.setdefault(c, set()).add(v)
    return list(communities.values())


# ------------------------------------------------------------------ hypergraph
def build_hypergraph(communities, n_nodes):
    """Incidence matrix and basic sizes for a community-derived hypergraph."""
    communities = [c for c in communities if len(c) > 0]
    H = np.zeros((n_nodes, len(communities)), dtype=np.int8)
    for j, comm in enumerate(communities):
        for v in comm:
            if 0 <= v < n_nodes:
                H[v, j] = 1
    sizes = H.sum(0).astype(float)
    deg = H.sum(1).astype(float)
    return {"H": H, "communities": communities, "sizes": sizes,
            "node_degree": deg, "n_nodes": n_nodes,
            "n_hyperedges": len(communities)}


def hyperedge_interaction(H):
    """Mean Jaccard overlap between hyperedge pairs (higher-order coupling)."""
    k = H.shape[1]
    if k < 2:
        return 0.0
    S = H.T.astype(float) @ H.astype(float)      # intersection sizes
    sz = np.diag(S).copy()
    denom = sz[:, None] + sz[None, :] - S
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(denom > 0, S / denom, 0.0)
    iu = np.triu_indices(k, 1)
    return float(J[iu].mean()) if len(iu[0]) else 0.0


def hypergraph_features(hg):
    """Topology descriptors used in the results tables."""
    H = hg["H"].astype(float)
    n, k = H.shape
    sizes = hg["sizes"]
    deg = hg["node_degree"]

    # vertices are joined when they share a hyperedge: the clique projection.
    A_v = csr_matrix((H @ H.T) > 0)
    A_v.setdiag(0)
    A_v.eliminate_zeros()
    n_comp, labels = connected_components(A_v, directed=False)
    gc = int(np.bincount(labels).max()) if n else 0

    feats = {
        "n_nodes": float(n),
        "n_hyperedges": float(k),
        "hedge_size_mean": float(sizes.mean()) if k else 0.0,
        "hedge_size_std": float(sizes.std()) if k else 0.0,
        "hedge_size_max": float(sizes.max()) if k else 0.0,
        "hedge_size_entropy": float(_norm_entropy(sizes)) if k else 0.0,
        "node_degree_mean": float(deg.mean()) if n else 0.0,
        "node_degree_std": float(deg.std()) if n else 0.0,
        "overlap_ratio": float((deg > 1).mean()) if n else 0.0,
        "hedge_interaction": hyperedge_interaction(hg["H"]),
        "n_components": float(n_comp),
        "giant_fraction": float(gc / n) if n else 0.0,
    }
    feats.update(_spectral(A_v.astype(float), n))
    return feats


def _norm_entropy(v):
    v = np.asarray(v, float)
    v = v[v > 0]
    if len(v) <= 1:
        return 0.0
    p = v / v.sum()
    return float(-np.sum(p * np.log(p)) / np.log(len(v)))


def _spectral(A, n):
    """Adjacency spectral radius/gap and Laplacian algebraic connectivity."""
    out = {"spectral_radius": 0.0, "algebraic_connectivity": 0.0,
           "spectral_gap": 0.0, "spectral_entropy": 0.0}
    if n < 4 or A.nnz == 0:
        return out
    try:
        vals = np.sort(np.abs(eigsh(A, k=min(3, n - 2), which="LM",
                                   return_eigenvectors=False)))[::-1]
        out["spectral_radius"] = float(vals[0])
        out["spectral_gap"] = float(vals[0] - vals[1]) if len(vals) > 1 else 0.0
        L = csr_matrix(np.diag(np.asarray(A.sum(1)).ravel()) - A)
        lv = np.sort(eigsh(L, k=2, which="SA",
                           return_eigenvectors=False))
        out["algebraic_connectivity"] = float(lv[1]) if len(lv) > 1 else 0.0
        # leading spectrum, used as a fast surrogate for the full spectrum
        ks = min(60, n - 2)
        ev = np.abs(eigsh(A, k=ks, which="LM", return_eigenvectors=False))
        p = ev / (ev.sum() + 1e-12)
        p = p[p > 0]
        out["spectral_entropy"] = float(-np.sum(p * np.log(p)) / np.log(ks))
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ pipeline
def signal_to_hypergraph(x, tau, dim, method="ni_louvain",
                         remove_ratio=REMOVE_RATIO,
                         percentile=REC_PERCENTILE, seed=42):
    """Full single-resolution pipeline for one signal."""
    traj = reconstruct(np.asarray(x, float), int(tau), int(dim))
    A, n = recurrence_network(traj, percentile=percentile, seed=seed)
    if method == "ni_louvain":
        comms = ni_louvain(A, remove_ratio=remove_ratio, seed=seed)
    elif method == "remove_ni_louvain":
        comms = ni_louvain(A, remove_ratio=0.0, seed=seed)
    else:
        comms = louvain_partition(A, seed=seed)
    hg = build_hypergraph(comms, n)
    feats = hypergraph_features(hg)
    return {"adjacency": A, "n_nodes": n, "hypergraph": hg,
            "features": feats}
