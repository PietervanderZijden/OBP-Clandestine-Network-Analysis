# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 14:30:34 2026

@author: athin
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import SpectralClustering
import scipy.io
import os
from sklearn.metrics import adjusted_rand_score
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



### Flow-based role similarity part 1

def prepare_adjacency(A: sp.spmatrix, sym_tol: float = 1e-12) -> tuple[sp.csr_matrix,dict]:
    """
    returns:
        A_dir: csr adjacency used as directed adjacency for the method
        meta: info about what we detected/changed for Dss transparency
    """
    
    if not sp.isspmatrix(A):
        A = sp.csr_matrix(A)
        
    A = A.tocsr().astype(float)
    
    # remove self-loops
    A.setdiag(0)
    A.eliminate_zeros()
    
    # nonnegative weights (flow interpretation)
    if A.data.size and (A.data < 0).any():
        raise ValueError("Adjacency contains negative weights; flow-based path counts assume nonnegative edges")
        
    
    #detect "undirected" by symmetry
    diff = (A - A.T).tocoo()
    is_symmetric = (diff.data.size == 0) or (np.max(np.abs(diff.data)) <= sym_tol)
    
    meta = {
        "is_symmetric_input": bool(is_symmetric),
        "treated_as_directed": True,
        "sym_tol": sym_tol
        }
    
    A_dir = A
    
    return A_dir, meta



### compute spectral radius lambda1

def spectral_radius(A: sp.csr_matrix) -> float:
    # largest real eigenvalue for nonnegative adjacency 
    vals = spla.eigs(A, k=1,which="LR", return_eigenvectors=False)
    lam1 = float(np.real(vals[0]))
    if lam1 <= 0:
        #if no edges lam1 might be 0, method degenerates
        raise ValueError("Largest eigenvaue <= 0. Graph may be empty or invalid for flow profiling")
    
    return lam1



### construct flow porfile x_i

def flow_profile_matrix(A: sp.csr_matrix, alpha: float, kmax: int) -> np.ndarray:
    if not(0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
        
    N = A.shape[0]
    lam1 = spectral_radius(A)
    beta = alpha / lam1
    
    ones = np.ones(N)
    
    X_in = np.zeros((N,kmax), dtype=float)
    X_out = np.zeros((N,kmax), dtype=float)
    
    v_in = ones.copy()
    v_out = ones.copy()
    
    AT = A.T.tocsr()
    
    for k in range(kmax):
        v_in = beta * (AT @ v_in)
        v_out = beta *(A @ v_out)
        X_in[:,k] = v_in
        X_out[:,k] = v_out
        
    
    X = np.hstack([X_in,X_out])
    
    return X


###node similarity

def cosine_similarity_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms,eps)
    Xn = X / norms
    Y = Xn@ Xn.T
    return np.clip(Y, 0.0, 1.0)


### cluster

def gaussian_affinity(Y: np.ndarray, sigma: float | None = None) -> np.ndarray:
    D = 1.0 - Y
    if sigma is None:
        off = D[~np.eye(D.shape[0], dtype=bool)]
        sigma = max(np.median(off), 1e-6)
        
    W = np.exp(-(D**2) / (2.0 * sigma**2))
    np.fill_diagonal(W, 0.0)
    
    return W

def role_clustering_from_similarity(Y: np.ndarray, n_roles: int, seed: int = 0) -> np.ndarray:
    W = gaussian_affinity(Y)
    sc = SpectralClustering(
        n_clusters=n_roles,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed
        )
    
    return sc.fit_predict(W)


### THE method

def flow_based_role_similarity(A_input: sp.spmatrix, alpha: float, kmax: int, n_roles: int):
    A, meta = prepare_adjacency(A_input)
    X = flow_profile_matrix(A, alpha=alpha, kmax=kmax)
    Y = cosine_similarity_matrix(X)
    labels = role_clustering_from_similarity(Y, n_roles=n_roles, seed=0)
    return {
        "A_used": A,
        "meta": meta,
        "X_flow_profiles": X,
        "Y_similarity": Y,
        "role_labels": labels,
        }
    

A = scipy.io.mmread("clandestine_network_example.mtx")
A = A.tocsr()


### multi-scale exploration 
alphas = np.linspace(0.10,0.99,10)

labels_by_alpha = {}

for a in alphas:
    res = flow_based_role_similarity(A_input=A, alpha=float(a), kmax=10, n_roles=4)
    labels_by_alpha[float(a)] = res["role_labels"]
    
    
### measure stability between consecutive a values

stability = []
alphas_sorted = sorted(labels_by_alpha.keys())

for i in range(1, len(alphas_sorted)):
    a0, a1 = alphas_sorted[i-1], alphas_sorted[i]
    ari = adjusted_rand_score(labels_by_alpha[a0], labels_by_alpha[a1])
    stability.append((a0,a1,ari))
    
#print("Alpha stability (ARI between consecutive alphas):")
#for a0,a1,ari in stability:
#    print(f"{a0:.2f} -> {a1:.2f}: ARI = {ari:.3f}")
    
alpha_star = 0.75
kmax = 10


res_star = flow_based_role_similarity(A_input=A, alpha=alpha_star, kmax=10, n_roles=4)

role_labels_star = res_star["role_labels"]
X_star = res_star["X_flow_profiles"]

def summarize_flow_roles(X,labels,kmax):
    df = pd.DataFrame({"node": np.arange(X.shape[0]), "role": labels})
    in_part = X[:, :kmax]
    out_part = X[:, kmax:]
    
    df["in_total"] = in_part.sum(axis=1)
    df["out_total"] = out_part.sum(axis=1)
    df["net_flow"] = df["out_total"] - df["in_total"]
    
    summary = df.groupby("role").agg(
        n_nodes=("node","count"),
        in_total_mean=("in_total","mean"),
        out_total_mean=("out_total","mean"),
        net_flow_mean=("net_flow","mean")
        ).reset_index()
    
    return df,summary
    

df_nodes, df_roles_summary = summarize_flow_roles(X_star, role_labels_star, kmax)
#print(df_roles_summary)

### map cluster id to interpretable names

def assign_role_names_by_embeddedness(df_nodes):
    role_order =(
        df_nodes.groupby("role")["embeddedness_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
        )
    
    role_names = [
        "Core-like (high embeddedness)",
        "Intermediate (moderate embeddedness)",
        "Peripheral (low embeddedness)",
        "Extreme peripheral / near-isolated"
        ]
    
    return {r: role_names[i] for i,r in enumerate(role_order)}


df_nodes["embeddedness_score"] = df_nodes["in_total"] + df_nodes["out_total"]
role_name_map = assign_role_names_by_embeddedness(df_nodes)
df_nodes["role_name"] = df_nodes["role"].map(role_name_map)


### robustness check for alpha =0.3

res_local = flow_based_role_similarity(A_input=A , alpha=0.3, kmax=10, n_roles=4)
labels_local = res_local["role_labels"]
X_local = res_local["X_flow_profiles"]

df_nodes_local, summary_local = summarize_flow_roles(X_local, labels_local , kmax)
#print(summary_local)


### Distance-Based Method 

def distance_based_roles(
        A_input: sp.spmatrix,
        core_quantile: float = 0.90,
        directed: str = "auto"
        ) -> tuple[pd.DataFrame, dict]:
    A, meta_prep = prepare_adjacency(A_input)
    
    if directed == "auto":
        is_undirected = meta_prep["is_symmetric_input"]
        use_directed = not is_undirected
    else:
        use_directed = bool(directed)
        
    # build graph
    if use_directed:
        G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
        # for core in directed use total degree
        deg = np.array([d for _, d in G.degree()], dtype=float)
    else:
        G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
        deg = np.array([d for _, d in G.degree()], dtype=float)
        
    
    n = A.shape[0]
    if n == 0:
        raise ValueError("Empty adjacency matrix")
        
    # choose core nodes; top 1-core quantile fraction by degree
    thresh = np.quantile(deg, core_quantile)
    core_nodes = np.where(deg >= thresh)[0].tolist()
    
    # edge case; if all degrees are equal, quantile may select too many
    # ensure core is not empty if there is at least one edge
    if len(core_nodes) == 0 and A.nnz > 0:
        core_nodes = [int(np.argmax(deg))]
        
    # compute distance to the core
    dist_dict = {}
    
    if len(core_nodes) > 0:
        G_temp = G.copy()
        super_source = "__CORE__"
        
        G_temp.add_node(super_source)
        for c in core_nodes:
            G_temp.add_edge(super_source, c)
            
        dist_dict_raw = nx.single_source_shortest_path_length(G_temp, super_source)
        
        # remove souper source and adjust the distance
        for node, d in dist_dict_raw.items():
            if node != super_source:
                dist_dict[int(node)] = d - 1
        
    dist_to_core = np.full(n,np.inf)
    for node, d in dist_dict.items():
        dist_to_core[int(node)] =  float(d)
        
    dist_from_core = dist_to_core.copy()
    
    if use_directed and len(core_nodes) > 0:
        G_rev = G.reverse(copy=True)
        
        G_temp2 = G_rev.copy()
        super_source2 = "__CORE__"
        G_temp2.add_node(super_source2)
        for c in core_nodes:
            G_temp2.add_edge(super_source2, c)
        
        dist_raw2 = nx.single_source_shortest_path_lenght(G_temp2,super_source2)
        
        dist_to_core2 = np.full(n,np.inf)
        for node, d in dist_raw2.items():
            if node != super_source2:
                dist_to_core2[int(node)] = float(d-1)
                
    else:
        dist_to_core2 = dist_from_core
        
        
    # assign roles
    roles = np.empty(n, dtype=object)
    for i in range(n):
        if i in core_nodes:
            roles[i] = "High-degree Core"
        elif np.isfinite(dist_to_core[i]):
            roles[i] = "Near high-degree core" if dist_to_core[i] == 1 else "Peripheral"
            
        else:
            roles[i] = "Isolated"
            
    df = pd.DataFrame({
        "node": np.arange(n),
        "degree": deg,
        "dist_from_core": dist_from_core,
        "dist_to_core": dist_to_core,
        "distance_role": roles
        })
    
    meta = {
        "directed_used_for_distance": use_directed,
        "core_quantile": core_quantile,
        "core_threshold_degree": float(thresh),
        "n_core": len(core_nodes),
        "core_nodes": core_nodes,
        **meta_prep
        }
    
    return df, meta

df_dist, meta_dist = distance_based_roles(A, core_quantile=0.90, directed="auto")

#print(df_dist["distance_role"].value_counts())
#print("\n")
#print(meta_dist["n_core"], meta_dist["core_nodes"][:10])


### cetrality profile roles clustering

def name_centrality_clusters(df):
    scores =(
        df.groupby("centrality_role")[["degree","betweenness","eigenvector","katz"]]
        .mean()
        .assign(score=lambda g: g["degree"] + g["katz"] + g["eigenvector"] + 2*g["betweenness"])
        .sort_values("score",ascending=False)
        )
    
    ordered = scores.index.tolist()
    names = ["Hub-like","Influential","Bridge-like","Peripheral"]
    names = names[:len(ordered)]
    
    return dict(zip(ordered, names))

def centrality_profile_roles(
        A_input: sp.spmatrix,
        n_roles: int = 4,
        directed: str = "auto",
        use_weights: bool = False,
        seed: int = 0,
        katz_alpha: float = 0.05,
        katz_beta: float = 1.0
        ) -> tuple[pd.DataFrame, dict]:
    A, meta_prep = prepare_adjacency(A_input)
    
    
    if directed == "auto":
        is_undirected = meta_prep["is_symmetric_input"]
        use_directed = not is_undirected
        
    else:
        use_directed = bool(directed)
        
    #build graph
    if use_directed:
        G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    else:
        G = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
        
    n = A.shape[0]
    if n == 0:
        raise ValueError("Empty adjacency matrix")
        
    weight_attr = "weight" if use_weights else None
    
    # degree
    if use_weights:
        deg = np.array([d for _, d in G.degree(weight="weight")], dtype=float)
    else:
        deg = np.array([d for _, d in G.degree()], dtype=float)
        
    # betweenness (only if weihts represents distances)
    bet = nx.betweenness_centrality(G, weight=weight_attr, normalized=True)
    bet = np.array([bet[i] for i in range(n)], dtype=float)
    
    
    # eigenvector centrality
    eig_failed = False
    try:
        eig_dict = nx.eigenvector_centrality_numpy(G, weight=weight_attr)
        eig = np.array([eig_dict[i] for i in range(n)], dtype=float)
        
    except Exception:
        try:
            eig_dict = nx.eigenvector_centrality(G,weight=weight_attr, max_iter=2000, tol=1e-6)
            eig = np.array([eig_dict[i] for i in range(n)], dtype=float)
            eig_failed = False
        except Exception:
            eig = np.zeros(n, dtype=float)
            eig_failed = True
    
    
    # Katz centrality
    lam1 = spectral_radius(A)
    if not (0 < katz_alpha < 1.0 / lam1):
        raise ValueError(f"katz_attenuation must satisfy 0 < a < 1/lam1 = {1.0/lam1:.6g}")
        
    a_eff = katz_alpha
    I = sp.eye(n, format="csr")
    b = np.ones(n) * katz_beta
    
    #solve (I-a_eff A)x =b
    try:
        katz = spla.spsolve(I - (a_eff * A), b)
        katz = np.asarray(katz).ravel()
        
        if np.max(np.abs(katz)) > 0:
            katz = katz / np.max(np.abs(katz))
            
    except Exception:
        katz = np.zeros(n, dtype=float)
        
        
    # profile matrix
    X = np.vstack([deg, bet, eig, katz]).T
    # standarize
    Xs = StandardScaler().fit_transform(X)
    
    km = KMeans(n_clusters=n_roles, n_init=20, random_state=seed)
    labels = km.fit_predict(Xs)
    
    df = pd.DataFrame({
        "node": np.arange(n),
        "degree": deg,
        "betweenness": bet,
        "eigenvector": eig,
        "katz": katz,
        "centrality_role": labels
        })
    
    role_name_map = name_centrality_clusters(df)
    df["centrality_role_name"] = df["centrality_role"].map(role_name_map)
    
    
    meta = {
        "n_roles": n_roles,
        "directed_used": use_directed,
        "use_weights": use_weights,
        "eigenvector_failed": eig_failed,
        "katz_alpha": katz_alpha,
        "katz_beta": katz_beta,
        "features": ["degree", "betweenness","eigenvector","katz"],
        **meta_prep
        }
    
    return df, meta


df_cent, meta_cent = centrality_profile_roles(A, n_roles=4, directed="auto", use_weights=False,seed=0)
#print(df_cent["centrality_role"].value_counts())

cent_summary = df_cent.groupby("centrality_role")[["degree","betweenness","eigenvector","katz"]].mean()
#print(cent_summary)


### neighborhood overlap

def neighborhood_overlap_similarity(
        A: sp.csr_matrix,
        directed_mode: str = "auto",
        similarity: str = "jaccard",
        eps: float = 1e-12
        ) -> np.ndarray:
    B = A.copy().tocsr()
    if B.nnz > 0:
        B.data[:] = 1.0 
        
    diff = (B - B.T).tocoo()
    is_symmetric = (diff.data.size == 0) or (np.max(np.abs(diff.data)) <= 1e-12)
    
    if directed_mode == "auto":
        mode = "out" if is_symmetric else "both"
    else:
        mode = directed_mode.lower()
        
    #build feature matrix X
    if mode == "out":
        X = B
    elif mode == "in":
        X = B.T.tocsr()
    elif mode == "both":
        X = sp.hstack([B, B.T.tocsr()], format="csr")
    else:
        raise ValueError("directed_mode must be one of {'auto','out','in','both'}")
        
    n = X.shape[0]
    
    if similarity.lower() == "cosine":
        #cosine for sparse X
        row_norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
        row_norms = np.maximum(row_norms, eps)
        Xn = sp.diags(1.0 / row_norms) @ X
        Y = (Xn @ Xn.T).toarray()
        return np.clip(Y, 0.0, 1.0)
    
    elif similarity.lower() == "jaccard":
        # jaccard for binary
        inter = (X @ X.T).toarray()
        deg = np.asarray(X.sum(axis=1)).ravel()
        union = deg[:, None] + deg[None,:] - inter
        Y = inter / np.maximum(union,eps)
        np.fill_diagonal(Y, 1.0)
        return np.clip(Y,0.0, 1.0)
    
    else:
        raise ValueError("similarity must be one of {'jaccard','cosine'}")
        

def structual_equivalence_roles(
        A_input: sp.spmatrix,
        n_roles: int = 4,
        directed: str = "auto",
        directed_mode: str = "auto",
        similarity: str = "jaccard",
        seed: int = 0
        ) -> tuple[pd.DataFrame, dict]:
    A ,meta_prep = prepare_adjacency(A_input)
    
    if directed == "auto":
        is_undirected = meta_prep["is_symmetric_input"]
        use_directed = not is_undirected
        
    else:
        use_directed = bool(directed)
        
    mode_used = directed_mode
    
    if not use_directed and directed_mode in ["in","both"]:
        mode_used = "out"
        
    # similarity matrix
    Y = neighborhood_overlap_similarity(A, directed_mode=mode_used if use_directed else "out", similarity=similarity)
    
    labels = role_clustering_from_similarity(Y, n_roles=n_roles, seed=seed)
    
    n = A.shape[0]
    
    deg = np.asarray((A > 0).sum(axis=1)).ravel()
    
    df = pd.DataFrame({
        "node": np.arange(n),
        "degree": deg,
        "overlap_role": labels
        })
    
    meta = {
        "n_roles": n_roles,
        "directed_used": use_directed,
        "directed_mode_used": (mode_used if use_directed else "out"),
        "similarity": similarity,
        **meta_prep
        }
    return df, meta


df_overlap, meta_overlap = structual_equivalence_roles(
    A, n_roles=4, directed="auto",directed_mode="auto",similarity="jaccard",seed=0
    )
#print(df_overlap["overlap_role"].value_counts())
        
        