"""
Dual Graph Clustering for Topic Modeling
Utilities for graph-based clustering with overlap via dual graphs
"""

import igraph as ig
import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from scipy.sparse import csr_matrix

# Optional torch import for GPU operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# PyTorch Setup (Optional)
# =============================================================================

def get_device():
    """
    Automatically select the best available device (CUDA, MPS, or CPU)
    Only works if PyTorch is installed.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Using CPU for any tensor operations.")
        return "cpu"
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Metal (MPS)")
    else:
        device = "cpu"
        print("Using CPU")
    return device


# =============================================================================
# Co-occurrence Matrix Construction
# =============================================================================

def build_cooccurrence_from_coordinations(
    coordinations: List[Tuple[str, str]], 
    weights: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[csr_matrix, Dict[str, int], Dict[int, str]]:
    """
    Build co-occurrence matrix from coordination pairs (e.g., "X og Y")
    
    Args:
        coordinations: List of (word1, word2) tuples
        weights: Optional dict of (word1, word2) -> weight (e.g., PMI scores)
    
    Returns:
        cooc_matrix: Sparse co-occurrence matrix
        word2idx: Dict mapping words to indices
        idx2word: Dict mapping indices to words
    """
    # Build vocabulary
    words = set()
    for w1, w2 in coordinations:
        words.add(w1)
        words.add(w2)
    
    word2idx = {word: idx for idx, word in enumerate(sorted(words))}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Build sparse matrix
    n = len(word2idx)
    row, col, data = [], [], []
    
    for w1, w2 in coordinations:
        i, j = word2idx[w1], word2idx[w2]
        weight = weights.get((w1, w2), 1.0) if weights else 1.0
        
        row.extend([i, j])
        col.extend([j, i])
        data.extend([weight, weight])
    
    cooc_matrix = csr_matrix((data, (row, col)), shape=(n, n))
    
    return cooc_matrix, word2idx, idx2word


# =============================================================================
# Graph Construction
# =============================================================================

def sparse_to_igraph(
    cooc_matrix: csr_matrix, 
    threshold: float = 0.0
) -> ig.Graph:
    """
    Convert sparse co-occurrence matrix to igraph Graph
    
    Args:
        cooc_matrix: Sparse co-occurrence matrix
        threshold: Minimum edge weight to include
    
    Returns:
        igraph.Graph object
    """
    # Get edges above threshold
    cooc_coo = cooc_matrix.tocoo()
    edges = []
    weights = []
    
    for i, j, w in zip(cooc_coo.row, cooc_coo.col, cooc_coo.data):
        if i < j and w > threshold:  # Undirected, so only keep i < j
            edges.append((i, j))
            weights.append(w)
    
    # Create igraph
    g = ig.Graph(n=cooc_matrix.shape[0])
    g.add_edges(edges)
    g.es['weight'] = weights
    
    return g


# =============================================================================
# Dual Graph Construction
# =============================================================================

def build_dual_graph(g: ig.Graph) -> Tuple[ig.Graph, List[Tuple[int, int]]]:
    """
    Build dual graph where edges become nodes
    Two dual nodes are connected if their original edges share a vertex
    
    Args:
        g: Original igraph Graph
    
    Returns:
        g_dual: Dual graph (edges as nodes)
        original_edges: List of (u, v) for each dual node
    """
    num_nodes = g.vcount()
    original_edges = [(e.source, e.target) for e in g.es]
    num_edges = len(original_edges)
    
    # Build node -> edges mapping
    node_to_edges = [[] for _ in range(num_nodes)]
    for edge_id, (u, v) in enumerate(original_edges):
        node_to_edges[u].append(edge_id)
        node_to_edges[v].append(edge_id)
    
    # Build dual edges
    dual_edges = []
    for node_edges in node_to_edges:
        # All pairs of edges meeting at this node
        for i in range(len(node_edges)):
            for j in range(i + 1, len(node_edges)):
                dual_edges.append((node_edges[i], node_edges[j]))
    
    # Create dual graph
    g_dual = ig.Graph(n=num_edges)
    g_dual.add_edges(dual_edges)
    
    return g_dual, original_edges


def build_weighted_line_graph(
    g: ig.Graph,
    weight_attr: str = 'weight',
    method: str = 'adamic_adar',
    threshold: float = 0.0,
    normalize_base_weights: bool = True
) -> Tuple[ig.Graph, List[Tuple[int, int]]]:
    """
    Modul A: Grad-penalisert linjegraf L(G).
    Hver kant i G blir en node i L(G).
    For hver felles node v mellom kant e1=(u, v) og e2=(v, w):
      - Beregn straff: 1.0 / log(deg(v)) hvis deg(v) > 2, ellers 1.0 (ved method='adamic_adar')
      - Eller 1.0 / deg(v) (ved method='resource_allocation')
      - Beregn kantvekt: W = w(e1) * w(e2) * straff
      - Filtrer bort kanter der W <= threshold
    """
    num_nodes = g.vcount()
    original_edges = [(e.source, e.target) for e in g.es]
    num_edges = len(original_edges)

    has_weights = (not normalize_base_weights) and (weight_attr in g.es.attributes())
    edge_weights = list(g.es[weight_attr]) if has_weights else [1.0] * num_edges

    node_to_edges = [[] for _ in range(num_nodes)]
    for edge_id, (u, v) in enumerate(original_edges):
        node_to_edges[u].append(edge_id)
        node_to_edges[v].append(edge_id)

    dual_edges = []
    dual_weights = []

    for v, incident in enumerate(node_to_edges):
        deg_v = len(incident)
        if deg_v < 2:
            continue
        if method == 'adamic_adar':
            penalty = 1.0 / np.log(deg_v) if deg_v > 2 else 1.0
        elif method == 'resource_allocation':
            penalty = 1.0 / deg_v
        else:
            penalty = 1.0

        for i in range(deg_v):
            e1 = incident[i]
            w1 = edge_weights[e1]
            for j in range(i + 1, deg_v):
                e2 = incident[j]
                w = w1 * edge_weights[e2] * penalty
                if w > threshold:
                    dual_edges.append((e1, e2))
                    dual_weights.append(float(w))

    g_dual = ig.Graph(n=num_edges)
    if dual_edges:
        g_dual.add_edges(dual_edges)
        g_dual.es['weight'] = dual_weights

    return g_dual, original_edges


# =============================================================================
# Clustering
# =============================================================================

def cluster_leiden(
    g: ig.Graph, 
    resolution: float = 1.0,
    objective: str = 'modularity'
) -> List[int]:
    """
    Perform Leiden clustering on graph
    
    Args:
        g: igraph Graph
        resolution: Resolution parameter (higher = more clusters)
        objective: 'modularity' or 'CPM'
    
    Returns:
        List of cluster assignments (one per node)
    """
    if objective == 'modularity':
        communities = g.community_leiden(
            objective_function='modularity',
            weights='weight' if 'weight' in g.es.attributes() else None
        )
    else:
        communities = g.community_leiden(
            objective_function='CPM',
            resolution_parameter=resolution,
            weights='weight' if 'weight' in g.es.attributes() else None
        )
    
    return communities.membership


def cluster_louvain(g: ig.Graph) -> List[int]:
    """
    Perform Louvain clustering on graph
    
    Args:
        g: igraph Graph
    
    Returns:
        List of cluster assignments (one per node)
    """
    communities = g.community_multilevel(
        weights='weight' if 'weight' in g.es.attributes() else None
    )
    return communities.membership


# =============================================================================
# Dual Cluster Matching
# =============================================================================

def expand_cluster_with_set_cover(
    adj: Dict[int, Set[int]],
    base_cluster: Set[int],
    candidate_dual_nodes: Optional[Set[int]] = None,
    min_internal_support: int = 2
) -> Tuple[Set[int], Set[int]]:
    """
    Modul B: Set-cover / klyngeutvidelse med forankringstest (Anti-lekkasje domenesperre).
    
    Logikk:
      1. La C være noder i det aktuelle grunn-clusteret (base_cluster).
      2. Identifiser kandidat-noder w ∉ C fra L(G) som henger på noder i C.
      3. For enhver ekstern node w ∉ C:
         Tell antall naboer w har internt i C:
             support = |N_G(w) ∩ C|
      4. Validering:
         - support < min_internal_support -> Dropp w (forhindrer kryss-domene-lekkasje)
         - support >= min_internal_support -> Inkluder w i rand-dekningen.

    Returns:
      expanded_cluster: Utvidet ordmengde C'
      blocked_leaks: Noder fra candidate_dual_nodes som ble stoppet av skjoldet
    """
    expanded = set(base_cluster)
    candidates = set()
    dual_external = set()

    if candidate_dual_nodes is not None:
        for w in candidate_dual_nodes:
            if w not in base_cluster:
                candidates.add(w)
                dual_external.add(w)

    for u in base_cluster:
        for w in adj.get(u, ()):
            if w not in base_cluster:
                candidates.add(w)

    blocked_leaks = set()
    for w in candidates:
        support = len(adj.get(w, set()) & base_cluster)
        if support >= min_internal_support:
            expanded.add(w)
        elif w in dual_external:
            blocked_leaks.add(w)

    return expanded, blocked_leaks


def deduplicate_expanded_clusters(
    clusters: Dict[int, Set[int]],
    jaccard_threshold: float = 0.85
) -> Dict[int, Set[int]]:
    """
    Modul C: Klynge-deduplisering (Jaccard-sanering).
    Hvis to utvidede klynger C_i og C_j har Jaccard-likhet
    |C_i ∩ C_j| / |C_i ∪ C_j| >= jaccard_threshold (standard 0.85),
    slås de sammen til én enhet.
    """
    ordered_ids = sorted(clusters.keys(), key=lambda cid: len(clusters[cid]), reverse=True)
    consumed = set()
    deduped: Dict[int, Set[int]] = {}

    for i, cid_i in enumerate(ordered_ids):
        if cid_i in consumed:
            continue
        merged_set = set(clusters[cid_i])
        for cid_j in ordered_ids[i + 1:]:
            if cid_j in consumed:
                continue
            other_set = clusters[cid_j]
            union_len = len(merged_set | other_set)
            jacc = len(merged_set & other_set) / union_len if union_len > 0 else 0.0
            if jacc >= jaccard_threshold:
                consumed.add(cid_j)
                merged_set |= other_set
        deduped[cid_i] = merged_set

    return deduped


def compute_cluster_entropy_and_innovation(
    clusters: Dict[int, Set[int]]
) -> Dict[int, Dict[str, float]]:
    """
    Modul D: Beregn normalisert medlemskapsentropi H_norm(C_i) og innovasjonsgrad I(C_i)
    for hver utvidet klynge C_i:
      P(C_j | C_i) = |C_i ∩ C_j| / sum_{k != i} |C_i ∩ C_k|
      H(C_i) = - sum_{j != i} P(C_j | C_i) * log2(P(C_j | C_i))
      H_norm(C_i) = H(C_i) / log2(K - 1)
      I(C_i) = |C_i \\ union_{j != i} C_j| / |C_i|
    """
    K = len(clusters)
    metrics: Dict[int, Dict[str, float]] = {}

    for cid_i, set_i in clusters.items():
        overlaps = []
        sum_overlap = 0
        union_others = set()

        for cid_j, set_j in clusters.items():
            if cid_j == cid_i:
                continue
            union_others |= set_j
            inter = len(set_i & set_j)
            if inter > 0:
                overlaps.append(inter)
                sum_overlap += inter

        H = 0.0
        if sum_overlap > 0:
            for inter in overlaps:
                p = inter / sum_overlap
                if p > 0:
                    H -= p * np.log2(p)

        H_norm = float(H / np.log2(K - 1)) if K > 2 else 0.0
        unique_words = set_i - union_others
        I = len(unique_words) / len(set_i) if len(set_i) > 0 else 0.0

        metrics[cid_i] = {
            'H_norm': H_norm,
            'I': float(I),
            'unique_count': len(unique_words),
            'size': len(set_i),
        }

    return metrics


def prune_hub_clusters_by_entropy(
    clusters: Dict[int, Set[int]],
    adj: Optional[Dict[int, Set[int]]] = None,
    tau_H: float = 0.75,
    tau_I: float = 0.10
) -> Tuple[Dict[int, Set[int]], List[int]]:
    """
    Modul D: Detekter og saner uoppløste parasitt-/hub-klynger (som f.eks. klassisk-skøyting)
    der H_norm(C_i) > tau_H (standard 0.75) og I(C_i) < tau_I (standard 0.10).
    Eventuelle unike rest-ord omplasseres til sterkeste naboklynge for å bevare 100 % dekning.
    """
    active = {cid: set(words) for cid, words in clusters.items()}
    pruned_ids: List[int] = []

    while len(active) > 2:
        metrics = compute_cluster_entropy_and_innovation(active)
        worst_cid = None
        worst_score = -1.0

        for cid, m in metrics.items():
            if m['H_norm'] > tau_H and m['I'] < tau_I:
                score = m['H_norm'] - m['I']
                if score > worst_score:
                    worst_score = score
                    worst_cid = cid

        if worst_cid is None:
            break

        victim_words = active.pop(worst_cid)
        pruned_ids.append(worst_cid)

        union_remaining = set().union(*active.values()) if active else set()
        orphans = victim_words - union_remaining
        for u in orphans:
            best_target = next(iter(active.keys()))
            best_support = -1
            if adj is not None:
                u_nbrs = adj.get(u, set())
                for cand_id, cand_words in active.items():
                    sup = len(u_nbrs & cand_words)
                    if sup > best_support:
                        best_support = sup
                        best_target = cand_id
            active[best_target].add(u)

    return active, pruned_ids


def match_clusters_with_overlap(
    communities_original: List[int],
    communities_dual: List[int],
    original_edges: List[Tuple[int, int]],
    strategy: str = 'shielded',
    k3_cliques: Optional[List[Set[int]]] = None,
    min_internal_support: int = 2,
    dedup_jaccard_threshold: float = 0.85
) -> Dict[int, Set[int]]:
    """
    Match original clusters with dual clusters (and anti-leakage shield) to add controlled overlap
    
    Args:
        communities_original: Cluster assignments for original graph nodes
        communities_dual: Cluster assignments for dual graph nodes (edges)
        original_edges: List of (u, v) for each dual node
        strategy: 'shielded' (Modul B + C anti-leakage, default), 'clique_hybrid', 'best_overlap', or 'threshold'
        k3_cliques: Optional list of 3-clique communities (as sets of node indices) for 'clique_hybrid'
        min_internal_support: Minimum internal neighbors in C required for external node w (default: 2)
        dedup_jaccard_threshold: Jaccard threshold for merging collapsed clusters (default: 0.85)
    
    Returns:
        Dict mapping cluster_id -> set of node_ids (with overlap)
    """
    # Map dual clusters to nodes
    dual_cluster_to_nodes = {}
    adj = defaultdict(set)
    for edge_id, dual_cluster in enumerate(communities_dual):
        if dual_cluster not in dual_cluster_to_nodes:
            dual_cluster_to_nodes[dual_cluster] = set()
        u, v = original_edges[edge_id]
        dual_cluster_to_nodes[dual_cluster].add(u)
        dual_cluster_to_nodes[dual_cluster].add(v)
        adj[u].add(v)
        adj[v].add(u)
    
    # Build overlapping communities
    overlapping_communities = {}
    
    for orig_cluster in set(communities_original):
        # Get nodes in this original cluster
        orig_nodes = set(i for i, c in enumerate(communities_original) if c == orig_cluster)
        
        best_dual = None
        best_overlap = 0
        for dual_cluster, dual_nodes in dual_cluster_to_nodes.items():
            overlap = len(orig_nodes & dual_nodes)
            if overlap > best_overlap:
                best_overlap = overlap
                best_dual = dual_cluster

        best_dual_nodes = dual_cluster_to_nodes[best_dual] if best_dual is not None else set()

        if strategy == 'shielded':
            combined, _ = expand_cluster_with_set_cover(
                adj=adj,
                base_cluster=orig_nodes,
                candidate_dual_nodes=best_dual_nodes,
                min_internal_support=min_internal_support
            )
            overlapping_communities[orig_cluster] = combined

        elif strategy in ('best_overlap', 'clique_hybrid'):
            combined = orig_nodes | best_dual_nodes if best_dual is not None else orig_nodes.copy()
            
            # Clique-informed bridge attachment: if a 3-clique community has >= 2 core nodes in orig_nodes,
            # attach any node w in that clique that forms at least one triangle (>= 2 links) with the core
            if strategy == 'clique_hybrid' and k3_cliques:
                for cl in k3_cliques:
                    core_in_orig = cl & orig_nodes
                    if len(core_in_orig) < 2:
                        continue
                    for w in cl:
                        if w not in orig_nodes and len(adj[w] & core_in_orig) >= 2:
                            combined.add(w)
            
            overlapping_communities[orig_cluster] = combined
        
        elif strategy == 'threshold':
            # Add all dual clusters with Jaccard > threshold
            threshold = 0.3
            combined = orig_nodes.copy()
            
            for dual_cluster, dual_nodes in dual_cluster_to_nodes.items():
                jaccard = len(orig_nodes & dual_nodes) / len(orig_nodes | dual_nodes)
                if jaccard > threshold:
                    combined |= dual_nodes
            
            overlapping_communities[orig_cluster] = combined
    
    if strategy == 'shielded' and dedup_jaccard_threshold < 1.0:
        overlapping_communities = deduplicate_expanded_clusters(
            overlapping_communities,
            jaccard_threshold=dedup_jaccard_threshold
        )

    return overlapping_communities


def compute_clique_cluster_spread(
    clique_nodes: Set[str],
    louvain_clusters: Dict[str, List[str]]
) -> List[Dict]:
    """
    Compute the distribution (spread) of a k-clique community across Louvain clusters.
    Classifies into:
      - 'Ren delmengde' (1 cluster, 100%)
      - 'Kjerne + bro' (>= 75% in one cluster + bridge word(s))
      - 'Tverrgående klikk' (< 75% dominant, spans multiple clusters)
    """
    total = max(1, len(clique_nodes))
    spread = []
    for cl_name, cl_words in louvain_clusters.items():
        shared = [w for w in cl_words if w in clique_nodes]
        if shared:
            spread.append({
                'cluster': cl_name,
                'count': len(shared),
                'share': len(shared) / total,
                'words': shared
            })
    spread.sort(key=lambda x: x['count'], reverse=True)
    return spread


def get_node_to_clusters(
    overlapping_communities: Dict[int, Set[int]]
) -> Dict[int, List[int]]:
    """
    Convert cluster->nodes mapping to node->clusters mapping
    
    Args:
        overlapping_communities: Dict of cluster_id -> set of node_ids
    
    Returns:
        Dict of node_id -> list of cluster_ids
    """
    node_to_clusters = defaultdict(list)
    for cluster_id, nodes in overlapping_communities.items():
        for node in nodes:
            node_to_clusters[node].append(cluster_id)
    return dict(node_to_clusters)


# =============================================================================
# Analysis & Metrics
# =============================================================================

def analyze_overlap(
    node_to_clusters: Dict[int, List[int]],
    idx2word: Optional[Dict[int, str]] = None
) -> Dict:
    """
    Analyze the overlap statistics
    
    Args:
        node_to_clusters: Dict of node_id -> list of cluster_ids
        idx2word: Optional mapping of node indices to words
    
    Returns:
        Dict with overlap statistics
    """
    overlap_counts = [len(clusters) for clusters in node_to_clusters.values()]
    
    stats = {
        'mean_clusters_per_node': np.mean(overlap_counts),
        'max_clusters_per_node': np.max(overlap_counts),
        'nodes_with_overlap': sum(1 for c in overlap_counts if c > 1),
        'total_nodes': len(node_to_clusters),
        'overlap_percentage': 100 * sum(1 for c in overlap_counts if c > 1) / len(node_to_clusters)
    }
    
    # Most ambiguous nodes
    if idx2word:
        ambiguous = sorted(
            node_to_clusters.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )[:10]
        stats['most_ambiguous'] = [
            (idx2word[node_id], clusters) 
            for node_id, clusters in ambiguous
        ]
    
    return stats


# =============================================================================
# Complete Pipeline
# =============================================================================

def dual_clustering_pipeline(
    cooc_matrix: csr_matrix,
    threshold: float = 0.0,
    resolution: float = 1.0,
    clustering_method: str = 'leiden',
    matching_strategy: str = 'best_overlap'
) -> Tuple[Dict[int, Set[int]], Dict[int, List[int]], Dict]:
    """
    Complete pipeline: co-occurrence -> dual graph -> overlapping clusters
    
    Args:
        cooc_matrix: Sparse co-occurrence matrix
        threshold: Minimum edge weight
        resolution: Resolution parameter for clustering
        clustering_method: 'leiden' or 'louvain'
        matching_strategy: 'best_overlap' or 'threshold'
    
    Returns:
        overlapping_communities: cluster_id -> set of nodes
        node_to_clusters: node_id -> list of clusters
        stats: Overlap statistics
    """
    print("Building original graph...")
    g = sparse_to_igraph(cooc_matrix, threshold=threshold)
    print(f"  Nodes: {g.vcount()}, Edges: {g.ecount()}")
    
    print("Clustering original graph...")
    if clustering_method == 'leiden':
        communities_orig = cluster_leiden(g, resolution=resolution)
    else:
        communities_orig = cluster_louvain(g)
    print(f"  Found {len(set(communities_orig))} communities")
    
    print("Building dual graph...")
    g_dual, original_edges = build_dual_graph(g)
    print(f"  Dual nodes: {g_dual.vcount()}, Dual edges: {g_dual.ecount()}")
    
    print("Clustering dual graph...")
    if clustering_method == 'leiden':
        communities_dual = cluster_leiden(g_dual, resolution=resolution)
    else:
        communities_dual = cluster_louvain(g_dual)
    print(f"  Found {len(set(communities_dual))} dual communities")
    
    print("Matching clusters...")
    overlapping = match_clusters_with_overlap(
        communities_orig, 
        communities_dual, 
        original_edges,
        strategy=matching_strategy
    )
    
    node_to_clusters = get_node_to_clusters(overlapping)
    stats = analyze_overlap(node_to_clusters)
    
    print(f"\nResults:")
    print(f"  Overlap: {stats['overlap_percentage']:.1f}% of nodes")
    print(f"  Avg clusters per node: {stats['mean_clusters_per_node']:.2f}")
    print(f"  Max clusters per node: {stats['max_clusters_per_node']}")
    
    return overlapping, node_to_clusters, stats
