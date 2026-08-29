"""Graph topology registry for landmark detection GCN models.

Provides edge-index factory functions for different landmark topologies.
"""

import torch


def make_chain_edge_index(num_landmarks: int) -> torch.Tensor:
    """Bidirectional chain: 0↔1↔2↔...↔(N-1).

    Args:
        num_landmarks: Number of landmarks in the chain.

    Returns:
        Edge index tensor of shape (2, 2*(num_landmarks-1)), dtype torch.long.
    """
    edges = []
    for i in range(num_landmarks - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_wflw_edge_index() -> torch.Tensor:
    """Anatomically correct WFLW 98-point facial landmark graph.

    Encodes facial regions with bidirectional edges:
      - Jaw contour (0-32): chain of 32 edges
      - Left eyebrow (33-41): chain of 8 edges
      - Right eyebrow (42-50): chain of 8 edges
      - Nose bridge (51-54): chain of 3 edges
      - Nose base (55-59): chain of 4 edges
      - Left eye (60-67): closed loop of 8 edges
      - Right eye (68-75): closed loop of 8 edges
      - Outer mouth (76-87): closed loop of 12 edges
      - Inner mouth (88-95): closed loop of 8 edges
      - Left pupil (96): edge to eye center (LM 64)
      - Right pupil (97): edge to eye center (LM 72)

    Total: 93 unique undirected edges → 186 directed edges.

    Returns:
        Edge index tensor of shape (2, 186), dtype torch.long.
    """
    edges = []

    def add_chain(start, end):
        for i in range(start, end):
            edges.append([i, i + 1])
            edges.append([i + 1, i])

    def add_loop(start, end):
        for i in range(start, end):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edges.append([end, start])
        edges.append([start, end])

    def add_edge(u, v):
        edges.append([u, v])
        edges.append([v, u])

    # Jaw contour: 0-32 (32 edges)
    add_chain(0, 32)
    # Left eyebrow: 33-41 (8 edges)
    add_chain(33, 41)
    # Right eyebrow: 42-50 (8 edges)
    add_chain(42, 50)
    # Nose bridge: 51-54 (3 edges)
    add_chain(51, 54)
    # Nose base: 55-59 (4 edges)
    add_chain(55, 59)
    # Left eye: 60-67 (closed loop, 8 edges)
    add_loop(60, 67)
    # Right eye: 68-75 (closed loop, 8 edges)
    add_loop(68, 75)
    # Outer mouth: 76-87 (closed loop, 12 edges)
    add_loop(76, 87)
    # Inner mouth: 88-95 (closed loop, 8 edges)
    add_loop(88, 95)
    # Left pupil (96) → left eye center (64)
    add_edge(96, 64)
    # Right pupil (97) → right eye center (72)
    add_edge(97, 72)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_subsampled_wflw_edge_index(landmark_indices: list) -> torch.Tensor:
    """Create WFLW topology for a subsampled landmark set.

    Preserves the anatomical groupings from the full WFLW graph. For each
    facial region, finds which members survive in the subset and chains them
    in order. Loops remain loops if ≥3 members survive, otherwise become chains.
    Cross-group edges (pupil→eye center) are kept if both endpoints survive.

    The output edge index uses 0-based indices into the subset (not original
    98-point indices). E.g., if landmark_indices=[0,4,8,60,64], then node 0
    in the subset is original LM 0, node 3 is original LM 60, etc.

    Args:
        landmark_indices: Sorted list of original landmark indices in the subset.

    Returns:
        Edge index tensor of shape (2, E), dtype torch.long, with indices into
        the subset array [0, len(landmark_indices)-1].
    """
    # Map from original index → position in subset
    idx_set = set(landmark_indices)
    orig_to_subset = {orig: pos for pos, orig in enumerate(sorted(landmark_indices))}

    # Define anatomical groups: (members, is_loop)
    groups = [
        (list(range(0, 33)), False),     # Jaw contour: chain
        (list(range(33, 42)), False),    # Left eyebrow: chain
        (list(range(42, 51)), False),    # Right eyebrow: chain
        (list(range(51, 55)), False),    # Nose bridge: chain
        (list(range(55, 60)), False),    # Nose base: chain
        (list(range(60, 68)), True),     # Left eye: loop
        (list(range(68, 76)), True),     # Right eye: loop
        (list(range(76, 88)), True),     # Outer mouth: loop
        (list(range(88, 96)), True),     # Inner mouth: loop
    ]

    # Cross-group edges (pupil → eye center)
    cross_edges = [(96, 64), (97, 72)]

    # Anatomical cross-group anchoring: connect eyebrows and nose to
    # nearest eye landmarks for information flow. This prevents isolated
    # nodes in sparse subsets. Uses explicit landmark pairs that are
    # spatially close on the face.
    # Right eyebrow endpoints → right eye landmarks
    cross_edges += [(36, 60), (40, 64)]
    # Left eyebrow endpoints → left eye landmarks
    cross_edges += [(44, 68), (48, 72)]
    # Nose bridge/base → eye landmarks (central anchoring)
    cross_edges += [(52, 60), (52, 68), (56, 60), (56, 72)]

    edges = []

    def add_edge(u_subset, v_subset):
        edges.append([u_subset, v_subset])
        edges.append([v_subset, u_subset])

    for members, is_loop in groups:
        # Find which members of this group survive in the subset
        surviving = [m for m in members if m in idx_set]
        if len(surviving) < 2:
            continue  # Single or no nodes — no edges possible

        # Chain the survivors in order
        for i in range(len(surviving) - 1):
            add_edge(orig_to_subset[surviving[i]], orig_to_subset[surviving[i + 1]])

        # Close the loop if ≥3 survivors and original was a loop
        if is_loop and len(surviving) >= 3:
            add_edge(orig_to_subset[surviving[-1]], orig_to_subset[surviving[0]])

    # Cross-group edges
    for u, v in cross_edges:
        if u in idx_set and v in idx_set:
            add_edge(orig_to_subset[u], orig_to_subset[v])

    if not edges:
        # Fallback: no edges (single isolated nodes)
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_cephalometric_edge_index() -> torch.Tensor:
    """Anatomically correct ISBI 2015 cephalometric 19-point landmark graph.

    Encodes the lateral skull structures with bidirectional edges connecting
    the cranial base, maxilla, mandible, dentition, and soft-tissue landmarks
    into a single connected graph (no isolated node). Landmark order:
      0 Sella, 1 Nasion, 2 Orbitale, 3 Porion, 4 A-point, 5 B-point,
      6 Pogonion, 7 Menton, 8 Gnathion, 9 Gonion, 10 L1 tip, 11 U1 tip,
      12 Upper Lip, 13 Lower Lip, 14 Subnasale, 15 Soft-tissue Pogonion,
      16 PNS, 17 ANS, 18 Articulare.

    The undirected edge list is de-duplicated before each pair is expanded
    into two directed edges [u, v] and [v, u].

    Returns:
        Edge index tensor of shape (2, E), dtype torch.long.
    """
    # Undirected anatomical adjacency pairs (index-mapped from the design).
    undirected = [
        (0, 1), (0, 3), (3, 18), (18, 9), (0, 18),
        (1, 2), (2, 3), (1, 17),
        (17, 4), (4, 16), (16, 2), (17, 11), (16, 0),
        (9, 7), (7, 8), (8, 6), (6, 5), (5, 10), (9, 18),
        (7, 6), (8, 7),
        (11, 10), (4, 11), (5, 10),
        (14, 12), (12, 13), (13, 15), (15, 6),
        (14, 17), (14, 4), (12, 11), (13, 10),
    ]

    # De-duplicate undirected pairs (treat (u,v) and (v,u) as the same edge).
    seen = set()
    edges = []
    for u, v in undirected:
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        edges.append([u, v])
        edges.append([v, u])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_subsampled_cephalometric_edge_index(landmark_indices: list) -> torch.Tensor:
    """Create cephalometric topology for a subsampled landmark set.

    Preserves the anatomical groupings from the full cephalometric graph. For
    each region, finds which members survive in the subset and chains them in
    sorted order (bidirectional). Cross-group anchor edges are kept if both
    endpoints survive.

    The output edge index uses 0-based indices into the subset (not original
    19-point indices). E.g., if landmark_indices=[0, 3, 9, 18], then node 0 in
    the subset is original LM 0, node 3 is original LM 18, etc.

    Args:
        landmark_indices: List of original landmark indices in the subset.

    Returns:
        Edge index tensor of shape (2, E), dtype torch.long, with indices into
        the subset array [0, len(landmark_indices)-1].
    """
    # Map from original index → position in subset
    idx_set = set(landmark_indices)
    orig_to_subset = {orig: pos for pos, orig in enumerate(sorted(landmark_indices))}

    # Define anatomical groups by original index
    groups = [
        [0, 1, 3, 18],       # Cranial base
        [2, 4, 16, 17],      # Maxilla
        [5, 6, 7, 8, 9],     # Mandible
        [10, 11],            # Dentition
        [12, 13, 14, 15],    # Soft tissue
    ]

    # Cross-group anchor edges (kept when both endpoints survive)
    cross_edges = [
        (0, 18), (3, 18), (18, 9), (1, 17), (17, 4), (16, 2), (16, 0),
        (17, 11), (4, 11), (5, 10), (11, 10), (14, 17), (14, 4), (15, 6),
        (12, 11), (13, 10),
    ]

    edges = []

    def add_edge(u_subset, v_subset):
        edges.append([u_subset, v_subset])
        edges.append([v_subset, u_subset])

    for members in groups:
        # Find which members of this group survive in the subset (sorted order)
        surviving = [m for m in sorted(members) if m in idx_set]
        if len(surviving) < 2:
            continue  # Single or no nodes — no edges possible

        # Chain the survivors in order
        for i in range(len(surviving) - 1):
            add_edge(orig_to_subset[surviving[i]], orig_to_subset[surviving[i + 1]])

    # Cross-group anchor edges
    for u, v in cross_edges:
        if u in idx_set and v in idx_set:
            add_edge(orig_to_subset[u], orig_to_subset[v])

    if not edges:
        # Fallback: no edges (single isolated nodes)
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def get_edge_index(topology_name: str, num_landmarks: int = None, landmark_indices: list = None) -> torch.Tensor:
    """Registry lookup for graph topologies.

    Args:
        topology_name: Name of the topology. Supported: 'chain', 'wflw',
                       'cephalometric'.
        num_landmarks: Required when topology_name == 'chain'. Ignored for
                       'wflw' and 'cephalometric'.
        landmark_indices: When provided with 'wflw' topology, creates a
                         subsampled WFLW graph preserving anatomical groupings.
                         When provided with 'cephalometric' topology and fewer
                         than 19 indices, creates a subsampled cephalometric
                         graph preserving anatomical groupings.

    Returns:
        Edge index tensor appropriate for the named topology.

    Raises:
        KeyError: If topology_name is not a known topology.
        ValueError: If topology_name == 'chain' and num_landmarks is None.
    """
    known = ["chain", "wflw", "cephalometric"]

    if topology_name == "chain":
        if num_landmarks is None:
            raise ValueError(
                "num_landmarks is required for the 'chain' topology"
            )
        return make_chain_edge_index(num_landmarks)
    elif topology_name == "wflw":
        if landmark_indices and len(landmark_indices) < 98:
            return make_subsampled_wflw_edge_index(landmark_indices)
        return make_wflw_edge_index()
    elif topology_name == "cephalometric":
        if landmark_indices and len(landmark_indices) < 19:
            return make_subsampled_cephalometric_edge_index(landmark_indices)
        return make_cephalometric_edge_index()
    else:
        raise KeyError(
            f"Unknown topology '{topology_name}'. Known topologies: {known}"
        )
