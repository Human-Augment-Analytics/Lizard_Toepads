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


def get_edge_index(topology_name: str, num_landmarks: int = None) -> torch.Tensor:
    """Registry lookup for graph topologies.

    Args:
        topology_name: Name of the topology. Supported: 'chain', 'wflw'.
        num_landmarks: Required when topology_name == 'chain'. Ignored for 'wflw'.

    Returns:
        Edge index tensor appropriate for the named topology.

    Raises:
        KeyError: If topology_name is not a known topology.
        ValueError: If topology_name == 'chain' and num_landmarks is None.
    """
    known = ["chain", "wflw"]

    if topology_name == "chain":
        if num_landmarks is None:
            raise ValueError(
                "num_landmarks is required for the 'chain' topology"
            )
        return make_chain_edge_index(num_landmarks)
    elif topology_name == "wflw":
        return make_wflw_edge_index()
    else:
        raise KeyError(
            f"Unknown topology '{topology_name}'. Known topologies: {known}"
        )
