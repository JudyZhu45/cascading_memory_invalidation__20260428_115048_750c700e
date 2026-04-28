"""
Memory graph data structures and cascade logic.

Implements:
- MemoryNode: a single memory item with content, timestamp, and weight
- MemoryGraph: NetworkX-based graph with typed edges (LOCATED_IN, CONFLICTS_WITH)
- Cascade methods: propagate invalidation via BFS with decay
"""

import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import json


@dataclass
class MemoryNode:
    """Represents a single memory item."""
    node_id: str
    content: str
    memory_type: str        # 'fact', 'preference', 'location', 'activity'
    timestamp: datetime
    weight: float = 1.0     # Confidence/relevance weight; decreases during cascade
    location_tags: List[str] = field(default_factory=list)  # Locations mentioned
    session_idx: int = 0    # Which conversation session this came from

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "weight": self.weight,
            "location_tags": self.location_tags,
            "session_idx": self.session_idx,
        }


class MemoryGraph:
    """
    Directed memory graph with typed dependency edges.

    Edge types:
      - LOCATED_IN: memory depends on a location node
      - CONFLICTS_WITH: semantic conflict between two memories
      - PRECEDES: temporal ordering between memories
    """

    def __init__(self, dialogue_id: str):
        self.dialogue_id = dialogue_id
        self.G = nx.DiGraph()
        self.nodes: Dict[str, MemoryNode] = {}
        self.location_root: Optional[str] = None  # Current primary location node

    def add_node(self, node: MemoryNode) -> None:
        self.nodes[node.node_id] = node
        self.G.add_node(
            node.node_id,
            content=node.content,
            memory_type=node.memory_type,
            weight=node.weight,
            timestamp=node.timestamp.isoformat(),
        )

    def add_edge(self, src_id: str, tgt_id: str, edge_type: str, strength: float = 1.0) -> None:
        self.G.add_edge(src_id, tgt_id, edge_type=edge_type, strength=strength)

    def set_location_root(self, node_id: str) -> None:
        self.location_root = node_id

    # ------------------------------------------------------------------
    # Cascade 1: Structural LOCATED_IN cascade
    # ------------------------------------------------------------------

    def structural_cascade(self, new_location_node_id: str, decay: float = 0.7,
                           max_depth: int = 3) -> Dict[str, float]:
        """
        When root location changes, down-weight all memories linked via LOCATED_IN
        to the OLD location.

        Returns:
            Dict mapping node_id -> new_weight for affected nodes
        """
        if self.location_root is None:
            return {}

        old_root = self.location_root
        affected: Dict[str, float] = {}

        # BFS from old root via LOCATED_IN edges
        queue = [(old_root, 0, 1.0)]  # (node_id, depth, cumulative_decay)
        visited: Set[str] = {old_root}

        while queue:
            current, depth, cum_decay = queue.pop(0)
            if depth > max_depth:
                continue

            # Down-weight the current node
            if current != old_root and current in self.nodes:
                new_weight = self.nodes[current].weight * (1.0 - cum_decay * decay)
                new_weight = max(0.0, new_weight)
                self.nodes[current].weight = new_weight
                self.G.nodes[current]['weight'] = new_weight
                affected[current] = new_weight

            # Traverse LOCATED_IN successors
            for succ in self.G.successors(current):
                edge_data = self.G.edges[current, succ]
                if edge_data.get('edge_type') == 'LOCATED_IN' and succ not in visited:
                    visited.add(succ)
                    edge_strength = edge_data.get('strength', 1.0)
                    queue.append((succ, depth + 1, cum_decay * edge_strength))

        # Update location root
        self.location_root = new_location_node_id
        return affected

    # ------------------------------------------------------------------
    # Cascade 2: Semantic CONFLICTS_WITH drift cascade
    # ------------------------------------------------------------------

    def drift_cascade(self, drifted_node_id: str, new_weight: float,
                      decay: float = 0.7, max_depth: int = 3) -> Dict[str, float]:
        """
        When a preference node gains/loses weight (preference drift),
        propagate invalidation to CONFLICTS_WITH neighbors.

        Returns:
            Dict mapping node_id -> new_weight for affected nodes
        """
        affected: Dict[str, float] = {}
        if drifted_node_id not in self.nodes:
            return affected

        # Set the new weight on the drifted node
        self.nodes[drifted_node_id].weight = new_weight
        self.G.nodes[drifted_node_id]['weight'] = new_weight

        # BFS over CONFLICTS_WITH edges
        queue = [(drifted_node_id, 0, 1.0 - new_weight)]  # invalidation signal = 1 - weight
        visited: Set[str] = {drifted_node_id}

        while queue:
            current, depth, signal = queue.pop(0)
            if depth >= max_depth or signal < 0.05:
                continue

            for succ in self.G.successors(current):
                edge_data = self.G.edges[current, succ]
                if edge_data.get('edge_type') == 'CONFLICTS_WITH' and succ not in visited:
                    visited.add(succ)
                    edge_strength = edge_data.get('strength', 1.0)
                    propagated_signal = signal * edge_strength * decay
                    if succ in self.nodes:
                        old_w = self.nodes[succ].weight
                        new_w = max(0.0, old_w - propagated_signal)
                        self.nodes[succ].weight = new_w
                        self.G.nodes[succ]['weight'] = new_w
                        affected[succ] = new_w
                    queue.append((succ, depth + 1, propagated_signal * decay))

        return affected

    def to_dict(self) -> dict:
        return {
            "dialogue_id": self.dialogue_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [
                {"src": u, "tgt": v, **data}
                for u, v, data in self.G.edges(data=True)
            ],
            "location_root": self.location_root,
        }

    @staticmethod
    def from_dict(d: dict) -> "MemoryGraph":
        g = MemoryGraph(d["dialogue_id"])
        g.location_root = d.get("location_root")
        for nid, nd in d["nodes"].items():
            node = MemoryNode(
                node_id=nd["node_id"],
                content=nd["content"],
                memory_type=nd["memory_type"],
                timestamp=datetime.fromisoformat(nd["timestamp"]),
                weight=nd["weight"],
                location_tags=nd.get("location_tags", []),
                session_idx=nd.get("session_idx", 0),
            )
            g.add_node(node)
        for edge in d["edges"]:
            g.add_edge(edge["src"], edge["tgt"],
                       edge["edge_type"], edge.get("strength", 1.0))
        return g
