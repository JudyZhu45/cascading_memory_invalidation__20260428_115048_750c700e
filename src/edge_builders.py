"""
Implementations of the three approaches to building CONFLICTS_WITH edges.

Method A: Embedding-based semantic similarity (sentence-transformers)
Method B: LLM inference at memory write time
Method C: Behavioral co-occurrence analysis

Also includes LOCATED_IN edge builder.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_utils import detect_conflicts_with_llm
from src.data_loader import LOCATION_KEYWORDS, QUIET_PREFERENCE_PATTERNS, NOISY_ACTIVITY_PATTERNS
from src.memory_graph import MemoryGraph, MemoryNode

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts using sentence-transformers."""
    model = get_embed_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# ------------------------------------------------------------------
# LOCATED_IN edge builder
# ------------------------------------------------------------------

def build_located_in_edges(graph: MemoryGraph) -> int:
    """
    Scan all memory nodes and add LOCATED_IN edges from location-dependent
    memories to the location root node.

    Heuristic: if a memory mentions a location keyword that matches
    the location root's content, add LOCATED_IN edge.

    Returns number of edges added.
    """
    if graph.location_root is None:
        return 0

    root_node = graph.nodes.get(graph.location_root)
    if root_node is None:
        return 0

    root_content_lower = root_node.content.lower()
    root_locs = [loc for loc in LOCATION_KEYWORDS if loc in root_content_lower]

    n_added = 0
    for nid, node in graph.nodes.items():
        if nid == graph.location_root:
            continue
        # Check if this memory is location-dependent
        node_lower = node.content.lower()
        overlap = any(loc in node_lower for loc in root_locs)
        if not overlap and node.location_tags:
            overlap = any(tag.lower() in root_locs for tag in node.location_tags)
        if overlap:
            graph.add_edge(graph.location_root, nid, edge_type='LOCATED_IN', strength=0.8)
            n_added += 1

    logger.debug(f"Added {n_added} LOCATED_IN edges for dialogue {graph.dialogue_id}")
    return n_added


# ------------------------------------------------------------------
# Method A: Embedding-based CONFLICTS_WITH
# ------------------------------------------------------------------

def build_conflicts_embedding(graph: MemoryGraph,
                               threshold: float = 0.3) -> Tuple[int, List[Dict]]:
    """
    Build CONFLICTS_WITH edges using embedding cosine DISTANCE.
    Two memories conflict if they are semantically dissimilar (high distance)
    AND belong to the same domain (preference/activity).

    Note: We use dissimilarity (1 - cosine_sim) as a distance measure.
    High distance = different semantic space = potential conflict.

    This is the "semantic bridging test": do "prefers quiet" and "likes bars"
    have cosine distance > threshold?

    Returns: (n_edges_added, list of conflict pairs with distances)
    """
    pref_nodes = [
        (nid, node) for nid, node in graph.nodes.items()
        if node.memory_type in ("preference", "activity")
    ]

    if len(pref_nodes) < 2:
        return 0, []

    node_ids = [nid for nid, _ in pref_nodes]
    contents = [node.content for _, node in pref_nodes]
    embeddings = embed_texts(contents)

    # Pairwise cosine similarity → distance
    sim_matrix = cosine_similarity(embeddings)
    conflict_pairs = []
    n_added = 0

    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            dist = 1.0 - sim_matrix[i, j]  # cosine distance
            if dist > threshold:
                # Check that they're in the same domain (to avoid spurious conflicts)
                # by verifying at least one quiet + one noisy pattern
                ci = contents[i].lower()
                cj = contents[j].lower()
                has_quiet = any(re.search(p, ci) for p in QUIET_PREFERENCE_PATTERNS) or \
                            any(re.search(p, cj) for p in QUIET_PREFERENCE_PATTERNS)
                has_noisy = any(re.search(p, ci) for p in NOISY_ACTIVITY_PATTERNS) or \
                            any(re.search(p, cj) for p in NOISY_ACTIVITY_PATTERNS)

                conflict_pairs.append({
                    "node_a": node_ids[i],
                    "content_a": contents[i],
                    "node_b": node_ids[j],
                    "content_b": contents[j],
                    "cosine_distance": float(dist),
                    "has_quiet_noisy_pair": has_quiet and has_noisy,
                })

                # Add bidirectional CONFLICTS_WITH edges
                strength = min(1.0, dist)  # stronger conflict = higher distance
                graph.add_edge(node_ids[i], node_ids[j], "CONFLICTS_WITH", strength)
                graph.add_edge(node_ids[j], node_ids[i], "CONFLICTS_WITH", strength)
                n_added += 2

    logger.debug(f"Embedding method: added {n_added//2} conflict pairs")
    return n_added, conflict_pairs


# ------------------------------------------------------------------
# Method B: LLM inference CONFLICTS_WITH
# ------------------------------------------------------------------

def build_conflicts_llm(graph: MemoryGraph,
                         confidence_threshold: float = 0.6,
                         max_pairs: int = 30) -> Tuple[int, List[Dict]]:
    """
    Build CONFLICTS_WITH edges using LLM inference.
    For each pair of preference/activity nodes, ask LLM if they conflict.

    max_pairs: cap on number of pairs checked (to control API cost).
    Returns: (n_edges_added, conflict pairs with LLM outputs)
    """
    pref_nodes = [
        (nid, node) for nid, node in graph.nodes.items()
        if node.memory_type in ("preference", "activity")
    ]

    if len(pref_nodes) < 2:
        return 0, []

    conflict_pairs = []
    n_added = 0
    pairs_checked = 0

    for i, (nid_a, node_a) in enumerate(pref_nodes):
        for j, (nid_b, node_b) in enumerate(pref_nodes):
            if j <= i:
                continue
            if pairs_checked >= max_pairs:
                break

            result = detect_conflicts_with_llm(node_a.content, node_b.content)
            pairs_checked += 1

            conflict_pairs.append({
                "node_a": nid_a,
                "content_a": node_a.content,
                "node_b": nid_b,
                "content_b": node_b.content,
                "conflicts": result["conflicts"],
                "confidence": result["confidence"],
                "explanation": result["explanation"],
            })

            if result["conflicts"] and result["confidence"] >= confidence_threshold:
                strength = result["confidence"]
                graph.add_edge(nid_a, nid_b, "CONFLICTS_WITH", strength)
                graph.add_edge(nid_b, nid_a, "CONFLICTS_WITH", strength)
                n_added += 2

        if pairs_checked >= max_pairs:
            break

    logger.debug(f"LLM method: checked {pairs_checked} pairs, added {n_added//2} conflicts")
    return n_added, conflict_pairs


# ------------------------------------------------------------------
# Method C: Behavioral co-occurrence CONFLICTS_WITH
# ------------------------------------------------------------------

def build_conflicts_behavioral(graph: MemoryGraph,
                                session_activities: List[List[str]],
                                correlation_threshold: float = -0.3) -> Tuple[int, List[Dict]]:
    """
    Build CONFLICTS_WITH edges using behavioral co-occurrence signals.

    If two activities/preferences have negative correlation across sessions
    (when one is present, the other tends to be absent), they conflict.

    session_activities: list of sessions, each session is a list of activity tags
    correlation_threshold: negative Pearson r below this → CONFLICTS_WITH edge
    """
    # Get unique activities
    all_activities = set()
    for session in session_activities:
        all_activities.update(session)
    activities = sorted(all_activities)

    if len(activities) < 2:
        return 0, []

    # Build binary presence matrix (sessions × activities)
    presence = np.zeros((len(session_activities), len(activities)))
    for s_idx, session in enumerate(session_activities):
        for a_idx, act in enumerate(activities):
            presence[s_idx, a_idx] = 1 if act in session else 0

    # Compute pairwise Pearson correlations
    n_added = 0
    conflict_pairs = []
    pref_node_ids = {node.content.lower(): nid
                     for nid, node in graph.nodes.items()
                     if node.memory_type in ("preference", "activity")}

    for i, act_a in enumerate(activities):
        for j, act_b in enumerate(activities):
            if j <= i:
                continue
            if presence[:, i].std() < 0.01 or presence[:, j].std() < 0.01:
                continue

            r = float(np.corrcoef(presence[:, i], presence[:, j])[0, 1])
            conflict_pairs.append({
                "activity_a": act_a,
                "activity_b": act_b,
                "pearson_r": r,
                "is_conflict": r < correlation_threshold,
            })

            if r < correlation_threshold:
                # Find matching nodes in graph
                nid_a = pref_node_ids.get(act_a)
                nid_b = pref_node_ids.get(act_b)
                if nid_a and nid_b:
                    strength = abs(r)
                    graph.add_edge(nid_a, nid_b, "CONFLICTS_WITH", strength)
                    graph.add_edge(nid_b, nid_a, "CONFLICTS_WITH", strength)
                    n_added += 2

    logger.debug(f"Behavioral method: added {n_added//2} conflict pairs")
    return n_added, conflict_pairs
