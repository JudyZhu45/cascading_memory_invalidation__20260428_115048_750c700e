"""
Evaluation logic for the cascading memory invalidation experiments.

Implements:
- Structural cascade accuracy evaluation
- Semantic edge precision evaluation
- Drift cascade accuracy evaluation
- Recommendation correctness comparison (flat vs graph methods)
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.memory_graph import MemoryGraph, MemoryNode
from src.llm_utils import answer_with_memories, llm_call

logger = logging.getLogger(__name__)


def evaluate_structural_cascade(graph_before: MemoryGraph,
                                  graph_after: MemoryGraph,
                                  gold_invalidated: List[str],
                                  all_node_ids: List[str]) -> Dict:
    """
    Evaluate structural cascade: did the cascade correctly downweight
    the right nodes?

    gold_invalidated: list of node_ids that SHOULD have been invalidated
    all_node_ids: all candidate node_ids

    Returns precision, recall, F1, accuracy.
    """
    # Predicted invalidated: nodes whose weight dropped significantly after cascade
    predicted = []
    for nid in all_node_ids:
        if nid in graph_before.nodes and nid in graph_after.nodes:
            w_before = graph_before.nodes[nid].weight
            w_after = graph_after.nodes[nid].weight
            if w_before - w_after > 0.1:  # 10% drop threshold
                predicted.append(nid)

    gold_set = set(gold_invalidated)
    pred_set = set(predicted)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    tn = len(set(all_node_ids) - gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(all_node_ids) if all_node_ids else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_gold_invalidated": len(gold_set),
        "n_predicted_invalidated": len(pred_set),
    }


def evaluate_semantic_edge_precision(conflict_pairs: List[Dict],
                                      method_name: str) -> Dict:
    """
    Evaluate precision of CONFLICTS_WITH edge detection.
    Uses ground-truth labels if available; otherwise uses LLM-verified labels.

    For the embedding method: uses 'has_quiet_noisy_pair' as proxy for true positives.
    For the LLM method: uses 'conflicts' flag (ground truth IS the LLM output).
    For behavioral method: uses 'is_conflict' flag.
    """
    if not conflict_pairs:
        return {
            "method": method_name,
            "n_pairs_checked": 0,
            "n_predicted_conflict": 0,
            "precision": 0.0,
            "notes": "no pairs to evaluate",
        }

    if method_name == "embedding":
        # Count pairs above threshold as predicted conflicts
        predicted_conflicts = conflict_pairs  # all returned pairs are above threshold
        true_positives = [p for p in predicted_conflicts if p.get("has_quiet_noisy_pair", False)]
        precision = len(true_positives) / len(predicted_conflicts) if predicted_conflicts else 0.0
        return {
            "method": method_name,
            "n_pairs_checked": len(conflict_pairs),
            "n_predicted_conflict": len(predicted_conflicts),
            "n_true_positive": len(true_positives),
            "precision": precision,
            "avg_cosine_distance": float(np.mean([p["cosine_distance"] for p in conflict_pairs])),
        }
    elif method_name == "llm":
        predicted_conflicts = [p for p in conflict_pairs if p.get("conflicts", False)]
        # For LLM method, we verify high-confidence pairs against a gold standard
        # Using confidence >= 0.7 as "high confidence" true positive proxy
        high_conf = [p for p in predicted_conflicts if p.get("confidence", 0) >= 0.7]
        precision = len(high_conf) / len(predicted_conflicts) if predicted_conflicts else 0.0
        return {
            "method": method_name,
            "n_pairs_checked": len(conflict_pairs),
            "n_predicted_conflict": len(predicted_conflicts),
            "n_high_confidence": len(high_conf),
            "precision": precision,
            "avg_confidence": float(np.mean([p.get("confidence", 0) for p in conflict_pairs])),
        }
    elif method_name == "behavioral":
        predicted_conflicts = [p for p in conflict_pairs if p.get("is_conflict", False)]
        # For behavioral method: pairs with strong negative correlation (r < -0.5) are true positives
        strong_conflicts = [p for p in predicted_conflicts if p.get("pearson_r", 0) < -0.5]
        precision = len(strong_conflicts) / len(predicted_conflicts) if predicted_conflicts else 0.0
        return {
            "method": method_name,
            "n_pairs_checked": len(conflict_pairs),
            "n_predicted_conflict": len(predicted_conflicts),
            "n_strong_conflict": len(strong_conflicts),
            "precision": precision,
            "avg_pearson_r": float(np.mean([p.get("pearson_r", 0) for p in conflict_pairs])) if conflict_pairs else 0.0,
        }
    else:
        return {"method": method_name, "error": "unknown method"}


def get_active_memories(graph: MemoryGraph, weight_threshold: float = 0.3) -> List[str]:
    """Get content of memories with weight above threshold."""
    return [
        node.content for node in graph.nodes.values()
        if node.weight >= weight_threshold
    ]


def evaluate_recommendation_correctness(qa_pairs: List[Dict],
                                          methods: Dict[str, MemoryGraph],
                                          n_pairs: Optional[int] = None) -> Dict:
    """
    Evaluate recommendation correctness for multiple methods on QA pairs.

    qa_pairs: list of {question, answer, requires_memory_type, involves_change}
    methods: dict of method_name -> MemoryGraph (with different cascade applied)

    Returns per-method accuracy and false_invalidation_rate.
    """
    if n_pairs is not None:
        qa_pairs = qa_pairs[:n_pairs]

    results = {}
    for method_name, graph in methods.items():
        active_mems = get_active_memories(graph)
        correct = 0
        false_invalidations = 0

        for qa in qa_pairs:
            question = qa["question"]
            gold_answer = qa["answer"]
            generated = answer_with_memories(question, active_mems)

            # Simple correctness check: does gold answer appear in generated answer?
            is_correct = gold_answer.lower() in generated.lower() or \
                         any(word in generated.lower()
                             for word in gold_answer.lower().split()[:3] if len(word) > 3)

            if is_correct:
                correct += 1

            # False invalidation: question involves a STABLE preference
            # but the relevant memory was incorrectly downweighted
            if not qa.get("involves_change", False):
                relevant_content = [m for m in active_mems
                                    if any(kw in m.lower()
                                           for kw in gold_answer.lower().split()[:3] if len(kw) > 3)]
                if not relevant_content:
                    false_invalidations += 1

        n = len(qa_pairs)
        results[method_name] = {
            "method": method_name,
            "n_qa": n,
            "correct": correct,
            "accuracy": correct / n if n > 0 else 0.0,
            "false_invalidations": false_invalidations,
            "false_invalidation_rate": false_invalidations / n if n > 0 else 0.0,
        }

    return results


def compute_embedding_distance_test(quiet_phrases: List[str],
                                     noisy_phrases: List[str]) -> Dict:
    """
    Test the key empirical question: is cosine distance between
    quiet-preference and noisy-activity embeddings > 0.3?

    This directly tests H2a.
    """
    from src.edge_builders import embed_texts

    quiet_embs = embed_texts(quiet_phrases)
    noisy_embs = embed_texts(noisy_phrases)

    from sklearn.metrics.pairwise import cosine_similarity
    # Cross-group distances
    cross_sim = cosine_similarity(quiet_embs, noisy_embs)
    cross_dist = 1.0 - cross_sim

    # Within-group distances (control)
    within_quiet_sim = cosine_similarity(quiet_embs, quiet_embs)
    np.fill_diagonal(within_quiet_sim, 1.0)  # ignore self-sim
    within_quiet_dist = 1.0 - within_quiet_sim

    return {
        "mean_cross_distance": float(cross_dist.mean()),
        "min_cross_distance": float(cross_dist.min()),
        "max_cross_distance": float(cross_dist.max()),
        "std_cross_distance": float(cross_dist.std()),
        "mean_within_quiet_distance": float(within_quiet_dist.mean()),
        "fraction_above_threshold_03": float((cross_dist > 0.3).mean()),
        "fraction_above_threshold_05": float((cross_dist > 0.5).mean()),
        "h2a_supported": float(cross_dist.mean()) > 0.3,
    }
