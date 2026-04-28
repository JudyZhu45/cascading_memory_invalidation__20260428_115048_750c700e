"""
Corrected HorizonBench drift cascade evaluation.

Fixes:
1. Parse JSON-encoded options field
2. Weight-aware option selection: score = similarity × node_weight
3. Evaluate flat vs cascade by comparing how often distractor is selected
"""

import copy
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/workspaces/cascading_memory_invalidation__20260428_115048_750c700e")

from src.config import (
    SEED, HORIZONBENCH_DIR, RESULTS_DIR,
    CASCADE_DECAY, CASCADE_MAX_DEPTH,
    CONFLICTS_WITH_EMBEDDING_THRESHOLD,
)
from src.data_loader import load_horizonbench
from src.memory_graph import MemoryGraph, MemoryNode
from src.edge_builders import embed_texts

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    handlers=[logging.StreamHandler(),
                               logging.FileHandler(f"{RESULTS_DIR}/horizonbench_eval.log")])
logger = logging.getLogger("horizonbench_eval")
np.random.seed(SEED)


def weight_aware_select(memories_with_weights: List[tuple],
                         options: List[dict]) -> str:
    """
    Select the option best supported by active memories, weighted by node weight.

    memories_with_weights: list of (content_str, weight_float)
    options: list of {"letter": ..., "value": ..., "option": ...}

    Score for option i = sum over memories of (sim(memory_i, option_i) * weight_i)
    """
    if not memories_with_weights or not options:
        return ""

    mem_contents = [m[0] for m in memories_with_weights]
    mem_weights = np.array([m[1] for m in memories_with_weights])
    option_values = [o["value"] for o in options]
    option_letters = [o["letter"] for o in options]

    all_texts = mem_contents + option_values
    embs = embed_texts(all_texts)
    mem_embs = embs[:len(mem_contents)]
    opt_embs = embs[len(mem_contents):]

    # sim matrix: (n_memories × n_options)
    sim_mat = cosine_similarity(mem_embs, opt_embs)  # shape (n_mem, n_opt)

    # Weight each memory's similarities by its weight
    weighted_sims = (sim_mat.T * mem_weights).T  # still (n_mem, n_opt)
    option_scores = weighted_sims.sum(axis=0)  # shape (n_opt,)

    best_idx = int(np.argmax(option_scores))
    return option_letters[best_idx]


def evaluate_one_item_all_methods(item: dict, is_evolved: bool) -> dict:
    """
    Evaluate one HorizonBench item under 4 memory methods.
    Returns dict of method → {correct, selected, distractor_selected}
    """
    options = item.get("options", [])
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            return {}

    if not options or not isinstance(options, list):
        return {}

    correct_letter = item.get("correct_letter", "")
    distractor_letter = item.get("distractor_letter", "")
    domain = item.get("preference_domain", "unknown").replace("_", " ")

    option_dict = {o["letter"]: o["value"] for o in options if isinstance(o, dict)}
    if correct_letter not in option_dict:
        return {}

    correct_value = option_dict[correct_letter]
    distractor_value = option_dict.get(distractor_letter, "")

    # Build mini memory graph
    graph = MemoryGraph(f"hb_{item.get('id', str(abs(hash(correct_value))))[:8]}")

    # Current (evolved/correct) preference node — most recent
    correct_node = MemoryNode(
        node_id="correct",
        content=f"User {domain} preference: {correct_value}",
        memory_type="preference",
        timestamp=datetime(2025, 11, 1),
        weight=1.0,
    )
    graph.add_node(correct_node)

    # Pre-evolution (distractor) preference node — older
    if distractor_value:
        distractor_node = MemoryNode(
            node_id="distractor",
            content=f"User {domain} preference: {distractor_value}",
            memory_type="preference",
            timestamp=datetime(2025, 8, 1),  # 3 months older
            weight=0.9,  # Slightly lower but still high initially
        )
        graph.add_node(distractor_node)

        # Add CONFLICTS_WITH edges if item has evolved
        if is_evolved:
            graph.add_edge("correct", "distractor", "CONFLICTS_WITH", strength=0.85)
            graph.add_edge("distractor", "correct", "CONFLICTS_WITH", strength=0.85)

    results = {}

    def get_mems_with_weights(g, threshold=0.1):
        return [(node.content, node.weight)
                for node in g.nodes.values()
                if node.weight >= threshold]

    # --- Method 1: Flat ---
    graph_flat = copy.deepcopy(graph)
    mems_flat = get_mems_with_weights(graph_flat)
    selected_flat = weight_aware_select(mems_flat, options)
    results["flat"] = {
        "selected": selected_flat,
        "correct": selected_flat == correct_letter,
        "distractor_selected": selected_flat == distractor_letter,
    }

    # --- Method 2: Recency decay ---
    graph_decay = copy.deepcopy(graph)
    for nid, node in graph_decay.nodes.items():
        delta_days = (datetime(2025, 12, 1) - node.timestamp).days
        node.weight = float(np.exp(-0.008 * delta_days))
        graph_decay.G.nodes[nid]['weight'] = node.weight
    mems_decay = get_mems_with_weights(graph_decay)
    selected_decay = weight_aware_select(mems_decay, options)
    results["recency_decay"] = {
        "selected": selected_decay,
        "correct": selected_decay == correct_letter,
        "distractor_selected": selected_decay == distractor_letter,
    }

    # --- Method 3: 1-hop cascade ---
    graph_1hop = copy.deepcopy(graph)
    if is_evolved and "correct" in graph_1hop.nodes:
        # Strengthen correct node, propagate invalidation 1 hop
        graph_1hop.drift_cascade("correct", new_weight=0.95, decay=0.9, max_depth=1)
    mems_1hop = get_mems_with_weights(graph_1hop)
    selected_1hop = weight_aware_select(mems_1hop, options)
    results["1hop_cascade"] = {
        "selected": selected_1hop,
        "correct": selected_1hop == correct_letter,
        "distractor_selected": selected_1hop == distractor_letter,
    }

    # --- Method 4: Full transitive cascade ---
    graph_full = copy.deepcopy(graph)
    if is_evolved and "correct" in graph_full.nodes:
        graph_full.drift_cascade("correct", new_weight=0.95,
                                  decay=CASCADE_DECAY, max_depth=CASCADE_MAX_DEPTH)
    mems_full = get_mems_with_weights(graph_full)
    selected_full = weight_aware_select(mems_full, options)
    results["full_cascade"] = {
        "selected": selected_full,
        "correct": selected_full == correct_letter,
        "distractor_selected": selected_full == distractor_letter,
    }

    return results


def run_horizonbench_full_eval(n_evolved: int = 150, n_static: int = 75) -> Dict:
    """
    Full HorizonBench evaluation comparing 4 methods.

    For evolved items: primary metric = correctly selecting current (post-evolution) preference
    For static items: primary metric = NOT incorrectly invalidating correct preference
    """
    logger.info("Loading HorizonBench data...")
    hb_data = load_horizonbench(HORIZONBENCH_DIR)
    evolved_items = hb_data["evolved"][:n_evolved]
    static_items = hb_data["static"][:n_static]

    # Only use items that have a distractor_letter (ground truth pre-evolution preference)
    evolved_with_distractor = [it for it in evolved_items if it.get("distractor_letter")]
    evolved_without_distractor = [it for it in evolved_items if not it.get("distractor_letter")]
    logger.info(f"Evolved: {len(evolved_items)} total, {len(evolved_with_distractor)} with distractor, "
                f"{len(evolved_without_distractor)} without")

    # Metrics
    methods = ["flat", "recency_decay", "1hop_cascade", "full_cascade"]
    evolved_stats = {m: {"correct": 0, "distractor_selected": 0, "n": 0} for m in methods}
    static_stats = {m: {"correct": 0, "distractor_selected": 0, "n": 0} for m in methods}

    logger.info(f"Evaluating {len(evolved_with_distractor)} evolved items with distractor...")
    per_item_results = []
    for i, item in enumerate(evolved_with_distractor):
        result = evaluate_one_item_all_methods(item, is_evolved=True)
        if not result:
            continue
        for m in methods:
            if m in result:
                evolved_stats[m]["correct"] += int(result[m]["correct"])
                evolved_stats[m]["distractor_selected"] += int(result[m]["distractor_selected"])
                evolved_stats[m]["n"] += 1
        per_item_results.append({"item_id": item.get("id"), "is_evolved": True, **result})
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i+1}/{len(evolved_with_distractor)} evolved items")

    logger.info(f"Evaluating {len(static_items)} static items...")
    for i, item in enumerate(static_items):
        result = evaluate_one_item_all_methods(item, is_evolved=False)
        if not result:
            continue
        for m in methods:
            if m in result:
                static_stats[m]["correct"] += int(result[m]["correct"])
                static_stats[m]["distractor_selected"] += int(result[m]["distractor_selected"])
                static_stats[m]["n"] += 1
        per_item_results.append({"item_id": item.get("id"), "is_evolved": False, **result})
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i+1}/{len(static_items)} static items")

    # Compute summary statistics
    summary = {}
    for m in methods:
        e_n = evolved_stats[m]["n"]
        s_n = static_stats[m]["n"]
        summary[m] = {
            "method": m,
            # Evolved items: accuracy of selecting correct post-evolution preference
            "evolved_accuracy": evolved_stats[m]["correct"] / e_n if e_n > 0 else 0.0,
            "evolved_distractor_rate": evolved_stats[m]["distractor_selected"] / e_n if e_n > 0 else 0.0,
            # Static items: accuracy of maintaining correct preference
            "static_accuracy": static_stats[m]["correct"] / s_n if s_n > 0 else 0.0,
            # Combined accuracy
            "combined_accuracy": (evolved_stats[m]["correct"] + static_stats[m]["correct"]) / max(1, e_n + s_n),
            # False invalidation: selecting wrong option on static (preference stable) items
            "false_invalidation_rate": 1.0 - (static_stats[m]["correct"] / s_n) if s_n > 0 else 0.0,
            "n_evolved": e_n,
            "n_static": s_n,
        }
        logger.info(f"  {m}: evolved_acc={summary[m]['evolved_accuracy']:.3f}, "
                    f"distractor_rate={summary[m]['evolved_distractor_rate']:.3f}, "
                    f"static_acc={summary[m]['static_accuracy']:.3f}")

    flat_acc = summary["flat"]["evolved_accuracy"]
    full_acc = summary["full_cascade"]["evolved_accuracy"]
    improvement = full_acc - flat_acc
    summary["improvement_full_vs_flat"] = improvement
    summary["h3_supported"] = improvement > 0.20

    logger.info(f"Improvement (full cascade vs flat): {improvement:.3f}")

    # Belief-update failure: how often EACH method selects the PRE-EVOLUTION distractor
    logger.info("\nBelief-update failure analysis (distractor selection rate on evolved items):")
    for m in methods:
        rate = summary[m]["evolved_distractor_rate"]
        logger.info(f"  {m}: {rate:.3f} (lower = better)")

    summary["per_item"] = per_item_results[:20]  # Save sample for debugging
    return summary


if __name__ == "__main__":
    result = run_horizonbench_full_eval(n_evolved=200, n_static=100)
    with open(f"{RESULTS_DIR}/horizonbench_eval.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Update metrics summary
    flat_acc = result.get("flat", {}).get("evolved_accuracy", 0.0)
    full_acc = result.get("full_cascade", {}).get("evolved_accuracy", 0.0)
    improvement = result.get("improvement_full_vs_flat", 0.0)

    logger.info(f"\nFINAL: flat={flat_acc:.3f}, full={full_acc:.3f}, improvement={improvement:.3f}")
    logger.info(f"H3 (improvement>20pp): {'SUPPORTED' if improvement > 0.20 else 'NOT SUPPORTED'}")
    print("Done.")
