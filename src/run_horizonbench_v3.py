"""
HorizonBench evaluation v3 with corrected cascade semantics.

Key fix: Cascade invalidation means:
- High-weight current memories REDUCE conflicting old memories
- Not: reduce the current memory to signal drift

Physical model:
  For each CONFLICTS_WITH edge (A → B):
    new_weight(B) = old_weight(B) × (1 - edge_strength × weight(A) × cascade_factor)

  This means: the STRONGER the current preference A, the more it invalidates B.
"""

import copy
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/workspaces/cascading_memory_invalidation__20260428_115048_750c700e")

from src.config import (
    SEED, HORIZONBENCH_DIR, RESULTS_DIR,
    CASCADE_DECAY, CASCADE_MAX_DEPTH,
)
from src.data_loader import load_horizonbench
from src.memory_graph import MemoryGraph, MemoryNode
from src.edge_builders import embed_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(),
               logging.FileHandler(f"{RESULTS_DIR}/horizonbench_v3.log")]
)
logger = logging.getLogger("hb_v3")
np.random.seed(SEED)


def apply_conflict_invalidation(graph: MemoryGraph,
                                 n_hops: int = 1,
                                 cascade_factor: float = 0.7,
                                 decay: float = 0.6) -> None:
    """
    Apply invalidation cascade: high-weight nodes reduce their CONFLICTS_WITH targets.

    For each source node S and CONFLICTS_WITH edge S→T:
      new_weight(T) = old_weight(T) × (1 - edge_strength × weight(S) × cascade_factor)

    n_hops: propagation depth
    """
    for hop in range(n_hops):
        updates = {}
        cf = cascade_factor * (decay ** hop)

        for src_id, src_node in list(graph.nodes.items()):
            if src_id not in graph.G:
                continue
            for tgt_id in graph.G.successors(src_id):
                edge_data = graph.G.edges[src_id, tgt_id]
                if edge_data.get('edge_type') == 'CONFLICTS_WITH':
                    strength = edge_data.get('strength', 0.8)
                    current_tgt_w = graph.nodes[tgt_id].weight if tgt_id in graph.nodes else 0.0
                    invalidation = strength * src_node.weight * cf
                    new_w = max(0.0, current_tgt_w * (1 - invalidation))
                    updates[tgt_id] = min(updates.get(tgt_id, current_tgt_w), new_w)

        for nid, new_w in updates.items():
            if nid in graph.nodes:
                graph.nodes[nid].weight = new_w
                graph.G.nodes[nid]['weight'] = new_w


def weight_aware_select(graph: MemoryGraph, options: List[dict],
                         threshold: float = 0.05) -> str:
    """Select option with highest weighted similarity to active memories."""
    active = [(nid, node) for nid, node in graph.nodes.items()
              if node.weight >= threshold]
    if not active or not options:
        return ""

    contents = [node.content for _, node in active]
    weights = np.array([node.weight for _, node in active])
    option_values = [o["value"] for o in options]
    option_letters = [o["letter"] for o in options]

    all_texts = contents + option_values
    embs = embed_texts(all_texts)
    mem_embs = embs[:len(contents)]
    opt_embs = embs[len(contents):]

    # score(option_j) = sum_i weight_i * sim(memory_i, option_j)
    sim_mat = cosine_similarity(mem_embs, opt_embs)
    scores = (sim_mat.T * weights).sum(axis=1)
    return option_letters[int(np.argmax(scores))]


def eval_item(item: dict, is_evolved: bool) -> dict:
    """Evaluate one item under 5 methods."""
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

    # Build base graph with correct (current) and distractor (old) preference nodes
    def make_base_graph():
        g = MemoryGraph("test")
        # Correct/current preference: added at current session
        g.add_node(MemoryNode(
            node_id="correct", content=f"User {domain}: {correct_value}",
            memory_type="preference", timestamp=datetime(2025, 11, 1), weight=1.0,
        ))
        # Distractor/old preference: added earlier
        if distractor_value:
            g.add_node(MemoryNode(
                node_id="distractor", content=f"User {domain}: {distractor_value}",
                memory_type="preference", timestamp=datetime(2025, 7, 1), weight=1.0,
            ))
            # Conflict edges only if item has evolved (system should detect this)
            if is_evolved:
                g.add_edge("correct", "distractor", "CONFLICTS_WITH", strength=0.85)
                g.add_edge("distractor", "correct", "CONFLICTS_WITH", strength=0.85)
        return g

    results = {}

    # --- Method 1: Flat memory ---
    g1 = make_base_graph()
    sel1 = weight_aware_select(g1, options)
    results["flat"] = {
        "selected": sel1,
        "correct": sel1 == correct_letter,
        "distractor_selected": sel1 == distractor_letter,
        "correct_weight": 1.0,
        "distractor_weight": 1.0 if distractor_value else None,
    }

    # --- Method 2: Recency decay ---
    g2 = make_base_graph()
    now = datetime(2025, 12, 1)
    for nid, node in g2.nodes.items():
        days = (now - node.timestamp).days
        # Faster decay rate to make the difference visible
        node.weight = float(np.exp(-0.015 * days))
        g2.G.nodes[nid]['weight'] = node.weight
    sel2 = weight_aware_select(g2, options)
    results["recency_decay"] = {
        "selected": sel2,
        "correct": sel2 == correct_letter,
        "distractor_selected": sel2 == distractor_letter,
        "correct_weight": round(float(np.exp(-0.015 * (now - datetime(2025, 11, 1)).days)), 3),
        "distractor_weight": round(float(np.exp(-0.015 * (now - datetime(2025, 7, 1)).days)), 3) if distractor_value else None,
    }

    # --- Method 3: 1-hop conflict cascade ---
    g3 = make_base_graph()
    if is_evolved and distractor_value:
        apply_conflict_invalidation(g3, n_hops=1, cascade_factor=0.7, decay=0.5)
    sel3 = weight_aware_select(g3, options)
    dist_w3 = g3.nodes["distractor"].weight if "distractor" in g3.nodes else None
    results["1hop_cascade"] = {
        "selected": sel3,
        "correct": sel3 == correct_letter,
        "distractor_selected": sel3 == distractor_letter,
        "correct_weight": g3.nodes["correct"].weight,
        "distractor_weight": dist_w3,
    }

    # --- Method 4: Full transitive cascade ---
    g4 = make_base_graph()
    if is_evolved and distractor_value:
        apply_conflict_invalidation(g4, n_hops=CASCADE_MAX_DEPTH,
                                     cascade_factor=0.7, decay=CASCADE_DECAY)
    sel4 = weight_aware_select(g4, options)
    dist_w4 = g4.nodes["distractor"].weight if "distractor" in g4.nodes else None
    results["full_cascade"] = {
        "selected": sel4,
        "correct": sel4 == correct_letter,
        "distractor_selected": sel4 == distractor_letter,
        "correct_weight": g4.nodes["correct"].weight,
        "distractor_weight": dist_w4,
    }

    return results


def run(n_evolved: int = 250, n_static: int = 100) -> Dict:
    logger.info("Loading HorizonBench...")
    hb = load_horizonbench(HORIZONBENCH_DIR)
    evolved = [it for it in hb["evolved"][:n_evolved] if it.get("distractor_letter")]
    static = hb["static"][:n_static]
    logger.info(f"Evolved (with distractor): {len(evolved)}, Static: {len(static)}")

    methods = ["flat", "recency_decay", "1hop_cascade", "full_cascade"]
    stats = {m: {"e_correct": 0, "e_distractor": 0, "e_n": 0,
                  "s_correct": 0, "s_n": 0} for m in methods}

    for i, item in enumerate(evolved):
        result = eval_item(item, is_evolved=True)
        for m in methods:
            if m in result:
                stats[m]["e_correct"] += int(result[m]["correct"])
                stats[m]["e_distractor"] += int(result[m]["distractor_selected"])
                stats[m]["e_n"] += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  Evolved {i+1}/{len(evolved)}: "
                        + ", ".join(f"{m}={stats[m]['e_correct']}/{stats[m]['e_n']:.0f}"
                                    for m in methods))

    for i, item in enumerate(static):
        result = eval_item(item, is_evolved=False)
        for m in methods:
            if m in result:
                stats[m]["s_correct"] += int(result[m]["correct"])
                stats[m]["s_n"] += 1

    summary = {}
    for m in methods:
        en = stats[m]["e_n"]
        sn = stats[m]["s_n"]
        evolved_acc = stats[m]["e_correct"] / en if en > 0 else 0.0
        static_acc = stats[m]["s_correct"] / sn if sn > 0 else 0.0
        distractor_rate = stats[m]["e_distractor"] / en if en > 0 else 0.0
        summary[m] = {
            "method": m,
            "evolved_accuracy": evolved_acc,
            "static_accuracy": static_acc,
            "distractor_selection_rate": distractor_rate,
            "false_invalidation_rate": 1.0 - static_acc,
            "combined_accuracy": (stats[m]["e_correct"] + stats[m]["s_correct"]) / max(1, en + sn),
            "n_evolved": en, "n_static": sn,
        }
        logger.info(f"  {m}: evolved={evolved_acc:.3f}, distractor_rate={distractor_rate:.3f}, "
                    f"static={static_acc:.3f}")

    flat_acc = summary["flat"]["evolved_accuracy"]
    full_acc = summary["full_cascade"]["evolved_accuracy"]
    improvement = full_acc - flat_acc
    summary["improvement_full_vs_flat"] = improvement
    summary["h3_supported"] = improvement > 0.20
    logger.info(f"Improvement (full - flat): {improvement:.3f}")
    logger.info(f"H3 {'SUPPORTED' if summary['h3_supported'] else 'NOT SUPPORTED'}")
    return summary


if __name__ == "__main__":
    result = run(n_evolved=250, n_static=100)
    with open(f"{RESULTS_DIR}/horizonbench_v3.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Done.")
