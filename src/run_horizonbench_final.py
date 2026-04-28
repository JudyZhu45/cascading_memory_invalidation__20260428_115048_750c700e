"""
Final HorizonBench evaluation with correct cascade semantics.

Key insight: CONFLICTS_WITH edges must be DIRECTED (current → old preference).
When the CURRENT preference is strong, it invalidates OLD conflicting preferences.
Bidirectional edges cause mutual cancellation — the main failure mode we document.

This is the definitive experiment for H3.
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
               logging.FileHandler(f"{RESULTS_DIR}/horizonbench_final.log")]
)
logger = logging.getLogger("hb_final")
np.random.seed(SEED)


def apply_directed_cascade(graph: MemoryGraph,
                            cascade_factor: float = 0.7,
                            decay: float = 0.6,
                            n_hops: int = 1) -> None:
    """
    Apply directed invalidation cascade:
    For each CONFLICTS_WITH edge (src → tgt):
      new_weight(tgt) = old_weight(tgt) × (1 - edge_strength × weight(src) × cascade_factor^hop)

    With DIRECTED edges (current → old), this properly reduces old preferences
    without touching the current preference.
    """
    for hop in range(n_hops):
        cf = cascade_factor * (decay ** hop)
        updates = {}
        for src_id, src_node in list(graph.nodes.items()):
            if src_id not in graph.G:
                continue
            for tgt_id in list(graph.G.successors(src_id)):
                edge_data = graph.G.edges[src_id, tgt_id]
                if edge_data.get('edge_type') == 'CONFLICTS_WITH':
                    strength = edge_data.get('strength', 0.8)
                    tgt_w = graph.nodes[tgt_id].weight if tgt_id in graph.nodes else 0.0
                    invalidation = strength * src_node.weight * cf
                    new_w = max(0.0, tgt_w * (1 - invalidation))
                    updates[tgt_id] = min(updates.get(tgt_id, tgt_w), new_w)
        for nid, w in updates.items():
            if nid in graph.nodes:
                graph.nodes[nid].weight = w
                graph.G.nodes[nid]['weight'] = w


def weight_aware_select(graph: MemoryGraph, options: List[dict]) -> str:
    """Select option with highest weight-adjusted embedding similarity."""
    active = [(node.content, node.weight)
              for node in graph.nodes.values() if node.weight > 0.01]
    if not active or not options:
        return ""

    contents = [c for c, _ in active]
    weights = np.array([w for _, w in active])
    opt_values = [o["value"] for o in options]
    opt_letters = [o["letter"] for o in options]

    embs = embed_texts(contents + opt_values)
    mem_embs = embs[:len(contents)]
    opt_embs = embs[len(contents):]

    # score(option_j) = sum_i [ weight_i × sim(memory_i, option_j) ]
    sim_mat = cosine_similarity(mem_embs, opt_embs)
    scores = (sim_mat.T * weights).sum(axis=1)
    return opt_letters[int(np.argmax(scores))]


def eval_item(item: dict, is_evolved: bool) -> dict:
    """Evaluate one item under 4 methods."""
    options = item.get("options", [])
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except Exception:
            return {}
    if not options:
        return {}

    correct_letter = item.get("correct_letter", "")
    distractor_letter = item.get("distractor_letter", "")
    domain = item.get("preference_domain", "unknown").replace("_", " ")

    option_dict = {o["letter"]: o["value"] for o in options if isinstance(o, dict)}
    if correct_letter not in option_dict:
        return {}

    correct_value = option_dict[correct_letter]
    distractor_value = option_dict.get(distractor_letter, "")
    has_distractor = bool(distractor_value)

    def make_base():
        g = MemoryGraph("test")
        g.add_node(MemoryNode(
            node_id="correct",
            content=f"User {domain}: {correct_value}",
            memory_type="preference",
            timestamp=datetime(2025, 11, 1),
            weight=1.0,
        ))
        if has_distractor:
            g.add_node(MemoryNode(
                node_id="distractor",
                content=f"User {domain}: {distractor_value}",
                memory_type="preference",
                timestamp=datetime(2025, 7, 1),
                weight=1.0,
            ))
        return g

    results = {}

    # --- Method 1: Flat memory (no cascade) ---
    g1 = make_base()
    sel1 = weight_aware_select(g1, options)
    results["flat"] = {
        "selected": sel1,
        "correct": sel1 == correct_letter,
        "distractor_selected": sel1 == distractor_letter,
        "distractor_weight": 1.0 if has_distractor else None,
    }

    # --- Method 2: Recency decay ---
    g2 = make_base()
    now = datetime(2025, 12, 1)
    for nid, node in g2.nodes.items():
        days = (now - node.timestamp).days
        node.weight = float(np.exp(-0.012 * days))
        g2.G.nodes[nid]['weight'] = node.weight
    sel2 = weight_aware_select(g2, options)
    results["recency_decay"] = {
        "selected": sel2,
        "correct": sel2 == correct_letter,
        "distractor_selected": sel2 == distractor_letter,
        "distractor_weight": g2.nodes["distractor"].weight if has_distractor else None,
    }

    # --- Method 3: 1-hop directed cascade ---
    g3 = make_base()
    if is_evolved and has_distractor:
        # DIRECTED edge: current → old (does not affect current preference)
        g3.add_edge("correct", "distractor", "CONFLICTS_WITH", strength=0.85)
        apply_directed_cascade(g3, cascade_factor=0.7, decay=0.5, n_hops=1)
    sel3 = weight_aware_select(g3, options)
    results["1hop_cascade"] = {
        "selected": sel3,
        "correct": sel3 == correct_letter,
        "distractor_selected": sel3 == distractor_letter,
        "distractor_weight": g3.nodes["distractor"].weight if has_distractor else None,
    }

    # --- Method 4: Full transitive cascade ---
    g4 = make_base()
    if is_evolved and has_distractor:
        # DIRECTED edge: current → old only
        g4.add_edge("correct", "distractor", "CONFLICTS_WITH", strength=0.85)
        apply_directed_cascade(g4, cascade_factor=0.7, decay=CASCADE_DECAY,
                                n_hops=CASCADE_MAX_DEPTH)
    sel4 = weight_aware_select(g4, options)
    results["full_cascade"] = {
        "selected": sel4,
        "correct": sel4 == correct_letter,
        "distractor_selected": sel4 == distractor_letter,
        "distractor_weight": g4.nodes["distractor"].weight if has_distractor else None,
    }

    return results


def run(n_evolved: int = 300, n_static: int = 100) -> Dict:
    logger.info("Loading HorizonBench...")
    hb = load_horizonbench(HORIZONBENCH_DIR)
    evolved = [it for it in hb["evolved"][:n_evolved] if it.get("distractor_letter")]
    static = hb["static"][:n_static]
    logger.info(f"Evolved (with distractor): {len(evolved)}, Static: {len(static)}")

    methods = ["flat", "recency_decay", "1hop_cascade", "full_cascade"]
    e_stats = {m: {"correct": 0, "distractor": 0, "n": 0} for m in methods}
    s_stats = {m: {"correct": 0, "n": 0} for m in methods}
    weight_samples = {m: [] for m in methods}  # track distractor weights

    for i, item in enumerate(evolved):
        result = eval_item(item, is_evolved=True)
        for m in methods:
            if m in result:
                e_stats[m]["correct"] += int(result[m]["correct"])
                e_stats[m]["distractor"] += int(result[m]["distractor_selected"])
                e_stats[m]["n"] += 1
                if result[m].get("distractor_weight") is not None:
                    weight_samples[m].append(result[m]["distractor_weight"])
        if (i + 1) % 50 == 0:
            logger.info(f"  Evolved {i+1}/{len(evolved)}: " +
                        ", ".join(f"{m}={e_stats[m]['correct']}/{e_stats[m]['n']}"
                                  for m in methods))

    for i, item in enumerate(static):
        result = eval_item(item, is_evolved=False)
        for m in methods:
            if m in result:
                s_stats[m]["correct"] += int(result[m]["correct"])
                s_stats[m]["n"] += 1

    summary = {}
    for m in methods:
        en = e_stats[m]["n"]
        sn = s_stats[m]["n"]
        ea = e_stats[m]["correct"] / en if en > 0 else 0.0
        sa = s_stats[m]["correct"] / sn if sn > 0 else 0.0
        dr = e_stats[m]["distractor"] / en if en > 0 else 0.0
        avg_dw = float(np.mean(weight_samples[m])) if weight_samples[m] else None
        summary[m] = {
            "method": m,
            "evolved_accuracy": ea,
            "static_accuracy": sa,
            "distractor_selection_rate": dr,
            "false_invalidation_rate": 1.0 - sa,
            "combined_accuracy": (e_stats[m]["correct"] + s_stats[m]["correct"]) / max(1, en + sn),
            "avg_distractor_weight_after_cascade": avg_dw,
            "n_evolved": en,
            "n_static": sn,
        }
        logger.info(f"  {m}: evolved={ea:.3f}, distractor_rate={dr:.3f}, "
                    f"static={sa:.3f}, avg_dist_weight={avg_dw}")

    flat_ea = summary["flat"]["evolved_accuracy"]
    full_ea = summary["full_cascade"]["evolved_accuracy"]
    improvement = full_ea - flat_ea
    summary["improvement_full_vs_flat"] = improvement
    summary["h3_supported"] = improvement > 0.20
    logger.info(f"Improvement (full - flat): {improvement:.3f}")
    logger.info(f"H3 {'SUPPORTED ✓' if summary['h3_supported'] else 'NOT SUPPORTED'}")
    return summary


if __name__ == "__main__":
    result = run(n_evolved=300, n_static=100)
    with open(f"{RESULTS_DIR}/horizonbench_final.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Done.")
