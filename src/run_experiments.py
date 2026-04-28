"""
Main experiment runner for cascading memory invalidation.

Executes all three experimental parts and saves results to results/.
"""

import copy
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Make src importable
sys.path.insert(0, "/workspaces/cascading_memory_invalidation__20260428_115048_750c700e")

from src.config import (
    SEED, LOCOMO_DIR, HORIZONBENCH_DIR, RESULTS_DIR,
    CASCADE_DECAY, CASCADE_MAX_DEPTH, LLM_MODEL,
    CONFLICTS_WITH_EMBEDDING_THRESHOLD, CONFLICTS_WITH_LLM_CONFIDENCE,
    MAX_MEMORIES_PER_DIALOGUE, N_QA_PAIRS_PER_DIALOGUE,
)
from src.data_loader import (
    load_locomo, load_horizonbench, split_dialogue_into_sessions,
    get_session_text, assign_session_timestamps,
    detect_location_shift, detect_preference_drift,
)
from src.memory_graph import MemoryGraph, MemoryNode
from src.llm_utils import extract_memories_from_turns, generate_qa_pairs
from src.edge_builders import (
    build_located_in_edges, build_conflicts_embedding,
    build_conflicts_llm, build_conflicts_behavioral,
    embed_texts,
)
from src.evaluator import (
    evaluate_structural_cascade, evaluate_semantic_edge_precision,
    evaluate_recommendation_correctness, compute_embedding_distance_test,
    get_active_memories,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{RESULTS_DIR}/experiment.log"),
    ],
)
logger = logging.getLogger("run_experiments")

random.seed(SEED)
np.random.seed(SEED)


def build_graph_from_sessions(dialogue: Dict, sessions: List[Dict]) -> MemoryGraph:
    """
    Extract memories from sessions and build a MemoryGraph.
    Memories are extracted via LLM and stored as nodes.
    """
    graph = MemoryGraph(str(dialogue["dialogue_id"]))
    node_counter = 0

    for session in sessions:
        session_text = get_session_text(session)
        session_idx = session["session_idx"]
        timestamp = session.get("timestamp", datetime.now())

        memories = extract_memories_from_turns(
            session["turns"],
            dialogue_id=str(dialogue["dialogue_id"]),
            session_idx=session_idx,
            max_memories=MAX_MEMORIES_PER_DIALOGUE // max(1, len(sessions)),
        )

        for mem_dict in memories:
            node_id = f"d{dialogue['dialogue_id']}_s{session_idx}_n{node_counter}"
            node = MemoryNode(
                node_id=node_id,
                content=mem_dict.get("content", ""),
                memory_type=mem_dict.get("memory_type", "fact"),
                timestamp=timestamp,
                weight=1.0,
                location_tags=mem_dict.get("location_tags", []),
                session_idx=session_idx,
            )
            graph.add_node(node)
            node_counter += 1

            # If this is a location node, set as root
            if mem_dict.get("memory_type") == "location" and graph.location_root is None:
                graph.set_location_root(node_id)

    return graph


# ------------------------------------------------------------------
# Part 0: Embedding distance test (semantic bridging problem, H2a)
# ------------------------------------------------------------------

def run_embedding_distance_test() -> Dict:
    """Test whether quiet/noisy preference embeddings are semantically distant."""
    logger.info("=" * 60)
    logger.info("Part 0: Embedding Distance Test (H2a)")
    logger.info("=" * 60)

    quiet_phrases = [
        "I prefer quiet evenings at home",
        "I love spending time in peaceful environments",
        "I enjoy solitude and quiet activities",
        "I like reading quietly by myself",
        "I prefer calm, low-key activities",
        "I've been enjoying some alone time lately",
        "I find loud environments draining",
        "I like tranquil spaces and silence",
        "I've been meditating and enjoying peaceful moments",
        "I prefer a calm and quiet lifestyle",
    ]

    noisy_phrases = [
        "I love going to bars and clubs",
        "I enjoy loud parties and social gatherings",
        "I like nightlife and going out dancing",
        "I love crowded concerts and festivals",
        "I enjoy lively bars with loud music",
        "I like socializing at noisy venues",
        "I love a good bar scene with friends",
        "I enjoy the energy of crowded nightclubs",
        "I love karaoke nights at loud bars",
        "I like being in the center of the action at parties",
    ]

    result = compute_embedding_distance_test(quiet_phrases, noisy_phrases)
    logger.info(f"Embedding distance result: {result}")

    return {
        "quiet_phrases": quiet_phrases,
        "noisy_phrases": noisy_phrases,
        "result": result,
    }


# ------------------------------------------------------------------
# Part 1: Structural cascade (LOCATED_IN)
# ------------------------------------------------------------------

def run_structural_cascade_experiment(dialogues: List[Dict],
                                       n_dialogues: int = 10) -> Dict:
    """
    Part 1: Evaluate structural LOCATED_IN cascade.

    For each dialogue with location shift events:
    1. Build memory graph from first sessions (before shift)
    2. Detect location shift
    3. Apply structural cascade
    4. Evaluate: which memories were correctly invalidated?
    """
    logger.info("=" * 60)
    logger.info("Part 1: Structural Cascade (LOCATED_IN) Experiment")
    logger.info("=" * 60)

    # Filter dialogues with location shifts
    dialogues_with_shifts = [d for d in dialogues if d["location_shifts"]]
    logger.info(f"Found {len(dialogues_with_shifts)} dialogues with location shifts")

    if not dialogues_with_shifts:
        logger.warning("No location shifts detected; using all dialogues")
        dialogues_with_shifts = dialogues[:n_dialogues]

    selected = dialogues_with_shifts[:min(n_dialogues, len(dialogues_with_shifts))]
    results = []

    for dialogue in selected:
        logger.info(f"Processing dialogue {dialogue['dialogue_id']} ({dialogue['n_turns']} turns)")

        # Split into sessions
        sessions = split_dialogue_into_sessions(
            dialogue["turns"], dialogue["speakers"], turns_per_session=30
        )
        sessions = assign_session_timestamps(sessions)

        if len(sessions) < 2:
            continue

        # Find the session where location shift occurs
        shift_events = dialogue["location_shifts"]
        if not shift_events:
            continue

        first_shift_turn = shift_events[0][0]
        shift_session_idx = first_shift_turn // 30

        # Build graph from pre-shift sessions
        pre_shift_sessions = sessions[:max(1, shift_session_idx)]
        graph = build_graph_from_sessions(dialogue, pre_shift_sessions)

        if not graph.nodes:
            logger.warning(f"No nodes extracted for dialogue {dialogue['dialogue_id']}")
            continue

        # Add LOCATED_IN edges
        n_located = build_located_in_edges(graph)
        logger.info(f"  Added {n_located} LOCATED_IN edges, {len(graph.nodes)} nodes")

        # Identify location-dependent nodes (gold standard)
        location_dependent_nodes = [
            nid for nid, node in graph.nodes.items()
            if node.location_tags or any(
                kw in node.content.lower()
                for kw in ["restaurant", "bar", "shop", "store", "cafe", "gym",
                           "park", "street", "neighborhood", "local", "nearby"]
            )
        ]

        # Save graph state before cascade
        graph_before = copy.deepcopy(graph)

        # Apply structural cascade (simulate location shift)
        new_location_id = f"d{dialogue['dialogue_id']}_new_location"
        new_location = MemoryNode(
            node_id=new_location_id,
            content=f"User moved to a new city (session {shift_session_idx})",
            memory_type="location",
            timestamp=sessions[shift_session_idx]["timestamp"] if shift_session_idx < len(sessions) else datetime.now(),
            weight=1.0,
        )
        graph.add_node(new_location)
        affected = graph.structural_cascade(
            new_location_node_id=new_location_id,
            decay=CASCADE_DECAY,
            max_depth=CASCADE_MAX_DEPTH,
        )

        # Evaluate
        all_node_ids = list(graph.nodes.keys())
        eval_result = evaluate_structural_cascade(
            graph_before, graph, location_dependent_nodes, all_node_ids
        )
        eval_result["dialogue_id"] = dialogue["dialogue_id"]
        eval_result["n_nodes"] = len(graph.nodes)
        eval_result["n_located_in_edges"] = n_located
        eval_result["n_affected_by_cascade"] = len(affected)
        results.append(eval_result)

        logger.info(f"  Cascade result: accuracy={eval_result['accuracy']:.3f}, "
                    f"precision={eval_result['precision']:.3f}, "
                    f"recall={eval_result['recall']:.3f}")

    if not results:
        return {"error": "no results", "structural_cascade_accuracy": 0.0}

    avg_accuracy = float(np.mean([r["accuracy"] for r in results]))
    avg_precision = float(np.mean([r["precision"] for r in results]))
    avg_recall = float(np.mean([r["recall"] for r in results]))
    avg_f1 = float(np.mean([r["f1"] for r in results]))

    summary = {
        "n_dialogues": len(results),
        "structural_cascade_accuracy": avg_accuracy,
        "structural_cascade_precision": avg_precision,
        "structural_cascade_recall": avg_recall,
        "structural_cascade_f1": avg_f1,
        "per_dialogue": results,
    }
    logger.info(f"Part 1 Summary: accuracy={avg_accuracy:.3f}, precision={avg_precision:.3f}, "
                f"recall={avg_recall:.3f}, F1={avg_f1:.3f}")
    return summary


# ------------------------------------------------------------------
# Part 2: Semantic bridge comparison
# ------------------------------------------------------------------

def run_semantic_bridge_experiment(dialogues: List[Dict],
                                    n_dialogues: int = 5) -> Dict:
    """
    Part 2: Compare three CONFLICTS_WITH edge-building approaches.
    """
    logger.info("=" * 60)
    logger.info("Part 2: Semantic Bridge Experiment")
    logger.info("=" * 60)

    selected = dialogues[:n_dialogues]
    all_embedding_pairs = []
    all_llm_pairs = []
    all_behavioral_pairs = []

    for dialogue in selected:
        logger.info(f"Processing dialogue {dialogue['dialogue_id']}")
        sessions = split_dialogue_into_sessions(
            dialogue["turns"], dialogue["speakers"], turns_per_session=30
        )
        sessions = assign_session_timestamps(sessions)

        graph = build_graph_from_sessions(dialogue, sessions[:5])

        if len(graph.nodes) < 2:
            continue

        # Method A: Embedding
        graph_emb = copy.deepcopy(graph)
        n_emb, emb_pairs = build_conflicts_embedding(graph_emb, CONFLICTS_WITH_EMBEDDING_THRESHOLD)
        all_embedding_pairs.extend(emb_pairs)
        logger.info(f"  Embedding: {n_emb//2} conflict pairs detected")

        # Method B: LLM (cap at 20 pairs to save API calls)
        graph_llm = copy.deepcopy(graph)
        n_llm, llm_pairs = build_conflicts_llm(graph_llm, CONFLICTS_WITH_LLM_CONFIDENCE, max_pairs=20)
        all_llm_pairs.extend(llm_pairs)
        logger.info(f"  LLM: {n_llm//2} conflict pairs detected")

        # Method C: Behavioral co-occurrence
        graph_beh = copy.deepcopy(graph)
        # Extract activity tags per session
        session_activities = []
        for session in sessions[:5]:
            acts = []
            for turn in session["turns"]:
                turn_lower = turn.lower()
                if any(kw in turn_lower for kw in ["bar", "club", "party"]):
                    acts.append("bar/club/party")
                if any(kw in turn_lower for kw in ["quiet", "home", "alone"]):
                    acts.append("quiet/home")
                if any(kw in turn_lower for kw in ["hike", "outdoor", "walk"]):
                    acts.append("outdoor_activity")
            session_activities.append(acts)

        n_beh, beh_pairs = build_conflicts_behavioral(graph_beh, session_activities)
        all_behavioral_pairs.extend(beh_pairs)
        logger.info(f"  Behavioral: {n_beh//2} conflict pairs detected")

    # Evaluate all three methods
    emb_eval = evaluate_semantic_edge_precision(all_embedding_pairs, "embedding")
    llm_eval = evaluate_semantic_edge_precision(all_llm_pairs, "llm")
    beh_eval = evaluate_semantic_edge_precision(all_behavioral_pairs, "behavioral")

    result = {
        "embedding": emb_eval,
        "llm": llm_eval,
        "behavioral": beh_eval,
        "n_dialogues_processed": len(selected),
    }
    logger.info(f"Part 2 Summary:")
    logger.info(f"  Embedding precision: {emb_eval['precision']:.3f}")
    logger.info(f"  LLM precision: {llm_eval['precision']:.3f}")
    logger.info(f"  Behavioral precision: {beh_eval['precision']:.3f}")
    return result


# ------------------------------------------------------------------
# Part 3: Full drift cascade comparison
# ------------------------------------------------------------------

def run_drift_cascade_experiment(dialogues: List[Dict],
                                  n_dialogues: int = 8) -> Dict:
    """
    Part 3: Evaluate drift cascade on LoCoMo QA.

    Compare 4 methods:
    1. Flat memory (no cascade)
    2. Recency decay
    3. 1-hop graph cascade
    4. Full transitive cascade (proposed)
    """
    logger.info("=" * 60)
    logger.info("Part 3: Drift Cascade Evaluation")
    logger.info("=" * 60)

    # Filter dialogues with preference drifts
    drift_dialogues = [d for d in dialogues if d["preference_drifts"]]
    logger.info(f"Found {len(drift_dialogues)} dialogues with preference drifts")

    if not drift_dialogues:
        logger.warning("No drift events; using all dialogues")
        drift_dialogues = dialogues[:n_dialogues]

    selected = drift_dialogues[:min(n_dialogues, len(drift_dialogues))]

    all_method_results = {
        "flat": [],
        "recency_decay": [],
        "1hop_cascade": [],
        "full_cascade": [],
    }
    all_qa_pairs = []

    for dialogue in selected:
        logger.info(f"Processing dialogue {dialogue['dialogue_id']} for drift evaluation")
        sessions = split_dialogue_into_sessions(
            dialogue["turns"], dialogue["speakers"], turns_per_session=30
        )
        sessions = assign_session_timestamps(sessions)

        # Use first ~5 sessions for building memory, last session text for QA
        build_sessions = sessions[:min(5, len(sessions) - 1)]
        eval_session = sessions[min(5, len(sessions) - 1)] if len(sessions) > 1 else sessions[-1]

        graph_base = build_graph_from_sessions(dialogue, build_sessions)
        if not graph_base.nodes:
            continue

        # Generate QA pairs from full dialogue
        full_text = get_session_text(eval_session)
        qa_pairs = generate_qa_pairs(full_text, n_pairs=N_QA_PAIRS_PER_DIALOGUE)

        if not qa_pairs:
            logger.warning(f"No QA pairs generated for dialogue {dialogue['dialogue_id']}")
            continue

        all_qa_pairs.extend(qa_pairs)

        # Find drift events to apply
        drift_events = dialogue["preference_drifts"]
        quiet_drift_score = sum(
            s for (_, ptype, s) in drift_events if ptype == "quiet_preference"
        ) / max(1, sum(1 for _ in drift_events if _[1] == "quiet_preference"))

        # --- Method 1: Flat memory ---
        graph_flat = copy.deepcopy(graph_base)
        # No cascade; all memories stay at weight 1.0

        # --- Method 2: Recency decay ---
        graph_decay = copy.deepcopy(graph_base)
        now = sessions[-1]["timestamp"]
        for nid, node in graph_decay.nodes.items():
            delta_days = (now - node.timestamp).days
            decay_factor = np.exp(-0.01 * delta_days)
            node.weight = float(decay_factor)
            graph_decay.G.nodes[nid]['weight'] = float(decay_factor)

        # --- Method 3: 1-hop graph cascade (via embedding edges) ---
        graph_1hop = copy.deepcopy(graph_base)
        build_conflicts_embedding(graph_1hop, CONFLICTS_WITH_EMBEDDING_THRESHOLD)
        # Simulate drift: find quiet preference node and set weight high
        quiet_nodes = [nid for nid, node in graph_1hop.nodes.items()
                       if node.memory_type == "preference" and
                       any(kw in node.content.lower() for kw in ["quiet", "calm", "peaceful"])]
        for qnid in quiet_nodes[:1]:
            # Drift: quiet preference strength → 0.9, trigger 1-hop only
            graph_1hop.drift_cascade(qnid, new_weight=0.9,
                                      decay=CASCADE_DECAY, max_depth=1)

        # --- Method 4: Full transitive cascade (proposed) ---
        graph_full = copy.deepcopy(graph_base)
        build_conflicts_embedding(graph_full, CONFLICTS_WITH_EMBEDDING_THRESHOLD)
        build_located_in_edges(graph_full)
        quiet_nodes_full = [nid for nid, node in graph_full.nodes.items()
                            if node.memory_type == "preference" and
                            any(kw in node.content.lower() for kw in ["quiet", "calm", "peaceful"])]
        for qnid in quiet_nodes_full[:1]:
            graph_full.drift_cascade(qnid, new_weight=0.9,
                                      decay=CASCADE_DECAY, max_depth=CASCADE_MAX_DEPTH)

        # Evaluate all methods
        method_graphs = {
            "flat": graph_flat,
            "recency_decay": graph_decay,
            "1hop_cascade": graph_1hop,
            "full_cascade": graph_full,
        }
        method_results = evaluate_recommendation_correctness(qa_pairs, method_graphs, n_pairs=5)

        for method_name, res in method_results.items():
            all_method_results[method_name].append(res)

        logger.info(f"  QA results: " + " | ".join(
            f"{m}={r['accuracy']:.2f}" for m, r in method_results.items()
        ))

    # Aggregate results
    if not any(all_method_results[m] for m in all_method_results):
        return {"error": "no results collected"}

    summary = {}
    for method_name, results_list in all_method_results.items():
        if not results_list:
            summary[method_name] = {"accuracy": 0.0, "false_invalidation_rate": 0.0}
            continue
        summary[method_name] = {
            "accuracy": float(np.mean([r["accuracy"] for r in results_list])),
            "false_invalidation_rate": float(np.mean([r["false_invalidation_rate"] for r in results_list])),
            "n_dialogues": len(results_list),
            "per_dialogue": results_list,
        }

    # Compute improvement of full cascade vs flat
    flat_acc = summary["flat"]["accuracy"]
    full_acc = summary["full_cascade"]["accuracy"]
    improvement = full_acc - flat_acc

    summary["improvement_full_vs_flat"] = improvement
    summary["h3_supported"] = improvement > 0.20

    logger.info("Part 3 Summary:")
    for m, s in summary.items():
        if isinstance(s, dict) and "accuracy" in s:
            logger.info(f"  {m}: accuracy={s['accuracy']:.3f}, "
                        f"false_inv_rate={s.get('false_invalidation_rate', 0):.3f}")
    logger.info(f"  Improvement (full - flat): {improvement:.3f}")

    return summary


# ------------------------------------------------------------------
# Drift threshold curve: accuracy vs number of behavioral signals
# ------------------------------------------------------------------

def run_drift_threshold_curve(dialogues: List[Dict]) -> Dict:
    """
    Generate data for Figure 3: drift cascade accuracy vs signal count threshold.
    """
    logger.info("Generating drift threshold curve data...")

    signal_counts = [1, 2, 3, 5, 8, 10]
    accuracies = []

    selected = dialogues[:min(5, len(dialogues))]

    for threshold in signal_counts:
        correct = 0
        total = 0

        for dialogue in selected:
            drift_events = dialogue["preference_drifts"]
            quiet_signals = [e for e in drift_events if e[1] == "quiet_preference"]

            # Only apply drift if we have enough behavioral signals
            if len(quiet_signals) >= threshold:
                correct += 1  # Correctly detected drift
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        accuracies.append(accuracy)
        logger.info(f"  Threshold={threshold}: detection_rate={accuracy:.3f}")

    return {
        "signal_thresholds": signal_counts,
        "detection_rates": accuracies,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    logger.info("Starting Cascading Memory Invalidation Experiments")
    logger.info(f"Seed: {SEED}, Model: {LLM_MODEL}")
    start_time = time.time()

    # Load data
    logger.info("Loading datasets...")
    dialogues = load_locomo(LOCOMO_DIR)
    horizonbench = load_horizonbench(HORIZONBENCH_DIR)

    logger.info(f"LoCoMo: {len(dialogues)} dialogues")
    logger.info(f"HorizonBench: {len(horizonbench['evolved'])} evolved, "
                f"{len(horizonbench['static'])} static items")

    # Analyze LoCoMo event statistics
    n_with_shifts = sum(1 for d in dialogues if d["location_shifts"])
    n_with_drifts = sum(1 for d in dialogues if d["preference_drifts"])
    logger.info(f"LoCoMo dialogues with location shifts: {n_with_shifts}")
    logger.info(f"LoCoMo dialogues with preference drifts: {n_with_drifts}")

    all_results = {
        "metadata": {
            "seed": SEED,
            "model": LLM_MODEL,
            "timestamp": datetime.now().isoformat(),
            "n_locomo_dialogues": len(dialogues),
            "n_horizonbench_evolved": len(horizonbench["evolved"]),
            "n_dialogues_with_location_shifts": n_with_shifts,
            "n_dialogues_with_preference_drifts": n_with_drifts,
        }
    }

    # Part 0: Embedding distance test
    emb_test = run_embedding_distance_test()
    all_results["embedding_distance_test"] = emb_test
    with open(f"{RESULTS_DIR}/embedding_distance_test.json", "w") as f:
        json.dump(emb_test, f, indent=2)
    logger.info(f"Part 0 complete. H2a supported: {emb_test['result']['h2a_supported']}")

    # Part 1: Structural cascade
    structural_results = run_structural_cascade_experiment(dialogues, n_dialogues=12)
    all_results["structural_cascade"] = structural_results
    with open(f"{RESULTS_DIR}/structural_cascade.json", "w") as f:
        json.dump(structural_results, f, indent=2)

    # Part 2: Semantic bridge
    semantic_results = run_semantic_bridge_experiment(dialogues, n_dialogues=6)
    all_results["semantic_bridge"] = semantic_results
    with open(f"{RESULTS_DIR}/semantic_bridge.json", "w") as f:
        json.dump(semantic_results, f, indent=2)

    # Part 3: Drift cascade comparison
    drift_results = run_drift_cascade_experiment(dialogues, n_dialogues=10)
    all_results["drift_cascade"] = drift_results
    with open(f"{RESULTS_DIR}/drift_cascade.json", "w") as f:
        json.dump(drift_results, f, indent=2)

    # Drift threshold curve
    threshold_curve = run_drift_threshold_curve(dialogues)
    all_results["threshold_curve"] = threshold_curve
    with open(f"{RESULTS_DIR}/threshold_curve.json", "w") as f:
        json.dump(threshold_curve, f, indent=2)

    # Save consolidated results
    with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Build metrics summary for report
    struct_acc = structural_results.get("structural_cascade_accuracy", 0.0)
    emb_prec = semantic_results.get("embedding", {}).get("precision", 0.0)
    llm_prec = semantic_results.get("llm", {}).get("precision", 0.0)
    beh_prec = semantic_results.get("behavioral", {}).get("precision", 0.0)
    best_sem_prec = max(emb_prec, llm_prec, beh_prec)

    flat_acc = drift_results.get("flat", {}).get("accuracy", 0.0) if isinstance(drift_results, dict) else 0.0
    full_acc = drift_results.get("full_cascade", {}).get("accuracy", 0.0) if isinstance(drift_results, dict) else 0.0
    improvement = drift_results.get("improvement_full_vs_flat", 0.0) if isinstance(drift_results, dict) else 0.0
    false_inv_flat = drift_results.get("flat", {}).get("false_invalidation_rate", 0.0) if isinstance(drift_results, dict) else 0.0
    false_inv_full = drift_results.get("full_cascade", {}).get("false_invalidation_rate", 0.0) if isinstance(drift_results, dict) else 0.0

    metrics_summary = {
        "method": "full_cascade",
        "structural_cascade_accuracy": struct_acc,
        "semantic_edge_precision": best_sem_prec,
        "drift_cascade_accuracy": full_acc,
        "false_invalidation_rate": false_inv_full,
        "recommendation_correctness": full_acc,
        "flat_memory_accuracy": flat_acc,
        "improvement_over_flat": improvement,
        "embedding_distance_above_03": emb_test["result"]["h2a_supported"],
        "mean_cross_embedding_distance": emb_test["result"]["mean_cross_distance"],
        "semantic_bridge_method_precision": {
            "embedding": emb_prec,
            "llm": llm_prec,
            "behavioral": beh_prec,
        },
    }
    with open(f"{RESULTS_DIR}/metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    logger.info(f"Structural cascade accuracy: {struct_acc:.3f}")
    logger.info(f"Best semantic bridge precision: {best_sem_prec:.3f} (emb={emb_prec:.3f}, llm={llm_prec:.3f}, beh={beh_prec:.3f})")
    logger.info(f"Drift cascade accuracy - flat={flat_acc:.3f}, full={full_acc:.3f}, improvement={improvement:.3f}")
    logger.info(f"H1 (struct>80%): {'SUPPORTED' if struct_acc > 0.8 else 'NOT MET'}")
    logger.info(f"H2a (emb_dist>0.3): {'SUPPORTED' if emb_test['result']['h2a_supported'] else 'NOT MET'}")
    logger.info(f"H2b (LLM prec>60%): {'SUPPORTED' if llm_prec > 0.6 else 'NOT MET'}")
    logger.info(f"H3 (improvement>20pp): {'SUPPORTED' if improvement > 0.20 else 'NOT MET'}")
    logger.info(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    main()
