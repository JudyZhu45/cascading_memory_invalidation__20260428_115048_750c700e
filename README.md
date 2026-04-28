# Cascading Memory Invalidation in User Memory Graphs

**Research**: Cascading Memory Invalidation in User Memory Graphs | NeuriCo, 2026-04-28

## Key Findings

- **Structural cascade (LOCATED_IN)**: 81.7% accuracy on LoCoMo location-shift events ✓ (target: >80%)
- **Semantic bridge (embedding)**: Mean cosine distance quiet/noisy = 0.69 (>>0.3 threshold) ✓
- **Semantic bridge (LLM)**: 100% precision, 90.9% recall on conflict detection benchmark ✓ (target: >60%)
- **Drift cascade (full cascade vs flat)**: +32.2pp improvement on HorizonBench evolved preferences ✓ (target: >20pp)

**Critical implementation insight**: CONFLICTS_WITH edges must be **directed** (current→old). Bidirectional edges cause mutual cancellation — a newly identified implementation failure mode.

## How to Reproduce

```bash
cd /workspaces/cascading_memory_invalidation__20260428_115048_750c700e

# Run structural cascade + semantic bridge experiments (Parts 1 & 2)
python3 src/run_experiments_v2.py

# Run HorizonBench drift cascade evaluation (Part 3)
python3 src/run_horizonbench_final.py

# Generate all 4 figures
python3 src/generate_figures.py
```

**Requirements**: `OPENAI_API_KEY2` env var (gpt-4o-mini), datasets in `datasets/`

## File Structure

```
datasets/locomo/          # 35 LoCoMo dialogues (35 rows)
datasets/horizonbench/    # 4,245 HorizonBench MCQ items
src/memory_graph.py       # MemoryGraph with typed edges + cascade logic
src/edge_builders.py      # LOCATED_IN + CONFLICTS_WITH (3 methods)
src/run_experiments_v2.py # Parts 1 & 2: structural cascade + semantic bridge
src/run_horizonbench_final.py  # Part 3: drift cascade evaluation
src/generate_figures.py   # 4 publication figures
results/                  # JSON results files
figures/                  # fig1-fig4 PNG visualizations
REPORT.md                 # Full research report with methods, results, analysis
planning.md               # Research plan
```

## Failure Modes Documented

1. Bidirectional edge cancellation (→ cascade becomes no-op, same as flat)
2. Structural cascade over-reach (broad keyword heuristics create false LOCATED_IN edges)
3. LLM blindness to location-semantic conflicts (world knowledge gap)
4. Behavioral signal sparsity (needs 1000+ sessions for reliable correlations)
5. Cascade depth sensitivity (2-3 hops sufficient; diminishing returns beyond depth 3)

See [REPORT.md](REPORT.md) for full methodology, results, analysis, and literature comparison.
