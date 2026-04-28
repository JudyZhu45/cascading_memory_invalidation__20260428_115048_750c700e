# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project on **Cascading Memory Invalidation in User Memory Graphs**.

**Research Hypothesis**: A memory graph with typed dependency edges can propagate invalidation signals to dependent memories without explicit user retraction — via structural LOCATED_IN-type edges (explicit context shift) and semantic CONFLICTS_WITH edges (implicit preference drift, requiring LLM inference at write time or behavioral co-occurrence signals).

---

## Papers

Total papers downloaded: **13**

| Title | Authors | Year | File | Relevance |
|-------|---------|------|------|-----------|
| HorizonBench: Long-Horizon Personalization with Evolving Preferences | Li et al. | 2026 | papers/2604.17283_HorizonBench_evolving_preferences.pdf | **HIGH** — implements mental state graph with typed dependency edges; identifies belief-update failure |
| LoCoMo: Evaluating Very Long-Term Conversational Memory | Maharana et al. | 2024 | papers/2402.17753_LoCoMo_long_term_conversational_memory.pdf | **HIGH** — user-specified benchmark dataset; temporal event graphs |
| PersonaMem-v2: Towards Personalized Intelligence | Jiang et al. | 2025 | papers/2512.06688_PersonaMem_v2_personalized_intelligence.pdf | **HIGH** — agentic memory framework; implicit preference evolution |
| RippleEdits: Evaluating Ripple Effects of Knowledge Editing | Cohen et al. | 2023 | papers/2307.12976_evaluating_ripple_effects_knowledge_editing.pdf | **HIGH** — 6-criteria evaluation framework for cascading graph updates |
| MemGPT: Towards LLMs as Operating Systems | Packer et al. | 2024 | papers/2310.08560_MemGPT_LLMs_as_operating_systems.pdf | **MEDIUM** — hierarchical memory baseline; working context for preferences |
| MemoryBank: Enhancing LLMs with Long-Term Memory | Zhong et al. | 2023 | papers/2305.10250_MemoryBank_enhancing_LLMs_long_term_memory.pdf | **MEDIUM** — Ebbinghaus temporal decay baseline; no graph structure |
| PAMU: Preference-Aware Memory Update for Long-Term LLM Agents | Sun et al. | 2025 | papers/2510.09720_preference_aware_memory_update_LLM_agents.pdf | **HIGH** — SW+EMA drift detection; evaluated on LoCoMo |
| ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing | (see paper) | 2025 | papers/2507.08427_ChainEdit_propagating_ripple_effects.pdf | **HIGH** — KG rules + LLM for multi-hop propagation; CONFLICTS_WITH establishment |
| PersistBench: When Should Long-Term Memories Be Forgotten? | (see paper) | 2026 | papers/2602.01146_PersistBench_long_term_memories_forgotten.pdf | **MEDIUM** — safety framing for memory invalidation |
| Position: Episodic Memory is the Missing Piece | (see paper) | 2025 | papers/2502.06975_episodic_memory_missing_piece_LLM_agents.pdf | **MEDIUM** — theoretical framing; single-shot learning |
| Dynamic Affective Memory Management | (see paper) | 2025 | papers/2510.27418_dynamic_affective_memory_management.pdf | **MEDIUM** — memory staleness and personalized LLM agents |
| MemoryCD: Benchmarking Long-Context User Memory | (see paper) | 2026 | papers/2603.25973_MemoryCD_benchmarking_long_context_user_memory.pdf | **MEDIUM** — real-world lifelong memory benchmark |
| Hypercyclic Composition Operators on OM(R) | (see paper) | 2023 | papers/2310.09104_hypercyclic_mixing_composition_operators.pdf | **LOW** — likely user-specified in error; pure mathematics paper |

See papers/README.md for detailed descriptions.

---

## Datasets

Total datasets downloaded: **2** (locally available)

| Name | Source | Size | Task | Location | Key Features |
|------|--------|------|------|----------|--------------|
| LoCoMo | Aman279/Locomo (HuggingFace) | 35 dialogues, 300+ turns avg | Long-term conversational memory QA | datasets/locomo/ | temporal event graphs, multi-session, temporal reasoning |
| HorizonBench | stellalisy/HorizonBench (HuggingFace) | 4,245 MCQ items, 360 users | Preference evolution tracking | datasets/horizonbench/ | has_evolved flag, distractor_letter, preference_domain, provenance |

Additional benchmark (included in code repository):
- **RippleEdits**: 5K factual edits in `code/RippleEdits/data/benchmark/` (recent.json, random.json, popular.json)

See datasets/README.md for detailed descriptions and download instructions.

### HorizonBench Key Fields for Experiments

```python
from datasets import load_from_disk
ds = load_from_disk('datasets/horizonbench')
test = ds['test']

# Key fields:
# - has_evolved: bool — primary split for cascading invalidation evaluation
# - distractor_letter: str — pre-evolution distractor (measures belief-update failure)
# - preference_domain: str — 30 preference domains (dietary, work_style, etc.)
# - preference_evolution: str — provenance (which life event caused the change)
# - correct_letter: str — ground truth current preference
# - conversation: str — full 6-month conversation history
# - options: dict — 5 response options (A-E)
# - generator: str — claude-sonnet-4.5 | o3 | gemini-3-flash
```

---

## Code Repositories

Total repositories cloned: **2**

| Name | URL | Purpose | Location | Key Files |
|------|-----|---------|----------|-----------|
| HorizonBench | github.com/stellalisy/HorizonBench | Mental state graph + preference evolution framework | code/HorizonBench/ | relation_propagator.py, preference_evolve_manager.py |
| RippleEdits | github.com/edenbiran/RippleEdits | Ripple effect evaluation benchmark | code/RippleEdits/ | data/benchmark/, src/evaluation.py |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **User-specified papers** (5 arXiv IDs): Downloaded directly from arXiv PDF endpoint
2. **Paper-finder service**: Unavailable (server error); fallback to manual search
3. **arXiv API search** (5 rounds): Keywords — memory graph invalidation, user preference drift, personalized memory LLM, knowledge graph temporal reasoning, long-term user modeling, memory invalidation belief update personalization, graph-based memory temporal decay, conversational memory RAG, user memory graph structured, knowledge editing propagation
4. **Targeted searches**: ripple effect knowledge editing, preference-aware memory update, MemoryBank LLM, episodic semantic memory LLM agent
5. **HuggingFace Hub**: Dataset search for LoCoMo and HorizonBench variants

### Selection Criteria

Papers selected based on:
- Direct relevance to typed dependency edges and cascading invalidation (HIGH priority)
- Preference evolution / belief update failure (HIGH priority)  
- Memory management architectures for LLM agents (MEDIUM priority)
- Knowledge editing propagation techniques (HIGH priority)
- Temporal decay / forgetting mechanisms (MEDIUM priority)

### Challenges Encountered

1. **paper-finder service**: Not available (server error 500); used manual arXiv API search instead
2. **snap-research/locomo**: Dataset no longer accessible on HuggingFace under original ID; used mirror at `Aman279/Locomo` (35 dialogues, same format)
3. **LoCoMo QA benchmark** (Percena/locomo-mc10): Download failed (generation error); the dialogue dataset itself is sufficient for experiment design
4. **2310.09104**: Appears to be a mathematics paper (functional analysis) not related to ML/AI; retained as user-specified

### Gaps and Workarounds

- **LoCoMo QA pairs**: The original LoCoMo benchmark includes QA annotations not present in the mirrored dataset. For QA evaluation, refer to the original paper's evaluation code or use the HorizonBench QA format instead.
- **PersonaMem-v2 dataset**: Not downloaded separately; available at `bowen-upenn/PersonaMem-v2` on HuggingFace if needed for fine-tuning experiments
- **HorizonBench full conversation histories**: The 360 user conversation histories with full 6-month timelines are available in the HorizonBench dataset; individual conversations average ~163K tokens

---

## Recommendations for Experiment Design

### 1. Primary Dataset
**HorizonBench** (`datasets/horizonbench/`) — 4,245 items with ground-truth typed dependency edges and preference evolution provenance. Use `has_evolved=True` items to evaluate cascading invalidation, `distractor_letter` to measure belief-update failure rate.

### 2. Baseline Methods
1. **Full context (direct)**: 163K+ tokens → LLM; no memory management
2. **RAG**: Embed conversation turns → retrieve top-k → LLM; fails at belief update
3. **MemGPT-style**: Hierarchical memory with working context; no dependency edges
4. **MemoryBank**: Ebbinghaus decay + event summaries; no graph propagation
5. **Proposed (HorizonBench RelationPropagator)**: Adapt for explicit LOCATED_IN edges

### 3. Novel Contribution
Add CONFLICTS_WITH edges via:
- **Method A**: LLM inference at memory write time (e.g., "Does preferring quiet conflict with going to bars?")
- **Method B**: PAMU-style SW+EMA behavioral co-occurrence detection

### 4. Evaluation Metrics
- Accuracy on evolved preferences (`has_evolved=True` subset)
- Pre-evolution distractor selection rate (≤25% = uniform error baseline; lower = better)
- Δevo = Accuracy(evolved) − Accuracy(static); target > 0 (improved belief update)
- RippleEdits six criteria (adapted for user preference facts)
- Cascade depth vs. accuracy trade-off

### 5. Code Starting Point
Use `code/HorizonBench/src/causal_framework/evolution/relation_propagator.py` as the graph propagation engine. Extend the dependency graph schema to include CONFLICTS_WITH edge type with semantic inference module.
