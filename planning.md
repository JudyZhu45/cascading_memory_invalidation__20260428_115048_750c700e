# Research Planning: Cascading Memory Invalidation in User Memory Graphs

## Motivation & Novelty Assessment

### Why This Research Matters
Current memory systems (MemGPT, Mem0, PersonaMem) store facts as flat key-value pairs and update only on explicit retraction. This creates two concrete failure modes: (A) location-dependent memories persist after a user moves (structural cascade failure), and (B) implicit preference drift is invisible to systems without semantic conflict detection (semantic bridging failure). These failures cause AI assistants to recommend irrelevant or contradictory services to users.

### Gap in Existing Work
From the literature review:
- HorizonBench implements typed dependency edges but uses synthetic data and focuses on preference prediction accuracy, not invalidation propagation
- LoCoMo provides real conversational data but lacks explicit cascade evaluation methodology
- PAMU addresses drift detection but not graph-based propagation
- RippleEdits addresses knowledge graph propagation for factual edits, not user preferences
- **No existing work evaluates cascading invalidation on real conversational data using both structural edges (LOCATED_IN) and semantic conflict edges (CONFLICTS_WITH)**

### Our Novel Contribution
We propose and evaluate a **Memory Graph with Typed Dependency Edges** on the LoCoMo real conversational dataset, testing:
1. Whether structural LOCATED_IN edges can correctly propagate location-change invalidation (Failure A)
2. Whether semantic CONFLICTS_WITH edges established via LLM inference and embedding similarity can detect and propagate preference drift (Failure B + Semantic Bridging Problem)
3. Quantitative comparison of 4 methods: flat memory, recency decay, 1-hop graph, full transitive graph

### Experiment Justification
- **Part 1 (Structural cascade)**: Needed to establish that graph-based invalidation works for explicit structural changes — serves as the "easy case" baseline
- **Part 2 (Semantic bridge)**: Tests all three approaches to CONFLICTS_WITH edge building — directly addresses the open question in the research hypothesis
- **Part 3 (Drift cascade evaluation)**: Full end-to-end evaluation combining structural + semantic cascades on LoCoMo ground truth

---

## Research Question
Can a memory graph with typed dependency edges (LOCATED_IN structural edges + CONFLICTS_WITH semantic edges) automatically invalidate outdated memories in two scenarios: (1) explicit context shifts (user moves location) and (2) implicit preference drift (quiet preference conflicting with bar recommendations)?

## Background and Motivation
See motivation section above. The core technical challenge is the **semantic bridging problem**: to build CONFLICTS_WITH("prefers quiet", "likes bars"), the system needs external knowledge that bars are loud/social — this is absent from conversation history alone.

## Hypothesis Decomposition

### H1: Structural cascade
- H1a: LOCATED_IN edges can correctly identify location-dependent memories (precision > 80%)
- H1b: Cascade propagation correctly invalidates location-dependent memories when root location changes (accuracy > 80%)

### H2: Semantic bridge for CONFLICTS_WITH edges
- H2a (Embedding): cosine distance between quiet/bar preference embeddings exceeds 0.3
- H2b (LLM): LLM inference correctly identifies conflict edges with precision > 60%
- H2c (Behavioral): behavioral co-occurrence signals (negative correlation) can serve as proxy

### H3: Drift cascade superiority
- H3: Full transitive graph cascade outperforms flat memory by > 20pp on recommendation correctness

---

## Proposed Methodology

### Approach
Three-part experiment using LoCoMo (35 multi-session dialogues) as primary dataset, supplemented by HorizonBench (4,245 preference evolution items) for controlled validation.

### Experimental Steps

**Step 1: Data Loading & Parsing**
- Load LoCoMo (35 dialogues) and HorizonBench (4,245 items)
- Parse LoCoMo sessions into structured memory items using LLM extraction
- Annotate location-shift events and preference-drift events manually/via pattern matching
- Build initial memory graphs using NetworkX

**Step 2: Structural Cascade (Part 1)**
- Implement LOCATED_IN edge detection: memory items mentioning locations → linked to root location node
- Simulate location shift events (extracted from LoCoMo dialogues)
- Evaluate: does cascade correctly downweight location-specific memories?
- Metric: precision/recall of correctly invalidated memories vs. gold annotations

**Step 3: Semantic Bridge (Part 2)**
- Method A — Embedding similarity: embed memories using sentence-transformers, compute cosine distance between "prefers quiet" and "likes bars" type pairs
- Method B — LLM inference: prompt GPT-4o-mini to identify CONFLICTS_WITH relationships at memory write time
- Method C — Behavioral co-occurrence: track session-level activity patterns, identify negative correlations
- Evaluate edge precision/recall against manually annotated LoCoMo QA pairs
- Key empirical question: does embedding distance "prefers quiet" vs "bars" > 0.3?

**Step 4: Drift Cascade Evaluation (Part 3)**
- Apply best semantic bridge approach from Part 2
- Implement full cascade: when "prefers quiet" grows stronger → invalidate conflicting "likes bars" memory
- Compare 4 methods: flat memory, recency decay, 1-hop graph, full transitive graph
- Evaluate on LoCoMo QA ground truth: recommendation correctness

**Step 5: Analysis and Visualization**
- Figure 1: Example memory graph before/after location-shift cascade (networkx)
- Figure 2: Semantic bridge approach comparison (bar chart of edge precision)
- Figure 3: Drift cascade accuracy vs. number of behavioral signals
- Figure 4: Cascade accuracy by method (structural vs. drift, flat vs. graph)

### Baselines
1. **Flat memory** (no edges, updates only on explicit retraction)
2. **Recency decay** (memory weight ∝ e^(-λ·Δt))
3. **1-hop graph cascade** (only direct neighbors invalidated)
4. **Full transitive cascade** (proposed, BFS with decay)

### Evaluation Metrics
- `structural_cascade_accuracy`: % of location-dependent memories correctly identified after location shift
- `semantic_edge_precision`: % of CONFLICTS_WITH edges that are valid (vs. gold annotations)
- `drift_cascade_accuracy`: % of QA answers correctly predicted after drift cascade
- `false_invalidation_rate`: % of memories incorrectly invalidated (false positives)
- `recommendation_correctness`: % of recommendations that match current user state

### Statistical Analysis Plan
- Bootstrap confidence intervals (1000 resamples) for all accuracy metrics
- McNemar's test for pairwise method comparisons
- Pearson correlation between behavioral signal count and drift detection accuracy
- Significance level: α = 0.05

---

## Expected Outcomes
- H1 supported if structural cascade accuracy > 80% (LoCoMo location events)
- H2a supported if cosine distance > 0.3 for quiet/noisy preference pairs
- H2b supported if LLM edge precision > 60%
- H3 supported if full cascade improves flat memory by > 20pp

---

## Timeline and Milestones
- Phase 1 (Planning): Complete — 30 min
- Phase 2 (Setup): 15 min (data loading, EDA)  
- Phase 3 (Core implementation): 60 min (graph construction, cascade logic, embedding)
- Phase 4 (Experiments): 45 min (run all parts, collect results)
- Phase 5 (Analysis + Viz): 30 min
- Phase 6 (Documentation): 20 min

Total: ~3.5 hours (within 5.4h limit)

---

## Potential Challenges
1. **LoCoMo has no explicit QA annotations** in the downloaded version → Use LLM to generate QA pairs from dialogues, or use HorizonBench ground truth directly
2. **API rate limits** → Batch requests, implement retry with exponential backoff, cache responses
3. **Embedding "quiet" vs "bars"** may be close in embedding space → Test empirically; try both general (all-MiniLM) and task-specific embeddings
4. **Small sample size (35 dialogues)** → Use bootstrap CIs; supplement with HorizonBench for statistical power

---

## Success Criteria
- [ ] Structural cascade accuracy > 80%
- [ ] At least one semantic bridge approach > 60% edge precision  
- [ ] Empirical answer to embedding distance question (quiet vs bar > 0.3?)
- [ ] Drift cascade outperforms flat memory by > 20pp
- [ ] Failure analysis with ≥ 3 distinct failure modes
