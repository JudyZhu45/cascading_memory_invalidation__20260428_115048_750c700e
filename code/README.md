# Cloned Repositories

## Repository 1: HorizonBench

- **URL**: https://github.com/stellalisy/HorizonBench
- **Purpose**: Complete implementation of mental state graph with typed dependency edges for preference evolution. Includes benchmark generation, evaluation pipeline, and analysis scripts.
- **Location**: code/HorizonBench/
- **Key Files**:
  - `src/causal_framework/evolution/relation_propagator.py`: **Core cascading invalidation mechanism** — propagates changes from source preferences to related preferences via dependency graph traversal. Configurable: propagation_depth=2, min_strength_threshold=0.2, propagation_decay=0.7, max_propagation_changes=5
  - `src/causal_framework/evolution/preference_evolve_manager.py`: Orchestrates preference evolution using four mechanisms: stability bias, contextual adaptation, experience-driven change, relational interdependence
  - `src/causal_framework/models/preference_model.py`: Preference node with evolution history, expression timestamp tracking, last_expressed_date
  - `src/causal_framework/models/user_model.py`: User agent state model
  - `src/causal_framework/models/event_model.py`: Life event representation (triggers for preference change)
  - `src/pipeline/generation_pipeline.py`: Full pipeline for generating benchmark conversations from mental state graph
  - `evaluate.py`: Evaluates any litellm-compatible model on HorizonBench
  - `scripts/analyze_accuracy.py`: Accuracy analysis by model, generator, preference type
  - `scripts/analyze_controlled_v2.py`: Controlled dimension analysis (evolution status, expression explicitness, context length)
- **Dependencies**: See requirements.txt and pyproject.toml
- **Potential Application**: 
  - Adapt RelationPropagator as the core engine for cascading memory invalidation
  - Use PreferenceEvolveManager to simulate preference drift scenarios for testing
  - Extend dependency graph schema to add CONFLICTS_WITH edges beyond existing structural types
  - Use expression tracking mechanism to implement the staleness threshold for invalidation eligibility

## Repository 2: RippleEdits

- **URL**: https://github.com/edenbiran/RippleEdits
- **Purpose**: Benchmark and evaluation code for ripple effects in knowledge editing. Provides 5K factual edits with 6-criteria ripple effect evaluation.
- **Location**: code/RippleEdits/
- **Key Files**:
  - `data/benchmark/recent.json`: Recently-changed factual edits with ripple effect annotations
  - `data/benchmark/random.json`: Random factual edits
  - `data/benchmark/popular.json`: Popular entity edits
  - `src/evaluation.py`: Evaluation code for six criteria (LG, CI, CII, SA, PV, RS)
  - `src/benchmark.py`: Benchmark data loading and manipulation
  - `src/build_benchmark.py`: Code for generating the benchmark from Wikidata
  - `src/modeleditor.py`: Knowledge editing methods (ROME, MEMIT, etc.)
  - `src/testrunner.py`: Test execution for ripple effect evaluation
- **Dependencies**: See requirements.txt
- **Potential Application**:
  - Adapt six evaluation criteria for typed dependency edges (LG → LOCATED_IN structural validation; CI/CII → CONFLICTS_WITH semantic chain validation)
  - Use severity metric concept to measure cascade depth in user memory graphs
  - Adapt benchmark format for user preference facts (entity = user, relation = preference_domain, object = preference_value)
