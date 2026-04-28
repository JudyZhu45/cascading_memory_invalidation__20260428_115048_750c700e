# Downloaded Papers

## User-Specified Papers

1. [LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents](2402.17753_LoCoMo_long_term_conversational_memory.pdf)
   - Authors: Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang
   - Year: 2024
   - arXiv: https://arxiv.org/abs/2402.17753
   - Why relevant: Introduces LoCoMo dataset (50 very long-term dialogues, 300 turns avg, temporal event graphs). Primary benchmark dataset for this research.

2. [Hypercyclic and mixing composition operators on OM(R)](2310.09104_hypercyclic_mixing_composition_operators.pdf)
   - Authors: Mathematics paper
   - Year: 2023
   - arXiv: https://arxiv.org/abs/2310.09104
   - Why relevant: **Likely a mismatch** — this is a mathematics paper about functional analysis operators, not ML/AI. May have been specified in error. Retained as user-specified.

3. [MemGPT: Towards LLMs as Operating Systems](2310.08560_MemGPT_LLMs_as_operating_systems.pdf)
   - Authors: Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez
   - Year: 2024 (Oct 2023)
   - arXiv: https://arxiv.org/abs/2310.08560
   - Why relevant: OS-inspired hierarchical memory management with working context for user preferences. Key baseline architecture lacking typed dependency edges and cascading invalidation.

4. [PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory](2512.06688_PersonaMem_v2_personalized_intelligence.pdf)
   - Authors: Bowen Jiang, Yuan Yuan, Maohao Shen, et al.
   - Year: 2025 (Dec)
   - arXiv: https://arxiv.org/abs/2512.06688
   - Why relevant: State-of-the-art LLM personalization with agentic memory framework, 20K+ implicit preferences, RFT-based memory updating. Most similar to target system architecture.

5. [HorizonBench: Long-Horizon Personalization with Evolving Preferences](2604.17283_HorizonBench_evolving_preferences.pdf)
   - Authors: Shuyue Stella Li, Bhargavi Paranjape, Kerem Oktar, et al.
   - Year: 2026 (April)
   - arXiv: https://arxiv.org/abs/2604.17283
   - Why relevant: **Most directly relevant paper.** Implements mental state graph with typed dependency edges for cascading preference evolution. Identifies belief-update failure as core problem. Provides benchmark (4,245 items) and full code with RelationPropagator.

## Additional Papers Found

6. [MemoryBank: Enhancing Large Language Models with Long-Term Memory](2305.10250_MemoryBank_enhancing_LLMs_long_term_memory.pdf)
   - Authors: Wanjun Zhong et al.
   - Year: 2023
   - arXiv: https://arxiv.org/abs/2305.10250
   - Why relevant: Ebbinghaus Forgetting Curve-based temporal decay mechanism for memory. Baseline for temporal downweighting without graph structure.

7. [Evaluating the Ripple Effects of Knowledge Editing in Language Models](2307.12976_evaluating_ripple_effects_knowledge_editing.pdf)
   - Authors: Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva
   - Year: 2023
   - arXiv: https://arxiv.org/abs/2307.12976
   - Why relevant: Formal framework for ripple effects in knowledge graphs (6 evaluation criteria). Direct methodological template for cascading memory invalidation evaluation.

8. [ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](2507.08427_ChainEdit_propagating_ripple_effects.pdf)
   - Authors: (see paper)
   - Year: 2025 (July)
   - arXiv: https://arxiv.org/abs/2507.08427
   - Why relevant: Addresses multi-hop propagation via KG rules + LLM reasoning — hybrid approach for CONFLICTS_WITH edge establishment.

9. [Preference-Aware Memory Update for Long-Term LLM Agents](2510.09720_preference_aware_memory_update_LLM_agents.pdf)
   - Authors: Haoran Sun, Zekun Zhang, Shaoning Zeng
   - Year: 2025 (Oct)
   - arXiv: https://arxiv.org/abs/2510.09720
   - Why relevant: SW+EMA approach for detecting preference drift in LoCoMo-based evaluation. Behavioral co-occurrence signal for implicit preference change detection.

10. [Dynamic Affective Memory Management for Personalized LLM Agents](2510.27418_dynamic_affective_memory_management.pdf)
    - Authors: (see paper)
    - Year: 2025 (Oct)
    - arXiv: https://arxiv.org/abs/2510.27418
    - Why relevant: Memory staleness and redundancy challenges in personalized LLM agents.

11. [MemoryCD: Benchmarking Long-Context User Memory of LLM Agents](2603.25973_MemoryCD_benchmarking_long_context_user_memory.pdf)
    - Authors: (see paper)
    - Year: 2026 (March)
    - arXiv: https://arxiv.org/abs/2603.25973
    - Why relevant: Large-scale real-world memory benchmark from Amazon Review data. Cross-domain lifelong personalization.

12. [PersistBench: When Should Long-Term Memories Be Forgotten by LLMs?](2602.01146_PersistBench_long_term_memories_forgotten.pdf)
    - Authors: (see paper)
    - Year: 2026 (Feb)
    - arXiv: https://arxiv.org/abs/2602.01146
    - Why relevant: Safety risks of memory persistence — motivates cascading invalidation. When should memories be forgotten (invalidated)?

13. [Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents](2502.06975_episodic_memory_missing_piece_LLM_agents.pdf)
    - Authors: (see paper)
    - Year: 2025 (Feb)
    - arXiv: https://arxiv.org/abs/2502.06975
    - Why relevant: Episodic memory as biological inspiration for instance-specific memory in LLM agents. Provides theoretical framing for memory structure.
