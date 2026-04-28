"""Global configuration for the cascading memory invalidation experiments."""

import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY2", os.environ.get("OPENAI_API_KEY", ""))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
LLM_MODEL = "gpt-4o-mini"

# Paths
WORKSPACE = "/workspaces/cascading_memory_invalidation__20260428_115048_750c700e"
DATASETS_DIR = f"{WORKSPACE}/datasets"
RESULTS_DIR = f"{WORKSPACE}/results"
FIGURES_DIR = f"{WORKSPACE}/figures"
LOCOMO_DIR = f"{DATASETS_DIR}/locomo"
HORIZONBENCH_DIR = f"{DATASETS_DIR}/horizonbench"

# Memory graph parameters
LOCATED_IN_THRESHOLD = 0.5       # Min keyword overlap to assign LOCATED_IN edge
CONFLICTS_WITH_EMBEDDING_THRESHOLD = 0.3  # Cosine distance threshold for semantic conflict
CONFLICTS_WITH_LLM_CONFIDENCE = 0.6       # LLM confidence threshold for CONFLICTS_WITH
CASCADE_DECAY = 0.7              # Decay per hop during cascade
CASCADE_MAX_DEPTH = 3            # Maximum cascade propagation depth
RECENCY_DECAY_LAMBDA = 0.01      # Recency decay rate (per day)

# Experiment parameters
N_LOCOMO_DIALOGUES = 35          # Use all available LoCoMo dialogues
MAX_MEMORIES_PER_DIALOGUE = 50   # Cap memory extraction per dialogue
N_QA_PAIRS_PER_DIALOGUE = 10     # QA pairs to generate per dialogue
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 2.0
