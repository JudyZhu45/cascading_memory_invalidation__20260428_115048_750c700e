"""
LLM utility functions for memory extraction and semantic inference.
Uses OpenAI API with retry logic and response caching.
"""

import os
import json
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import openai

logger = logging.getLogger(__name__)

# Initialize OpenAI client
_client = None

def get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY2", os.environ.get("OPENAI_API_KEY", "")),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1"),
        )
    return _client


# Simple file-based cache for LLM responses
_CACHE_DIR = Path("/workspaces/cascading_memory_invalidation__20260428_115048_750c700e/results/llm_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(prompt: str, model: str) -> str:
    h = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
    return h


def _load_cache(key: str) -> Optional[str]:
    cache_file = _CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["response"]
    return None


def _save_cache(key: str, response: str) -> None:
    cache_file = _CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps({"response": response}))


def llm_call(prompt: str, model: str = "gpt-4o-mini",
             system: str = "You are a helpful assistant.",
             max_tokens: int = 512,
             temperature: float = 0.0,
             use_cache: bool = True,
             retry_attempts: int = 3) -> str:
    """Call the LLM with caching and retry logic."""
    cache_key = _cache_key(f"{system}\n{prompt}", model)

    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    client = get_client()
    last_error = None

    for attempt in range(retry_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = resp.choices[0].message.content.strip()
            if use_cache:
                _save_cache(cache_key, result)
            return result
        except openai.RateLimitError:
            wait = 2 ** attempt * 2
            logger.warning(f"Rate limit hit; waiting {wait}s (attempt {attempt+1})")
            time.sleep(wait)
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
            time.sleep(1)

    raise RuntimeError(f"LLM call failed after {retry_attempts} attempts: {last_error}")


def extract_memories_from_turns(turns: List[str], dialogue_id: str,
                                 session_idx: int = 0,
                                 max_memories: int = 15) -> List[Dict]:
    """
    Extract structured memory items from a list of conversation turns.
    Returns a list of memory dicts with keys: content, memory_type, location_tags.
    """
    # Combine turns into a compact representation
    combined = "\n".join(f"[Turn {i+1}]: {t}" for i, t in enumerate(turns[:30]))

    prompt = f"""Extract up to {max_memories} distinct memory items from this conversation excerpt.
Focus on: locations mentioned, stated preferences, activities, relationships, and personal facts.

Conversation (session {session_idx} of dialogue {dialogue_id}):
{combined[:3000]}

Return a JSON array where each item has:
- "content": the memory fact (1-2 sentences)
- "memory_type": one of ["location", "preference", "activity", "fact", "relationship"]
- "location_tags": list of place names mentioned (empty list if none)

Return ONLY valid JSON, no markdown:"""

    try:
        response = llm_call(prompt, max_tokens=1024)
        # Strip potential markdown code blocks
        response = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        memories = json.loads(response)
        if not isinstance(memories, list):
            return []
        return memories[:max_memories]
    except Exception as e:
        logger.warning(f"Memory extraction failed for dialogue {dialogue_id} session {session_idx}: {e}")
        return []


def detect_conflicts_with_llm(memory_a: str, memory_b: str,
                               model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Ask LLM whether two memories conflict with each other.
    Returns: {"conflicts": bool, "confidence": float, "explanation": str}
    """
    prompt = f"""Determine whether these two user memory items CONFLICT with each other.
A conflict means: if both are true simultaneously, they would be contradictory or strongly inconsistent.

Memory A: "{memory_a}"
Memory B: "{memory_b}"

Consider: Would a reasonable person say these two facts cannot both be true at the same time,
or that one strongly undermines the relevance/desirability of the other?

Return JSON only:
{{"conflicts": true/false, "confidence": 0.0-1.0, "explanation": "brief reason"}}"""

    try:
        response = llm_call(prompt, max_tokens=200)
        response = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(response)
        return {
            "conflicts": bool(result.get("conflicts", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "explanation": str(result.get("explanation", "")),
        }
    except Exception as e:
        logger.warning(f"Conflict detection failed: {e}")
        return {"conflicts": False, "confidence": 0.0, "explanation": f"error: {e}"}


def generate_qa_pairs(dialogue_text: str, n_pairs: int = 5) -> List[Dict]:
    """
    Generate QA pairs from a dialogue to use as evaluation ground truth.
    Returns list of {"question": ..., "answer": ..., "requires_memory_type": ...}
    """
    prompt = f"""Given this long-term conversation between two people, generate {n_pairs}
question-answer pairs that test MEMORY and PREFERENCE TRACKING.

Focus on questions where the CURRENT answer might differ from an EARLIER answer (preference evolution,
location changes, activity changes).

Dialogue excerpt:
{dialogue_text[:4000]}

Return JSON array only:
[{{"question": "...", "answer": "...", "requires_memory_type": "location/preference/activity/fact",
   "involves_change": true/false, "change_description": "what changed"}}]"""

    try:
        response = llm_call(prompt, max_tokens=1024)
        response = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        pairs = json.loads(response)
        return pairs[:n_pairs]
    except Exception as e:
        logger.warning(f"QA generation failed: {e}")
        return []


def answer_with_memories(question: str, memories: List[str],
                          model: str = "gpt-4o-mini") -> str:
    """Given a question and a list of active memory strings, generate an answer."""
    mem_text = "\n".join(f"- {m}" for m in memories[:20]) if memories else "(no memories)"
    prompt = f"""Using ONLY the following memory items, answer the question about this user.
If the memories don't contain enough information, say "UNKNOWN".

Memory items:
{mem_text}

Question: {question}

Answer concisely (1-2 sentences):"""

    try:
        return llm_call(prompt, max_tokens=200)
    except Exception as e:
        return f"ERROR: {e}"
