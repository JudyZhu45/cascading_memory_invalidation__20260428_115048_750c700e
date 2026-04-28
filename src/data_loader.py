"""
Data loading and preprocessing for LoCoMo and HorizonBench datasets.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from datasets import load_from_disk

logger = logging.getLogger(__name__)


# Known location keywords for LOCATED_IN detection
LOCATION_KEYWORDS = [
    "city", "town", "neighborhood", "country", "state", "province",
    "street", "district", "region", "area", "place", "location",
    "moved", "move", "moving", "relocated", "relocation", "live", "living",
    "home", "apartment", "house", "address",
    # Common city names that appear in LoCoMo
    "shanghai", "beijing", "new york", "london", "paris", "tokyo",
    "chicago", "boston", "seattle", "austin", "denver", "miami",
    "portland", "san francisco", "los angeles", "la", "sf",
    "montreal", "toronto", "vancouver", "sydney", "melbourne",
    "colorado", "california", "texas", "washington",
]

# Location shift trigger phrases
LOCATION_SHIFT_PATTERNS = [
    r"\bmov(?:ed|ing|e)\b.*(?:to|from)",
    r"\brelocate(?:d|ing)?\b",
    r"\bnew (?:city|town|place|home|apartment)\b",
    r"\bjust moved\b",
    r"\bi'?m (?:now|currently)? (?:in|at|living in)\b",
    r"\b(?:moved|move) to\b",
]

# Quiet/noise preference patterns for drift detection
QUIET_PREFERENCE_PATTERNS = [
    r"\bquiet\b", r"\bpeaceful\b", r"\bsolitude\b", r"\balone time\b",
    r"\bintrovert\b", r"\bhomebody\b", r"\bstay(?:ing)? (?:in|home)\b",
    r"\brecharge\b", r"\bsolitary\b", r"\bmindful\b", r"\bmeditat\b",
]

NOISY_ACTIVITY_PATTERNS = [
    r"\bbar\b", r"\bclub\b", r"\bparty\b", r"\bnightlife\b",
    r"\bconcert\b", r"\bfestival\b", r"\bcrowd\b", r"\boutgoing\b",
    r"\bsocial event\b", r"\bgoing out\b", r"\bnoise\b", r"\bloud\b",
]


def extract_location_from_text(text: str) -> Optional[str]:
    """Extract primary location mention from text."""
    text_lower = text.lower()
    for loc in LOCATION_KEYWORDS[12:]:  # City/place names
        if loc in text_lower:
            return loc
    return None


def detect_location_shift(turns: List[str]) -> List[Tuple[int, str, str]]:
    """
    Detect location-shift events across turns.
    Returns: list of (turn_idx, old_loc_hint, new_loc_hint)
    """
    events = []
    for i, turn in enumerate(turns):
        turn_lower = turn.lower()
        for pattern in LOCATION_SHIFT_PATTERNS:
            if re.search(pattern, turn_lower):
                new_loc = extract_location_from_text(turn) or "unknown"
                events.append((i, "previous", new_loc))
                break
    return events


def detect_preference_drift(turns: List[str]) -> List[Tuple[int, str, float]]:
    """
    Detect preference drift signals across turns.
    Returns: list of (turn_idx, preference_type, drift_score)
    """
    events = []
    for i, turn in enumerate(turns):
        turn_lower = turn.lower()
        quiet_score = sum(1 for p in QUIET_PREFERENCE_PATTERNS if re.search(p, turn_lower))
        noisy_score = sum(1 for p in NOISY_ACTIVITY_PATTERNS if re.search(p, turn_lower))

        if quiet_score > 0:
            events.append((i, "quiet_preference", quiet_score / len(QUIET_PREFERENCE_PATTERNS)))
        if noisy_score > 0:
            events.append((i, "noisy_activity", noisy_score / len(NOISY_ACTIVITY_PATTERNS)))

    return events


def load_locomo(data_dir: str) -> List[Dict]:
    """
    Load LoCoMo dataset and parse into structured dialogue objects.

    Returns list of dicts:
      {
        dialogue_id: int,
        turns: List[str],
        speakers: List[str],
        location_shifts: List[Tuple],
        preference_drifts: List[Tuple],
        n_turns: int,
      }
    """
    ds = load_from_disk(data_dir)
    train = ds['train']
    dialogues = []

    for row in train:
        dialogue_id = row['dialogue_id']
        turns_data = json.loads(row['turns'])
        utterances = turns_data['utterance']
        speakers = turns_data['speaker_role']

        # Detect events
        location_shifts = detect_location_shift(utterances)
        preference_drifts = detect_preference_drift(utterances)

        dialogues.append({
            "dialogue_id": dialogue_id,
            "turns": utterances,
            "speakers": speakers,
            "location_shifts": location_shifts,
            "preference_drifts": preference_drifts,
            "n_turns": len(utterances),
        })

    logger.info(f"Loaded {len(dialogues)} LoCoMo dialogues")
    return dialogues


def load_horizonbench(data_dir: str) -> Dict:
    """
    Load HorizonBench dataset.
    Returns dict with 'evolved' and 'static' splits.
    """
    ds = load_from_disk(data_dir)
    test = ds['test']

    evolved = [item for item in test if item['has_evolved']]
    static = [item for item in test if not item['has_evolved']]

    logger.info(f"HorizonBench: {len(evolved)} evolved, {len(static)} static items")
    return {
        "all": list(test),
        "evolved": evolved,
        "static": static,
    }


def split_dialogue_into_sessions(turns: List[str], speakers: List[str],
                                  turns_per_session: int = 20) -> List[Dict]:
    """
    Split a long dialogue into sessions (simulating multiple meetings).
    Each session is a dict with 'turns', 'speakers', 'session_idx'.
    """
    sessions = []
    n = len(turns)
    for i in range(0, n, turns_per_session):
        session_turns = turns[i:i + turns_per_session]
        session_speakers = speakers[i:i + turns_per_session]
        sessions.append({
            "session_idx": i // turns_per_session,
            "turns": session_turns,
            "speakers": session_speakers,
            "start_turn": i,
        })
    return sessions


def get_session_text(session: Dict) -> str:
    """Combine session turns into a single text string."""
    parts = []
    for speaker, turn in zip(session["speakers"], session["turns"]):
        parts.append(f"{speaker}: {turn}")
    return "\n".join(parts)


def assign_session_timestamps(sessions: List[Dict],
                               start_date: datetime = None) -> List[Dict]:
    """Assign simulated timestamps to sessions (weekly intervals)."""
    if start_date is None:
        start_date = datetime(2024, 1, 1)
    for i, session in enumerate(sessions):
        session["timestamp"] = start_date + timedelta(weeks=i)
    return sessions
