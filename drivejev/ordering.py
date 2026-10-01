"""Label-free canonical ordering of semantic candidates (opaque IDs never reach the prompt)."""
from __future__ import annotations

import json
from dataclasses import dataclass

SEMANTIC_FIELDS = ("action_type", "target_id", "speed_profile", "description")
CANDIDATE_FIELDS = {"candidate_id", *SEMANTIC_FIELDS}


def semantic_key(candidate):
    unknown = set(candidate) - CANDIDATE_FIELDS
    if unknown:
        raise ValueError(f"Unapproved candidate fields: {sorted(unknown)}")
    if not isinstance(candidate.get("action_type"), str) or not candidate["action_type"]:
        raise ValueError("Candidate action_type must be a non-empty string")
    return tuple(json.dumps(candidate.get(f), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                 for f in SEMANTIC_FIELDS)


@dataclass(frozen=True)
class CandidateLayout:
    request_ids: tuple
    compiled_ids: tuple


def canonicalize(candidates):
    """Sort candidates by their semantic body; returns (sorted candidates, layout)."""
    if not 2 <= len(candidates) <= 12:
        raise ValueError("A decision needs 2-12 candidates")
    ids = [c.get("candidate_id") for c in candidates]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique non-empty strings")
    keys = [semantic_key(c) for c in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("Two candidates have the same semantic body")
    order = sorted(range(len(candidates)), key=keys.__getitem__)
    return [candidates[i] for i in order], CandidateLayout(tuple(ids), tuple(ids[i] for i in order))
