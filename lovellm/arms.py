"""The §6 evaluation arms as frozen configs over ONE pipeline (no per-arm code paths).

1. full                     — caring + particularity (the mechanism)
2. empty_r                  — stranger baseline: layer present, R empty, floor resources
3. mismatched_r             — WRONG user's populated R (C5 manipulation-check hardening;
                              expected ordering: known > mismatched > stranger)
4. caring_only              — allocation bias with empty other-model (prior() only)
5. particularity_only       — OWN estimator, FLAT allocation (floor only) — NOT the same
                              as generic_personalization (arm-collapse guard, PROTOCOL)
6. generic_personalization  — relevance-only retrieval, NO estimator, NO allocation
                              (the crowded-field baseline)
7. base_unmodified          — no layer at all: the §8 floor comparator
plus the R-depth grid: the full arm probed at every session checkpoint (C2's curve).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmConfig:
    name: str
    use_rstore: bool
    use_othermodel: bool
    use_allocation: bool
    mismatched_r: bool = False
    generic_retrieval: bool = False


FULL = ArmConfig("full", True, True, True)
EMPTY_R = ArmConfig("empty_r", False, False, False)
MISMATCHED_R = ArmConfig("mismatched_r", True, True, True, mismatched_r=True)
CARING_ONLY = ArmConfig("caring_only", True, False, True)
PARTICULARITY_ONLY = ArmConfig("particularity_only", True, True, False)
GENERIC_PERSONALIZATION = ArmConfig(
    "generic_personalization", True, False, False, generic_retrieval=True
)
BASE_UNMODIFIED = ArmConfig("base_unmodified", False, False, False)

CORE_ARMS = [
    FULL,
    EMPTY_R,
    MISMATCHED_R,
    CARING_ONLY,
    PARTICULARITY_ONLY,
    GENERIC_PERSONALIZATION,
]
