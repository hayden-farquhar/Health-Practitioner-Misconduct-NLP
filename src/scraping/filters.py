"""Health practitioner decision filtering logic.

Shared by both the HuggingFace corpus download and AustLII scraper pipelines.
Provides citation-pattern matching, keyword content filtering, and
party-name pattern matching to identify health practitioner decisions
from general tribunal databases.
"""

import re
from typing import Optional

from .config import HEALTH_KEYWORDS, STRONG_HEALTH_KEYWORDS, load_tribunals


# Compiled regex patterns (built once at import time)
_TRIBUNAL_PATTERNS: dict[str, re.Pattern] = {}
_PARTY_PATTERNS: list[re.Pattern] = []
_NON_HEALTH_PATTERN: Optional[re.Pattern] = None
_KEYWORD_PATTERN: Optional[re.Pattern] = None
_STRONG_KEYWORD_PATTERN: Optional[re.Pattern] = None

# Non-health professions that share general tribunals (NSWCATOD, VCAT, etc.)
# and whose cases contain terms like "professional misconduct" that would
# otherwise pass our keyword filter.
NON_HEALTH_PARTIES = [
    r"law\s+society",
    r"legal\s+services\s+commissioner",
    r"council\s+of\s+the\s+law\s+society",
    r"commissioner\s+for\s+fair\s+trading",
    r"building\s+professionals?\s+board",
    r"architects?\s+registration\s+board",
    r"surveyors?\s+registration\s+board",
    r"board\s+of\s+surveying",
    r"veterinary\s+practitioners?\s+board",
    r"veterinary\s+surgeons?\s+board",
]


def _ensure_compiled():
    """Lazily compile regex patterns from tribunal config."""
    global _TRIBUNAL_PATTERNS, _PARTY_PATTERNS, _NON_HEALTH_PATTERN
    global _KEYWORD_PATTERN, _STRONG_KEYWORD_PATTERN

    if _KEYWORD_PATTERN is not None:
        return

    # Compile non-health party pattern (used to exclude false positives)
    _NON_HEALTH_PATTERN = re.compile(
        r"\b(?:" + "|".join(NON_HEALTH_PARTIES) + r")\b",
        re.IGNORECASE,
    )

    # Compile tribunal citation patterns
    config = load_tribunals()
    for tribunal_id, info in config["tribunals"].items():
        _TRIBUNAL_PATTERNS[tribunal_id] = re.compile(
            info["citation_pattern"], re.IGNORECASE
        )

    # Party-name patterns for health practitioner cases
    _PARTY_PATTERNS.extend([
        # "X Board of Australia v [Name]"
        re.compile(
            r"\b(?:medical|nursing|dental|pharmacy|psychology|physiotherapy|"
            r"optometry|chiropractic|osteopathy|podiatry|occupational therapy|"
            r"chinese medicine|paramedicine|medical radiation)"
            r"\s+board\s+of\s+australia\s+v\b",
            re.IGNORECASE,
        ),
        # "Health Care Complaints Commission v [Name]"
        re.compile(
            r"\bhealth\s+care\s+complaints\s+commission\s+v\b",
            re.IGNORECASE,
        ),
        # "Health Ombudsman v [Name]"
        re.compile(
            r"\bhealth\s+ombudsman\s+v\b",
            re.IGNORECASE,
        ),
        # "[Professional] Council of New South Wales v [Name]"
        re.compile(
            r"\b(?:medical|nursing|dental|pharmacy|psychology|physiotherapy|"
            r"optometry|chiropractic|osteopathy|podiatry)"
            r"\s+council\s+of\s+new\s+south\s+wales\s+v\b",
            re.IGNORECASE,
        ),
        # "AHPRA v [Name]" or "[Name] v AHPRA"
        re.compile(r"\bahpra\s+v\b|\bv\s+ahpra\b", re.IGNORECASE),
    ])

    # Build keyword patterns — single compiled regex for efficiency
    escaped_keywords = [re.escape(kw) for kw in HEALTH_KEYWORDS]
    _KEYWORD_PATTERN = re.compile(
        r"\b(?:" + "|".join(escaped_keywords) + r")\b",
        re.IGNORECASE,
    )

    escaped_strong = [re.escape(kw) for kw in STRONG_HEALTH_KEYWORDS]
    _STRONG_KEYWORD_PATTERN = re.compile(
        r"\b(?:" + "|".join(escaped_strong) + r")\b",
        re.IGNORECASE,
    )


def match_citation(citation: str) -> Optional[str]:
    """Match a citation string to a tribunal ID.

    Args:
        citation: Legal citation, e.g. "[2023] NSWCATOD 456"

    Returns:
        Tribunal ID if matched, None otherwise.
    """
    _ensure_compiled()
    for tribunal_id, pattern in _TRIBUNAL_PATTERNS.items():
        if pattern.search(citation):
            return tribunal_id
    return None


def is_health_tribunal(tribunal_id: str) -> bool:
    """Check if a tribunal is health-specific (no keyword filtering needed).

    Args:
        tribunal_id: Tribunal identifier from tribunals.json

    Returns:
        True if the tribunal only hears health practitioner matters.
    """
    config = load_tribunals()
    tribunal = config["tribunals"].get(tribunal_id, {})
    return tribunal.get("health_specific", False)


def has_health_keywords(text: str, min_matches: int = 2) -> bool:
    """Check if text contains health practitioner keywords.

    Uses a two-tier approach:
    - One strong keyword match is sufficient (high precision terms)
    - Otherwise requires min_matches of general keywords

    Args:
        text: Full text or title to search.
        min_matches: Minimum general keyword matches required.

    Returns:
        True if the text appears to be a health practitioner decision.
    """
    _ensure_compiled()

    # A single strong keyword is sufficient
    if _STRONG_KEYWORD_PATTERN.search(text):
        return True

    # Otherwise, require multiple general keyword matches
    matches = _KEYWORD_PATTERN.findall(text[:50000])  # limit search length
    return len(matches) >= min_matches


def has_party_match(text: str) -> bool:
    """Check if text contains health practitioner party-name patterns.

    Looks for patterns like "Medical Board of Australia v [Name]" in the
    first portion of text (typically the title or first few paragraphs).

    Args:
        text: Citation, title, or full text to search.

    Returns:
        True if a health practitioner party pattern is found.
    """
    _ensure_compiled()
    search_text = text[:5000]  # party names appear early
    return any(p.search(search_text) for p in _PARTY_PATTERNS)


def is_non_health_decision(citation: str, text: str) -> bool:
    """Check if a decision belongs to a non-health profession.

    Identifies cases involving lawyers, building professionals, surveyors,
    veterinarians, etc. that share general tribunals with health matters
    and can produce false positives from keyword matching.

    Args:
        citation: Legal citation string.
        text: Full text of the decision.

    Returns:
        True if this is a non-health professional disciplinary matter.
    """
    _ensure_compiled()
    # Check citation/title first (fast)
    if _NON_HEALTH_PATTERN.search(citation):
        return True
    # Check early text for party names
    if _NON_HEALTH_PATTERN.search(text[:3000]):
        return True
    return False


def is_health_decision(
    citation: str,
    text: str,
    tribunal_id: Optional[str] = None,
) -> bool:
    """Determine whether a decision is a health practitioner matter.

    Main entry point for filtering. Applies a cascade of checks:
    1. If tribunal is health-specific, always include
    2. Exclude known non-health professions (lawyers, building, etc.)
    3. Check party-name patterns (high precision)
    4. Check keyword content (requires multiple matches)

    Args:
        citation: Legal citation string.
        text: Full text of the decision.
        tribunal_id: Known tribunal ID, or None to detect from citation.

    Returns:
        True if this appears to be a health practitioner decision.
    """
    # Detect tribunal from citation if not provided
    if tribunal_id is None:
        tribunal_id = match_citation(citation)

    # Health-specific tribunals: always include
    if tribunal_id and is_health_tribunal(tribunal_id):
        return True

    # Exclude known non-health professions
    if is_non_health_decision(citation, text):
        return False

    # Party-name match in citation or early text
    if has_party_match(citation) or has_party_match(text):
        return True

    # Keyword content filtering
    if has_health_keywords(text):
        return True

    return False
