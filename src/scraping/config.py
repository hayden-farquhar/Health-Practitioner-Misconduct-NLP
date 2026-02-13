"""Configuration constants for scraping pipeline."""

import json
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
EXTRACTED_DIR = DATA_DIR / "extracted"
METADATA_DIR = DATA_DIR / "metadata"

# Output files
NSW_DECISIONS_PATH = EXTRACTED_DIR / "nsw_decisions.jsonl"
NONNSW_HF_DECISIONS_PATH = EXTRACTED_DIR / "nonnsw_hf_decisions.jsonl"
AUSTLII_DECISIONS_PATH = EXTRACTED_DIR / "austlii_decisions.jsonl"
ALL_DECISIONS_PATH = EXTRACTED_DIR / "all_decisions.jsonl"
CORPUS_STATS_PATH = METADATA_DIR / "corpus_statistics.json"
HF_EXPLORE_PATH = METADATA_DIR / "hf_corpus_survey.json"

# Scraping progress tracking
SCRAPE_PROGRESS_PATH = METADATA_DIR / "scrape_progress.json"

# AustLII settings
# classic.austlii.edu.au is used because www.austlii.edu.au times out
# for programmatic requests. Both serve the same content.
AUSTLII_BASE_URL = "https://classic.austlii.edu.au"
AUSTLII_CRAWL_DELAY = 120  # seconds — conservative rate limit for academic scraping
USER_AGENT = (
    "HealthPracMisconductResearch/1.0 "
    "(Academic research; University of Sydney; "
    "contact: hayden.farquhar@sydney.edu.au)"
)

# HuggingFace dataset
HUGGINGFACE_DATASET_ID = "umarbutler/open-australian-legal-corpus"

# Health practitioner keywords for filtering general tribunal decisions.
# Used to identify health-related cases from tribunals that also hear
# non-health matters (e.g. NSWCATOD, VCAT, QCAT, SACAT, WASAT).
HEALTH_KEYWORDS = [
    # Regulatory bodies
    "ahpra",
    "australian health practitioner regulation agency",
    "health care complaints commission",
    "hccc",
    "health ombudsman",
    "office of the health ombudsman",
    "health professional councils authority",
    "hpca",
    "medical board of australia",
    "nursing and midwifery board",
    "dental board of australia",
    "pharmacy board of australia",
    "psychology board of australia",
    "physiotherapy board of australia",
    "optometry board of australia",
    "chiropractic board of australia",
    "osteopathy board of australia",
    "podiatry board of australia",
    "occupational therapy board",
    "chinese medicine board",
    "aboriginal and torres strait islander health practice board",
    "medical radiation practice board",
    "paramedicine board",
    # Legislation
    "health practitioner regulation national law",
    "national law",
    "health practitioner regulation",
    "health professions registration act",
    "medical practice act",
    "medical practitioners act",
    "health professionals act",
    # Registration/practice terms
    "registered health practitioner",
    "medical registration",
    "nursing registration",
    "conditions on registration",
    "cancellation of registration",
    "suspension of registration",
    "professional misconduct",
    "unsatisfactory professional conduct",
    "unprofessional conduct",
    "impaired registrant",
    # Professions
    "medical practitioner",
    "registered nurse",
    "enrolled nurse",
    "midwife",
    "dentist",
    "dental practitioner",
    "pharmacist",
    "psychologist",
    "physiotherapist",
    "optometrist",
    "chiropractor",
    "osteopath",
    "podiatrist",
    "occupational therapist",
    "medical radiation practitioner",
    "chinese medicine practitioner",
    "paramedic",
    # Common parties
    "council of new south wales",
    "board of australia v",
    "health care complaints",
]

# Keywords that strongly indicate a health practitioner case (high precision).
# A match on any of these is sufficient for inclusion.
STRONG_HEALTH_KEYWORDS = [
    "ahpra",
    "health practitioner regulation national law",
    "medical board of australia",
    "nursing and midwifery board",
    "dental board of australia",
    "pharmacy board of australia",
    "psychology board of australia",
    "health care complaints commission",
    "hccc",
    "health ombudsman",
    "professional misconduct",
    "unsatisfactory professional conduct",
    "cancellation of registration",
    "suspension of registration",
    "conditions on registration",
]


def load_tribunals():
    """Load tribunal registry from config/tribunals.json."""
    tribunals_path = CONFIG_DIR / "tribunals.json"
    with open(tribunals_path, "r") as f:
        return json.load(f)
