"""Rule-based entity extraction for health practitioner decisions.

Extracts:
- MEDICATION: Drug names (focus on Schedule 4/8 drugs)
- SCHEDULE: Drug schedule references (S4, S8, controlled substance)
- CONDITION: Health conditions mentioned
- BOARD: Regulatory board references

Uses spaCy EntityRuler + PhraseMatcher for efficiency.

Usage:
    python -m src.ner.entity_extraction
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.tokens import Span

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "extracted" / "all_decisions.jsonl"
OUTPUT_PATH = DATA_DIR / "extracted" / "entities.jsonl"

# -------------------------------------------------------------------------
# Medication dictionaries (Australian PBS Schedule 4/8 drugs commonly
# encountered in disciplinary decisions)
# -------------------------------------------------------------------------

# The two lists below are matched together (ALL_DRUGS); the split is for readability only and
# is not an accurate scheduling (e.g. tramadol and compounded codeine are Schedule 4). Schedules
# reported in the paper come from config/poisons_schedule_map_2026-10.json.
# Schedule 8 (Controlled drugs) — high relevance to prescribing misconduct
SCHEDULE_8_DRUGS = [
    # Opioids
    "oxycodone", "oxycontin", "endone", "targin",
    "morphine", "ms contin", "kapanol",
    "fentanyl", "durogesic", "actiq",
    "methadone", "physeptone", "biodone",
    "hydromorphone", "dilaudid", "jurnista",
    "codeine", "panadeine forte",
    "tramadol",
    "buprenorphine", "suboxone", "subutex", "temgesic",
    "pethidine", "meperidine",
    "tapentadol", "palexia",
    # Stimulants
    "dexamphetamine", "dextroamphetamine",
    "methylphenidate", "ritalin", "concerta",
    "lisdexamfetamine", "vyvanse",
    # Other S8
    "ketamine",
    "cannabis", "medicinal cannabis", "thc", "dronabinol",
    "alprazolam", "xanax",
]

# Schedule 4 — common in prescribing misconduct decisions
SCHEDULE_4_DRUGS = [
    # Benzodiazepines
    "diazepam", "valium",
    "temazepam", "temaze", "normison",
    "oxazepam", "serepax",
    "clonazepam", "rivotril",
    "nitrazepam", "mogadon",
    "lorazepam", "ativan",
    "midazolam",
    "flunitrazepam",
    # Z-drugs
    "zolpidem", "stilnox",
    "zopiclone", "imovane",
    # Antidepressants
    "sertraline", "zoloft",
    "fluoxetine", "prozac",
    "paroxetine", "paxil",
    "escitalopram", "lexapro",
    "venlafaxine", "efexor",
    "duloxetine", "cymbalta",
    "mirtazapine", "avanza",
    "amitriptyline", "endep",
    # Antipsychotics
    "quetiapine", "seroquel",
    "olanzapine", "zyprexa",
    "risperidone", "risperdal",
    "aripiprazole", "abilify",
    "clozapine", "clozaril",
    # Other
    "pregabalin", "lyrica",
    "gabapentin", "neurontin",
    "propranolol", "inderal",
    "prednisone", "prednisolone",
    "testosterone",
    "sildenafil", "viagra",
    "modafinil",
]

ALL_DRUGS = SCHEDULE_8_DRUGS + SCHEDULE_4_DRUGS

# -------------------------------------------------------------------------
# Health conditions commonly mentioned in decisions
# -------------------------------------------------------------------------
HEALTH_CONDITIONS = [
    # Mental health
    "depression", "major depressive disorder",
    "anxiety", "anxiety disorder", "generalised anxiety disorder",
    "bipolar disorder", "bipolar affective disorder",
    "schizophrenia", "psychosis", "psychotic",
    "post-traumatic stress disorder", "ptsd",
    "personality disorder", "borderline personality disorder",
    "eating disorder", "anorexia", "bulimia",
    # Substance use
    "substance abuse", "substance use disorder",
    "alcohol dependence", "alcohol abuse", "alcoholism",
    "drug dependence", "drug addiction", "opioid dependence",
    "opioid use disorder",
    # Pain
    "chronic pain", "back pain", "lower back pain",
    "fibromyalgia", "neuropathic pain",
    # Other
    "diabetes", "type 2 diabetes",
    "hypertension", "high blood pressure",
    "cancer", "malignancy",
    "dementia", "cognitive impairment",
    "chronic fatigue syndrome",
    "insomnia", "sleep disorder",
]

# -------------------------------------------------------------------------
# Regulatory boards
# -------------------------------------------------------------------------
BOARD_NAMES = [
    "medical board of australia",
    "nursing and midwifery board of australia",
    "dental board of australia",
    "pharmacy board of australia",
    "psychology board of australia",
    "physiotherapy board of australia",
    "chiropractic board of australia",
    "osteopathy board of australia",
    "optometry board of australia",
    "podiatry board of australia",
    "occupational therapy board of australia",
    "chinese medicine board of australia",
    "medical radiation practice board of australia",
    "paramedicine board of australia",
    "aboriginal and torres strait islander health practice board of australia",
    # Shortened forms
    "medical board",
    "nursing board",
    "dental board",
    "pharmacy board",
    "psychology board",
    "physiotherapy board",
    "chiropractic board",
    "paramedicine board",
    # State-specific
    "health care complaints commission",
    "hccc",
    "office of the health ombudsman",
    "health ombudsman",
    "ahpra",
]

# -------------------------------------------------------------------------
# Schedule patterns (regex-based)
# -------------------------------------------------------------------------
SCHEDULE_PATTERNS = [
    (re.compile(r"\bschedule\s*8\b", re.IGNORECASE), "Schedule 8"),
    (re.compile(r"\bschedule\s*4\b", re.IGNORECASE), "Schedule 4"),
    (re.compile(r"\bS8\s+(?:drug|substance|medication|poison)", re.IGNORECASE), "Schedule 8"),
    (re.compile(r"\bS4\s+(?:drug|substance|medication|poison)", re.IGNORECASE), "Schedule 4"),
    (re.compile(r"\bdrug[s]?\s+of\s+dependence\b", re.IGNORECASE), "drug of dependence"),
    (re.compile(r"\bcontrolled\s+substance[s]?\b", re.IGNORECASE), "controlled substance"),
    (re.compile(r"\bcontrolled\s+drug[s]?\b", re.IGNORECASE), "controlled drug"),
    (re.compile(r"\brestricted\s+substance[s]?\b", re.IGNORECASE), "restricted substance"),
    (re.compile(r"\bschedule\s*4\s*(?:appendix\s*)?D\b", re.IGNORECASE), "Schedule 4 Appendix D"),
]


def build_nlp_pipeline() -> Language:
    """Build spaCy pipeline with entity ruler for medications, conditions, boards."""
    nlp = spacy.blank("en")

    # Add entity ruler with phrase patterns
    ruler = nlp.add_pipe("entity_ruler")

    patterns = []

    # Medication patterns
    for drug in ALL_DRUGS:
        patterns.append({"label": "MEDICATION", "pattern": drug})
        # Also match capitalised form
        patterns.append({"label": "MEDICATION", "pattern": drug.title()})

    # Condition patterns
    for condition in HEALTH_CONDITIONS:
        patterns.append({"label": "CONDITION", "pattern": condition})
        patterns.append({"label": "CONDITION", "pattern": condition.title()})

    # Board patterns
    for board in BOARD_NAMES:
        patterns.append({"label": "BOARD", "pattern": board})
        patterns.append({"label": "BOARD", "pattern": board.title()})
        patterns.append({"label": "BOARD", "pattern": board.upper()})

    ruler.add_patterns(patterns)

    return nlp


def extract_schedule_refs(text: str) -> list[dict]:
    """Extract drug schedule references using regex patterns."""
    results = []
    for pattern, label in SCHEDULE_PATTERNS:
        for match in pattern.finditer(text):
            results.append({
                "text": match.group(),
                "label": "SCHEDULE",
                "normalised": label,
                "start": match.start(),
                "end": match.end(),
            })
    return results


def extract_entities(text: str, nlp: Language) -> dict:
    """Extract all entity types from a decision text.

    Returns dict with lists of entities by type.
    """
    # Limit text length for spaCy processing (very long docs cause memory issues)
    doc = nlp(text[:200000])

    entities = defaultdict(list)
    seen = set()

    for ent in doc.ents:
        key = (ent.label_, ent.text.lower())
        if key not in seen:
            seen.add(key)
            entities[ent.label_].append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

    # Add schedule references (regex-based, not in spaCy pipeline)
    schedule_refs = extract_schedule_refs(text)
    schedule_seen = set()
    for ref in schedule_refs:
        norm = ref["normalised"]
        if norm not in schedule_seen:
            schedule_seen.add(norm)
            entities["SCHEDULE"].append(ref)

    return dict(entities)


def main():
    print("=== NER Entity Extraction ===\n")

    # Build pipeline
    print("Building spaCy NER pipeline...")
    nlp = build_nlp_pipeline()

    # Load corpus
    print("Loading corpus...")
    records = []
    with open(CORPUS_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  {len(records)} decisions loaded")

    # Extract entities
    print("\nExtracting entities...")
    results = []
    total_counts = Counter()

    for i, record in enumerate(records):
        entities = extract_entities(record["full_text"], nlp)
        result = {
            "id": record["id"],
            "entities": entities,
        }
        results.append(result)

        # Count entity types
        for ent_type, ents in entities.items():
            total_counts[ent_type] += len(ents)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(records)}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\nEntities saved to {OUTPUT_PATH}")

    # Summary statistics
    print("\n=== Entity Extraction Summary ===")
    docs_with_entities = sum(1 for r in results if any(r["entities"].values()))
    print(f"Documents with entities: {docs_with_entities}/{len(results)} ({docs_with_entities/len(results)*100:.1f}%)")

    for ent_type in ["MEDICATION", "SCHEDULE", "CONDITION", "BOARD"]:
        docs_with = sum(1 for r in results if r["entities"].get(ent_type))
        unique_values = set()
        for r in results:
            for ent in r["entities"].get(ent_type, []):
                text = ent.get("normalised", ent["text"]).lower()
                unique_values.add(text)
        print(f"\n{ent_type}:")
        print(f"  Documents with {ent_type}: {docs_with} ({docs_with/len(results)*100:.1f}%)")
        print(f"  Unique values: {len(unique_values)}")

    # Top medications
    med_counter = Counter()
    for r in results:
        for ent in r["entities"].get("MEDICATION", []):
            med_counter[ent["text"].lower()] += 1
    print("\nTop 20 medications:")
    for drug, count in med_counter.most_common(20):
        print(f"  {drug}: {count}")

    # Top conditions
    cond_counter = Counter()
    for r in results:
        for ent in r["entities"].get("CONDITION", []):
            cond_counter[ent["text"].lower()] += 1
    print("\nTop 15 conditions:")
    for cond, count in cond_counter.most_common(15):
        print(f"  {cond}: {count}")


if __name__ == "__main__":
    main()
