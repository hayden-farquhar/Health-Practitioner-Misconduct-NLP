"""Map each NER medication term to its schedule(s) in the current Poisons Standard.

Source: Therapeutic Goods (Poisons Standard—October 2026) Instrument 2026 (Cth), Federal
Register F2026L01327, in force from 1 October 2026; Word file downloaded 5 Nov 2026 from
https://www.legislation.gov.au/F2026L01327/asmade/2026-09-30/text/original/word and kept at
data/reference/poisons_standard_F2026L01327.docx. Schedules are read from the instrument's
Index, which lists every schedule and appendix entry for each substance.

Brand names and synonyms are mapped to the indexed substance through ALIASES. The four brand
names in the top-20 medication table (Endone, OxyContin, Valium, Panadeine Forte) were checked
against Australian product information on 5 Nov 2026; the others follow the pairings in
entity_extraction.py and are marked unverified.

Scheduling is current scheduling; it may differ from scheduling at the time of the conduct.

Usage (from project root; needs pandoc):
    .venv-ml/bin/python -m src.ner.build_poisons_schedule_map
Writes config/poisons_schedule_map_2026-10.json.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from src.ner.entity_extraction import ALL_DRUGS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "data" / "reference" / "poisons_standard_F2026L01327.docx"
OUT = PROJECT_ROOT / "config" / "poisons_schedule_map_2026-10.json"

# term -> (indexed substance, checked against product information?)
ALIASES = {
    "oxycontin": ("OXYCODONE", True), "endone": ("OXYCODONE", True), "targin": ("OXYCODONE", False),
    "ms contin": ("MORPHINE", False), "kapanol": ("MORPHINE", False),
    "durogesic": ("FENTANYL", False), "actiq": ("FENTANYL", False),
    "physeptone": ("METHADONE", False), "biodone": ("METHADONE", False),
    "dilaudid": ("HYDROMORPHONE", False), "jurnista": ("HYDROMORPHONE", False),
    "panadeine forte": ("CODEINE", True),
    "suboxone": ("BUPRENORPHINE", False), "subutex": ("BUPRENORPHINE", False),
    "temgesic": ("BUPRENORPHINE", False),
    "meperidine": ("PETHIDINE", False), "palexia": ("TAPENTADOL", False),
    "dexamphetamine": ("DEXAMFETAMINE", False), "dextroamphetamine": ("DEXAMFETAMINE", False),
    "ritalin": ("METHYLPHENIDATE", False), "concerta": ("METHYLPHENIDATE", False),
    "vyvanse": ("LISDEXAMFETAMINE", False),
    "medicinal cannabis": ("CANNABIS", False), "thc": ("TETRAHYDROCANNABINOLS", False),
    "xanax": ("ALPRAZOLAM", False), "valium": ("DIAZEPAM", True),
    "temaze": ("TEMAZEPAM", False), "normison": ("TEMAZEPAM", False),
    "serepax": ("OXAZEPAM", False), "rivotril": ("CLONAZEPAM", False),
    "mogadon": ("NITRAZEPAM", False), "ativan": ("LORAZEPAM", False),
    "stilnox": ("ZOLPIDEM", False), "imovane": ("ZOPICLONE", False),
    "zoloft": ("SERTRALINE", False), "prozac": ("FLUOXETINE", False),
    "paxil": ("PAROXETINE", False), "lexapro": ("ESCITALOPRAM", False),
    "efexor": ("VENLAFAXINE", False), "cymbalta": ("DULOXETINE", False),
    "avanza": ("MIRTAZAPINE", False), "endep": ("AMITRIPTYLINE", False),
    "seroquel": ("QUETIAPINE", False), "zyprexa": ("OLANZAPINE", False),
    "risperdal": ("RISPERIDONE", False), "abilify": ("ARIPIPRAZOLE", False),
    "clozaril": ("CLOZAPINE", False), "lyrica": ("PREGABALIN", False),
    "neurontin": ("GABAPENTIN", False), "inderal": ("PROPRANOLOL", False),
    "viagra": ("SILDENAFIL", False),
}

# Labels where the index lists several schedules and the instrument's entries say which applies
OVERRIDES = {
    "codeine": ("S4/S8", "S4 when compounded with other active substances at 30 mg or less per "
                         "dosage unit (Schedule 4 entry); otherwise S8"),
    "panadeine forte": ("S4", "paracetamol 500 mg + codeine 30 mg per tablet: compounded codeine, "
                              "30 mg or less per dosage unit (Schedule 4 entry)"),
    "testosterone": ("S4 (App D)", "Schedule 6 entry covers only implant preparations for use in "
                                   "animals"),
}


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def parse_index(text: str) -> dict:
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip() == "Index")
    idx, cur = {}, None
    for raw in lines[start + 1:]:
        s = re.sub(r"\s+", " ", raw.replace("‑", "-")).strip()
        if not s or s.lower().startswith("cross reference"):
            continue
        if re.match(r"^(Schedule \d+|Appendix [A-Z])", s):
            if cur:
                idx[cur].append(s)
        elif re.fullmatch(r"[A-Z]", s):
            cur = None
        else:
            cur = s.upper()
            idx.setdefault(cur, [])
    return idx


def label(entries: list[str]) -> str:
    scheds = sorted({int(m.group(1)) for e in entries if (m := re.match(r"Schedule (\d+)", e))})
    lab = "/".join(f"S{n}" for n in scheds)
    if any(e.startswith("Appendix D") for e in entries):
        lab += " (App D)"
    return lab


def main():
    text = subprocess.run(["pandoc", str(SOURCE), "-t", "plain", "--wrap=none"],
                          capture_output=True, text=True, check=True).stdout
    idx = parse_index(text)
    out, missing = {}, []
    for term in ALL_DRUGS:
        substance, verified = ALIASES.get(term, (term.upper(), True))
        entries = idx.get(substance)
        if not entries:
            missing.append(term)
            continue
        lab, note = OVERRIDES.get(term, (label(entries), ""))
        out[term] = {"substance": substance, "index_entries": entries, "label": lab, "note": note,
                     "alias_checked": verified if term in ALIASES else None}
    if missing:
        raise SystemExit(f"not found in the Poisons Standard index: {missing}")
    OUT.write_text(json.dumps({
        "source": "Therapeutic Goods (Poisons Standard—October 2026) Instrument 2026 (Cth), F2026L01327",
        "url": "https://www.legislation.gov.au/F2026L01327/latest/text",
        "downloaded": "2026-11-05",
        "source_sha1": sha1(SOURCE),
        "label_key": "S4/S8 etc = schedules listed in the instrument's Index; (App D) = also listed in "
                     "Appendix D (additional controls on possession or supply). Current scheduling, "
                     "which may differ from scheduling at the time of the conduct.",
        "terms": out,
    }, indent=1, ensure_ascii=False))
    print(f"{len(out)} terms mapped -> {OUT}")


if __name__ == "__main__":
    main()
