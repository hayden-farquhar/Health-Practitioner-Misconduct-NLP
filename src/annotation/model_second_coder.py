"""Independent model second-coder pass for the Nov 2026 gold-standard rebuild.

Codes the same 300 decision texts the author re-coded (data/annotations/recode_2026-10/texts),
using the same written guidelines, so that inter-rater agreement can be computed against a
single, fully logged model source. Replaces the Feb 2026 llm_annotations.json, whose API and
agent-produced records cannot be told apart.

Every request is logged: model ID, prompt SHA-256, request parameters, batch ID, raw response,
stop reason and token usage. Nothing is truncated; nothing falls back to another model (a
refusal is recorded as missing, so all codes come from one coder).

Usage (from project root, with .venv-ml):
    .venv-ml/bin/python -m src.annotation.model_second_coder count     # free token count + cost estimate
    .venv-ml/bin/python -m src.annotation.model_second_coder submit    # create the batch (spends credits)
    .venv-ml/bin/python -m src.annotation.model_second_coder collect   # poll, then write results
Add --set supplementary to code the 300 supplementary training decisions instead (retrain
protocol, config/retrain_protocol_2026-11.md), or --set test2 to code the 160 second-test-set
decisions (config/llm_coder_protocol_2026-11.md); prompt, model and schema are unchanged.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"  # optional; ANTHROPIC_API_KEY may also be set in the shell
RECODE_DIR = PROJECT_ROOT / "data" / "annotations" / "recode_2026-10"
TEXTS_DIR = RECODE_DIR / "texts"
GUIDELINES = PROJECT_ROOT / "manuscript" / "supplementary" / "annotation_guidelines.md"
QUICKSTART = PROJECT_ROOT / "manuscript" / "supplementary" / "annotation_quickstart.md"
INSTRUCTIONS = RECODE_DIR / "CODING_INSTRUCTIONS.md"
OUT_DIR = PROJECT_ROOT / "data" / "annotations" / "model_pass_2026-11"
SUP_DIR = PROJECT_ROOT / "data" / "annotations" / "supplementary_2026-11"
# set name -> (texts dir, output dir, item prefix)
T2_DIR = PROJECT_ROOT / "data" / "annotations" / "test2_2026-11"
SETS = {"gold": (TEXTS_DIR, OUT_DIR, "HPM"),
        "supplementary": (SUP_DIR / "texts", SUP_DIR / "model_pass", "SUP"),
        "test2": (T2_DIR / "texts", T2_DIR / "model_pass", "T2")}
EXPECTED = {"HPM": 300, "SUP": 300, "T2": 160}

MODEL = "claude-sonnet-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
# Claude API list prices, USD per million tokens (claude-api skill table cached 2026-10-06);
# the Batches API bills at 50% of these.
PRICE_IN, PRICE_OUT, BATCH_DISCOUNT = 2.00, 10.00, 0.5

MISCONDUCT = ["CLIN", "PRESC", "BOUND", "IMPR", "FRAUD", "COMM", "PROF"]
BOUND_SUB = ["BOUND_SEXUAL", "BOUND_EMOTIONAL", "BOUND_FINANCIAL"]
OUTCOMES = ["CANCEL", "SUSPEND", "CONDITIONS", "REPRIMAND", "FINE", "CAUTION",
            "UNDERTAKING", "NO_ADVERSE", "OTHER"]
FINDINGS = ["PM", "UPC_SAT", "UPC_GEN", "NO_FINDING", "NA"]
PROFESSIONS = [
    "medical_practitioner", "nurse", "midwife", "dentist", "pharmacist", "psychologist",
    "physiotherapist", "chiropractor", "osteopath", "optometrist", "podiatrist", "paramedic",
    "occupational_therapist", "chinese_medicine_practitioner", "medical_radiation_practitioner",
    "aboriginal_health_practitioner", "other", "unknown",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "is_disciplinary": {"type": "boolean"},
        "finding": {"type": "string", "enum": FINDINGS},
        "misconduct": {"type": "array", "items": {"type": "string", "enum": MISCONDUCT}},
        "boundary_subtypes": {"type": "array", "items": {"type": "string", "enum": BOUND_SUB}},
        "outcomes": {"type": "array", "items": {"type": "string", "enum": OUTCOMES}},
        "profession": {"type": "string", "enum": PROFESSIONS},
        "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
        "notes": {"type": "string"},
    },
    "required": ["is_disciplinary", "finding", "misconduct", "boundary_subtypes", "outcomes",
                 "profession", "confidence", "notes"],
    "additionalProperties": False,
}


def _field_rules() -> str:
    """The 'Method' and 'Fields' sections of the author's coding instructions, verbatim."""
    text = INSTRUCTIONS.read_text(encoding="utf-8")
    start = text.index("## Method (per decision)")
    end = text.index("### date_coded")
    return text[start:end].strip()


def build_system_prompt() -> str:
    return "\n\n".join([
        "You are coding Australian health practitioner tribunal and disciplinary decisions "
        "for a research study. Apply the coding guidelines below exactly as written. Code "
        "what the decision itself describes and orders; do not infer beyond the text.",
        "# Field rules\n\n" + _field_rules(),
        "# Annotation guidelines\n\n" + GUIDELINES.read_text(encoding="utf-8"),
        "# Quick reference\n\n" + QUICKSTART.read_text(encoding="utf-8"),
        "# Output\n\nReturn one JSON object. `misconduct`, `boundary_subtypes` and `outcomes` "
        "list every code that applies (empty lists when none apply). If is_disciplinary is "
        "false, set finding to NA and leave misconduct, boundary_subtypes and outcomes empty. "
        "List boundary_subtypes only when BOUND is in misconduct. Keep notes to one or two "
        "sentences, covering the matters the field rules say belong in notes.",
    ])


def items():
    paths = sorted(TEXTS_DIR.glob(f"{PREFIX}-*.txt"))
    if len(paths) != EXPECTED[PREFIX]:
        sys.exit(f"Expected {EXPECTED[PREFIX]} texts in {TEXTS_DIR}, found {len(paths)}")
    return [(p.stem, p.read_text(encoding="utf-8")) for p in paths]


def params(system: str, text: str) -> dict:
    return dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "Code this decision.\n\n" + text}],
    )


def client():
    load_dotenv(ENV_FILE)
    return anthropic.Anthropic()


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cmd_count():
    c, system = client(), build_system_prompt()
    total, per_item = 0, {}
    for item, text in items():
        p = params(system, text)
        n = c.messages.count_tokens(model=MODEL, system=p["system"], messages=p["messages"],
                                    output_config=p["output_config"]).input_tokens
        per_item[item] = n
        total += n
    sys_tokens = c.messages.count_tokens(
        model=MODEL, system=system, messages=[{"role": "user", "content": "x"}]).input_tokens
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "token_count.json").write_text(json.dumps(
        {"model": MODEL, "counted_at": datetime.now(timezone.utc).isoformat(),
         "system_prompt_sha256": sha256(system), "system_prompt_tokens_approx": sys_tokens,
         "total_input_tokens": total, "per_item": per_item}, indent=2))
    big = max(per_item.values())
    print(f"Input tokens, all {len(per_item)} requests: {total:,} (largest single request {big:,}; "
          f"system prompt ~{sys_tokens:,} per request)")
    for label, out_per in [("low", 1500), ("mid", 4000), ("high", MAX_TOKENS)]:
        out = 300 * out_per
        cost = (total * PRICE_IN + out * PRICE_OUT) / 1e6 * BATCH_DISCOUNT
        print(f"  {label:4} output ({out_per:,}/request incl. thinking): ~US${cost:,.2f} with batch")
    print("Caching the shared system prompt lowers these figures; the upper row is a hard ceiling "
          "set by max_tokens.")


def cmd_submit():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "batch_manifest.json"
    if manifest_path.exists():
        sys.exit(f"A batch was already submitted ({manifest_path}); use 'collect'.")
    c, system = client(), build_system_prompt()
    if OUT_DIR != SETS["gold"][1]:
        first = json.loads((SETS["gold"][1] / "batch_manifest.json").read_text())
        if sha256(system) != first["system_prompt_sha256"]:
            sys.exit("System prompt differs from the gold-standard pass; refusing to submit.")
    (OUT_DIR / "system_prompt.txt").write_text(system, encoding="utf-8")
    (OUT_DIR / "output_schema.json").write_text(json.dumps(SCHEMA, indent=2))
    reqs, input_hashes = [], {}
    for item, text in items():
        reqs.append(Request(custom_id=item,
                            params=MessageCreateParamsNonStreaming(**params(system, text))))
        input_hashes[item] = sha256(text)
    batch = c.messages.batches.create(requests=reqs)
    manifest_path.write_text(json.dumps({
        "batch_id": batch.id, "submitted_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "effort": EFFORT, "max_tokens": MAX_TOKENS,
        "anthropic_sdk": anthropic.__version__, "system_prompt_sha256": sha256(system),
        "n_requests": len(reqs), "input_text_sha256": input_hashes,
    }, indent=2))
    print(f"Submitted batch {batch.id} with {len(reqs)} requests.")


def _to_row(item: str, coded: dict) -> dict:
    yn = lambda present: "Y" if present else ""
    row = {"item": item,
           "is_disciplinary": "Y" if coded["is_disciplinary"] else "N",
           "finding": coded["finding"],
           "profession": coded["profession"],
           "confidence": coded["confidence"],
           "notes": coded["notes"]}
    for code in MISCONDUCT:
        row[code] = yn(code in coded["misconduct"])
    for code in BOUND_SUB:
        row[code] = yn(code in coded["boundary_subtypes"])
    for code in OUTCOMES:
        row[code] = yn(code in coded["outcomes"])
    return row


def cmd_collect():
    manifest = json.loads((OUT_DIR / "batch_manifest.json").read_text())
    c = client()
    while True:
        batch = c.messages.batches.retrieve(manifest["batch_id"])
        if batch.processing_status == "ended":
            break
        print(f"{datetime.now():%H:%M:%S} {batch.request_counts}")
        time.sleep(60)

    raw_log, rows, problems = [], [], []
    usage_tot = {"input_tokens": 0, "output_tokens": 0,
                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    for res in c.messages.batches.results(manifest["batch_id"]):
        entry = {"item": res.custom_id, "result_type": res.result.type}
        if res.result.type == "succeeded":
            msg = res.result.message
            entry.update(model=msg.model, message_id=msg.id, stop_reason=msg.stop_reason,
                         usage=msg.usage.model_dump(), content=[b.model_dump() for b in msg.content])
            for k in usage_tot:
                usage_tot[k] += getattr(msg.usage, k) or 0
            text = "".join(b.text for b in msg.content if b.type == "text")
            if msg.stop_reason == "end_turn":
                try:
                    rows.append(_to_row(res.custom_id, json.loads(text)))
                except (json.JSONDecodeError, KeyError) as e:
                    problems.append(f"{res.custom_id}: unparseable output ({e})")
            else:
                problems.append(f"{res.custom_id}: stop_reason={msg.stop_reason}; recorded as missing")
        else:
            entry["error"] = res.result.model_dump()
            problems.append(f"{res.custom_id}: {res.result.type}")
        raw_log.append(entry)

    raw_log.sort(key=lambda e: e["item"])
    rows.sort(key=lambda r: r["item"])
    with open(OUT_DIR / "raw_responses.jsonl", "w", encoding="utf-8") as f:
        for e in raw_log:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (OUT_DIR / "model_codes.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    cost = ((usage_tot["input_tokens"] * PRICE_IN
             + usage_tot["cache_creation_input_tokens"] * PRICE_IN * 1.25
             + usage_tot["cache_read_input_tokens"] * PRICE_IN * 0.1
             + usage_tot["output_tokens"] * PRICE_OUT) / 1e6 * BATCH_DISCOUNT)
    (OUT_DIR / "run_summary.json").write_text(json.dumps({
        "batch_id": manifest["batch_id"], "collected_at": datetime.now(timezone.utc).isoformat(),
        "n_coded": len(rows), "problems": problems, "usage": usage_tot,
        "estimated_cost_usd": round(cost, 2)}, indent=2))
    print(f"Coded {len(rows)}/{len(raw_log)}; {len(problems)} problem(s); usage {usage_tot}; "
          f"estimated cost US${cost:,.2f}")
    for p in problems:
        print(" -", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["count", "submit", "collect"])
    ap.add_argument("--set", choices=list(SETS), default="gold")
    args = ap.parse_args()
    TEXTS_DIR, OUT_DIR, PREFIX = SETS[args.set]
    {"count": cmd_count, "submit": cmd_submit, "collect": cmd_collect}[args.command]()
