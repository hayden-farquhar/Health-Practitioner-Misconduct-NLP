"""Annotate decisions using Claude API.

Sends each decision's full text to the Claude API along with the taxonomy
and coding guidelines, then parses structured annotations back.

Supports:
- Annotating the 300-decision sample (for IRR comparison with human coder)
- Annotating the full corpus (after IRR validation)
- Resumable processing (saves progress after each decision)
- Cost tracking

Usage:
    # Annotate the 300-decision sample
    python -m src.preprocessing.llm_annotation --sample

    # Annotate the full corpus
    python -m src.preprocessing.llm_annotation --full

    # Dry run (process 5 decisions only)
    python -m src.preprocessing.llm_annotation --sample --limit 5

Note: The script always resumes automatically. Existing annotations are never
overwritten — it is safe to re-run at any time.

Prerequisites:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from src.scraping.config import ALL_DECISIONS_PATH, DATA_DIR

logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = DATA_DIR / "annotations"
SAMPLE_PATH = ANNOTATIONS_DIR / "annotation_sample.jsonl"
LLM_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "llm_annotations.json"
LLM_FULL_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "llm_annotations_full.json"
LLM_PROGRESS_PATH = ANNOTATIONS_DIR / "llm_annotation_progress.json"
TAXONOMY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "taxonomy.json"

# Claude API settings
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 2000
MAX_TEXT_CHARS = 80000  # ~20K tokens — fits comfortably in context with prompt

# Rate limiting
REQUESTS_PER_MINUTE = 40
REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE


def _load_taxonomy() -> str:
    """Load taxonomy as a formatted string for the prompt."""
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    lines = []

    lines.append("MISCONDUCT CATEGORIES (select all applicable L1 codes; L2 optional):")
    for code, cat in taxonomy["misconduct_categories"].items():
        lines.append(f"  {code} — {cat['name']}: {cat['definition']}")
        for sub_code, sub in cat.get("subcategories", {}).items():
            lines.append(f"    {sub_code} — {sub['name']}: {sub['definition']}")

    lines.append("")
    lines.append("FINDING CATEGORIES (select one):")
    for code, cat in taxonomy["finding_categories"].items():
        lines.append(f"  {code} — {cat['name']}: {cat['definition']}")

    lines.append("")
    lines.append("OUTCOME CATEGORIES (select all applicable):")
    for code, cat in taxonomy["outcome_categories"].items():
        lines.append(f"  {code} — {cat['name']}: {cat['definition']}")

    lines.append("")
    lines.append("PROFESSIONS (select one):")
    for prof in taxonomy["professions"]:
        lines.append(f"  {prof}")

    lines.append("")
    lines.append("CODING RULES:")
    for rule_name, rule in taxonomy["coding_rules"].items():
        lines.append(f"  {rule_name}: {rule}")

    return "\n".join(lines)


SYSTEM_PROMPT = """You are an expert legal coder annotating Australian health practitioner disciplinary tribunal decisions.

Your task is to read each decision and assign structured codes. You must respond ONLY with valid JSON matching the exact schema specified.

Key principles:
- Read the catchwords line first (usually states the finding type)
- Read the orders section (lists sanctions imposed)
- Code based on what is described in the decision, not inferences
- If the decision is not a disciplinary matter (e.g., privacy dispute, registration appeal, procedural ruling), set is_disciplinary to false
- If text is too truncated to determine a field, set that field to null
- Multiple misconduct types and outcomes are common — code all that apply
- For finding, code the most serious finding made
- Trust the terminology the tribunal uses (cancelled = CANCEL, suspended = SUSPEND)"""


def _build_user_prompt(taxonomy_text: str, decision_text: str, decision_id: str) -> str:
    """Build the user prompt for a single decision."""
    return f"""Code the following tribunal decision using the taxonomy below.

{taxonomy_text}

---

DECISION ID: {decision_id}
DECISION TEXT:
{decision_text}

---

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{
  "id": "{decision_id}",
  "is_disciplinary": true,
  "misconduct_L1": ["CLIN"],
  "misconduct_L2": ["CLIN_DIAG"],
  "finding": "PM",
  "outcomes": ["CANCEL"],
  "profession": "medical_practitioner",
  "confidence": "high",
  "notes": "Brief notes on any ambiguity or edge cases"
}}

Rules for the JSON:
- "is_disciplinary": false if this is not a disciplinary decision (privacy dispute, registration appeal, procedural ruling). If false, set misconduct/finding/outcomes to null.
- "misconduct_L1": array of L1 codes (e.g., ["CLIN", "PRESC"]). Use null if cannot determine.
- "misconduct_L2": array of L2 codes (e.g., ["CLIN_DIAG", "PRESC_OPIOID"]). Use empty array if L2 not determinable.
- "finding": one of PM, UPC_SAT, UPC_GEN, NO_FINDING, or null if cannot determine.
- "outcomes": array of outcome codes (e.g., ["SUSPEND", "CONDITIONS"]). Use null if cannot determine.
- "profession": one profession string from the list, or null if cannot determine.
- "confidence": "high", "medium", or "low".
- "notes": brief string explaining any ambiguity, edge cases, or truncation issues."""


def annotate_decision(client, taxonomy_text: str, record: dict) -> dict | None:
    """Send a single decision to Claude API and parse the response."""
    decision_id = record["id"]
    text = record.get("full_text", "")

    if not text:
        logger.warning(f"No text for {decision_id}, skipping")
        return None

    # Truncate if needed
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"\n\n[... truncated at {MAX_TEXT_CHARS} characters ...]"

    prompt = _build_user_prompt(taxonomy_text, text, decision_id)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        response_text = response.content[0].text.strip()

        # Parse JSON — handle potential markdown wrapping
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()

        annotation = json.loads(response_text)

        # Track token usage
        annotation["_usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return annotation

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for {decision_id}: {e}")
        logger.error(f"Raw response: {response_text[:500]}")
        return {
            "id": decision_id,
            "error": f"JSON parse error: {e}",
            "raw_response": response_text[:1000],
        }
    except Exception as e:
        logger.error(f"API error for {decision_id}: {e}")
        return {
            "id": decision_id,
            "error": str(e),
        }


def _load_progress(progress_path: Path) -> set:
    """Load set of already-processed decision IDs."""
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _save_progress(progress_path: Path, completed_ids: set):
    """Save set of completed decision IDs."""
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(sorted(completed_ids), f)


def annotate_corpus(
    mode: str = "sample",
    limit: int | None = None,
) -> dict:
    """Annotate decisions using Claude API.

    Automatically resumes from where it left off — existing annotations are
    never overwritten.

    Args:
        mode: "sample" for 300-decision sample, "full" for entire corpus.
        limit: Maximum number of decisions to process (for testing).

    Returns:
        Summary statistics.
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load taxonomy
    taxonomy_text = _load_taxonomy()

    # Load decisions
    if mode == "sample":
        # Load sample IDs, then get full text from corpus
        sample_ids = set()
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_ids.add(json.loads(line)["id"])

        records = []
        with open(ALL_DECISIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record["id"] in sample_ids:
                    records.append(record)

        output_path = LLM_ANNOTATIONS_PATH
        logger.info(f"Loaded {len(records)} sample decisions")
    else:
        records = []
        with open(ALL_DECISIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        output_path = LLM_FULL_ANNOTATIONS_PATH
        logger.info(f"Loaded {len(records)} decisions from full corpus")

    # Always load existing progress to avoid overwriting previous work
    completed_ids = _load_progress(LLM_PROGRESS_PATH)
    annotations = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        # Sync progress file with annotations file in case they diverged
        annot_ids = {a["id"] for a in annotations if "id" in a}
        completed_ids = completed_ids | annot_ids
        _save_progress(LLM_PROGRESS_PATH, completed_ids)

    if completed_ids:
        logger.info(f"Found {len(completed_ids)} existing annotations — skipping these")

    # Apply limit
    remaining = [r for r in records if r["id"] not in completed_ids]
    if limit:
        remaining = remaining[:limit]

    logger.info(f"Processing {len(remaining)} decisions...")

    # Track costs
    total_input_tokens = 0
    total_output_tokens = 0
    errors = 0

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(remaining):
        decision_id = record["id"]

        if (i + 1) % 10 == 0 or i == 0:
            cost_estimate = (total_input_tokens * 3 / 1_000_000) + (total_output_tokens * 15 / 1_000_000)
            logger.info(
                f"  [{i + 1}/{len(remaining)}] Processing {decision_id}... "
                f"(est. cost so far: ${cost_estimate:.2f})"
            )

        annotation = annotate_decision(client, taxonomy_text, record)

        if annotation:
            if "error" in annotation:
                errors += 1
            else:
                usage = annotation.pop("_usage", {})
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)

            annotations.append(annotation)
            completed_ids.add(decision_id)

            # Save progress after each decision
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)
            _save_progress(LLM_PROGRESS_PATH, completed_ids)

        # Rate limiting
        time.sleep(REQUEST_INTERVAL)

    # Final cost estimate (Sonnet pricing: $3/MTok input, $15/MTok output)
    cost_estimate = (total_input_tokens * 3 / 1_000_000) + (total_output_tokens * 15 / 1_000_000)

    stats = {
        "total_processed": len(remaining),
        "total_annotations": len(annotations),
        "errors": errors,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "estimated_cost_usd": round(cost_estimate, 2),
    }

    logger.info("")
    logger.info("=" * 50)
    logger.info("LLM ANNOTATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Processed: {stats['total_processed']}")
    logger.info(f"  Annotations: {stats['total_annotations']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info(f"  Input tokens: {stats['input_tokens']:,}")
    logger.info(f"  Output tokens: {stats['output_tokens']:,}")
    logger.info(f"  Estimated cost: ${stats['estimated_cost_usd']:.2f}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 50)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Annotate decisions using Claude API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="Annotate 300-decision sample")
    group.add_argument("--full", action="store_true", help="Annotate full corpus")
    parser.add_argument("--resume", action="store_true", help="(Ignored — always resumes automatically)")
    parser.add_argument("--limit", type=int, help="Process only N decisions (for testing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mode = "sample" if args.sample else "full"
    stats = annotate_corpus(mode=mode, limit=args.limit)

    print(f"\nDone. {stats['total_annotations']} annotations saved.")
    print(f"Estimated cost: ${stats['estimated_cost_usd']:.2f}")


if __name__ == "__main__":
    main()
