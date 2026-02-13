"""Extract structured data from saved AustLII HTML files.

Parses raw HTML decision files into the project's standard JSONL schema,
extracting metadata, full text (via inscriptis), section segmentation,
and outcome information.

Usage:
    python -m src.scraping.austlii_extractor
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from .config import (
    AUSTLII_BASE_URL,
    AUSTLII_DECISIONS_PATH,
    EXTRACTED_DIR,
    RAW_HTML_DIR,
    load_tribunals,
)
from .filters import is_health_decision

logger = logging.getLogger(__name__)

# Section heading patterns for segmentation
SECTION_PATTERNS = [
    (r"(?:^|\n)\s*(?:BACKGROUND|Background)\s*\n", "background"),
    (r"(?:^|\n)\s*(?:ALLEGATIONS?|Allegations?|THE COMPLAINTS?|The Complaints?|CHARGES?|Charges?)\s*\n", "allegations"),
    (r"(?:^|\n)\s*(?:EVIDENCE|Evidence|THE EVIDENCE|The Evidence|SUBMISSIONS?|Submissions?)\s*\n", "evidence"),
    (r"(?:^|\n)\s*(?:FINDINGS?|Findings?|THE (?:TRIBUNAL'?S?\s+)?FINDINGS?|REASONS?\s+FOR\s+DECISION|DISCUSSION|Discussion)\s*\n", "findings"),
    (r"(?:^|\n)\s*(?:ORDERS?|Orders?|THE ORDERS?|The Orders?|DETERMINATION|Determination|DECISION|Decision)\s*\n", "orders"),
]

# Outcome extraction patterns
OUTCOME_PATTERNS = {
    "cancellation": re.compile(
        r"\b(?:cancel(?:led|lation)|registration\s+(?:is|be)\s+cancelled)\b",
        re.IGNORECASE,
    ),
    "suspension": re.compile(
        r"\b(?:suspend(?:ed)?|suspension)\s+(?:of\s+)?(?:registration|for\s+a\s+period)\b",
        re.IGNORECASE,
    ),
    "conditions": re.compile(
        r"\b(?:conditions?\s+(?:imposed|placed)|(?:impose|place)\s+conditions?|"
        r"subject\s+to\s+(?:the\s+following\s+)?conditions?)\b",
        re.IGNORECASE,
    ),
    "reprimand": re.compile(
        r"\b(?:reprimand(?:ed)?|censure[d]?)\b",
        re.IGNORECASE,
    ),
    "caution": re.compile(
        r"\b(?:caution(?:ed)?)\b",
        re.IGNORECASE,
    ),
    "fine": re.compile(
        r"\b(?:fine[d]?\s+(?:the\s+sum\s+of\s+)?\$|pay\s+a\s+(?:fine|penalty)\s+of)\b",
        re.IGNORECASE,
    ),
    "undertaking": re.compile(
        r"\b(?:undertaking|undertake)\b",
        re.IGNORECASE,
    ),
    "no_adverse_finding": re.compile(
        r"\b(?:no\s+(?:adverse\s+)?finding|complaint\s+dismissed|"
        r"application\s+dismissed|not\s+(?:established|proven|made\s+out))\b",
        re.IGNORECASE,
    ),
}

# Profession extraction patterns
PROFESSION_PATTERNS = {
    "medical_practitioner": re.compile(
        r"\b(?:medical\s+practitioner|doctor|physician|surgeon|general\s+practitioner|GP)\b",
        re.IGNORECASE,
    ),
    "nurse": re.compile(
        r"\b(?:registered\s+nurse|enrolled\s+nurse|nurse\s+practitioner|nursing)\b",
        re.IGNORECASE,
    ),
    "midwife": re.compile(r"\bmidwi(?:fe|fery|ves)\b", re.IGNORECASE),
    "dentist": re.compile(
        r"\b(?:dentist|dental\s+practitioner|dental\s+surgeon)\b", re.IGNORECASE
    ),
    "pharmacist": re.compile(r"\bpharmacist\b", re.IGNORECASE),
    "psychologist": re.compile(r"\bpsychologist\b", re.IGNORECASE),
    "physiotherapist": re.compile(r"\bphysiotherapist\b", re.IGNORECASE),
    "chiropractor": re.compile(r"\bchiropractor\b", re.IGNORECASE),
    "osteopath": re.compile(r"\bosteopath\b", re.IGNORECASE),
    "optometrist": re.compile(r"\boptometrist\b", re.IGNORECASE),
    "podiatrist": re.compile(r"\bpodiatrist\b", re.IGNORECASE),
    "paramedic": re.compile(r"\bparamedic\b", re.IGNORECASE),
    "occupational_therapist": re.compile(
        r"\boccupational\s+therapist\b", re.IGNORECASE
    ),
    "chinese_medicine_practitioner": re.compile(
        r"\bchinese\s+medicine\s+practitioner\b", re.IGNORECASE
    ),
}

# Board extraction pattern
BOARD_PATTERN = re.compile(
    r"((?:Medical|Nursing and Midwifery|Dental|Pharmacy|Psychology|"
    r"Physiotherapy|Chiropractic|Osteopathy|Optometry|Podiatry|"
    r"Occupational Therapy|Chinese Medicine|Paramedicine|"
    r"Medical Radiation Practice|Aboriginal and Torres Strait Islander "
    r"Health Practice)\s+Board\s+of\s+Australia)",
    re.IGNORECASE,
)


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using inscriptis."""
    try:
        from inscriptis import get_text

        return get_text(html)
    except ImportError:
        logger.warning(
            "inscriptis not installed; falling back to BeautifulSoup text extraction"
        )
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n")


def _rtf_to_text(rtf_bytes: bytes) -> str:
    """Convert RTF content to plain text using striprtf."""
    try:
        from striprtf.striprtf import rtf_to_text

        # striprtf expects a string
        rtf_str = rtf_bytes.decode("utf-8", errors="replace")
        return rtf_to_text(rtf_str)
    except ImportError:
        logger.error("striprtf not installed — run: pip install striprtf")
        return ""


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Convert PDF content to plain text using PyMuPDF.

    Falls back to OCR via pytesseract for scanned (image-based) PDFs.
    """
    try:
        import pymupdf
    except ImportError:
        logger.error("PyMuPDF not installed — run: pip install pymupdf")
        return ""

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    text = "\n\n".join(pages).strip()

    # If text extraction yielded very little, try OCR
    if len(text) < 100 and len(doc) > 0:
        logger.info(f"  PDF has no text layer ({len(doc)} pages) — attempting OCR...")
        try:
            import pytesseract
            from PIL import Image
            import io

            ocr_pages = []
            for page in doc:
                # Render page to image at 300 DPI
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_text = pytesseract.image_to_string(img)
                ocr_pages.append(page_text)
            text = "\n\n".join(ocr_pages).strip()
            logger.info(f"  OCR extracted {len(text)} chars from {len(doc)} pages")
        except ImportError:
            logger.warning(
                "  pytesseract/Pillow not installed for OCR — run: "
                "pip install pytesseract Pillow"
            )
        except Exception as e:
            logger.warning(f"  OCR failed: {e}")

    doc.close()
    return text


def extract_metadata_from_html(html: str, tribunal_id: str) -> dict:
    """Extract structured metadata from AustLII HTML.

    Returns a dict with: citation, date, catchwords, practitioner_name, url.
    """
    soup = BeautifulSoup(html, "lxml")
    meta = {
        "citation": None,
        "date": None,
        "catchwords": None,
        "practitioner_name": None,
        "url": None,
    }

    # Citation — look in <title>, or <span class="citation">
    title_tag = soup.find("title")
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        # AustLII titles often have format: "Party v Party [Year] TRIBUNAL Num"
        citation_match = re.search(
            r"\[\d{4}\]\s*\w+\s+\d+", title_text
        )
        if citation_match:
            meta["citation"] = citation_match.group(0)

    citation_span = soup.find("span", class_="citation")
    if citation_span:
        meta["citation"] = citation_span.get_text(strip=True)

    # URL — canonical link
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        meta["url"] = canonical["href"]

    # Date — look for date metadata or extract from text
    date_tag = soup.find("meta", attrs={"name": "DC.Date"})
    if date_tag and date_tag.get("content"):
        meta["date"] = date_tag["content"][:10]

    if not meta["date"]:
        # Try to extract from the decision text
        text = soup.get_text()[:3000]
        date_match = re.search(
            r"(?:decided|dated|delivered|published)\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})",
            text,
            re.IGNORECASE,
        )
        if date_match:
            try:
                from dateutil.parser import parse as parse_date

                dt = parse_date(date_match.group(1))
                meta["date"] = dt.strftime("%Y-%m-%d")
            except (ImportError, ValueError):
                pass

    # Catchwords
    catchwords_tag = soup.find("meta", attrs={"name": "DC.Subject"})
    if catchwords_tag and catchwords_tag.get("content"):
        meta["catchwords"] = catchwords_tag["content"]

    if not meta["catchwords"]:
        # Look for a "Catchwords:" section in the text
        text = soup.get_text()[:5000]
        cw_match = re.search(
            r"(?:CATCHWORDS?|Catchwords?)\s*:?\s*\n(.+?)(?:\n\s*\n|\n[A-Z])",
            text,
            re.DOTALL,
        )
        if cw_match:
            meta["catchwords"] = cw_match.group(1).strip()

    # Practitioner name — from party names in title
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        # Pattern: "Board v [Name]" or "HCCC v [Name]"
        name_match = re.search(
            r"(?:Board of Australia|HCCC|Health Care Complaints Commission|"
            r"Health Ombudsman|Council of New South Wales)\s+v\s+(.+?)(?:\s*\[|\s*$)",
            title_text,
            re.IGNORECASE,
        )
        if name_match:
            meta["practitioner_name"] = name_match.group(1).strip()

    return meta


def extract_sections(text: str) -> dict[str, str]:
    """Segment decision text into standard sections.

    Returns a dict mapping section names to their text content.
    Sections: background, allegations, evidence, findings, orders.
    """
    sections = {}

    # Find all section boundaries
    boundaries = []
    for pattern_str, section_name in SECTION_PATTERNS:
        pattern = re.compile(pattern_str)
        for m in pattern.finditer(text):
            boundaries.append((m.start(), m.end(), section_name))

    if not boundaries:
        return sections

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    # Extract text between boundaries
    for i, (start, end, name) in enumerate(boundaries):
        if i + 1 < len(boundaries):
            section_text = text[end : boundaries[i + 1][0]]
        else:
            section_text = text[end:]

        section_text = section_text.strip()
        if section_text:
            sections[name] = section_text

    return sections


def extract_outcomes(text: str) -> list[str]:
    """Extract outcome types from decision text.

    Focuses on the orders/determination section if available,
    otherwise searches the full text.
    """
    # Prefer searching only in the last portion of the text (orders section)
    search_text = text[-5000:]

    outcomes = []
    for outcome_type, pattern in OUTCOME_PATTERNS.items():
        if pattern.search(search_text):
            outcomes.append(outcome_type)

    return outcomes


def extract_profession(text: str) -> str | None:
    """Extract the practitioner's profession from decision text."""
    # Search early in the text where profession is typically stated
    search_text = text[:5000]
    for profession, pattern in PROFESSION_PATTERNS.items():
        if pattern.search(search_text):
            return profession
    return None


def extract_board(text: str) -> str | None:
    """Extract the regulatory board name from decision text."""
    match = BOARD_PATTERN.search(text[:5000])
    return match.group(1) if match else None


def _make_document_id(tribunal_id: str, filename: str, citation: str | None) -> str:
    """Generate a document ID."""
    if citation:
        m = re.search(r"\[(\d{4})\]\s*(\w+)\s+(\d+)", citation)
        if m:
            year, trib, num = m.groups()
            return f"{trib.lower()}_{year}_{num}"

    # Fallback: use tribunal_id + filename
    stem = Path(filename).stem
    return f"{tribunal_id.lower()}_{stem}"


def extract_decision(filepath: Path, tribunal_id: str) -> dict | None:
    """Extract structured data from a single AustLII HTML file.

    Returns a dict in the project's standard JSONL schema, or None if
    the file cannot be parsed.
    """
    try:
        html = filepath.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None

    config = load_tribunals()
    tribunal_info = config["tribunals"].get(tribunal_id, {})

    # Extract metadata from HTML structure
    meta = extract_metadata_from_html(html, tribunal_id)

    # Convert to plain text
    full_text = _html_to_text(html)
    if not full_text or len(full_text.strip()) < 100:
        logger.warning(f"Skipping {filepath}: extracted text too short")
        return None

    # Section segmentation
    sections = extract_sections(full_text)

    # Outcome extraction
    outcomes = extract_outcomes(full_text)

    # Profession and board extraction
    profession = extract_profession(full_text)
    board = extract_board(full_text)

    # Build URL if not found in metadata
    url = meta["url"]
    if not url:
        url = AUSTLII_BASE_URL + tribunal_info.get("austlii_path", "") + filepath.name

    return {
        "id": _make_document_id(tribunal_id, filepath.name, meta["citation"]),
        "source": "austlii",
        "tribunal": tribunal_id,
        "jurisdiction": tribunal_info.get("jurisdiction", "UNKNOWN"),
        "date": meta["date"],
        "citation": meta["citation"],
        "url": url,
        "practitioner_name": meta["practitioner_name"],
        "profession": profession,
        "board": board,
        "catchwords": meta["catchwords"],
        "full_text": full_text,
        "sections": sections,
        "outcome_raw": outcomes if outcomes else None,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_decision_from_file(filepath: Path, tribunal_id: str) -> dict | None:
    """Extract structured data from an HTML, RTF, or PDF decision file.

    Dispatches to the appropriate converter based on file extension,
    then applies the same metadata/section/outcome extraction pipeline.

    Returns a dict in the standard JSONL schema, or None if unparseable.
    """
    ext = filepath.suffix.lower()

    if ext == ".html" or ext == ".htm":
        return extract_decision(filepath, tribunal_id)

    # For RTF and PDF, read as bytes and convert to text
    try:
        raw = filepath.read_bytes()
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None

    if ext == ".rtf":
        full_text = _rtf_to_text(raw)
    elif ext == ".pdf":
        full_text = _pdf_to_text(raw)
    else:
        logger.warning(f"Unsupported file type: {filepath}")
        return None

    if not full_text or len(full_text.strip()) < 100:
        logger.warning(f"Skipping {filepath}: extracted text too short ({len(full_text.strip())} chars)")
        return None

    config = load_tribunals()
    tribunal_info = config["tribunals"].get(tribunal_id, {})

    # Extract metadata from plain text (no HTML structure available)
    citation = None
    citation_match = re.search(r"\[\d{4}\]\s*\w+\s+\d+", full_text[:3000])
    if citation_match:
        citation = citation_match.group(0)

    date = None
    date_match = re.search(
        r"(?:decided|dated|delivered|published|heard)\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})",
        full_text[:5000],
        re.IGNORECASE,
    )
    if date_match:
        try:
            from dateutil.parser import parse as parse_date
            dt = parse_date(date_match.group(1))
            date = dt.strftime("%Y-%m-%d")
        except (ImportError, ValueError):
            pass

    # If no date from text, try year from filename (e.g. 2014_3.rtf)
    if not date:
        year_match = re.match(r"(\d{4})_", filepath.name)
        if year_match:
            date = f"{year_match.group(1)}-01-01"

    practitioner_name = None
    name_match = re.search(
        r"(?:Board of Australia|HCCC|Health Care Complaints Commission|"
        r"Health Ombudsman|Council of New South Wales)\s+v\s+(.+?)(?:\s*\[|\s*\n)",
        full_text[:3000],
        re.IGNORECASE,
    )
    if name_match:
        practitioner_name = name_match.group(1).strip()

    sections = extract_sections(full_text)
    outcomes = extract_outcomes(full_text)
    profession = extract_profession(full_text)
    board = extract_board(full_text)

    return {
        "id": _make_document_id(tribunal_id, filepath.name, citation),
        "source": "wayback",
        "tribunal": tribunal_id,
        "jurisdiction": tribunal_info.get("jurisdiction", "UNKNOWN"),
        "date": date,
        "citation": citation,
        "url": None,
        "practitioner_name": practitioner_name,
        "profession": profession,
        "board": board,
        "catchwords": None,
        "full_text": full_text,
        "sections": sections,
        "outcome_raw": outcomes if outcomes else None,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_all():
    """Extract structured data from all downloaded AustLII HTML files.

    Writes to data/extracted/austlii_decisions.jsonl.
    """
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    config = load_tribunals()
    tribunals = config["tribunals"]

    total_files = 0
    total_extracted = 0
    total_skipped = 0
    total_filtered = 0

    with open(AUSTLII_DECISIONS_PATH, "w", encoding="utf-8") as f:
        for tribunal_id, info in tribunals.items():
            if info.get("source") != "austlii":
                continue

            tribunal_dir = RAW_HTML_DIR / tribunal_id.lower()
            if not tribunal_dir.exists():
                continue

            all_files = sorted(
                list(tribunal_dir.glob("*.html"))
                + list(tribunal_dir.glob("*.rtf"))
                + list(tribunal_dir.glob("*.pdf"))
            )
            if not all_files:
                continue

            health_specific = info.get("health_specific", False)
            logger.info(
                f"Extracting {len(all_files)} files from {tribunal_id}"
                + ("" if health_specific else " (with health filter)")
                + "..."
            )

            for filepath in all_files:
                total_files += 1
                doc = extract_decision_from_file(filepath, tribunal_id)

                if doc is None:
                    total_skipped += 1
                    continue

                # For general tribunals, filter out non-health decisions
                if not health_specific:
                    citation = doc.get("citation") or ""
                    text = doc.get("full_text") or ""
                    if not is_health_decision(citation, text, tribunal_id):
                        total_filtered += 1
                        logger.debug(f"  Filtered (non-health): {filepath.name}")
                        continue

                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_extracted += 1

    logger.info(
        f"Extraction complete. "
        f"Files processed: {total_files}. "
        f"Extracted: {total_extracted}. "
        f"Skipped: {total_skipped}. "
        f"Filtered (non-health): {total_filtered}."
    )

    return {
        "total_files": total_files,
        "total_extracted": total_extracted,
        "total_skipped": total_skipped,
        "total_filtered": total_filtered,
        "output_path": str(AUSTLII_DECISIONS_PATH),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stats = extract_all()
    print(f"\nExtraction complete: {stats['total_extracted']} decisions extracted.")


if __name__ == "__main__":
    main()
