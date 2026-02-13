"""Download tribunal decisions from the Wayback Machine for AustLII 410-Gone databases.

Covers NTHPRT (21 PDFs), TasHPT (13 RTF/HTML), and TasCAT (64 RTFs).
These tribunals have been permanently removed from AustLII but are
partially archived by the Internet Archive.

Six tribunals (VicMPB, VicPRB, VicPsyRB, QldMPT, SADenPCT, SAPsyB)
have no archive anywhere and are unrecoverable.

Usage:
    python -m src.scraping.wayback_scraper
    python -m src.scraping.wayback_scraper --dry-run
    python -m src.scraping.wayback_scraper --tribunals NTHPRT TasHPT
"""

import argparse
import json
import logging
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

from .config import METADATA_DIR, RAW_HTML_DIR

logger = logging.getLogger(__name__)

CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_RAW = "https://web.archive.org/web/{ts}id_/{url}"

# AustLII hosts to search across — the same content was mirrored on many subdomains
AUSTLII_HOSTS = [
    "www.austlii.edu.au",
    "www5.austlii.edu.au",
    "www6.austlii.edu.au",
    "www7.austlii.edu.au",
    "www8.austlii.edu.au",
    "classic.austlii.edu.au",
]

# Tribunals recoverable from Wayback, with their AustLII path segments
WAYBACK_TRIBUNALS = {
    "NTHPRT": {
        "path": "nt/NTHPRT",
        "health_specific": True,
        "name": "NT Health Professional Review Tribunal",
    },
    "TasHPT": {
        "path": "tas/TasHPT",
        "health_specific": True,
        "name": "Health Practitioners Tribunal of Tasmania",
    },
    "TasCAT": {
        "path": "tas/TasCAT",
        "health_specific": False,
        "name": "Tasmanian Civil and Administrative Tribunal",
    },
}

PROGRESS_PATH = METADATA_DIR / "wayback_progress.json"


class WaybackProgress:
    """Track Wayback download progress for resumability."""

    def __init__(self, path: Path = PROGRESS_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"downloaded": [], "discovered": {}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_downloaded(self, path_key: str) -> bool:
        return path_key in self.data["downloaded"]

    def mark_downloaded(self, path_key: str):
        if path_key not in self.data["downloaded"]:
            self.data["downloaded"].append(path_key)

    def set_discovered(self, tribunal_id: str, entries: list[dict]):
        self.data["discovered"][tribunal_id] = entries

    def get_discovered(self, tribunal_id: str) -> list[dict]:
        return self.data["discovered"].get(tribunal_id, [])


def discover_decisions(tribunal_id: str, austlii_path: str) -> list[dict]:
    """Query Wayback CDX API to find archived decision files for a tribunal.

    Searches all known AustLII mirror hostnames, deduplicates by path,
    and keeps the most recent snapshot timestamp for each file.

    Returns a list of dicts: {path, timestamp, original_url, wayback_url, ext}
    """
    logger.info(f"  Querying Wayback CDX for {tribunal_id}...")

    # Collect all CDX entries across all hosts
    all_entries = []
    for host in AUSTLII_HOSTS:
        url = (
            f"{CDX_API}?url={host}/au/cases/{austlii_path}"
            f"&matchType=prefix&output=json"
            f"&filter=statuscode:200"
            f"&fl=timestamp,original,mimetype"
        )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                if len(data) > 1:
                    all_entries.extend(data[1:])
        except Exception as e:
            logger.debug(f"  CDX query failed for {host}: {e}")

    # Deduplicate by AustLII path, keep most recent timestamp
    best: dict[str, tuple[str, str, str]] = {}
    for ts, orig, mime in all_entries:
        m = re.search(r"austlii\.edu\.au(/au/cases/.+)", orig)
        if not m:
            continue
        path = m.group(1)

        # Only keep decision files
        filename = path.split("/")[-1]
        if "." not in filename:
            continue
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in ("rtf", "pdf", "html", "htm"):
            continue
        # Skip index/toc pages
        if "toc-" in filename or filename == "recent.html":
            continue

        if path not in best or ts > best[path][0]:
            best[path] = (ts, orig, mime)

    decisions = []
    for path, (ts, orig, mime) in sorted(best.items()):
        ext = path.rsplit(".", 1)[-1].lower()
        wayback_url = WAYBACK_RAW.format(ts=ts, url=orig)

        # Extract year from path for filename
        year_match = re.search(r"/(\d{4})/", path)
        year = year_match.group(1) if year_match else "unknown"

        decisions.append({
            "path": path,
            "timestamp": ts,
            "original_url": orig,
            "wayback_url": wayback_url,
            "ext": ext,
            "year": year,
            "tribunal_id": tribunal_id,
        })

    logger.info(f"  Found {len(decisions)} unique decision files for {tribunal_id}")
    return decisions


def download_decision(
    decision: dict,
    progress: WaybackProgress,
    dry_run: bool = False,
) -> bool:
    """Download a single decision file from the Wayback Machine.

    Saves to data/raw_html/{tribunal_id}/{year}_{filename}.{ext}

    Returns True if newly downloaded.
    """
    path_key = decision["path"]
    tribunal_id = decision["tribunal_id"]

    if progress.is_downloaded(path_key):
        return False

    if dry_run:
        return False

    wayback_url = decision["wayback_url"]
    logger.info(f"  Fetching: {wayback_url}")

    try:
        req = urllib.request.Request(
            wayback_url,
            headers={"User-Agent": "HealthPracMisconductResearch/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
    except Exception as e:
        logger.warning(f"  Failed to fetch {wayback_url}: {e}")
        return False

    # Save file
    tribunal_dir = RAW_HTML_DIR / tribunal_id.lower()
    tribunal_dir.mkdir(parents=True, exist_ok=True)

    original_filename = decision["path"].split("/")[-1]
    year = decision["year"]
    filename = f"{year}_{original_filename}"

    filepath = tribunal_dir / filename
    filepath.write_bytes(content)

    progress.mark_downloaded(path_key)
    progress.save()

    logger.info(f"  Saved: {filepath} ({len(content):,} bytes)")
    return True


def run(
    tribunal_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the Wayback Machine download pipeline.

    Args:
        tribunal_ids: Specific tribunals to download. If None, does all three.
        dry_run: If True, discover only; don't download.
    """
    progress = WaybackProgress()

    targets = WAYBACK_TRIBUNALS
    if tribunal_ids:
        targets = {k: v for k, v in targets.items() if k in tribunal_ids}

    logger.info(f"Wayback scraper targeting: {list(targets.keys())}")

    total_discovered = 0
    total_downloaded = 0

    for tribunal_id, info in targets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Tribunal: {info['name']} ({tribunal_id})")
        logger.info(f"Health-specific: {info['health_specific']}")
        logger.info(f"{'='*60}")

        # Use cached discovery if available
        decisions = progress.get_discovered(tribunal_id)
        if not decisions:
            decisions = discover_decisions(tribunal_id, info["path"])
            progress.set_discovered(tribunal_id, decisions)
            progress.save()

        total_discovered += len(decisions)
        logger.info(f"  {len(decisions)} decisions to process")

        if dry_run:
            for d in decisions:
                logger.info(f"    [DRY] {d['path']} ({d['ext']})")
            continue

        for i, decision in enumerate(decisions):
            if progress.is_downloaded(decision["path"]):
                continue
            logger.info(f"  [{i+1}/{len(decisions)}] {decision['path']}")
            if download_decision(decision, progress):
                total_downloaded += 1
            # Brief pause between Wayback requests
            time.sleep(1)

    logger.info(f"\nWayback scraping complete.")
    logger.info(f"  Discovered: {total_discovered}")
    logger.info(f"  Downloaded: {total_downloaded}")

    return {
        "total_discovered": total_discovered,
        "total_downloaded": total_downloaded,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download health practitioner decisions from the Wayback Machine"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover only; don't download files",
    )
    parser.add_argument(
        "--tribunals",
        nargs="+",
        choices=list(WAYBACK_TRIBUNALS.keys()),
        default=None,
        help="Specific tribunals to download",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    stats = run(tribunal_ids=args.tribunals, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] Discovered {stats['total_discovered']} decisions.")
    else:
        print(f"\nWayback download complete: {stats['total_downloaded']} new files.")


if __name__ == "__main__":
    main()
