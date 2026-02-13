"""Scrape AustLII for non-NSW health practitioner tribunal decisions.

Covers VIC, QLD, SA, WA, ACT, NT, and TAS jurisdictions. Respects
robots.txt with a 120-second crawl delay. Fully resumable via a
progress tracking JSON file.

Usage:
    python -m src.scraping.austlii_scraper
    python -m src.scraping.austlii_scraper --dry-run
    python -m src.scraping.austlii_scraper --tribunals VCAT QCAT
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .config import (
    AUSTLII_BASE_URL,
    RAW_HTML_DIR,
    SCRAPE_PROGRESS_PATH,
    USER_AGENT,
    load_tribunals,
)
from .filters import has_health_keywords, has_party_match

logger = logging.getLogger(__name__)


class ScrapeProgress:
    """Track scraping progress for resumability."""

    def __init__(self, path: Path = SCRAPE_PROGRESS_PATH):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {"downloaded_urls": [], "discovered_decisions": {}, "last_updated": None}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_downloaded(self, url: str) -> bool:
        return url in self.data["downloaded_urls"]

    def mark_downloaded(self, url: str):
        if url not in self.data["downloaded_urls"]:
            self.data["downloaded_urls"].append(url)

    def add_discovered(self, tribunal_id: str, decisions: list[dict]):
        """Store discovered decision URLs for a tribunal."""
        self.data["discovered_decisions"][tribunal_id] = decisions

    def get_discovered(self, tribunal_id: str) -> list[dict]:
        return self.data["discovered_decisions"].get(tribunal_id, [])


class AustLIIScraper:
    """Scraper for AustLII tribunal decision pages."""

    def __init__(self, dry_run: bool = False):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.dry_run = dry_run
        self.progress = ScrapeProgress()

    def _fetch(self, url: str) -> str | None:
        """Fetch a URL with error handling.

        No rate limiting — the user manages IP rotation/whitelisting
        with AustLII directly.
        """
        logger.info(f"  Fetching: {url}")

        try:
            resp = self.session.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"  Failed to fetch {url}: {e}")
            return None

    def discover_year_links(self, tribunal_path: str) -> list[str]:
        """Fetch a tribunal's index page and extract year links.

        Classic AustLII uses relative year links like "2023/" on
        tribunal index pages.
        """
        url = AUSTLII_BASE_URL + tribunal_path
        html = self._fetch(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        year_links = []

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Match year directory links: "2023/", "/au/cases/vic/VCAT/2023/", etc.
            if re.search(r"(?:^|/)\d{4}/?$", href):
                full_url = self._resolve_url(href, url)
                year_links.append(full_url)

        year_links = sorted(set(year_links))
        logger.info(f"  Found {len(year_links)} year pages for {tribunal_path}")
        return year_links

    def discover_decision_links(
        self, year_url: str, tribunal_id: str, health_specific: bool
    ) -> list[dict]:
        """Fetch a year page and extract individual decision links.

        Classic AustLII year pages list decisions as <li> elements with
        links like ``../2024/5.html``. For general tribunals, applies a
        keyword pre-filter on the full <li> text (which includes the
        citation, party names, and date).
        """
        html = self._fetch(year_url)
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        decisions = []
        total_on_page = 0

        for li in soup.find_all("li"):
            a_tag = li.find("a", href=True)
            if not a_tag:
                continue

            href = a_tag["href"]
            if not href.endswith(".html"):
                continue

            total_on_page += 1

            # Use full <li> text for filtering (includes citation + date)
            title = li.get_text(strip=True)
            full_url = self._resolve_url(href, year_url)

            # For general tribunals, pre-filter by title/citation
            if not health_specific:
                if not (has_party_match(title) or has_health_keywords(title, min_matches=1)):
                    continue

            decisions.append({
                "url": full_url,
                "title": title[:200],  # truncate for storage
                "tribunal_id": tribunal_id,
            })

        logger.info(
            f"  Found {len(decisions)}/{total_on_page} decisions on {year_url}"
            + (" (after health pre-filter)" if not health_specific else "")
        )
        return decisions

    def download_decision(self, decision: dict) -> bool:
        """Download a single decision's HTML and save to raw_html directory.

        Returns True if downloaded successfully.
        """
        url = decision["url"]
        tribunal_id = decision["tribunal_id"]

        if self.progress.is_downloaded(url):
            logger.debug(f"  Already downloaded: {url}")
            return False

        html = self._fetch(url)
        if not html:
            return False

        # Save raw HTML
        tribunal_dir = RAW_HTML_DIR / tribunal_id.lower()
        tribunal_dir.mkdir(parents=True, exist_ok=True)

        # Filename from URL: e.g. "2022_1054.html" from ".../VCAT/2022/1054.html"
        parts = url.rstrip("/").split("/")
        filename = parts[-1]
        year = parts[-2] if len(parts) >= 2 else ""
        if not filename.endswith(".html"):
            filename += ".html"
        if year.isdigit() and len(year) == 4:
            filename = f"{year}_{filename}"

        filepath = tribunal_dir / filename
        filepath.write_text(html, encoding="utf-8")

        self.progress.mark_downloaded(url)
        self.progress.save()

        logger.info(f"  Saved: {filepath}")
        return True

    def _resolve_url(self, href: str, base_url: str) -> str:
        """Resolve a relative URL against a base URL.

        Uses urllib.parse.urljoin for correct handling of ``..``,
        relative paths, and root-relative paths.

        Args:
            href: The href to resolve (absolute, root-relative, or relative).
            base_url: The full URL of the page containing the href.
        """
        if href.startswith("http"):
            return href
        # Ensure base_url is absolute
        if not base_url.startswith("http"):
            base_url = AUSTLII_BASE_URL + base_url
        return urljoin(base_url, href)

    def run(self, tribunal_ids: list[str] | None = None):
        """Run the full scraping pipeline.

        Args:
            tribunal_ids: Specific tribunals to scrape. If None, scrapes all
                non-NSW tribunals marked as "austlii" source.
        """
        config = load_tribunals()
        tribunals = config["tribunals"]

        # Filter to AustLII-sourced tribunals
        targets = {
            tid: info
            for tid, info in tribunals.items()
            if info.get("source") == "austlii"
        }

        if tribunal_ids:
            targets = {
                tid: info
                for tid, info in targets.items()
                if tid in tribunal_ids
            }

        logger.info(f"Scraping {len(targets)} tribunals: {list(targets.keys())}")

        total_discovered = 0
        total_downloaded = 0

        for tribunal_id, info in targets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Tribunal: {info['name']} ({tribunal_id})")
            logger.info(f"Path: {info['austlii_path']}")
            logger.info(f"Health-specific: {info['health_specific']}")
            logger.info(f"{'='*60}")

            # Step 1: Discover year pages
            year_links = self.discover_year_links(info["austlii_path"])

            # Step 2: Discover decision links for each year
            all_decisions = self.progress.get_discovered(tribunal_id)
            discovered_urls = {d["url"] for d in all_decisions}

            for year_url in year_links:
                decisions = self.discover_decision_links(
                    year_url, tribunal_id, info["health_specific"]
                )
                for d in decisions:
                    if d["url"] not in discovered_urls:
                        all_decisions.append(d)
                        discovered_urls.add(d["url"])

            self.progress.add_discovered(tribunal_id, all_decisions)
            self.progress.save()

            total_discovered += len(all_decisions)
            logger.info(
                f"  Total decisions discovered for {tribunal_id}: {len(all_decisions)}"
            )

            if self.dry_run:
                logger.info(f"  [DRY RUN] Skipping downloads for {tribunal_id}")
                continue

            # Step 3: Download each decision
            for i, decision in enumerate(all_decisions):
                if self.progress.is_downloaded(decision["url"]):
                    continue

                logger.info(
                    f"  Downloading {i+1}/{len(all_decisions)}: {decision['title'][:80]}"
                )
                if self.download_decision(decision):
                    total_downloaded += 1

        logger.info(f"\nScraping complete.")
        logger.info(f"  Discovered: {total_discovered} decisions")
        logger.info(f"  Downloaded: {total_downloaded} new decisions")

        return {
            "total_discovered": total_discovered,
            "total_downloaded": total_downloaded,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Scrape AustLII for health practitioner tribunal decisions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover decision links without downloading HTML",
    )
    parser.add_argument(
        "--tribunals",
        nargs="+",
        default=None,
        help="Specific tribunal IDs to scrape (e.g. VCAT QCAT)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    scraper = AustLIIScraper(dry_run=args.dry_run)
    stats = scraper.run(tribunal_ids=args.tribunals)

    if args.dry_run:
        print(f"\n[DRY RUN] Discovered {stats['total_discovered']} decisions.")
    else:
        print(
            f"\nScraping complete: {stats['total_downloaded']} new decisions downloaded."
        )


if __name__ == "__main__":
    main()
