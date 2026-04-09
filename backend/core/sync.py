import hashlib
import json
import logging
import os
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from backend.config import settings

logger = logging.getLogger(__name__)


class SOPSync:
    def __init__(self):
        self.base_url = settings.sop_base_url.rstrip("/")
        self.download_dir = settings.sop_documents_dir
        self.sync_log_file = str(
            settings.sop_documents_dir.replace("sop_documents", "") + "sync_log.json"
        ).replace("\\", "/")
        # Fix: use project root for sync_log
        from backend.config import PROJECT_ROOT
        self.sync_log_file = str(PROJECT_ROOT / "sync_log.json")

        os.makedirs(self.download_dir, exist_ok=True)
        self.sync_log = self._load_sync_log()

    def _load_sync_log(self) -> dict:
        if os.path.exists(self.sync_log_file):
            with open(self.sync_log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"documents": {}, "last_sync": None}

    def _save_sync_log(self):
        with open(self.sync_log_file, "w", encoding="utf-8") as f:
            json.dump(self.sync_log, f, indent=4)

    @staticmethod
    def _compute_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def discover_html_pages(self) -> set[str]:
        toc_url = f"{self.base_url}/_toc.json"
        try:
            response = requests.get(toc_url, timeout=15)
            response.raise_for_status()
            toc_data = response.json()
        except Exception as e:
            logger.error("Failed to fetch TOC: %s", e)
            return set()

        pages = set()
        for item in toc_data:
            if "a_attr" in item and "href" in item["a_attr"]:
                href = item["a_attr"]["href"]
                if href.lower().endswith(".html"):
                    pages.add(urljoin(self.base_url + "/", href))

        logger.info("Discovered %d HTML pages", len(pages))
        return pages

    def extract_pdf_links(self, html_pages: set[str]) -> set[str]:
        pdf_links = set()
        for url in html_pages:
            try:
                response = requests.get(url, timeout=15)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.content, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if href.lower().endswith(".pdf"):
                        pdf_links.add(urljoin(self.base_url + "/", href))
            except Exception:
                continue

        logger.info("Found %d PDF links", len(pdf_links))
        return pdf_links

    def download_pdf(self, pdf_url: str) -> tuple[str | None, str | None]:
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code != 200:
                return None, None

            content = response.content
            content_hash = self._compute_hash(content)
            filename = os.path.basename(urlparse(pdf_url).path)
            filepath = os.path.join(self.download_dir, filename)

            is_new = pdf_url not in self.sync_log["documents"]

            if not is_new:
                if self.sync_log["documents"][pdf_url]["hash"] == content_hash:
                    return "unchanged", None

            with open(filepath, "wb") as f:
                f.write(content)

            self.sync_log["documents"][pdf_url] = {
                "hash": content_hash,
                "path": filepath,
                "last_updated": datetime.now().isoformat(),
            }

            return ("new" if is_new else "updated"), filepath

        except Exception as e:
            logger.error("Error downloading %s: %s", pdf_url, e)
            return None, None

    def sync(self) -> dict:
        logger.info("Starting SOP sync at %s", datetime.now().isoformat())

        html_pages = self.discover_html_pages()
        pdf_links = self.extract_pdf_links(html_pages)

        result = {"new": 0, "updated": 0, "unchanged": 0, "changed_files": []}

        for pdf_url in pdf_links:
            status, filepath = self.download_pdf(pdf_url)
            if status == "new":
                result["new"] += 1
                result["changed_files"].append(filepath)
            elif status == "updated":
                result["updated"] += 1
                result["changed_files"].append(filepath)
            elif status == "unchanged":
                result["unchanged"] += 1

        self.sync_log["last_sync"] = datetime.now().isoformat()
        self._save_sync_log()

        logger.info(
            "Sync complete: %d new, %d updated, %d unchanged",
            result["new"], result["updated"], result["unchanged"],
        )
        return result

    def get_status(self) -> dict:
        return {
            "last_sync": self.sync_log.get("last_sync"),
            "total_documents": len(self.sync_log.get("documents", {})),
        }
