import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime


class SOPAutoSync:
    def __init__(self, base_url, download_dir):
        self.base_url = base_url.rstrip("/")
        self.download_dir = download_dir
        self.sync_log_file = "sync_log.json"

        os.makedirs(self.download_dir, exist_ok=True)
        self.sync_log = self._load_sync_log()

    # ------------------------------------------
    # Sync Log
    # ------------------------------------------

    def _load_sync_log(self):
        if os.path.exists(self.sync_log_file):
            with open(self.sync_log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"documents": {}, "last_sync": None}

    def _save_sync_log(self):
        with open(self.sync_log_file, "w", encoding="utf-8") as f:
            json.dump(self.sync_log, f, indent=4)

    def _compute_hash(self, content):
        return hashlib.sha256(content).hexdigest()

    # ------------------------------------------
    # Step 1: Get HTML pages from _toc.json
    # ------------------------------------------

    def discover_html_pages_from_toc(self):
        print("🔍 Fetching HTML pages from _toc.json...")

        toc_url = f"{self.base_url}/_toc.json"

        try:
            response = requests.get(toc_url, timeout=10)
            response.raise_for_status()
            toc_data = response.json()
        except Exception as e:
            print(f"❌ Failed to fetch TOC: {e}")
            return set()

        html_pages = set()

        for item in toc_data:
            if "a_attr" in item and "href" in item["a_attr"]:
                href = item["a_attr"]["href"]

                if href.lower().endswith(".html"):
                    full_url = urljoin(self.base_url + "/", href)
                    html_pages.add(full_url)

        print(f" Found {len(html_pages)} HTML pages")
        return html_pages

    # ------------------------------------------
    # Step 2: Extract PDF links
    # ------------------------------------------

    def extract_pdf_links(self, html_pages):
        print("🔎 Extracting PDF links...")
        pdf_links = set()

        for url in html_pages:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.content, "html.parser")

                for link in soup.find_all("a", href=True):
                    href = link["href"]

                    if href.lower().endswith(".pdf"):
                        full_url = urljoin(self.base_url + "/", href)
                        pdf_links.add(full_url)

            except Exception:
                continue

        print(f" Found {len(pdf_links)} PDF links")
        return pdf_links

    # ------------------------------------------
    # Step 3: Download & Compare
    # ------------------------------------------

    def download_and_track(self, pdf_url):
        try:
            response = requests.get(pdf_url, timeout=20)
            if response.status_code != 200:
                return None, None

            content = response.content
            content_hash = self._compute_hash(content)

            filename = os.path.basename(urlparse(pdf_url).path)
            filepath = os.path.join(self.download_dir, filename)

            is_new = pdf_url not in self.sync_log["documents"]

            # If exists and unchanged
            if not is_new:
                if self.sync_log["documents"][pdf_url]["hash"] == content_hash:
                    return "unchanged", None

            # Save / overwrite file
            with open(filepath, "wb") as f:
                f.write(content)

            # Update sync log
            self.sync_log["documents"][pdf_url] = {
                "hash": content_hash,
                "path": filepath,
                "last_updated": datetime.now().isoformat(),
            }

            if is_new:
                return "new", filepath
            else:
                return "updated", filepath

        except Exception as e:
            print(f"  ⚠ Error downloading {pdf_url}: {e}")
            return None, None

    # ------------------------------------------
    # MAIN SYNC
    # ------------------------------------------

    def sync(self):
        print("\n" + "=" * 60)
        print(f"SOP PDF Sync Started: {datetime.now()}")
        print("=" * 60)

        html_pages = self.discover_html_pages_from_toc()
        pdf_links = self.extract_pdf_links(html_pages)

        new = 0
        updated = 0
        unchanged = 0
        changed_files = []

        for pdf_url in pdf_links:
            status, filepath = self.download_and_track(pdf_url)

            if status == "new":
                print(f"  ✓ NEW: {os.path.basename(filepath)}")
                new += 1
                changed_files.append(filepath)

            elif status == "updated":
                print(f"  ✓ UPDATED: {os.path.basename(filepath)}")
                updated += 1
                changed_files.append(filepath)

            elif status == "unchanged":
                unchanged += 1

        self.sync_log["last_sync"] = datetime.now().isoformat()
        self._save_sync_log()

        print("\n" + "=" * 60)
        print("Sync Summary:")
        print(f"  New: {new}")
        print(f"  Updated: {updated}")
        print(f"  Unchanged: {unchanged}")
        print("=" * 60 + "\n")

        # Now returning changed files also
        return new, updated, changed_files

# Helper function for rebuilding vector store
def rebuild_vectorstore():
    """
    Rebuild the vector store after syncing new PDFs
    Call this after sync() completes with changes
    """
    from rag.loader import load_pdfs
    from rag.splitter import split_docs
    from rag.vectorstore import create_vectorstore

    print("🔄 Rebuilding vector store...")

    # Load all PDFs
    docs = load_pdfs()

    # Split into chunks
    chunks = split_docs(docs)

    # Recreate vector store (this will save it automatically)
    vectorstore = create_vectorstore(chunks)

    print(" Vector store rebuilt successfully!")

    return vectorstore


# ------------------------------------------
# Run standalone
# ------------------------------------------

if __name__ == "__main__":
    BASE_URL = "https://upaygoa.com/geltm/helpndoc"
    DOWNLOAD_DIR = "./sop_documents"

    syncer = SOPAutoSync(BASE_URL, DOWNLOAD_DIR)
    new_count, updated_count = syncer.sync()

    # Rebuild vector store if there were changes
    if new_count > 0 or updated_count > 0:
        print("\n📦 Changes detected - rebuilding vector store...")
        rebuild_vectorstore()
    else:
        print("\n No changes - vector store is up to date")