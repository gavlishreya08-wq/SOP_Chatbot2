import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime
from pathlib import Path


class SOPAutoSyncHybrid:
    def __init__(self, base_url, download_dir, local_html_dir=None):
        """
        Hybrid sync: Fetches from API + scans local HTML files
        
        Args:
            base_url: Base URL for API sync
            download_dir: Where to save PDFs
            local_html_dir: Optional local folder with HTML files and images
        """
        self.base_url = base_url.rstrip("/")
        self.download_dir = Path(download_dir)
        self.local_html_dir = Path(local_html_dir) if local_html_dir else None
        self.sync_log_file = "sync_log.json"

        # Create directories
        self.download_dir.mkdir(parents=True, exist_ok=True)
        if self.local_html_dir:
            (self.download_dir / "images").mkdir(exist_ok=True)
        
        self.sync_log = self._load_sync_log()
        
        # Track processed PDFs to avoid duplicates
        self.processed_pdfs = set()

    # ------------------------------------------
    # Sync Log
    # ------------------------------------------

    def _load_sync_log(self):
        if os.path.exists(self.sync_log_file):
            with open(self.sync_log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"documents": {}, "images": {}, "last_sync": None}

    def _save_sync_log(self):
        with open(self.sync_log_file, "w", encoding="utf-8") as f:
            json.dump(self.sync_log, f, indent=4)

    def _compute_hash(self, content):
        return hashlib.sha256(content).hexdigest()

    # ------------------------------------------
    # PART 1: API Sync (from _toc.json)
    # ------------------------------------------

    def discover_html_pages_from_toc(self):
        """Fetch HTML pages from API"""
        print("🔍 [API] Fetching HTML pages from _toc.json...")

        toc_url = f"{self.base_url}/_toc.json"

        try:
            response = requests.get(toc_url, timeout=10)
            response.raise_for_status()
            toc_data = response.json()
        except Exception as e:
            print(f"  ❌ Failed to fetch TOC: {e}")
            return set()

        html_pages = set()
        for item in toc_data:
            if "a_attr" in item and "href" in item["a_attr"]:
                href = item["a_attr"]["href"]
                if href.lower().endswith(".html"):
                    full_url = urljoin(self.base_url + "/", href)
                    html_pages.add(full_url)

        print(f"  ✅ Found {len(html_pages)} HTML pages from API")
        return html_pages

    def extract_pdf_links_from_api(self, html_pages):
        """Extract PDF links from API HTML pages"""
        print("🔎 [API] Extracting PDF links...")
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

        print(f"  ✅ Found {len(pdf_links)} PDF links from API")
        return pdf_links

    # ------------------------------------------
    # PART 2: Local HTML Sync
    # ------------------------------------------

    def discover_local_html_files(self):
        """Find all HTML files in local directory"""
        if not self.local_html_dir or not self.local_html_dir.exists():
            return []

        print(f"🔍 [LOCAL] Scanning HTML files in: {self.local_html_dir}")
        
        html_files = list(self.local_html_dir.rglob("*.html")) + list(self.local_html_dir.rglob("*.htm"))
        print(f"  ✅ Found {len(html_files)} local HTML files")
        return html_files

    def extract_pdf_links_from_local(self, html_files):
        """Extract PDF links from local HTML files"""
        print("🔎 [LOCAL] Extracting PDF links from local HTML...")
        pdf_links = set()

        for html_path in html_files:
            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    
                    if href.lower().endswith(".pdf"):
                        # Construct full URL based on href format
                        if href.startswith("../"):
                            full_url = f"https://upaygoa.com/geltm/{href.replace('../', '')}"
                        elif href.startswith("/"):
                            full_url = f"https://upaygoa.com{href}"
                        elif href.startswith("http"):
                            full_url = href
                        else:
                            full_url = f"{self.base_url}/{href}"
                        
                        pdf_links.add(full_url)
            except Exception as e:
                print(f"  ⚠ Error reading {html_path.name}: {e}")

        print(f"  ✅ Found {len(pdf_links)} PDF links from local HTML")
        return pdf_links

    def discover_local_images(self):
        """Find all images in local directory"""
        if not self.local_html_dir or not self.local_html_dir.exists():
            return []

        print(f"🔍 [LOCAL] Scanning images in: {self.local_html_dir}")
        
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]:
            image_files.extend(self.local_html_dir.rglob(ext))
        
        print(f"  ✅ Found {len(image_files)} local images")
        return image_files

    # ------------------------------------------
    # PART 3: Download with Duplicate Detection
    # ------------------------------------------

    def download_and_track_pdf(self, pdf_url):
        """Download PDF with duplicate detection"""
        # Get filename
        filename = os.path.basename(urlparse(pdf_url).path.split("?")[0])
        
        # Check if already processed in this session
        if filename in self.processed_pdfs:
            return "duplicate", None
        
        self.processed_pdfs.add(filename)
        
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code != 200:
                return None, None

            content = response.content
            content_hash = self._compute_hash(content)
            filepath = self.download_dir / filename

            is_new = pdf_url not in self.sync_log["documents"]

            # Check if unchanged
            if not is_new:
                if self.sync_log["documents"][pdf_url]["hash"] == content_hash:
                    return "unchanged", None

            # Save file
            with open(filepath, "wb") as f:
                f.write(content)

            # Update log
            self.sync_log["documents"][pdf_url] = {
                "hash": content_hash,
                "path": str(filepath),
                "last_updated": datetime.now().isoformat(),
            }

            return "new" if is_new else "updated", filepath

        except Exception as e:
            print(f"  ⚠ Error downloading {filename}: {e}")
            return None, None

    # ------------------------------------------
    # PART 4: Image Processing with OCR
    # ------------------------------------------

    def process_image(self, image_path):
        """Process image: Copy to images folder + Extract text (OCR)"""
        try:
            # Copy image to images folder
            dest_path = self.download_dir / "images" / image_path.name
            
            # Check if already processed
            image_hash = self._compute_hash(image_path.read_bytes())
            
            if str(image_path) in self.sync_log.get("images", {}):
                if self.sync_log["images"][str(image_path)]["hash"] == image_hash:
                    return "unchanged", None
            
            # Copy image
            import shutil
            shutil.copy2(image_path, dest_path)
            
            # Try to extract text using OCR (optional - requires pytesseract)
            extracted_text = ""
            try:
                from PIL import Image
                import pytesseract
                
                img = Image.open(image_path)
                extracted_text = pytesseract.image_to_string(img)
                
                # Save extracted text alongside image
                text_file = dest_path.with_suffix('.txt')
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"Image: {image_path.name}\n")
                    f.write(f"Extracted at: {datetime.now().isoformat()}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(extracted_text)
                
                print(f"    → Extracted {len(extracted_text)} chars of text")
                
            except ImportError:
                print(f"    ⚠ pytesseract not installed - skipping OCR")
            except Exception as e:
                print(f"    ⚠ OCR failed: {e}")
            
            # Update log
            if "images" not in self.sync_log:
                self.sync_log["images"] = {}
            
            self.sync_log["images"][str(image_path)] = {
                "hash": image_hash,
                "path": str(dest_path),
                "text_extracted": len(extracted_text) > 0,
                "last_updated": datetime.now().isoformat(),
            }
            
            return "new", dest_path
            
        except Exception as e:
            print(f"  ⚠ Error processing {image_path.name}: {e}")
            return None, None

    # ------------------------------------------
    # MAIN SYNC
    # ------------------------------------------

    def sync(self):
        """Main hybrid sync operation"""
        print("\n" + "=" * 60)
        print(f"SOP HYBRID SYNC - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")

        # Counters
        pdf_new = 0
        pdf_updated = 0
        pdf_unchanged = 0
        pdf_duplicates = 0
        img_new = 0
        img_unchanged = 0

        # ========== PART 1: API Sync ==========
        print("📡 PHASE 1: API Sync")
        print("-" * 60)
        
        api_html_pages = self.discover_html_pages_from_toc()
        api_pdf_links = self.extract_pdf_links_from_api(api_html_pages)

        if api_pdf_links:
            print(f"\n📥 Downloading PDFs from API...\n")
            for i, pdf_url in enumerate(api_pdf_links, 1):
                filename = os.path.basename(urlparse(pdf_url).path)
                print(f"  [{i}/{len(api_pdf_links)}] {filename[:50]}", end=" ... ")
                
                status, filepath = self.download_and_track_pdf(pdf_url)
                
                if status == "new":
                    print("✅ NEW")
                    pdf_new += 1
                elif status == "updated":
                    print("🔄 UPDATED")
                    pdf_updated += 1
                elif status == "unchanged":
                    print("⏭️  SKIP")
                    pdf_unchanged += 1
                elif status == "duplicate":
                    print("🔁 DUP")
                    pdf_duplicates += 1
                else:
                    print("❌ FAIL")

        # ========== PART 2: Local HTML Sync ==========
        if self.local_html_dir:
            print(f"\n📁 PHASE 2: Local HTML Sync")
            print("-" * 60)
            
            local_html_files = self.discover_local_html_files()
            local_pdf_links = self.extract_pdf_links_from_local(local_html_files)

            if local_pdf_links:
                print(f"\n📥 Downloading PDFs from local HTML...\n")
                for i, pdf_url in enumerate(local_pdf_links, 1):
                    filename = os.path.basename(urlparse(pdf_url).path)
                    print(f"  [{i}/{len(local_pdf_links)}] {filename[:50]}", end=" ... ")
                    
                    status, filepath = self.download_and_track_pdf(pdf_url)
                    
                    if status == "new":
                        print("✅ NEW")
                        pdf_new += 1
                    elif status == "updated":
                        print("🔄 UPDATED")
                        pdf_updated += 1
                    elif status == "unchanged":
                        print("⏭️  SKIP")
                        pdf_unchanged += 1
                    elif status == "duplicate":
                        print("🔁 DUP")
                        pdf_duplicates += 1
                    else:
                        print("❌ FAIL")

            # ========== PART 3: Image Processing ==========
            print(f"\n🖼️  PHASE 3: Image Processing")
            print("-" * 60)
            
            image_files = self.discover_local_images()
            
            if image_files:
                print(f"\n📸 Processing images...\n")
                for i, img_path in enumerate(image_files, 1):
                    print(f"  [{i}/{len(image_files)}] {img_path.name[:50]}", end=" ... ")
                    
                    status, dest = self.process_image(img_path)
                    
                    if status == "new":
                        print("✅ NEW")
                        img_new += 1
                    elif status == "unchanged":
                        print("⏭️  SKIP")
                        img_unchanged += 1
                    else:
                        print("❌ FAIL")

        # Save log
        self.sync_log["last_sync"] = datetime.now().isoformat()
        self._save_sync_log()

        # Print summary
        print("\n" + "=" * 60)
        print("SYNC SUMMARY")
        print("=" * 60)
        print("\n📄 PDFs:")
        print(f"  ✅ New:        {pdf_new}")
        print(f"  🔄 Updated:    {pdf_updated}")
        print(f"  ⏭️  Unchanged:  {pdf_unchanged}")
        print(f"  🔁 Duplicates: {pdf_duplicates}")
        print(f"\n🖼️  Images:")
        print(f"  ✅ New:        {img_new}")
        print(f"  ⏭️  Unchanged:  {img_unchanged}")
        print("=" * 60 + "\n")

        return pdf_new, pdf_updated


# ------------------------------------------
# Helper function for rebuilding vector store
# ------------------------------------------

def rebuild_vectorstore():
    """Rebuild vector store including PDFs and image text"""
    from rag.loader import load_pdfs
    from rag.splitter import split_docs
    from rag.vectorstore import create_vectorstore

    print("🔄 Rebuilding vector store...")

    # Load all PDFs
    docs = load_pdfs()

    # Split into chunks
    chunks = split_docs(docs)

    # Recreate vector store
    vectorstore = create_vectorstore(chunks)

    print("✅ Vector store rebuilt successfully!")

    return vectorstore


# ------------------------------------------
# Run standalone
# ------------------------------------------

if __name__ == "__main__":
    BASE_URL = "https://upaygoa.com/geltm/helpndoc"
    DOWNLOAD_DIR = "./sop_documents"
    
    # Point to your Code HelpN Doc folder
    # Adjust this path based on where Code HelpN Doc is relative to this script
    LOCAL_HTML_DIR = "../Code HelpN Doc"  # One folder up, then into Code HelpN Doc

    syncer = SOPAutoSyncHybrid(
        base_url=BASE_URL,
        download_dir=DOWNLOAD_DIR,
        local_html_dir=LOCAL_HTML_DIR
    )
    
    new_count, updated_count = syncer.sync()

    # Rebuild vector store if there were changes
    if new_count > 0 or updated_count > 0:
        print("\n📦 Changes detected - rebuilding vector store...")
        rebuild_vectorstore()
    else:
        print("\n✅ No changes - vector store is up to date")
