"""
Auto-sync module for SOP Chatbot
Monitors https://upaygoa.com/geltm/helpndoc/ for new/updated SOPs
and automatically updates the vector store
"""

import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import hashlib
from pathlib import Path
import time
from typing import List, Dict, Set
import schedule

class SOPAutoSync:
    def __init__(self, base_url: str, download_dir: str, sync_log_file: str = "sync_log.json"):
        """
        Initialize the auto-sync system
        
        Args:
            base_url: Base URL of the SOP website (e.g., https://upaygoa.com/geltm/helpndoc/)
            download_dir: Local directory to store downloaded SOPs
            sync_log_file: JSON file to track synced documents
        """
        self.base_url = base_url.rstrip('/')
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.sync_log_file = sync_log_file
        self.sync_log = self._load_sync_log()
        
    def _load_sync_log(self) -> Dict:
        """Load the sync log from disk"""
        if os.path.exists(self.sync_log_file):
            with open(self.sync_log_file, 'r') as f:
                return json.load(f)
        return {
            "last_sync": None,
            "documents": {}  # {url: {hash, path, last_updated}}
        }
    
    def _save_sync_log(self):
        """Save the sync log to disk"""
        with open(self.sync_log_file, 'w') as f:
            json.dump(self.sync_log, indent=2, fp=f)
    
    def _compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def discover_sop_pages(self) -> List[str]:
        """
        Crawl the website to discover all SOP document URLs
        Returns list of full URLs to SOP pages
        """
        discovered_urls = set()
        pages_to_visit = [f"{self.base_url}/introduction.html"]
        visited = set()
        
        print("🔍 Discovering SOP pages...")
        
        while pages_to_visit:
            url = pages_to_visit.pop(0)
            
            if url in visited:
                continue
            
            visited.add(url)
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Add this page to discovered URLs
                discovered_urls.add(url)
                
                # Find all links on this page
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Skip external links, anchors, javascript, etc.
                    if (href.startswith('#') or 
                        href.startswith('javascript:') or 
                        href.startswith('mailto:') or
                        'http://' in href and 'upaygoa.com' not in href or
                        'https://' in href and 'upaygoa.com' not in href):
                        continue
                    
                    # Convert relative URLs to absolute
                    if not href.startswith('http'):
                        href = href.lstrip('/')
                        full_url = f"{self.base_url}/{href}"
                    else:
                        full_url = href
                    if (full_url.startswith(self.base_url) and 
                        (full_url.endswith('.html') or full_url.endswith('.htm')) and
                        full_url not in visited):
                        pages_to_visit.append(full_url)
                
                # Progress indicator
                if len(discovered_urls) % 10 == 0:
                    print(f"  ... Found {len(discovered_urls)} pages so far")
                    
            except Exception as e:
                print(f"  ⚠ Error crawling {url}: {str(e)}")
                continue
        
        print(f"✓ Discovered {len(discovered_urls)} SOP pages")
        return list(discovered_urls)
    
    def fetch_page_content(self, url: str) -> tuple[str, str]:
        """
        Fetch content from a URL
        Returns (title, content_text)
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Untitled"
            
            # Check for PDF/Word file links on the page
            pdf_link = None
            word_link = None
            
            # Look for "Click Here for PDF Files" or similar
            for link in soup.find_all('a', href=True):
                link_text = link.get_text().lower()
                href = link['href']
                
                if 'pdf' in link_text or href.endswith('.pdf'):
                    if not href.startswith('http'):
                        pdf_link = f"{self.base_url}/{href.lstrip('/')}"
                    else:
                        pdf_link = href
                
                if 'word' in link_text or href.endswith('.doc') or href.endswith('.docx'):
                    if not href.startswith('http'):
                        word_link = f"{self.base_url}/{href.lstrip('/')}"
                    else:
                        word_link = href
            
            # If PDF found, try to extract text from it
            if pdf_link:
                try:
                    print(f"    → Found PDF: {pdf_link}")
                    pdf_response = requests.get(pdf_link, timeout=15)
                    pdf_response.raise_for_status()
                    
                    # Save PDF temporarily and extract text
                    import io
                    try:
                        from PyPDF2 import PdfReader
                        pdf_file = io.BytesIO(pdf_response.content)
                        pdf_reader = PdfReader(pdf_file)
                        pdf_text = []
                        for page in pdf_reader.pages:
                            pdf_text.append(page.extract_text())
                        content_text = '\n\n'.join(pdf_text)
                        if content_text.strip():
                            return title_text, content_text
                    except ImportError:
                        print("    ⚠ PyPDF2 not installed, skipping PDF extraction")
                    except Exception as e:
                        print(f"    ⚠ Could not extract PDF text: {str(e)}")
                except Exception as e:
                    print(f"    ⚠ Could not fetch PDF: {str(e)}")
            
            # Fallback: Extract HTML content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content area
            content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_='content') or
                soup.find('div', id='content') or
                soup.body
            )
            
            if content:
                content_text = content.get_text(separator='\n', strip=True)
            else:
                content_text = soup.get_text(separator='\n', strip=True)
            
            # Clean up content
            lines = [line.strip() for line in content_text.split('\n') if line.strip()]
            content_text = '\n'.join(lines)
            
            return title_text, content_text
            
        except Exception as e:
            print(f"✗ Error fetching {url}: {str(e)}")
            return "", ""
    
    def check_for_updates(self) -> Dict[str, str]:
        """
        Check all discovered SOPs for new or updated content
        Returns dict of {url: status} where status is 'new', 'updated', or 'unchanged'
        """
        updates = {}
        discovered_urls = self.discover_sop_pages()
        
        for url in discovered_urls:
            title, content = self.fetch_page_content(url)
            
            if not content:
                continue
            
            content_hash = self._compute_hash(content)
            
            # Check if this is a new document
            if url not in self.sync_log["documents"]:
                updates[url] = "new"
                print(f"  → NEW: {title}")
            
            # Check if content has changed
            elif self.sync_log["documents"][url]["hash"] != content_hash:
                updates[url] = "updated"
                print(f"  → UPDATED: {title}")
            
            else:
                updates[url] = "unchanged"
        
        return updates
    
    def download_and_save(self, url: str) -> str:
        """
        Download SOP content and save as text file
        Returns path to saved file
        """
        title, content = self.fetch_page_content(url)
        
        if not content:
            return None
        
        # Create safe filename from title or URL
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
        safe_title = safe_title[:100]  # Limit length
        
        filename = f"{safe_title}.txt"
        filepath = self.download_dir / filename
        
        # Save content with metadata header
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Source: {url}\n")
            f.write(f"Title: {title}\n")
            f.write(f"Downloaded: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(content)
        
        print(f"  ✓ Saved to: {filepath}")
        
        # Update sync log
        content_hash = self._compute_hash(content)
        self.sync_log["documents"][url] = {
            "hash": content_hash,
            "path": str(filepath),
            "title": title,
            "last_updated": datetime.now().isoformat()
        }
        
        return str(filepath)
    
    def sync(self) -> Dict[str, List[str]]:
        """
        Main sync operation: check for updates and download new/changed SOPs
        Returns dict with 'new', 'updated', 'unchanged' lists of URLs
        """
        print(f"\n{'='*60}")
        print(f"Starting SOP sync at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target URL: {self.base_url}")
        print(f"Download to: {self.download_dir}")
        print(f"{'='*60}")
        
        # Test connection first
        try:
            test_url = f"{self.base_url}/introduction.html"
            print(f"\n🔗 Testing connection to: {test_url}")
            test_response = requests.get(test_url, timeout=10)
            test_response.raise_for_status()
            print(f"✓ Connection successful (Status: {test_response.status_code})")
        except Exception as e:
            print(f"✗ Connection failed: {str(e)}")
            print(f"  Please check:")
            print(f"  1. Is the URL correct? {self.base_url}")
            print(f"  2. Is the website accessible?")
            print(f"  3. Do you have internet connection?")
            return {"new": [], "updated": [], "unchanged": []}
        
        updates = self.check_for_updates()
        
        result = {
            "new": [],
            "updated": [],
            "unchanged": []
        }
        
        for url, status in updates.items():
            result[status].append(url)
            
            if status in ("new", "updated"):
                self.download_and_save(url)
        
        # Update last sync time
        self.sync_log["last_sync"] = datetime.now().isoformat()
        self._save_sync_log()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Sync Summary:")
        print(f"  New documents:       {len(result['new'])}")
        print(f"  Updated documents:   {len(result['updated'])}")
        print(f"  Unchanged documents: {len(result['unchanged'])}")
        print(f"{'='*60}\n")
        return result


def rebuild_vectorstore_after_sync(sop_directory: str):
    """
    Rebuild the vector store after new SOPs are downloaded
    This should be called after sync() completes
    """
    from rag.loader import load_pdfs
    from rag.splitter import split_docs
    from rag.vectorstore import create_vectorstore
    
    print("Rebuilding vector store with new documents...")
    
    # Load all documents (including newly downloaded ones)
    docs = load_pdfs(sop_directory)
    
    # Split into chunks
    chunks = split_docs(docs)
    
    # Recreate vector store
    vectorstore = create_vectorstore(chunks)
    
    print("✓ Vector store rebuilt successfully!")
    
    return vectorstore


# ============================================================================
# Scheduled Auto-Sync
# ============================================================================

def schedule_auto_sync(base_url: str, download_dir: str, interval_hours: int = 24):
    """
    Set up automatic syncing on a schedule
    
    Args:
        base_url: SOP website URL
        download_dir: Where to save downloaded SOPs
        interval_hours: How often to check (default: 24 hours)
    """
    syncer = SOPAutoSync(base_url, download_dir)
    
    def sync_job():
        result = syncer.sync()
        
        # Rebuild vector store if there were any updates
        if result["new"] or result["updated"]:
            print("Changes detected - rebuilding vector store...")
            rebuild_vectorstore_after_sync(download_dir)
        else:
            print("No changes detected - skipping vector store rebuild")
    
    # Run immediately on start
    print(f"Running initial sync...")
    sync_job()
    
    # Schedule periodic syncs
    schedule.every(interval_hours).hours.do(sync_job)
    
    print(f"\n✓ Auto-sync scheduled to run every {interval_hours} hours")
    print("Press Ctrl+C to stop\n")
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


# ============================================================================
# Manual Sync (run this script directly)
# ============================================================================

if __name__ == "__main__":
    # Configuration
    BASE_URL = "https://upaygoa.com/geltm/helpndoc"
    DOWNLOAD_DIR = "./sop_documents"
    
    # Create syncer
    syncer = SOPAutoSync(BASE_URL, DOWNLOAD_DIR)
    
    # Run sync
    result = syncer.sync()
    
    # Optionally rebuild vector store
    if result["new"] or result["updated"]:
        print("\nRebuilding vector store...")
        rebuild_vectorstore_after_sync(DOWNLOAD_DIR)