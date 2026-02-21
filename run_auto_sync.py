from sop_auto_sync import schedule_auto_sync
import sys

BASE_URL = "https://upaygoa.com/geltm/helpndoc"
DOWNLOAD_DIR = "./sop_documents"
SYNC_INTERVAL = 24

if __name__ == "__main__":
    print("=" * 60)
    print("    SOP CHATBOT - AUTO-SYNC SERVICE")
    print("=" * 60)

    try:
        schedule_auto_sync(
            base_url=BASE_URL,
            download_dir=DOWNLOAD_DIR,
            interval_hours=SYNC_INTERVAL,
        )
    except KeyboardInterrupt:
        print("\n✓ Auto-sync stopped")
        sys.exit(0)
