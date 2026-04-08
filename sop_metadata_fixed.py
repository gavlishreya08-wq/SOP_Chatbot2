from collections import Counter
from html import escape
from pathlib import Path
from urllib.parse import urlparse


class SOPMetadata:

    @staticmethod
    def _is_valid_external_link(link: str) -> bool:
        if not link or not isinstance(link, str):
            return False
        parsed = urlparse(link.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def format_sources_html(self, docs):

        if not docs:
            return ""

        # ⭐ count most relevant SOP
        sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")]
        if not sources:
            return ""

        main_source = Counter(sources).most_common(1)[0][0]

        for d in docs:
            if d.metadata.get("source") == main_source:

                link = (d.metadata.get("link") or d.metadata.get("pdf_link") or "").strip()
                version = d.metadata.get("version", "NA")
                created = d.metadata.get("created_date", "NA")

                title = escape(Path(main_source).stem.replace("_", " ").replace("-", " "))
                version = escape(str(version))
                created = escape(str(created))
                valid_link = self._is_valid_external_link(link)
                safe_link = escape(link, quote=True) if valid_link else ""
                link_html = (
                    f"🔗 <a href='{safe_link}' target='_blank' rel='noopener noreferrer' "
                    "style='color:#60a5fa;text-decoration:none;'>Open SOP</a><br>"
                    if valid_link
                    else "🔗 <span style='color:#94a3b8;'>Link unavailable</span><br>"
                )

                html = (
                    "<div style='margin-top:20px'>"
                    "<div style='background:#0f172a;"
                    "padding:10px;"
                    "border-radius:8px;"
                    "margin-top:8px;"
                    "border-left:3px solid #3b82f6;"
                    "font-size:0.85em;"
                    "color:#cbd5e1'>"

                    f"📄 <b>{title}</b><br>"
                    f"{link_html}"
                    f"🧾 Version: {version}<br>"
                    f"📅 Created: {created}"

                    "</div></div>"
                )

                return html

        return ""


# ⭐ VERY IMPORTANT SINGLETON FUNCTION
_metadata_instance = None

def get_metadata_handler():
    global _metadata_instance
    if _metadata_instance is None:
        _metadata_instance = SOPMetadata()
    return _metadata_instance
