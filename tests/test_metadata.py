import unittest

from backend.core.metadata import format_sources
from tests.helpers import make_doc


class FormatSourcesTests(unittest.TestCase):
    def test_returns_none_for_empty_docs(self):
        self.assertIsNone(format_sources([]))

    def test_formats_title_and_sorts_pages(self):
        docs = [
            make_doc(
                "step a",
                "GELJira_IssueCreation(Annexure2).pdf",
                page=3,
                page_label="4",
                section_title="Procedure",
                pdf_link="https://example.com/sop",
                version="v1.0",
                created_date="10 Jan 2024",
            ),
            make_doc(
                "step b",
                "GELJira_IssueCreation(Annexure2).pdf",
                page=1,
                page_label="2",
                section_title="Participants",
                pdf_link="https://example.com/sop",
                version="v1.0",
                created_date="10 Jan 2024",
            ),
            make_doc(
                "step c",
                "GELJira_IssueCreation(Annexure2).pdf",
                page=0,
                page_label="1",
                section_title="Overview",
                pdf_link="https://example.com/sop",
                version="v1.0",
                created_date="10 Jan 2024",
            ),
        ]

        source = format_sources(docs)

        self.assertIsNotNone(source)
        self.assertEqual(source["title"], "GELJira IssueCreation Annexure2")
        self.assertEqual(source["filename"], "GELJira_IssueCreation(Annexure2).pdf")
        self.assertEqual(source["pages"], ["1", "2", "4"])
        self.assertEqual(source["link"], "https://example.com/sop")
        self.assertEqual(source["version"], "v1.0")
        self.assertEqual(
            source["citations"],
            [
                {"page": "1", "section": "Overview"},
                {"page": "2", "section": "Participants"},
                {"page": "4", "section": "Procedure"},
            ],
        )

    def test_sorts_page_ranges_after_numeric_pages(self):
        docs = [
            make_doc("step a", "SOP_TestAutomation_V1.0.pdf", page_label="3-4"),
            make_doc("step b", "SOP_TestAutomation_V1.0.pdf", page_label="1"),
            make_doc("step c", "SOP_TestAutomation_V1.0.pdf", page_label="2"),
        ]

        source = format_sources(docs)

        self.assertEqual(source["pages"], ["1", "2", "3-4"])


if __name__ == "__main__":
    unittest.main()
