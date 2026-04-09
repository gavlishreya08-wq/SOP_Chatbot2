import unittest

from backend.rag.preprocess import build_source_profile, clean_lines, extract_sections
from backend.rag.splitter import split_docs
from tests.helpers import make_doc


class IngestionPipelineTests(unittest.TestCase):
    def test_build_source_profile_extracts_title_aliases_sections_and_intents(self):
        text = """
        TEST-LEAD
        JOB OBJECTIVES
        To ensure an error free product.
        RESPONSIBILITIES
        Lead the testing team.
        """

        profile = build_source_profile("RR_TestLead_V1.pdf", text)

        self.assertEqual(profile["source_title"], "TEST-LEAD")
        self.assertIn("Test Lead", profile["source_aliases"])
        self.assertIn("roles and responsibilities of Test Lead", profile["source_intents"])
        self.assertIn("Job Objectives", profile["section_titles"])
        self.assertIn("Responsibilities", profile["section_titles"])

    def test_extract_sections_splits_heading_blocks(self):
        text = """
        OBJECTIVE
        Ensure high quality delivery.
        RESPONSIBILITIES
        Lead the team.
        Review test plans.
        """

        sections = extract_sections(text)

        self.assertEqual(sections[0][0], "Objective")
        self.assertIn("Ensure high quality delivery.", sections[0][1])
        self.assertEqual(sections[1][0], "Responsibilities")
        self.assertIn("Lead the team.", sections[1][1])

    def test_split_docs_creates_profile_focus_and_section_chunks(self):
        docs = [
            make_doc(
                "TEST-LEAD\nJOB OBJECTIVES\nEnsure quality delivery.\nRESPONSIBILITIES\nLead the testing team.",
                "RR_TestLead_V1.pdf",
                page=0,
                page_label="1",
                source_title="TEST-LEAD",
                source_kind="role",
                source_aliases="Test Lead | RR Test Lead",
                source_intents="roles and responsibilities of Test Lead | Test Lead role",
                source_section_titles="Job Objectives | Responsibilities",
                source_summary="Job Objectives: Ensure quality delivery.",
            )
        ]

        chunks = split_docs(docs)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].metadata["content_type"], "profile")
        self.assertIn("DOCUMENT PROFILE", chunks[0].page_content)
        self.assertTrue(any(chunk.metadata["content_type"] == "focus" for chunk in chunks))
        section_chunks = [chunk for chunk in chunks if chunk.metadata["content_type"] == "section"]
        self.assertTrue(section_chunks)
        self.assertTrue(any("SECTION: Job Objectives" in chunk.page_content for chunk in section_chunks))

    def test_split_docs_preserves_cross_page_section_continuity(self):
        docs = [
            make_doc(
                "TECHNICAL LEAD\nRESPONSIBILITIES\n1. Lead the team.\n2. Coordinate delivery.",
                "3_RR_TechnicalLead_version2.pdf",
                page=0,
                page_label="1",
                source_title="TECHNICAL LEAD",
                source_kind="role",
                source_aliases="Technical Lead",
                source_intents="roles and responsibilities of Technical Lead | Technical Lead role",
                source_section_titles="Responsibilities",
                source_summary="Responsibilities: Lead the team.",
            ),
            make_doc(
                "TECHNICAL LEAD\n3. Review progress weekly.\n4. Ensure quality is maintained.",
                "3_RR_TechnicalLead_version2.pdf",
                page=1,
                page_label="2",
                source_title="TECHNICAL LEAD",
                source_kind="role",
                source_aliases="Technical Lead",
                source_intents="roles and responsibilities of Technical Lead | Technical Lead role",
                source_section_titles="Responsibilities",
                source_summary="Responsibilities: Lead the team.",
            ),
        ]

        chunks = split_docs(docs)

        responsibility_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata["content_type"] == "section"
            and chunk.metadata["section_title"] == "Responsibilities"
        ]

        self.assertTrue(responsibility_chunks)
        self.assertTrue(any(chunk.metadata["page_label"] == "1-2" for chunk in responsibility_chunks))
        combined_text = "\n".join(chunk.page_content for chunk in responsibility_chunks)
        self.assertIn("3. Review progress weekly.", combined_text)


if __name__ == "__main__":
    unittest.main()
