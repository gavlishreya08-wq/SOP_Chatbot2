import unittest

from backend.config import settings
from backend.rag.retriever import build_source_catalog, retrieve
from backend.rag.vectorstore import load_existing_vectorstore


class RetrievalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectorstore = load_existing_vectorstore()
        if cls.vectorstore is None:
            raise unittest.SkipTest(f"No vectorstore found in {settings.chroma_db_dir}")
        cls.source_catalog = build_source_catalog(cls.vectorstore)

    def test_jira_issue_query_routes_to_issue_creation_annexure(self):
        docs, source = retrieve(
            self.vectorstore,
            "How to create a Jira issue?",
            source_catalog=self.source_catalog,
        )

        self.assertEqual(source, "GELJira_IssueCreation(Annexure2).pdf")
        self.assertTrue(docs)

    def test_change_management_query_routes_to_workflow(self):
        docs, source = retrieve(
            self.vectorstore,
            "Explain the change management workflow",
            source_catalog=self.source_catalog,
        )

        self.assertEqual(source, "ChangeManagementWorkflow_version2.pdf")
        self.assertTrue(docs)

    def test_unknown_policy_query_returns_no_documents(self):
        docs, source = retrieve(
            self.vectorstore,
            "Leave policy overview",
            source_catalog=self.source_catalog,
        )

        self.assertEqual(docs, [])
        self.assertIsNone(source)

    def test_test_lead_roles_query_routes_to_test_lead_sop(self):
        docs, source = retrieve(
            self.vectorstore,
            "Tell me Roles and responsibility of Test Lead",
            source_catalog=self.source_catalog,
        )

        self.assertEqual(source, "RR_TestLead_V1.pdf")
        self.assertTrue(docs)

    def test_technical_lead_roles_query_routes_to_technical_lead_sop(self):
        docs, source = retrieve(
            self.vectorstore,
            "Tell me the roles and responsibilities of Technical Lead",
            source_catalog=self.source_catalog,
        )

        self.assertEqual(source, "3_RR_TechnicalLead_version2.pdf")
        self.assertTrue(docs)

    def test_explicit_role_switch_overrides_active_sop(self):
        docs, source = retrieve(
            self.vectorstore,
            "roles and responsibility of technical lead",
            active_sop="RR_TestLead_V1.pdf",
            source_catalog=self.source_catalog,
        )

        self.assertEqual(source, "3_RR_TechnicalLead_version2.pdf")
        self.assertTrue(docs)


if __name__ == "__main__":
    unittest.main()
