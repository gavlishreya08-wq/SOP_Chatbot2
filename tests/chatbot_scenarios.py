from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioGroup:
    name: str
    category: str
    expected_source: str | None
    variants: tuple[str, ...]
    active_sop: str | None = None


POSITIVE_SCENARIOS: tuple[ScenarioGroup, ...] = (
    ScenarioGroup(
        name="test lead",
        category="roles",
        expected_source="RR_TestLead_V1.pdf",
        variants=(
            "Tell me Roles and responsibility of Test Lead",
            "What are the job objectives of the Test Lead?",
            "Explain the Test Lead role and responsibilities",
        ),
    ),
    ScenarioGroup(
        name="test engineer",
        category="roles",
        expected_source="RR_TestEngineer_V1.pdf",
        variants=(
            "What are the roles and responsibilities of Test Engineer?",
            "Explain the Test Engineer role",
        ),
    ),
    ScenarioGroup(
        name="technical lead",
        category="roles",
        expected_source="3_RR_TechnicalLead_version2.pdf",
        variants=(
            "Tell me the roles and responsibilities of Technical Lead",
            "What are the responsibilities of Technical Lead?",
        ),
    ),
    ScenarioGroup(
        name="development engineer",
        category="roles",
        expected_source="4_RR_DevelopmentEng_version2.pdf",
        variants=(
            "Tell me the responsibilities of Development Engineer",
            "Explain the Development Engineer SOP role",
        ),
    ),
    ScenarioGroup(
        name="technical resource manager",
        category="roles",
        expected_source="2.RR_TechnicalResourceManager_version1.pdf",
        variants=(
            "What are the roles of Technical Resource Process and Training Manager?",
            "Explain Technical Resource Process and Training Manager responsibilities",
        ),
    ),
    ScenarioGroup(
        name="dba",
        category="roles",
        expected_source="RR DBA.pdf",
        variants=(
            "Tell me the Database Administrator DBA role and responsibilities",
            "What is the DBA role in the SOP?",
        ),
    ),
    ScenarioGroup(
        name="change workflow",
        category="workflow",
        expected_source="ChangeManagementWorkflow_version2.pdf",
        variants=(
            "Explain the change management workflow",
            "What is the change request workflow?",
            "Show me the change management process",
        ),
    ),
    ScenarioGroup(
        name="jira issue creation",
        category="jira",
        expected_source="GELJira_IssueCreation(Annexure2).pdf",
        variants=(
            "How to create a Jira issue?",
            "Explain GEL Jira issue creation",
            "What issue types are available while creating a Jira ticket?",
        ),
    ),
    ScenarioGroup(
        name="gitlab scm",
        category="workflow",
        expected_source="SOP Source Code Management Gitlab.pdf",
        variants=(
            "Explain source code management process in GitLab",
            "What is the SOP for source code management in GitLab?",
        ),
    ),
    ScenarioGroup(
        name="application deployment",
        category="workflow",
        expected_source="SOP_APPLICATION DEPLOYMENT _ RELEASE.pdf",
        variants=(
            "Explain the application deployment and release process",
            "What is the SOP for application deployment release?",
        ),
    ),
    ScenarioGroup(
        name="backend data update",
        category="workflow",
        expected_source="SOP_Backend_Data_Update_Request_20260224175914.pdf",
        variants=(
            "How to manage backend data update request?",
            "Explain the backend data update request process",
        ),
    ),
    ScenarioGroup(
        name="production issue",
        category="workflow",
        expected_source="SOP_Production Issue.pdf",
        variants=(
            "How to manage production issue?",
            "Explain the production issue SOP",
        ),
    ),
    ScenarioGroup(
        name="root cause fixture",
        category="workflow",
        expected_source="SOP_Root Cause Fixture.pdf",
        variants=(
            "Explain the production issue root cause fixture process",
            "What is the SOP for root cause fixture?",
        ),
    ),
    ScenarioGroup(
        name="standup meeting",
        category="meeting",
        expected_source="SOP_Standup_meeting.pdf",
        variants=(
            "What is the SOP for stand up meeting?",
            "Explain the standup meeting procedure",
        ),
    ),
    ScenarioGroup(
        name="issue resolution meeting",
        category="meeting",
        expected_source="SOP_Issue_resolution_meeting.pdf",
        variants=(
            "Explain issue resolution meeting SOP",
            "What is the issue resolution meeting procedure?",
        ),
    ),
    ScenarioGroup(
        name="detailed work review",
        category="meeting",
        expected_source="SOP_Detailed_Work_Review.pdf",
        variants=(
            "Explain detailed work review meeting SOP",
            "What is the detailed work review process?",
        ),
    ),
    ScenarioGroup(
        name="tl review meet",
        category="meeting",
        expected_source="SOP TL Review Meet.pdf",
        variants=(
            "Explain TL review meet SOP",
            "What is the TL review meet procedure?",
        ),
    ),
    ScenarioGroup(
        name="server whitelisting",
        category="workflow",
        expected_source="SOP_ServerWhitelistingg.pdf",
        variants=(
            "Explain the server whitelisting SOP",
            "What is the process for server whitelisting?",
        ),
    ),
    ScenarioGroup(
        name="test management",
        category="workflow",
        expected_source="SOP_Test Management _V1.0.pdf",
        variants=(
            "Explain the test management SOP",
            "What is the test management process?",
        ),
    ),
    ScenarioGroup(
        name="test automation",
        category="workflow",
        expected_source="SOP_TestAutomation_V1.0.pdf",
        variants=(
            "Explain the test automation SOP",
            "What is the test automation process?",
        ),
    ),
    ScenarioGroup(
        name="database design",
        category="workflow",
        expected_source="SOP_DATABASE DESIGN AND MODELING.pdf",
        variants=(
            "Explain database design and modeling SOP",
            "What is the process for database design and modeling?",
        ),
    ),
    ScenarioGroup(
        name="database migration",
        category="workflow",
        expected_source="SOP_Database Migration and Porting.pdf",
        variants=(
            "Explain database migration and porting procedure",
            "What is the SOP for database migration and porting?",
        ),
    ),
    ScenarioGroup(
        name="system architecture",
        category="workflow",
        expected_source="SOP How to design system architecture.pdf",
        variants=(
            "How to design system architecture?",
            "Explain the system architecture design SOP",
        ),
    ),
    ScenarioGroup(
        name="react standards",
        category="standards",
        expected_source="Coding React Standards.pdf",
        variants=(
            "What are the React coding standards SOP?",
            "Explain coding standards for React",
        ),
    ),
    ScenarioGroup(
        name="dotnet standards",
        category="standards",
        expected_source="Final_v2_Standards_Coding_DotNet.pdf",
        variants=(
            "What are the .NET coding standards SOP?",
            "Explain the .NET coding standards document",
        ),
    ),
    ScenarioGroup(
        name="website seo standards",
        category="standards",
        expected_source="Website-SEO-Standards.pdf",
        variants=(
            "What are the website SEO standards?",
            "Explain website SEO standards SOP",
        ),
    ),
)


NEGATIVE_SCENARIOS: tuple[ScenarioGroup, ...] = (
    ScenarioGroup(
        name="leave policy",
        category="negative",
        expected_source=None,
        variants=(
            "Leave policy overview",
            "What is the leave policy?",
        ),
    ),
    ScenarioGroup(
        name="dress code",
        category="negative",
        expected_source=None,
        variants=(
            "What is the dress code?",
            "Explain the office dress code policy",
        ),
    ),
    ScenarioGroup(
        name="holiday list",
        category="negative",
        expected_source=None,
        variants=(
            "Holiday calendar for this year",
            "What are the company holidays?",
        ),
    ),
    ScenarioGroup(
        name="salary policy",
        category="negative",
        expected_source=None,
        variants=(
            "Travel reimbursement policy",
            "Maternity leave eligibility policy",
        ),
    ),
    ScenarioGroup(
        name="work from home",
        category="negative",
        expected_source=None,
        variants=(
            "Remote work laptop allowance policy",
            "Office cab pickup policy",
        ),
    ),
)


ACTIVE_SOP_SCENARIOS: tuple[ScenarioGroup, ...] = (
    ScenarioGroup(
        name="test lead followup",
        category="context",
        expected_source="RR_TestLead_V1.pdf",
        active_sop="RR_TestLead_V1.pdf",
        variants=(
            "What are the job objectives?",
            "List the responsibilities",
        ),
    ),
    ScenarioGroup(
        name="explicit source switch from test lead",
        category="context",
        expected_source="3_RR_TechnicalLead_version2.pdf",
        active_sop="RR_TestLead_V1.pdf",
        variants=(
            "roles and responsibility of technical lead",
            "What are the responsibilities of Technical Lead?",
        ),
    ),
    ScenarioGroup(
        name="change workflow followup",
        category="context",
        expected_source="ChangeManagementWorkflow_version2.pdf",
        active_sop="ChangeManagementWorkflow_version2.pdf",
        variants=(
            "Who prepares the impact analysis?",
            "What happens after the PM reviews the request?",
        ),
    ),
    ScenarioGroup(
        name="jira followup",
        category="context",
        expected_source="GELJira_IssueCreation(Annexure2).pdf",
        active_sop="GELJira_IssueCreation(Annexure2).pdf",
        variants=(
            "What issue types are available?",
            "List the visible issue types",
        ),
    ),
    ScenarioGroup(
        name="gitlab followup",
        category="context",
        expected_source="SOP Source Code Management Gitlab.pdf",
        active_sop="SOP Source Code Management Gitlab.pdf",
        variants=(
            "What is the objective?",
            "Tell me the process objective",
        ),
    ),
    ScenarioGroup(
        name="production issue followup",
        category="context",
        expected_source="SOP_Production Issue.pdf",
        active_sop="SOP_Production Issue.pdf",
        variants=(
            "What is the workflow?",
            "Explain the process steps",
        ),
    ),
    ScenarioGroup(
        name="test automation followup",
        category="context",
        expected_source="SOP_TestAutomation_V1.0.pdf",
        active_sop="SOP_TestAutomation_V1.0.pdf",
        variants=(
            "What is the objective?",
            "Explain the scope",
        ),
    ),
)


CLARIFICATION_SCENARIOS: tuple[ScenarioGroup, ...] = (
    ScenarioGroup(
        name="generic process clarification",
        category="clarification",
        expected_source=None,
        variants=(
            "What is the process?",
            "Explain the procedure",
        ),
    ),
    ScenarioGroup(
        name="generic role clarification",
        category="clarification",
        expected_source=None,
        variants=(
            "What is the role?",
            "What are the responsibilities?",
        ),
    ),
)
