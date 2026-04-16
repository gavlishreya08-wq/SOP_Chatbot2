from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from backend.config import settings
from backend.core.rag_chain import RAGChain
from backend.rag.retriever import build_source_catalog, retrieve
from backend.rag.vectorstore import load_existing_vectorstore
from tests.chatbot_scenarios import (
    ACTIVE_SOP_SCENARIOS,
    CLARIFICATION_SCENARIOS,
    NEGATIVE_SCENARIOS,
    POSITIVE_SCENARIOS,
    ScenarioGroup,
)
from tests.helpers import FakeLLM


def default_log_dir() -> Path:
    return Path(settings.data_dir) / "eval_logs"


def evaluate_group(group: ScenarioGroup, vectorstore, source_catalog) -> list[dict]:
    rows = []
    for question in group.variants:
        docs, source = retrieve(
            vectorstore,
            question,
            active_sop=group.active_sop,
            source_catalog=source_catalog,
        )
        passed = source == group.expected_source and bool(docs) if group.expected_source else (not docs and source is None)
        rows.append(
            {
                "group": group.name,
                "category": group.category,
                "question": question,
                "expected": group.expected_source or "NO_MATCH",
                "actual": source or "NO_MATCH",
                "doc_count": len(docs),
                "passed": passed,
            }
        )
    return rows


def evaluate_clarification_group(group: ScenarioGroup, rag_chain: RAGChain) -> list[dict]:
    import asyncio

    rows = []
    for question in group.variants:
        result = asyncio.run(rag_chain.query(question, [], None))
        passed = result["answer"].startswith("Which") and bool(result.get("suggestions"))
        rows.append(
            {
                "group": group.name,
                "category": group.category,
                "question": question,
                "expected": "CLARIFICATION",
                "actual": "CLARIFICATION" if passed else result["answer"],
                "doc_count": 0,
                "passed": passed,
            }
        )
    return rows


def build_report(rows: list[dict]) -> str:
    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    failed_rows = [row for row in rows if not row["passed"]]
    by_category = Counter(row["category"] for row in rows)

    lines = [
        "# Chatbot Evaluation Report",
        "",
        f"- Total scenarios: {total}",
        f"- Passed: {passed}",
        f"- Failed: {total - passed}",
        "",
        "## Coverage",
        "",
    ]
    for category, count in sorted(by_category.items()):
        lines.append(f"- {category}: {count}")

    lines.extend(["", "## Failures", ""])
    if not failed_rows:
        lines.append("- None")
    else:
        for row in failed_rows:
            lines.append(
                f"- [{row['category']}] {row['group']} :: `{row['question']}`"
                f" -> expected `{row['expected']}`, got `{row['actual']}`"
            )

    lines.extend(["", "## Detailed Results", ""])
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"- {status} | [{row['category']}] `{row['question']}`"
            f" | expected `{row['expected']}` | actual `{row['actual']}` | docs {row['doc_count']}"
        )

    return "\n".join(lines) + "\n"


def write_log_files(rows: list[dict], report: str, log_dir: Path) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = log_dir / f"chatbot_eval_{timestamp}.md"
    json_path = log_dir / f"chatbot_eval_{timestamp}.json"
    latest_markdown_path = log_dir / "chatbot_eval_latest.md"
    latest_json_path = log_dir / "chatbot_eval_latest.json"

    summary = {
        "generated_at": datetime.now().isoformat(),
        "total": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "failed": sum(1 for row in rows if not row["passed"]),
        "rows": rows,
    }

    markdown_path.write_text(report, encoding="utf-8")
    json_text = json.dumps(summary, indent=2, ensure_ascii=True)
    json_path.write_text(json_text, encoding="utf-8")
    latest_markdown_path.write_text(report, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")

    return markdown_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SOP chatbot retrieval evaluation matrix.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the markdown report.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(default_log_dir()),
        help="Directory where evaluation log files are written.",
    )
    args = parser.parse_args()

    vectorstore = load_existing_vectorstore()
    if vectorstore is None:
        raise SystemExit(f"No vectorstore found in {settings.chroma_db_dir}")

    source_catalog = build_source_catalog(vectorstore)
    rag_chain = RAGChain(FakeLLM(), vectorstore)
    rows: list[dict] = []
    for group in POSITIVE_SCENARIOS + NEGATIVE_SCENARIOS + ACTIVE_SOP_SCENARIOS:
        rows.extend(evaluate_group(group, vectorstore, source_catalog))
    for group in CLARIFICATION_SCENARIOS:
        rows.extend(evaluate_clarification_group(group, rag_chain))

    report = build_report(rows)
    print(report)

    markdown_log_path, json_log_path = write_log_files(rows, report, Path(args.log_dir))
    print(f"Saved evaluation logs to {markdown_log_path} and {json_log_path}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
