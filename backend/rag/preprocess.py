from __future__ import annotations

import re
from pathlib import Path

SECTION_HINTS = {
    "agenda",
    "best practice",
    "best practices",
    "checklist",
    "conclusion",
    "department",
    "department / vertical",
    "designation",
    "details",
    "guidelines",
    "inputs",
    "job objective",
    "objective",
    "objectives",
    "overview",
    "outputs",
    "participants",
    "process",
    "procedure",
    "procedures",
    "profile details",
    "reporting authority",
    "responsibility",
    "introduction",
    "workflow",
    "responsibilities",
    "review history",
    "revision history",
    "role",
    "roles",
    "scope",
    "training requirement",
    "minimum qualification",
    "minimum experience",
}

SECTION_TITLE_MAP = {
    "agenda": "Agenda",
    "best practice": "Best Practices",
    "best practices": "Best Practices",
    "checklist": "Checklist",
    "conclusion": "Conclusion",
    "department": "Department",
    "department vertical": "Department / Vertical",
    "department / vertical": "Department / Vertical",
    "designation": "Designation",
    "details": "Details",
    "guidelines": "Guidelines",
    "inputs": "Inputs",
    "introduction": "Introduction",
    "job objective": "Job Objectives",
    "job objectives": "Job Objectives",
    "minimum experience": "Minimum Experience",
    "minimum qualification": "Minimum Qualification",
    "objective": "Objective",
    "objectives": "Objectives",
    "outputs": "Outputs",
    "overview": "Overview",
    "participants": "Participants",
    "process": "Process",
    "procedure": "Procedure",
    "procedures": "Procedure",
    "profile details": "Profile Details",
    "reporting authority": "Reporting Authority",
    "responsibility": "Responsibilities",
    "responsibilities": "Responsibilities",
    "review history": "Review History",
    "revision history": "Revision History",
    "role": "Role",
    "roles": "Roles",
    "scope": "Scope",
    "training requirement": "Training Requirement",
    "workflow": "Workflow",
}

SUMMARY_PREFERENCE = {
    "role": (
        "Job Objectives",
        "Responsibilities",
        "Reporting Authority",
        "Minimum Qualification",
        "Minimum Experience",
        "Training Requirement",
    ),
    "workflow": (
        "Objective",
        "Objectives",
        "Overview",
        "Process",
        "Procedure",
        "Workflow",
        "Responsibilities",
    ),
    "meeting": (
        "Objective",
        "Objectives",
        "Agenda",
        "Participants",
        "Procedure",
        "Workflow",
    ),
    "standards": (
        "Objective",
        "Objectives",
        "Overview",
        "Guidelines",
        "Best Practices",
        "Checklist",
    ),
}

ACRONYMS = {
    "API",
    "BA",
    "CMS",
    "CMMI",
    "CTO",
    "DBA",
    "FRS",
    "GEL",
    "GITLAB",
    "IT",
    "JIRA",
    "LAQ",
    "MIS",
    "OM",
    "PM",
    "QA",
    "RR",
    "RTI",
    "RTM",
    "SEO",
    "SMS",
    "SOP",
    "SQL",
    "TL",
    "UI",
    "UX",
    ".NET",
}

NOISE_PATTERNS = [
    r"^page\s+\d+(\s+of\s+\d+)?$",
    r"^\d+\s*$",
    r"^important:\s+the information contained in this document",
    r"^the information contained in this document should not be passed on",
    r"^gel[-\s].*ver[-\s]?\d+(\.\d+)?(\s*\|\s*\d{2}/\d{2}/\d{4})?$",
    r"^\|?\s*\d{2}/\d{2}/\d{4}\s*\|?$",
]

UNICODE_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\uf0a7": "-",
        "\uf0b7": "-",
        "\uf0d8": "-",
        "\xa0": " ",
    }
)


def humanize_source_name(source: str) -> str:
    stem = Path(source).stem
    stem = stem.replace("&", " and ")
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    stem = re.sub(r"[_\-.()/]+", " ", stem)
    stem = re.sub(r"^\d+\s+", " ", stem)
    stem = re.sub(r"\b(ver|version|v)\s*\d+(\.\d+)?\b", " ", stem, flags=re.I)
    stem = re.sub(r"\b\d{8,}\b", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.translate(UNICODE_REPLACEMENTS)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    line = normalize_text(line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip(" -\t")


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    normalized = normalize_line(line).lower()
    if not normalized:
        return True
    for pattern in NOISE_PATTERNS:
        if re.match(pattern, normalized):
            return True
    return False


def is_heading(line: str) -> bool:
    normalized = normalize_line(line)
    if not normalized or len(normalized) > 90:
        return False

    lowered = normalized.lower()
    if lowered in SECTION_HINTS:
        return True

    alpha_words = [word for word in re.split(r"\s+", normalized) if re.search(r"[A-Za-z]", word)]
    if not alpha_words:
        return False

    if len(alpha_words) <= 6:
        uppercase_ratio = sum(1 for char in normalized if char.isupper()) / max(
            1, sum(1 for char in normalized if char.isalpha())
        )
        if uppercase_ratio >= 0.7:
            return True

    if any(hint in lowered for hint in SECTION_HINTS):
        return len(alpha_words) <= 8

    return False


def canonical_section_title(line: str) -> str | None:
    normalized = normalize_text(line).lower()
    normalized = re.sub(r"[^a-z0-9/ ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    if normalized in SECTION_TITLE_MAP:
        return SECTION_TITLE_MAP[normalized]
    if is_heading(line):
        return normalize_line(line).title()
    return None


def clean_lines(text: str) -> list[str]:
    text = normalize_text(text)
    raw_lines = [normalize_line(line) for line in text.splitlines()]
    raw_lines = [line for line in raw_lines if not is_noise_line(line)]

    merged: list[str] = []
    for line in raw_lines:
        if not merged:
            merged.append(line)
            continue

        previous = merged[-1]
        if is_heading(previous) or is_heading(line):
            merged.append(line)
            continue

        if previous.endswith("-") and len(line) > 0:
            merged[-1] = f"{previous[:-1]}{line}"
            continue

        if should_join_lines(previous, line):
            merged[-1] = f"{previous} {line}"
        else:
            merged.append(line)

    return merged


def should_join_lines(previous: str, current: str) -> bool:
    if not previous or not current:
        return False

    if previous.endswith((".", ":", "?", "!", ";")):
        return False
    if current[0].islower():
        return True
    if len(current.split()) <= 3 and not is_heading(current):
        return True
    if previous.lower().endswith(("and", "or", "to", "with", "for", "of", "by")):
        return True
    return False


def extract_sections(text: str) -> list[tuple[str, str]]:
    lines = clean_lines(text)
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    buffer: list[str] = []

    for line in lines:
        canonical_heading = canonical_section_title(line)
        if canonical_heading:
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = canonical_heading
            continue
        buffer.append(line)

    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))

    return [(heading, body) for heading, body in sections if body]


def detect_source_title(source: str, lines: list[str]) -> str:
    for line in lines[:20]:
        if is_heading(line) and line.lower() not in SECTION_HINTS:
            return line

    for line in lines[:20]:
        if 3 <= len(line) <= 80 and re.search(r"[A-Za-z]", line):
            if line.lower() not in SECTION_HINTS:
                return line

    return humanize_source_name(source)


def infer_source_kind(source: str, title: str) -> str:
    haystack = f"{source} {title}".lower()
    if "rr" in haystack or "role" in haystack or "responsibilit" in haystack:
        return "role"
    if "standard" in haystack or "best practice" in haystack:
        return "standards"
    if "meeting" in haystack or "review meet" in haystack:
        return "meeting"
    if "workflow" in haystack or "process" in haystack or "sop" in haystack:
        return "workflow"
    return "document"


def _display_alias(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts: list[str] = []
    for token in text.split():
        key = re.sub(r"[^A-Za-z0-9.]+", "", token).upper()
        if key in ACRONYMS:
            parts.append(key)
        else:
            parts.append(token[:1].upper() + token[1:].lower())
    return " ".join(parts)


def is_alias_line(line: str) -> bool:
    normalized = normalize_line(line)
    if not normalized:
        return False
    if canonical_section_title(normalized):
        return False
    if len(normalized) > 70:
        return False
    if normalized.endswith("."):
        return False
    lowered = normalized.lower()
    if any(term in lowered for term in ("addition", "updation", "prepared by", "approved by")):
        return False
    letters = sum(1 for char in normalized if char.isalpha())
    digits = sum(1 for char in normalized if char.isdigit())
    punctuation = sum(1 for char in normalized if char in "/|:;,.()")
    if letters < 3:
        return False
    if digits > max(2, letters // 3):
        return False
    if punctuation > 4:
        return False
    return True


def build_intent_phrases(title: str, source_kind: str) -> list[str]:
    base = _display_alias(title)
    if not base:
        return []

    if source_kind == "role":
        return [
            f"{base} role",
            f"{base} responsibilities",
            f"roles and responsibilities of {base}",
            f"{base} job objectives",
            f"{base} reporting authority",
        ]
    if source_kind == "meeting":
        return [
            f"{base} meeting",
            f"{base} procedure",
            f"{base} agenda",
            f"{base} SOP",
        ]
    if source_kind == "standards":
        return [
            f"{base} standards",
            f"{base} best practices",
            f"{base} guidelines",
        ]
    if source_kind == "workflow":
        return [
            f"{base} SOP",
            f"{base} process",
            f"{base} workflow",
            f"{base} procedure",
        ]
    return [
        f"{base} SOP",
        f"{base} document",
    ]


def build_source_aliases(
    source: str,
    title: str,
    lines: list[str],
    source_kind: str,
) -> list[str]:
    aliases = {
        humanize_source_name(source),
        title,
        title.replace("-", " "),
        _display_alias(title),
    }

    haystack = f"{source} {title}".lower()
    if "dba" in haystack:
        aliases.add("Database Administrator DBA")
        aliases.add("Database Administrator")
    if "testlead" in haystack or "test lead" in haystack:
        aliases.add("Test Lead")
    if "technicallead" in haystack or "technical lead" in haystack:
        aliases.add("Technical Lead")
    if "developmenteng" in haystack or "development engineer" in haystack:
        aliases.add("Development Engineer")
    if "dotnet" in haystack or ".net" in haystack or "dot net" in haystack:
        aliases.add(".NET Coding Standards")
        aliases.add("DotNet Coding Standards")
    if "react" in haystack:
        aliases.add("React Coding Standards")
    if "gitlab" in haystack:
        aliases.add("Source Code Management GitLab")
    if "tl review meet" in haystack:
        aliases.add("Technical Lead Review Meeting")
    if "jira" in haystack:
        aliases.add("GEL Jira Issue Creation")

    aliases.update(build_intent_phrases(title, source_kind))

    for line in lines[:10]:
        if is_alias_line(line):
            aliases.add(line)

    cleaned = sorted(
        {
            re.sub(r"\s+", " ", alias).strip()
            for alias in aliases
            if alias and len(alias.strip()) >= 3
        }
    )
    return cleaned[:20]


def _summary_sections(
    source_kind: str,
    sections: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    if not sections:
        return []

    preferred_titles = SUMMARY_PREFERENCE.get(source_kind, ())
    if not preferred_titles:
        return sections

    ranked: list[tuple[int, str, str]] = []
    for heading, body in sections:
        try:
            rank = preferred_titles.index(heading)
        except ValueError:
            rank = len(preferred_titles) + 1
        ranked.append((rank, heading, body))

    ranked.sort(key=lambda item: item[0])
    return [(heading, body) for _, heading, body in ranked]


def build_source_profile(source: str, text: str) -> dict[str, str | list[str]]:
    lines = clean_lines(text)
    title = detect_source_title(source, lines)
    sections = extract_sections(text)
    source_kind = infer_source_kind(source, title)
    aliases = build_source_aliases(source, title, lines, source_kind)
    intents = build_intent_phrases(title, source_kind)
    section_titles = []
    for heading, _ in sections:
        if heading not in section_titles:
            section_titles.append(heading)

    summary_lines: list[str] = []
    for heading, body in _summary_sections(source_kind, sections)[:4]:
        snippet = body[:240].strip()
        if snippet:
            summary_lines.append(f"{heading}: {snippet}")
    if not summary_lines:
        summary_lines = lines[:8]

    return {
        "source_title": title,
        "source_kind": source_kind,
        "source_aliases": aliases,
        "source_intents": intents[:12],
        "section_titles": section_titles[:12],
        "summary_text": "\n".join(summary_lines[:6]).strip(),
    }
