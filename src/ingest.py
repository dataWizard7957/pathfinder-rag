from docx import Document
from typing import List, Dict
from pathlib import Path
import re
from collections import defaultdict


def _is_heading(paragraph) -> bool:
    """
    Detect headings using:
    1) Word heading styles (Heading 1/2/3...)
    2) Fallback: mostly-bold, short line (common in hand-formatted docs)
    """
    if paragraph.style and paragraph.style.name:
        if paragraph.style.name.lower().startswith("heading"):
            return True

    text = (paragraph.text or "").strip()
    if not text:
        return False

    runs = paragraph.runs
    if runs:
        bold_runs = sum(1 for r in runs if r.bold)
        # mostly bold + short => treat as heading
        return (bold_runs / len(runs)) > 0.6 and len(text) < 80

    return False


def _normalize_text(text: str) -> str:
    """
    Normalize for matching age bands and other markers.
    (This is ingestion hygiene, not business logic.)
    """
    t = (text or "").lower().strip()
    # normalize common dash variants to "-"
    for ch in ["–", "—", "−", "‒", "―"]:
        t = t.replace(ch, "-")
    t = re.sub(r"\s+", " ", t)
    return t


def _extract_age_band(text: str) -> str | None:
    """
    Return "18_24", "25_34", ..., or "65_plus" if found.
    """
    t = _normalize_text(text)

    # matches 18-24, 25-34, etc (allow spaces)
    m = re.search(r"\b(18|25|35|45|55)\s*-\s*(24|34|44|54|64)\b", t)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    # match 65+ / 65 +
    if re.search(r"\b65\s*\+\b", t):
        return "65_plus"

    return None


def _semantic_prefix(heading: str, body: str) -> str:
    """
    Minimal semantic labeling for chunk_id.
    - age_<band> for age-band chunks
    - income_type for income type section(s)
    - misc for everything else
    """
    combined = _normalize_text(f"{heading} {body}")

    age_band = _extract_age_band(combined)
    if age_band:
        return f"age_{age_band}"

    if "income type" in combined:
        return "income_type"

    return "misc"


def ingest_docx(doc_path: Path) -> List[Dict]:
    """
    Read DOCX and split into heading-based chunks.
    Generates chunk_id like:
      age_25_34_01, age_65_plus_01, income_type_01, misc_01 ...
    """
    document = Document(doc_path)
    chunks: List[Dict] = []

    current_heading: str | None = None
    current_text: List[str] = []

    counters = defaultdict(int)

    def add_chunk(heading: str, text_parts: List[str]) -> None:
        full_text = " ".join(text_parts).strip()
        if not full_text:
            return

        prefix = _semantic_prefix(heading, full_text)
        counters[prefix] += 1

        chunks.append({
            "chunk_id": f"{prefix}_{counters[prefix]:02d}",
            "heading": heading,
            "text": full_text,
        })

    for para in document.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue

        if _is_heading(para):
            # close previous chunk
            if current_heading and current_text:
                add_chunk(current_heading, current_text)
                current_text = []
            current_heading = text
        else:
            # if body text appears before any heading, attach to a default heading
            if current_heading is None:
                current_heading = "Untitled Section"
            current_text.append(text)

    # flush last chunk
    if current_heading and current_text:
        add_chunk(current_heading, current_text)

    return chunks
