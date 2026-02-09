from typing import Dict, Any, List, Tuple

from src.config import (
    DOC_PATH,
    TOP_K,
    BM25_TOP_N,
    SUPPORTED_AGE_RANGES,
    SUPPORTED_INCOME_TYPES,
)
from src.ingest import ingest_docx
from src.retrieval import HybridRetriever
from src.generator import generate_with_llm

_chunks_cache: List[Dict[str, Any]] | None = None
_retriever_cache: HybridRetriever | None = None


def _load_retriever() -> Tuple[HybridRetriever, List[Dict[str, Any]]]:
    """Load chunks + build retriever once (cached)."""
    global _chunks_cache, _retriever_cache
    if _retriever_cache is None:
        _chunks_cache = ingest_docx(DOC_PATH)
        _retriever_cache = HybridRetriever(_chunks_cache)
    return _retriever_cache, _chunks_cache  # type: ignore[return-value]


def _age_variants(age_range: str) -> List[str]:
    """Simple textual variants to match DOCX formatting differences."""
    variants = [age_range]
    if "-" in age_range:
        a, b = age_range.split("-", 1)
        variants.append(age_range.replace("-", "–"))  # en-dash
        variants.append(f"{a} to {b}")
        variants.append(f"{a} to {b} years")
        variants.append(f"{a}-{b} years")
        variants.append(f"{a}–{b} years")
    return variants


def _matches_any(haystack: str, needles: List[str]) -> bool:
    h = (haystack or "").lower()
    return any(n.lower() in h for n in needles)


def _best_chunk_by_text(
    chunks: List[Dict[str, Any]],
    needles: List[str],
) -> Dict[str, Any] | None:
    """
    Pick the first chunk whose (heading OR text) contains any needle.
    Retrieval hygiene only (not business logic).
    """
    for c in chunks:
        ht = f"{c.get('heading', '')}\n{c.get('text', '')}"
        if _matches_any(ht, needles):
            return c
    return None


def _normalize_age_text(s: str) -> str:
    """
    Normalize dash/minus variants so '55–64', '55—64', '55−64' match '55-64'.
    Also normalizes 'a to b' to 'a-b'.
    Retrieval hygiene only (not business logic).
    """
    t = (s or "").lower()
    for ch in ["–", "—", "−", "‒", "―"]:
        t = t.replace(ch, "-")
    t = t.replace(" to ", "-")
    return t


def _is_other_age_chunk(chunk: Dict[str, Any], age_range: str) -> bool:
    """
    True if chunk clearly belongs to a DIFFERENT age band.
    String-only checks. No regex. No business logic.
    Checks both heading and body (age band may appear in body like 'Clients 55–64...').
    """
    heading = _normalize_age_text(chunk.get("heading") or "")
    text = _normalize_age_text(chunk.get("text") or "")
    haystack = f"{heading}\n{text}"

    age_bands = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    # If it doesn't mention ANY age band, treat as general info (keep it)
    if not any(band in haystack for band in age_bands):
        return False

    # If it mentions age band(s) but not the requested one, it's cross-age (drop it)
    requested_variants = [_normalize_age_text(v) for v in _age_variants(age_range)]
    matches_requested = any(v in haystack for v in requested_variants)
    return not matches_requested


def _safe_context_set(
    retrieved: List[Dict[str, Any]],
    age_chunk: Dict[str, Any] | None,
    income_chunk: Dict[str, Any] | None,
    rationale_chunk: Dict[str, Any] | None,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Build a stable, grounded context set:
    - age chunk (if found)
    - income chunk (if found)
    - general "why age range matters" chunk (if found)
    - then fill with remaining retrieved chunks
    """
    final: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(ch: Dict[str, Any] | None) -> None:
        if not ch:
            return
        cid = ch.get("chunk_id")
        if cid and cid not in seen:
            final.append(ch)
            seen.add(cid)

    add(age_chunk)
    add(income_chunk)
    add(rationale_chunk)

    for c in retrieved:
        add(c)
        if len(final) >= top_k:
            break

    return final[:top_k]


def generate_pathfinder_suggestions(age_range: str, income_type: str) -> Dict[str, Any]:
    """
    Main interface required by the assignment.
    Returns dict with:
      profile, suggestions, follow_up_questions, retrieved_evidence
    If insufficient context, adds insufficient_context + clarifying_questions.
    """
    result: Dict[str, Any] = {
        "profile": {"age_range": age_range, "income_type": income_type},
        "suggestions": [],
        "follow_up_questions": [],
        "retrieved_evidence": [],
    }

    # Validate supported inputs (no guessing)
    if age_range not in SUPPORTED_AGE_RANGES or income_type not in SUPPORTED_INCOME_TYPES:
        result["insufficient_context"] = True
        result["clarifying_questions"] = [
            f"Can you confirm the client’s age range from the supported options ({', '.join(SUPPORTED_AGE_RANGES)})?",
            f"Can you confirm which income type best matches the client’s primary income ({', '.join(SUPPORTED_INCOME_TYPES)})?",
        ]
        return result

    retriever, all_chunks = _load_retriever()

    # Retrieval query (no business rules)
    age_q = " ".join(_age_variants(age_range))
    query = f"Age range {age_q} Age {age_q} Income Type {income_type}"

    retrieved = retriever.retrieve(query, top_k=TOP_K, bm25_top_n=BM25_TOP_N)

    if not retrieved:
        result["insufficient_context"] = True
        result["clarifying_questions"] = [
            "I couldn’t retrieve relevant sections from the training document for this profile. Can you confirm the age range?",
            "Can you confirm which income type best matches the client’s primary income?",
        ]
        return result

    # Prefer explicit age + income + rationale chunks if they exist (selection for grounding)
    age_chunk = _best_chunk_by_text(all_chunks, _age_variants(age_range))

    income_chunk = _best_chunk_by_text(all_chunks, [income_type])
    if income_chunk is None:
        income_chunk = _best_chunk_by_text(all_chunks, ["income type"])

    rationale_chunk = _best_chunk_by_text(all_chunks, ["what is your age range"])

    # Filter out other-age chunks from retrieved list (prevents cross-age leakage incl. 65+)
    retrieved_filtered = [c for c in retrieved if not _is_other_age_chunk(c, age_range)]

    # Build final context set (stable + grounded)
    final_context = _safe_context_set(
        retrieved=retrieved_filtered,
        age_chunk=age_chunk,
        income_chunk=income_chunk,
        rationale_chunk=rationale_chunk,
        top_k=TOP_K,
    )

    # Generation (grounded in final_context only)
    gen = generate_with_llm(final_context, age_range, income_type)

    result["suggestions"] = gen.get("suggestions", [])[:5]
    result["follow_up_questions"] = gen.get("follow_up_questions", [])

    # Evidence (semantic chunk_id comes from ingestion; pipeline just passes it through)
    result["retrieved_evidence"] = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "text_excerpt": ((c.get("text", "")[:300] + "...") if c.get("text") else ""),
            "reason_used": "Retrieved as relevant to the provided age range and/or income type",
        }
        for c in final_context
    ]

    return result
