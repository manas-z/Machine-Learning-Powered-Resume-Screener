"""High level orchestration helpers for the resume screening workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from . import matching
from .pdf_extractor import batch_extract_text


def load_job_description(path: Path | str) -> str:
    """Load a job description from a plain text file."""

    job_path = Path(path)
    if not job_path.exists():
        raise FileNotFoundError(f"Job description file not found: {job_path}")
    return job_path.read_text(encoding="utf-8")


def load_resume_texts_from_directory(directory: Path | str) -> Dict[str, str]:
    """Read all PDF files in a directory and extract their text."""

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Resume directory not found: {dir_path}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in resume directory: {dir_path}"
        )

    return batch_extract_text(pdf_files)


def screen_resumes(
    job_description_text: str,
    resume_text_by_id: Dict[str, str],
    *,
    top_k: int | None = None,
    min_score: float = 0.0,
    enable_rerank: bool | None = None,
    rerank_top_n: int | None = None,
    rerank_model: str | None = None,
    rerank_device: str | None = None,
) -> List[matching.ResumeMatch]:
    """Score resumes against a job description and return ranked matches."""

    return matching.score_resumes(
        job_description=job_description_text,
        resume_text_by_id=resume_text_by_id,
        top_k=top_k,
        min_score=min_score,
        enable_rerank=enable_rerank,
        rerank_top_n=rerank_top_n,
        rerank_model=(rerank_model or matching.DEFAULT_RERANK_MODEL),
        rerank_device=rerank_device,
    )


def summarise_matches(matches: Iterable[matching.ResumeMatch]) -> List[dict[str, str]]:
    """Convert :class:`ResumeMatch` objects into serialisable dictionaries."""

    summary: List[dict[str, str]] = []
    for match in matches:
        summary.append(
            {
                "resume_id": match.resume_id,
                "score": f"{match.score:.3f}",
                "highlights": ", ".join(match.highlights),
            }
        )
    return summary
