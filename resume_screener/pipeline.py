"""Lightweight helpers for the resume screening workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from . import matching
from .pdf_extractor import batch_extract_text


def load_job_description(path: Path | str) -> str:
    """Read a plain-text job description from disk."""

    job_path = Path(path)
    if not job_path.exists():
        raise FileNotFoundError(f"Job description file not found: {job_path}")
    return job_path.read_text(encoding="utf-8")


def load_resume_texts_from_directory(directory: Path | str) -> Dict[str, str]:
    """Extract text from all PDF files in a directory."""

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Resume directory not found: {dir_path}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files were found in resume directory: {dir_path}")

    return batch_extract_text(pdf_files)


def screen_resumes(
    job_description_text: str,
    resume_text_by_id: Dict[str, str],
    *,
    top_k: int | None = None,
    min_score: float = 0.0,
) -> List[matching.ResumeMatch]:
    """Score resumes against the job description using TF-IDF cosine similarity."""

    return matching.score_resumes(
        job_description=job_description_text,
        resume_text_by_id=resume_text_by_id,
        top_k=top_k,
        min_score=min_score,
    )


def summarise_matches(matches: Iterable[matching.ResumeMatch]) -> List[dict[str, str]]:
    """Convert matches into dictionaries suitable for export."""

    return matching.summarise_matches(matches)
