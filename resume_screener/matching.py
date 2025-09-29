"""Simple TF-IDF based resume matching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from . import preprocessing


@dataclass(frozen=True)
class ResumeMatch:
    """Represents a resume and its similarity score."""

    resume_id: str
    score: float
    keywords: Tuple[str, ...]


def _extract_keywords(
    vectoriser: TfidfVectorizer,
    resume_vector,
    job_tokens: set[str],
    *,
    top_n: int = 5,
) -> Tuple[str, ...]:
    """Return the strongest overlapping terms between a resume and the job description."""

    if resume_vector.nnz == 0:
        return ()

    feature_names = vectoriser.get_feature_names_out()

    keyword_weights: List[Tuple[str, float]] = []
    row = resume_vector.tocoo()
    for column_index, weight in zip(row.col, row.data):
        term = feature_names[column_index]
        if term not in job_tokens:
            continue
        keyword_weights.append((term, float(weight)))

    keyword_weights.sort(key=lambda item: item[1], reverse=True)
    return tuple(term for term, _ in keyword_weights[:top_n])


def score_resumes(
    job_description: str,
    resume_text_by_id: Dict[str, str],
    *,
    top_k: int | None = None,
    min_score: float = 0.0,
) -> List[ResumeMatch]:
    """Rank resumes using cosine similarity between TF-IDF vectors."""

    if not job_description.strip():
        raise ValueError("Job description text cannot be empty.")

    if not resume_text_by_id:
        return []

    documents = [job_description] + list(resume_text_by_id.values())
    vectoriser = TfidfVectorizer(stop_words="english")
    matrix = vectoriser.fit_transform(documents)

    job_vector = matrix[0]
    resume_matrix = matrix[1:, :]
    scores = linear_kernel(resume_matrix, job_vector).ravel()

    job_tokens = set(preprocessing.tokenize(job_description))
    resume_ids = list(resume_text_by_id.keys())

    matches: List[ResumeMatch] = []
    for index, resume_id in enumerate(resume_ids):
        score = float(scores[index])
        if score < min_score:
            continue

        resume_vector = resume_matrix[index]
        keywords = _extract_keywords(vectoriser, resume_vector, job_tokens)
        matches.append(ResumeMatch(resume_id=resume_id, score=score, keywords=keywords))

    matches.sort(key=lambda match: match.score, reverse=True)

    if top_k is not None:
        matches = matches[:top_k]

    return matches


def summarise_matches(matches: Iterable[ResumeMatch]) -> List[dict[str, str]]:
    """Represent matches as dictionaries suitable for serialisation."""

    summary: List[dict[str, str]] = []
    for match in matches:
        summary.append(
            {
                "resume_id": match.resume_id,
                "score": f"{match.score:.3f}",
                "keywords": ", ".join(match.keywords),
            }
        )
    return summary
