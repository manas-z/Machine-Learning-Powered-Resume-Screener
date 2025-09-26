"""Core resume matching logic using a light-weight TF-IDF implementation."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from . import preprocessing


@dataclass(frozen=True)
class ResumeMatch:
    """Represents the similarity between a resume and a job description."""

    resume_id: str
    score: float
    highlights: Tuple[str, ...]


@dataclass(frozen=True)
class CorpusSnapshot:
    """Normalised view of the job description and resumes used for scoring."""

    job_tokens: Tuple[str, ...]
    job_weights: Dict[str, float]
    resume_tokens: Dict[str, Tuple[str, ...]]
    resume_weights: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class KeywordInsight:
    """Summarises how important a keyword is and how often it appears in resumes."""

    term: str
    weight: float
    resume_count: int
    coverage_ratio: float


def _term_frequencies(tokens: Sequence[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = float(sum(counts.values())) or 1.0
    return {term: freq / total for term, freq in counts.items()}


def _inverse_document_frequencies(documents: Iterable[Sequence[str]]) -> Dict[str, float]:
    document_frequency: Dict[str, int] = defaultdict(int)
    num_documents = 0
    for tokens in documents:
        num_documents += 1
        for term in set(tokens):
            document_frequency[term] += 1

    if num_documents == 0:
        return {}

    idf: Dict[str, float] = {}
    for term, df in document_frequency.items():
        idf[term] = math.log((1 + num_documents) / (1 + df)) + 1.0
    return idf


def _tfidf_vector(tokens: Sequence[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = _term_frequencies(tokens)
    return {term: tf_val * idf.get(term, 0.0) for term, tf_val in tf.items()}


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    common_terms = set(vec_a).intersection(vec_b)
    numerator = sum(vec_a[term] * vec_b[term] for term in common_terms)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return numerator / (norm_a * norm_b)


def prepare_corpus(
    job_description: str, resume_text_by_id: Dict[str, str]
) -> CorpusSnapshot:
    """Create TF-IDF representations used to compare resumes."""

    if not job_description.strip():
        raise ValueError("Job description text must not be empty.")

    job_tokens = tuple(preprocessing.tokenize(job_description))
    resume_tokens: Dict[str, Tuple[str, ...]] = {
        resume_id: tuple(preprocessing.tokenize(text))
        for resume_id, text in resume_text_by_id.items()
        if text.strip()
    }

    documents: List[Sequence[str]] = list(resume_tokens.values()) + [job_tokens]
    idf = _inverse_document_frequencies(documents)
    job_weights = _tfidf_vector(job_tokens, idf)
    resume_weights = {
        resume_id: _tfidf_vector(tokens, idf)
        for resume_id, tokens in resume_tokens.items()
    }

    return CorpusSnapshot(
        job_tokens=job_tokens,
        job_weights=job_weights,
        resume_tokens=resume_tokens,
        resume_weights=resume_weights,
    )


def score_resumes(
    job_description: str,
    resume_text_by_id: Dict[str, str],
    *,
    top_k: int | None = None,
    min_score: float = 0.0,
) -> List[ResumeMatch]:
    """Rank resumes according to their similarity with a job description.

    Parameters
    ----------
    job_description:
        Raw text of the job specification.
    resume_text_by_id:
        Mapping of resume identifiers to their raw text.
    top_k:
        Limit the number of returned matches. ``None`` returns all matches.
    min_score:
        Minimum similarity score required for a resume to be returned.
    """

    corpus = prepare_corpus(job_description, resume_text_by_id)

    if not corpus.resume_tokens:
        return []

    matches: List[ResumeMatch] = []
    for resume_id, tokens in corpus.resume_tokens.items():
        resume_vector = corpus.resume_weights.get(resume_id, {})
        score = _cosine_similarity(corpus.job_weights, resume_vector)
        if score < min_score:
            continue

        # Highlight keywords that carry the most weight for both the resume and job description.
        candidate_terms = []
        for term in set(tokens):
            if term not in corpus.job_weights:
                continue
            weight = resume_vector.get(term, 0.0) * corpus.job_weights.get(term, 0.0)
            if weight <= 0.0:
                continue
            candidate_terms.append((term, weight))

        candidate_terms.sort(key=lambda item: item[1], reverse=True)
        highlights = tuple(term for term, _ in candidate_terms[:5])
        matches.append(ResumeMatch(resume_id=resume_id, score=score, highlights=highlights))

    matches.sort(key=lambda match: match.score, reverse=True)

    if top_k is not None:
        matches = matches[:top_k]

    return matches


def extract_keyword_insights(
    corpus: CorpusSnapshot, *, limit: int = 15
) -> List[KeywordInsight]:
    """Return the most important job description keywords and their coverage."""

    if not corpus.job_weights:
        return []

    sorted_terms = sorted(
        corpus.job_weights.items(), key=lambda item: item[1], reverse=True
    )
    total_resumes = len(corpus.resume_tokens)

    insights: List[KeywordInsight] = []
    for term, weight in sorted_terms[:limit]:
        resume_count = sum(1 for tokens in corpus.resume_tokens.values() if term in tokens)
        coverage_ratio = (resume_count / total_resumes) if total_resumes else 0.0
        insights.append(
            KeywordInsight(
                term=term,
                weight=weight,
                resume_count=resume_count,
                coverage_ratio=coverage_ratio,
            )
        )

    return insights


def missing_keywords_for_resume(
    corpus: CorpusSnapshot, resume_id: str, *, top_n: int = 10
) -> Tuple[str, ...]:
    """Identify the most important job keywords missing from a resume."""

    if resume_id not in corpus.resume_tokens or not corpus.job_weights:
        return ()

    ordered_terms = [
        term for term, _ in sorted(corpus.job_weights.items(), key=lambda item: item[1], reverse=True)
    ]
    resume_terms = set(corpus.resume_tokens[resume_id])
    missing = [term for term in ordered_terms if term not in resume_terms][:top_n]
    return tuple(missing)
