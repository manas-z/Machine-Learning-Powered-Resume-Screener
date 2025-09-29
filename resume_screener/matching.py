"""Core resume matching logic using a light-weight TF-IDF implementation."""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Protocol, Sequence, Tuple

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


class _CrossEncoderLike(Protocol):
    """Protocol describing the interface expected from cross-encoder models."""

    def predict(self, sentence_pairs: Sequence[Tuple[str, str]]) -> Sequence[float]:
        ...


DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, device: str | None = None) -> _CrossEncoderLike:
    """Load a cross-encoder model using sentence-transformers or transformers."""

    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        return CrossEncoder(model_name, device=device)
    except ImportError:
        pass

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError(
            "Cross-encoder re-ranking requires the 'sentence-transformers' or "
            "'transformers' packages to be installed."
        ) from exc

    class _TransformersCrossEncoder:
        def __init__(self) -> None:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if device is not None:
                self._model.to(device)

        def predict(self, sentence_pairs: Sequence[Tuple[str, str]]) -> Sequence[float]:
            inputs = self._tokenizer(
                [pair[0] for pair in sentence_pairs],
                [pair[1] for pair in sentence_pairs],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if device is not None:
                inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            logits = outputs.logits.squeeze(-1)
            scores = logits.detach().cpu().tolist()
            if isinstance(scores, float):
                return [scores]
            return scores

    return _TransformersCrossEncoder()


def rerank_matches(
    job_description: str,
    resume_text_by_id: Dict[str, str],
    matches: Sequence[ResumeMatch],
    *,
    top_n: int | None = None,
    model_name: str = DEFAULT_RERANK_MODEL,
    device: str | None = None,
    model_loader: Callable[[str], _CrossEncoderLike] | None = None,
) -> List[ResumeMatch]:
    """Refine similarity scores using a cross-encoder over the strongest matches.

    Parameters
    ----------
    job_description:
        Normalised job description text used as the query.
    resume_text_by_id:
        Mapping of resume identifiers to their raw text.
    matches:
        Preliminary matches produced by the embedding stage.
    top_n:
        Number of candidates to re-score. ``None`` defaults to the number of matches.
    model_name:
        Name of the cross-encoder model to load.
    device:
        Optional device identifier (e.g. ``"cpu"`` or ``"cuda"``).
    model_loader:
        Optional factory for providing a pre-loaded cross-encoder. Primarily useful
        for injecting lightweight stubs in tests.
    """

    if not matches:
        return list(matches)

    limit = len(matches) if top_n is None else max(0, min(len(matches), top_n))
    if limit == 0:
        return list(matches)

    selected_matches = list(matches[:limit])

    if model_loader is None:
        model = _load_cross_encoder(model_name, device)
    else:
        model = model_loader(model_name)

    sentence_pairs: list[Tuple[str, str]] = []
    selected_ids: list[str] = []
    for match in selected_matches:
        resume_text = resume_text_by_id.get(match.resume_id)
        if not resume_text:
            continue
        sentence_pairs.append((job_description, resume_text))
        selected_ids.append(match.resume_id)

    if not sentence_pairs:
        return list(matches)

    scores = model.predict(sentence_pairs)
    refined_scores = {
        resume_id: float(score)
        for resume_id, score in zip(selected_ids, scores)
    }

    reranked_matches: List[ResumeMatch] = []
    for match in matches:
        score = refined_scores.get(match.resume_id, match.score)
        reranked_matches.append(
            ResumeMatch(
                resume_id=match.resume_id,
                score=score,
                highlights=match.highlights,
            )
        )

    return reranked_matches


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
    enable_rerank: bool | None = None,
    rerank_top_n: int | None = None,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    rerank_device: str | None = None,
    rerank_model_loader: Callable[[str], _CrossEncoderLike] | None = None,
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

    if enable_rerank is None:
        enable_rerank = _env_flag("RESUME_SCREENER_RERANK", default=False)

    if rerank_top_n is None:
        rerank_top_n = _env_int("RESUME_SCREENER_RERANK_TOP_N")

    if enable_rerank and matches:
        matches = rerank_matches(
            job_description,
            resume_text_by_id,
            matches,
            top_n=rerank_top_n,
            model_name=rerank_model,
            device=rerank_device,
            model_loader=rerank_model_loader,
        )
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
