from __future__ import annotations

import numpy as np
import pytest

from resume_screener import matching


@pytest.fixture(autouse=True)
def _fake_sentence_embeddings(monkeypatch):
    def _embed(text: str) -> np.ndarray:
        tokens = matching.preprocessing.tokenize(text)
        vector = np.zeros(16, dtype=float)
        for token in tokens:
            index = sum(ord(char) for char in token) % vector.size
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vector /= norm
        return vector

    def _build_embeddings(job_description, resume_text_by_id, *, model_name=matching.DEFAULT_EMBEDDING_MODEL):
        job_vector = _embed(job_description)
        resume_vectors = {
            resume_id: _embed(text)
            for resume_id, text in resume_text_by_id.items()
        }
        return job_vector, resume_vectors

    monkeypatch.setattr(matching, "build_sentence_embeddings", _build_embeddings)


def _build_sample_corpus():
    job_description = "Data scientist with python, sql and machine learning"
    resume_texts = {
        "alice": "Python data scientist skilled in SQL and cloud machine learning.",
        "bob": "Front-end developer with javascript skills.",
    }
    corpus = matching.prepare_corpus(job_description, resume_texts)
    return job_description, resume_texts, corpus


def test_score_resumes_ranks_by_similarity():
    job_description = """
    Looking for a data scientist with python, machine learning and NLP experience.
    """
    resume_texts = {
        "alice": "Experienced data scientist skilled in Python, NLP, and TensorFlow.",
        "bob": "Front-end developer with React and CSS expertise.",
        "carol": "Machine learning engineer proficient in Python and data analysis.",
    }

    matches = matching.score_resumes(job_description, resume_texts)
    assert [match.resume_id for match in matches] == ["alice", "carol", "bob"]
    assert matches[0].score >= matches[1].score >= matches[2].score


def test_score_resumes_filters_by_threshold():
    job_description = "Cloud engineer Kubernetes"
    resume_texts = {
        "alice": "Kubernetes administrator and DevOps specialist.",
        "bob": "Graphic designer experienced with Adobe tools.",
    }

    matches = matching.score_resumes(job_description, resume_texts, min_score=0.1)
    assert [match.resume_id for match in matches] == ["alice"]


def test_prepare_corpus_returns_tokens_and_embeddings():
    _, _, corpus = _build_sample_corpus()

    assert corpus.job_tokens
    assert corpus.job_embedding.size > 0
    assert "alice" in corpus.resume_tokens
    assert corpus.resume_embeddings["alice"].size > 0


def test_extract_keyword_insights_reports_coverage():
    _, _, corpus = _build_sample_corpus()

    insights = matching.extract_keyword_insights(corpus, limit=3)
    assert insights
    top_terms = {insight.term for insight in insights}
    assert "python" in top_terms
    python_insight = next(insight for insight in insights if insight.term == "python")
    assert python_insight.resume_count == 1


def test_missing_keywords_for_resume_identifies_gaps():
    _, _, corpus = _build_sample_corpus()
    missing = matching.missing_keywords_for_resume(corpus, "alice", top_n=2)
    # Alice already covers the major keywords so only unrelated terms remain.
    assert isinstance(missing, tuple)
    missing_bob = matching.missing_keywords_for_resume(corpus, "bob", top_n=5)
    assert "python" in missing_bob
