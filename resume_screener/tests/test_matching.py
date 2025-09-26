from __future__ import annotations

from resume_screener import matching


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
