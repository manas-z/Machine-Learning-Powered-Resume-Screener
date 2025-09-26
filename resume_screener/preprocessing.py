"""Text normalisation and tokenisation helpers."""
from __future__ import annotations

import re
from typing import Iterable, List

# A lightweight set of stop words that helps focus on meaningful tokens.
STOP_WORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> List[str]:
    """Tokenise text into a list of alphanumeric words.

    The output is lower-cased and filters out a compact list of stop words to
    emphasise keywords contained in resumes and job descriptions.
    """

    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in raw_tokens if token not in STOP_WORDS]


def normalise(text: str) -> str:
    """Collapse whitespace and lower-case the text."""

    tokens = tokenize(text)
    return " ".join(tokens)


def flatten_documents(documents: Iterable[str]) -> str:
    """Concatenate multiple documents into a single text blob."""

    return " \n".join(documents)
