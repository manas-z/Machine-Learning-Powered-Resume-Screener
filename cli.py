"""Command line interface for the simple resume screener."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from resume_screener import matching, pipeline


def _parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank resume PDFs against a job description using TF-IDF cosine similarity.",
    )
    parser.add_argument(
        "job_description",
        type=Path,
        help="Path to a plain text file containing the job description.",
    )
    parser.add_argument(
        "resume_dir",
        type=Path,
        help="Directory containing PDF resumes to screen.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only return the top K matches.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Ignore resumes with a similarity score below this threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the results as JSON or CSV (based on extension).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _serialise_results(matches: Iterable[matching.ResumeMatch], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = matching.summarise_matches(matches)

    if destination.suffix.lower() == ".json":
        destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    elif destination.suffix.lower() == ".csv":
        with destination.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["resume_id", "score", "keywords"])
            writer.writeheader()
            for row in summary:
                writer.writerow(row)
    else:
        raise ValueError("Unsupported output format. Use a .json or .csv file extension.")


def main(argv: Iterable[str] | None = None) -> list[matching.ResumeMatch]:
    args = _parse_arguments(argv)

    job_description_text = pipeline.load_job_description(args.job_description)
    resume_texts = pipeline.load_resume_texts_from_directory(args.resume_dir)

    matches = pipeline.screen_resumes(
        job_description_text,
        resume_texts,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if args.output:
        _serialise_results(matches, args.output)

    for match in matches:
        keyword_text = f" (keywords: {', '.join(match.keywords)})" if match.keywords else ""
        print(f"{match.resume_id}: {match.score:.3f}{keyword_text}")

    return matches


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
