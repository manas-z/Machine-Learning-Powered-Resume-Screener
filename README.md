# Simple Resume Screener

This project showcases a lightweight resume screener that uses TF-IDF and cosine similarity to compare resume PDFs with a job description. It is intentionally focused on the fundamentals so it is easy to explain in a portfolio or demo setting.

## Features

- Extract text from resume PDFs with `pdfminer.six`.
- Score resumes against a job description using TF-IDF cosine similarity.
- Command line script to rank resumes and export the results.
- Simple Streamlit interface for quick experimentation.

## Project structure

```
.
├── cli.py                     # Command line entry point
├── resume_screener/
│   ├── __init__.py
│   ├── matching.py            # TF-IDF scoring helper
│   ├── pdf_extractor.py       # PDF text extraction utilities
│   ├── pipeline.py            # Small orchestration helpers
│   └── preprocessing.py       # Tokenisation helper used for keyword highlights
└── streamlit_app.py           # Minimal Streamlit UI
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Command line usage

Prepare a plain-text job description and place resume PDFs in a folder:

```bash
python cli.py job.txt resumes/ --top-k 5 --output results.csv
```

The command prints the ranked resumes and can optionally save the results to JSON or CSV.

## Streamlit app

Launch the interactive interface with:

```bash
streamlit run streamlit_app.py
```

Paste your job description, upload PDFs, and view the similarity scores directly in the browser.
