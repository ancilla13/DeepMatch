# DeepMatch 

An AI-powered resume ranking system that evaluates and ranks candidates based on their relevance to a given job description.

## Overview

DeepMatch automates the initial resume screening process by extracting information from resumes, comparing candidate skills and experience with job requirements, and generating relevance scores. It helps recruiters identify the most suitable candidates quickly and consistently.

## Features

- Resume parsing and preprocessing
- Job description analysis
- Candidate scoring based on multiple factors
- Resume ranking from best to least suitable
- Configurable scoring pipeline
- Easy-to-extend architecture for custom ranking logic

## Tech Stack

- Python
- Natural Language Processing (NLP)
- Machine Learning based similarity scoring
- Pandas
- NumPy

## Project Structure

```text
DeepMatch/
│
├── app.py                  # Application entry point
├── prepare.py              # Resume preprocessing pipeline
├── scorer.py               # Candidate scoring engine
├── rank.py                 # Ranking logic
├── requirements.txt        # Project dependencies
├── README.md
└── pre_integration_audit.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ancilla13/DeepMatch.git
cd DeepMatch
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Prepare the resumes and job description:

```bash
python prepare.py
```

Generate candidate scores:

```bash
python scorer.py
```

Rank candidates:

```bash
python rank.py
```

Or run the application:

```bash
python app.py
```

## How It Works

1. Resume data is collected and preprocessed.
2. Relevant skills, experience, and keywords are extracted.
3. Each resume is compared against the target job description.
4. A weighted scoring algorithm calculates candidate relevance.
5. Candidates are ranked based on their final scores.

## Future Improvements

- Semantic matching using transformer models
- Support for PDF and DOCX parsing
- Interactive recruiter dashboard
- Explainable AI score breakdown
- ATS compatibility
- REST API for integration

## License

This project is intended for educational and research purposes.
