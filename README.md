# Allie Accuracy Checks

Gemini-based judge pipeline for evaluating Allie bot responses against benchmark CSVs. Classifies solver failures (RAG issues, reasoning errors, input misreads, etc.) and supports multimodal judging with diagram images.

## Features

- Batch judge runs from CSV input with resume support
- Failure taxonomy defined in `judge_pipeline/Prompt_store.py`
- Optional S3 presigned URL to static URL conversion for image inputs
- Parallel Gemini workers with retries

## Project structure

```
allie_accuracy_checks/
├── judge_pipeline/          # Core Python package
├── notebooks/               # Exploratory / utility notebooks
├── samples/                 # Small example input CSV
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
cd allie_accuracy_checks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GOOGLE_GEMINI_API
```

## Run the judge

```bash
cd judge_pipeline

python run_judge.py \
  --input_csv ../samples/sample_input.csv \
  --output_csv ../samples/sample_input_judged.csv
```

### Common flags

| Flag | Description |
|------|-------------|
| `--input_csv` | Input benchmark CSV |
| `--output_csv` | Output path (merged on resume if file exists) |
| `--limit` | Process only first N rows |
| `--skip-static-url` | Skip S3 presigned -> static URL step |
| `--workers` | Parallel judge workers |

## Input CSV columns

The pipeline auto-maps common column aliases. Expected fields include:

- `combined_input` or `user_input` — student doubt text
- `bot_response_solution` — Allie solver response being judged
- `gemini_response` — frontier reference response (optional)
- `qa_complete_context`, `theory_complete_context` — RAG context (optional)
- `image_url` or `static_image_url` — diagram image URL (optional)

Judge output columns: `judge_failure_category`, `judge_reason`, `judge_raw_output`, `judge_parse_error`.

## Notebooks

- `notebooks/s3_presigned_to_static_url.ipynb` — convert presigned S3 URLs to stable public URLs
- `notebooks/PRO_CHECKS_verifier.ipynb` — verifier accuracy checks

## Data policy

Weekly benchmark CSV dumps (`week_*_accuracy/`, `verifier_checks/`, etc.) are **gitignored**. Do not commit proprietary student or production data to a public repository.

## License

Private / internal use unless otherwise specified.
