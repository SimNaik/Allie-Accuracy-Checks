from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JudgeConfig:
    # Gemini model name for "Gemini Pro 3.1"
    gemini_model: str = "gemini-3.1-pro-preview"

    # Environment variable holding the Gemini API key
    gemini_api_key_env: str = "GOOGLE_GEMINI_API"

    # CSV columns expected in the input
    col_user_input: str = "user_input"
    col_reference_response: str = "reference_response"
    col_qa_context: str = "qa_complete_context"
    col_theory_context: str = "theory_complete_context"

    # ground_truth is optional in your CSV. If missing, we can infer it from openai_response / gemini_response.
    col_ground_truth: str = "ground_truth"

    # Optional anchors (present in your trial CSV)
    col_openai_response: str = "openai_response"
    col_gemini_response: str = "frontier_model_response"
    col_openai_gemini_match: str = "openai_gemini_match"

    # Static image URL column for judge multimodal input (from s3_presigned_to_static_url flow)
    col_static_urll: str = "static_urll"
    col_static_image_url: str = "static_image_url"
    col_presigned_image_url: str = "image_url"

    # S3 presigned -> static URL (see notebooks/s3_presigned_to_static_url.ipynb)
    # Override via environment variables — see .env.example
    aws_region: str = os.getenv("AWS_REGION", "ap-south-1")
    s3_source_bucket: str = os.getenv("S3_SOURCE_BUCKET", "")
    s3_dest_bucket: str = os.getenv("S3_DEST_BUCKET", "")
    s3_upload_prefix: str = os.getenv("S3_UPLOAD_PREFIX", "llm_evaluation_framework")
    s3_download_profile: str = os.getenv("S3_DOWNLOAD_PROFILE", "")
    s3_upload_profile: str = os.getenv("S3_UPLOAD_PROFILE", "")
    s3_download_dir: str = ""
    s3_workers: int = 8

    # Output paths
    default_output_csv_name: str = "allie_gemini_judge_output.csv"

    # Generation params
    temperature: float = 0.0
    max_output_tokens: int | None = 4096

    # Gemini reasoning effort:
    # For Gemini 3.x models, thinking_level supports MEDIUM to balance consistency and reasoning depth.
    thinking_level: str = "MEDIUM"

    # Retry params
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0

    # Parallel Gemini judge workers
    judge_workers: int = 3


def load_env_dotenv_path() -> Path:
    """Load .env from the repository root (parent of judge_pipeline/)."""
    return Path(__file__).resolve().parents[1] / ".env"
