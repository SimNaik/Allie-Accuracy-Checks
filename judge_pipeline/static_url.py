from __future__ import annotations

import os
import re
import tempfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from config import JudgeConfig

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore


def extract_image_key(url: str) -> str:
    """Extract file name like AD-2026-07-27-20-42-30-945.jpeg from a presigned URL."""
    if not isinstance(url, str) or not url.strip():
        return ""
    match = re.search(r"/([^/?#]+\.[A-Za-z0-9]{3,4})(?:\?|$)", url.strip())
    return match.group(1) if match else ""


def normalize_url(url: Any) -> str:
    if url is None or pd.isna(url):
        return ""
    s = str(url).strip()
    if not s or s.lower() in {"nan", "none", "n/a", "<na>"}:
        return ""
    if " " in s:
        s = urllib.parse.quote(s, safe=":/?#[]@!$&'()*+,;=")
    return s


def _pick_presigned_url(row: pd.Series, cfg: JudgeConfig) -> str:
    for col in (cfg.col_presigned_image_url, "current_input_image_url", "image_url"):
        if col in row.index:
            s = normalize_url(row.get(col))
            if s and "X-Amz-" in s:
                return s
    for col in (cfg.col_presigned_image_url, "current_input_image_url", "image_url"):
        if col in row.index:
            s = normalize_url(row.get(col))
            if s:
                return s
    return ""


def _copy_existing_static(row: pd.Series, cfg: JudgeConfig) -> str:
    for col in (cfg.col_static_image_url, "static_image_url"):
        if col in row.index:
            s = normalize_url(row.get(col))
            if s:
                return s
    return ""


def _download_from_prod(s3_client, image_key: str, local_dir: str, cfg: JudgeConfig) -> str:
    if not image_key:
        return ""
    decoded_key = urllib.parse.unquote(image_key.strip())
    local_filename = os.path.basename(decoded_key)
    local_path = os.path.join(local_dir, local_filename)
    try:
        s3_client.download_file(cfg.s3_source_bucket, decoded_key, local_path)
        return os.path.abspath(local_path)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", str(e))
        print(f"  download failed {decoded_key}: {code}", flush=True)
        return ""


def _upload_and_get_url(s3_client, local_path: str, image_key: str, cfg: JudgeConfig) -> str:
    if not local_path or not os.path.exists(local_path):
        return ""
    decoded_key = urllib.parse.unquote(image_key.strip())
    prefix = cfg.s3_upload_prefix.strip("/")
    s3_key = f"{prefix}/{decoded_key}" if prefix else decoded_key
    try:
        s3_client.upload_file(local_path, cfg.s3_dest_bucket, s3_key)
        return (
            f"https://{cfg.s3_dest_bucket}.s3.{cfg.aws_region}.amazonaws.com/{s3_key}"
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", str(e))
        print(f"  upload failed {decoded_key}: {code}", flush=True)
        return ""


def populate_static_urll(
    df: pd.DataFrame,
    cfg: JudgeConfig,
    *,
    output_csv: str | Path | None = None,
    run_s3: bool = False,
    force_s3: bool = False,
) -> pd.DataFrame:
    """
    Fill `static_urll` for judge multimodal input.

    Priority per row (skip if static_urll already set):
    1. Copy existing `static_image_url` (space-safe encoding)
    2. S3 download from prod + upload to doubt-qc (presigned `image_url` / `current_input_image_url`)
    """
    df = df.copy()
    col = cfg.col_static_urll
    if col not in df.columns:
        df[col] = pd.NA

    # Pass 1: copy existing static_image_url where possible
    copied = 0
    for idx, row in df.iterrows():
        if normalize_url(row.get(col)):
            continue
        existing = _copy_existing_static(row, cfg)
        if existing and not force_s3:
            df.at[idx, col] = existing
            copied += 1
    if copied:
        print(f"static_urll: copied {copied} rows from static_image_url", flush=True)

    # Pass 2: S3 pipeline for rows still missing static_urll but with presigned URL
    if not run_s3:
        filled = df[col].notna() & (df[col].astype(str).str.strip() != "")
        print(
            f"static_urll: {filled.sum()}/{len(df)} rows filled (S3 upload skipped; use --build_static_urls to refresh presigned URLs)",
            flush=True,
        )
        if output_csv:
            df.to_csv(output_csv, index=False)
        return df

    if boto3 is None:
        pending = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if pending.any():
            print(
                "static_urll: boto3 not installed — skipping S3 upload for remaining rows",
                flush=True,
            )
        return df

    need_s3: list[tuple[Any, str, str]] = []
    for idx, row in df.iterrows():
        if normalize_url(row.get(col)):
            continue
        presigned = _pick_presigned_url(row, cfg)
        if not presigned:
            continue
        image_key = extract_image_key(presigned)
        if image_key:
            need_s3.append((idx, image_key, presigned))

    if not need_s3:
        print(f"static_urll: nothing left to build via S3", flush=True)
        return df

    print(
        f"static_urll: building {len(need_s3)} URLs via S3 "
        f"(download {cfg.s3_source_bucket} -> upload {cfg.s3_dest_bucket}/{cfg.s3_upload_prefix})",
        flush=True,
    )

    download_dir = cfg.s3_download_dir or tempfile.mkdtemp(prefix="judge_static_url_")
    os.makedirs(download_dir, exist_ok=True)

    download_session = boto3.Session(
        profile_name=cfg.s3_download_profile,
        region_name=cfg.aws_region,
    )
    upload_session = boto3.Session(
        profile_name=cfg.s3_upload_profile,
        region_name=cfg.aws_region,
    )
    s3_download = download_session.client("s3")
    s3_upload = upload_session.client("s3")

    def _process_one(item: tuple[Any, str, str]) -> tuple[Any, str]:
        idx, image_key, _presigned = item
        local_path = _download_from_prod(s3_download, image_key, download_dir, cfg)
        if not local_path:
            return idx, ""
        return idx, _upload_and_get_url(s3_upload, local_path, image_key, cfg)

    done = 0
    with ThreadPoolExecutor(max_workers=cfg.s3_workers) as executor:
        futures = {executor.submit(_process_one, item): item[0] for item in need_s3}
        for fut in as_completed(futures):
            idx, static_url = fut.result()
            if static_url:
                df.at[idx, col] = static_url
            done += 1
            if output_csv and done % 10 == 0:
                df.to_csv(output_csv, index=False)
            if done % 25 == 0 or done == len(need_s3):
                filled = df[col].notna() & (df[col].astype(str).str.strip() != "")
                print(
                    f"  static_urll S3 progress: {done}/{len(need_s3)} | total filled: {filled.sum()}",
                    flush=True,
                )

    if output_csv:
        df.to_csv(output_csv, index=False)

    filled = df[col].notna() & (df[col].astype(str).str.strip() != "")
    print(
        f"static_urll: done — {filled.sum()}/{len(df)} rows have a static URL",
        flush=True,
    )
    return df
