from __future__ import annotations

import argparse
import sys
import threading
import time
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from config import JudgeConfig, load_env_dotenv_path
from gemini_client import call_gemini_judge, judge_failure_categories
from prompting import build_judge_prompt
from static_url import normalize_url, populate_static_urll


COLUMN_ALIASES: dict[str, list[str]] = {
    "user_input": ["combined_input", "user_input_text", "current_input_text", "message"],
    # Solver answer being diagnosed (Allie). Prefer bot_response_solution over
    # gemini_response — gemini is the frontier reference, not the solver.
    "reference_response": [
        "bot_response_solution",
        "solver_response",
        "new_allie_response",
        "reference_response",
    ],
    "frontier_model_response": [
        "gemini_response",
        "FRONTIER_MODEL_RESPONSE",
        "frontier_model_response",
    ],
    "current_input_image_url": ["static_urll", "static_image_url", "image_url"],
}

JUDGE_OUTPUT_COLS = [
    "judge_model_used",
    "judge_failure_category",
    "judge_reason",
    "judge_raw_output",
    "judge_parse_error",
]


def _is_filled(val: Any) -> bool:
    if val is None or pd.isna(val):
        return False
    s = str(val).strip()
    return bool(s) and s.lower() not in {"nan", "none", "n/a", "<na>"}


def row_already_judged(row: pd.Series) -> bool:
    """Skip rows that already have a failure category from a prior run."""
    return _is_filled(row.get("judge_failure_category"))


def load_resume_dataframe(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    limit: int | None = None,
    resume_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load input CSV and merge prior judge (and static_urll) results from output if present."""
    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit).copy()

    out_path = Path(output_csv)
    if not out_path.exists() or out_path.resolve() == Path(input_csv).resolve():
        return df

    prev = pd.read_csv(out_path)
    cols_to_merge = list(resume_cols or [])
    for c in JUDGE_OUTPUT_COLS + ["static_urll"]:
        if c in prev.columns and c not in cols_to_merge:
            cols_to_merge.append(c)

    merge_cols = [c for c in cols_to_merge if c in prev.columns]
    if not merge_cols:
        print(f"Resume: {out_path.name} exists but has no merge columns — starting fresh", flush=True)
        return df

    if "message_id" in df.columns and "message_id" in prev.columns:
        prev_subset = prev[["message_id", *merge_cols]].drop_duplicates("message_id", keep="last")
        df = df.drop(columns=[c for c in merge_cols if c in df.columns], errors="ignore")
        df = df.merge(prev_subset, on="message_id", how="left")
        matched = df[merge_cols[0]].notna().sum() if merge_cols else 0
        print(
            f"Resume: merged {matched}/{len(df)} rows from {out_path.name} on message_id",
            flush=True,
        )
    elif len(prev) == len(df):
        for c in merge_cols:
            df[c] = prev[c].values
        print(f"Resume: copied prior results from {out_path.name} by row index", flush=True)
    else:
        print(
            f"Resume: WARNING — output has {len(prev)} rows, input has {len(df)} rows; "
            "cannot merge safely. Starting judge columns fresh.",
            flush=True,
        )

    return df


def ensure_columns_from_aliases(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    df = df.copy()
    mapped: list[str] = []
    for target, sources in aliases.items():
        if target not in df.columns:
            for src in sources:
                if src in df.columns:
                    df[target] = df[src]
                    mapped.append(f"{target} <- {src}")
                    break
    if mapped:
        print("Mapped columns from aliases: " + ", ".join(mapped), flush=True)
    return df


def infer_ground_truth(row: dict[str, Any], cfg: JudgeConfig) -> str:
    """
    Strict ground-truth inference:
    - If `ground_truth` exists and is non-empty, use it.
    - Otherwise use ONLY `gemini_response` as the benchmark.
    """
    if cfg.col_ground_truth in row and str(row.get(cfg.col_ground_truth, "")).strip():
        return str(row[cfg.col_ground_truth])

    # IMPORTANT: Do not fall back to any other column.
    gemini_resp = str(row.get(cfg.col_gemini_response, "") or "").strip()
    return gemini_resp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini judge pipeline over a CSV.")
    parser.add_argument(
        "--input_csv",
        default="/Users/simrannaik/Desktop/bots/Allie_/Allie Accuracy deep dive- series - 2026-06-08-NNY-series_trial_10.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="/Users/simrannaik/Desktop/bots/Allie_/Allie Accuracy deep dive- series - 2026-06-08-NNY-series_trial_10_judged.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many rows to judge (defaults to all rows in --input_csv)",
    )
    parser.add_argument("--sleep_seconds", type=float, default=0.5)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--build_static_urls",
        action="store_true",
        help="Build static_urll via S3 download/upload for rows missing a static URL",
    )
    parser.add_argument(
        "--static_url_upload_prefix",
        default="llm_evaluation_framework",
        help="S3 prefix in doubt-qc for uploaded static images",
    )
    parser.add_argument(
        "--force_s3_static_urls",
        action="store_true",
        help="Re-run S3 upload even when static_image_url exists (refresh static_urll)",
    )
    parser.add_argument(
        "--force_rejudge",
        action="store_true",
        help="Re-judge all rows even if judge_failure_category is already set",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel Gemini judge workers (default: 3)",
    )
    parser.add_argument(
        "--prompt",
        choices=("prompt_4", "prompt_4"),
        default="prompt_4",
        help="Prompt template from Prompt_store.py (default: prompt_4)",
    )
    args = parser.parse_args()

    # Ensure bots root is on sys.path so imports work.
    this_dir = Path(__file__).resolve().parent  # .../Allie_accuracy_checks/judge_pipeline
    parent_dir = this_dir.parent  # .../Allie_accuracy_checks
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Load env vars (Gemini key)
    load_dotenv(load_env_dotenv_path())

    cfg = JudgeConfig(s3_upload_prefix=args.static_url_upload_prefix)

    from Prompt_store import prompt_4, prompt_4

    prompt_templates = {
        "prompt_4": prompt_4,
    }
    prompt_name = args.prompt
    prompt_text_template = prompt_templates[prompt_name]
    failure_categories = judge_failure_categories(prompt_name)

    gemini_api_key = os.getenv(cfg.gemini_api_key_env)
    if not gemini_api_key:
        raise RuntimeError(
            f"Missing Gemini API key. Expected env var: {cfg.gemini_api_key_env}"
        )

    df = load_resume_dataframe(args.input_csv, args.output_csv, limit=args.limit)
    effective_limit = args.limit if args.limit is not None else len(df)

    df = ensure_columns_from_aliases(df, COLUMN_ALIASES)

    # Build static_urll: always copy static_image_url; optional S3 for presigned URLs
    df = populate_static_urll(
        df,
        cfg,
        output_csv=args.output_csv,
        run_s3=args.build_static_urls,
        force_s3=args.force_s3_static_urls,
    )

    # Legacy export: frontier output was stored in `reference_response` while
    # solver output was in `bot_response_solution`.
    if (
        "bot_response_solution" in df.columns
        and "reference_response" in df.columns
        and "frontier_model_response" not in df.columns
        and "gemini_response" not in df.columns
    ):
        df["frontier_model_response"] = df["reference_response"]
        df["reference_response"] = df["bot_response_solution"]
        print(
            "Mapped legacy columns: "
            "reference_response <- bot_response_solution, "
            "frontier_model_response <- prior reference_response",
            flush=True,
        )

    # ---- Input slimming + required column mapping ----
    # What we will keep + send to the judge prompt/output.
    required_in_cols = [
        "user_input",
        "current_input_image_ocr",
        "reference_response",
        "frontier_model_response",
        "qa_complete_context",
        "theory_complete_context",
        "request_timestamp",
    ]
    missing = [c for c in required_in_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required input columns: {missing}")

    image_cols = (
        cfg.col_static_urll,
        "static_urll",
        "current_input_image_url",
        "static_image_url",
        "image",
    )
    if not any(c in df.columns for c in image_cols):
        raise RuntimeError(
            "Need an image URL column: static_urll, current_input_image_url, static_image_url, or image."
        )

    # Frontier model response aliases already handled above.
    if "frontier_model_response" not in df.columns:
        raise RuntimeError(
            "Need `frontier_model_response` (or legacy `gemini_response` / prior `reference_response`)."
        )

    # Compute min/max date from request_timestamp and attach to every row.
    ts = pd.to_datetime(df["request_timestamp"], errors="coerce")
    if ts.isna().all():
        raise RuntimeError("request_timestamp could not be parsed as datetime.")
    min_dt = ts.min()
    max_dt = ts.max()
    # These date columns are used for downstream grouping / file naming.
    df["min_date"] = min_dt.date().isoformat()
    df["max_date"] = max_dt.date().isoformat()
    # Backward-compatible aliases.
    df["min_response_date"] = df["min_date"]
    df["max_response_date"] = df["max_date"]

    # IMPORTANT: we keep the full input schema in the final output.
    # We only add columns above (min/max dates, plus optional mapped columns),
    # but we do not drop any existing columns from the input CSV.

    total = len(df)
    skipped = sum(
        1
        for _, row in df.iterrows()
        if not args.force_rejudge and row_already_judged(row)
    )
    to_judge = total - skipped
    start_all = time.time()
    model = cfg.gemini_model
    print(
        f"Total rows: {total} | To judge: {to_judge} | Skipped (done): {skipped} | "
        f"workers={args.workers} | prompt={prompt_name} | model={model} | output={args.output_csv}",
        flush=True,
    )

    # Pre-create output columns (so CSV keeps consistent schema)
    for c in JUDGE_OUTPUT_COLS:
        if c not in df.columns:
            df[c] = None

    def _pick_image_url(row_dict: dict[str, Any]) -> str:
        for k in (
            cfg.col_static_urll,
            "static_urll",
            "current_input_image_url",
            "static_image_url",
            "image",
        ):
            s = normalize_url(row_dict.get(k, ""))
            if s:
                return s
        return ""

    def _truncate(s: Any, n: int = 220) -> str:
        s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
        s = s.replace("\n", " ").replace("\r", " ").strip()
        return s[:n]

    def _judge_one_row(task: dict[str, Any]) -> dict[str, Any]:
        idx = task["idx"]
        row_dict = task["row_dict"]
        t0 = time.time()

        prompt_text = build_judge_prompt(
            row=row_dict,
            cfg=cfg,
            prompt_text=prompt_text_template,
        )
        image_url_for_judge = _pick_image_url(row_dict)
        raw_text, parsed = call_gemini_judge(
            prompt_text,
            gemini_api_key=gemini_api_key,
            model=model,
            image_url=image_url_for_judge if image_url_for_judge else None,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            thinking_level=cfg.thinking_level,
            max_retries=args.max_retries,
            retry_backoff_seconds=cfg.retry_backoff_seconds,
            failure_categories=failure_categories,
        )
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

        elapsed = time.time() - t0
        updates: dict[str, Any] = {
            "idx": idx,
            "judge_model_used": model,
            "judge_raw_output": raw_text,
            "elapsed": elapsed,
            "log": "",
        }

        call_error = parsed.get("_call_error") or parsed.get("_unreachable")
        parse_error = parsed.get("_parse_error")

        if call_error:
            updates["judge_parse_error"] = str(call_error)
            if parsed.get("failure_category") is not None:
                updates["judge_failure_category"] = parsed.get("failure_category")
            updates["judge_reason"] = (
                parsed.get("diagnostic")
                or parsed.get("evidence")
                or parsed.get("judge_reason")
                or parsed.get("reasoning")
            )
            updates["log"] = f"CALL_ERROR: {_truncate(call_error)} | {elapsed:.2f}s"
        elif parse_error:
            updates["judge_parse_error"] = str(parse_error)
            if parsed.get("failure_category") is not None:
                updates["judge_failure_category"] = parsed.get("failure_category")
            updates["judge_reason"] = (
                parsed.get("diagnostic")
                or parsed.get("evidence")
                or parsed.get("judge_reason")
                or parsed.get("reasoning")
            )
            cat = updates.get("judge_failure_category")
            reason = updates.get("judge_reason")
            cat_ok = _is_filled(cat)
            reason_ok = _is_filled(reason)
            if cat_ok and reason_ok:
                updates["judge_parse_error"] = None
                updates["log"] = f"DONE_REPAIRED: category={cat} | {elapsed:.2f}s"
            elif cat_ok:
                updates["judge_parse_error"] = "repaired_partial"
                updates["log"] = f"REPAIRED_PARTIAL: category={cat} | {elapsed:.2f}s"
            else:
                updates["log"] = (
                    f"PARSE_ERROR: {_truncate(parse_error)} | raw={_truncate(raw_text)} | {elapsed:.2f}s"
                )
        else:
            updates["judge_failure_category"] = parsed.get("failure_category")
            updates["judge_reason"] = (
                parsed.get("diagnostic")
                or parsed.get("evidence")
                or parsed.get("judge_reason")
                or parsed.get("reasoning")
            )
            updates["log"] = (
                f"DONE: category={updates['judge_failure_category']} | {elapsed:.2f}s"
            )

        return updates

    pending: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        if not args.force_rejudge and row_already_judged(row):
            continue
        pending.append({"idx": i, "row_dict": row.to_dict()})

    scored = 0
    done = 0
    save_lock = threading.Lock()

    if not pending:
        df.to_csv(args.output_csv, index=False)
        print("Nothing to judge — all rows already have judge_failure_category.", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_judge_one_row, task): task for task in pending}
            for fut in as_completed(futures):
                result = fut.result()
                done += 1
                idx = result["idx"]

                with save_lock:
                    for col in JUDGE_OUTPUT_COLS:
                        if col in result:
                            df.at[idx, col] = result[col]
                    df.to_csv(args.output_csv, index=False)
                    scored += 1
                    print(
                        f"[{done}/{len(pending)}] row_index={idx} | {result.get('log', '')}",
                        flush=True,
                    )

    # ---- Summary counts by failure category ----
    if "judge_failure_category" in df.columns:
        vc = df["judge_failure_category"].value_counts(dropna=True)
        desired_order = list(failure_categories)
        print(f"Failure category counts (judge_model={model}):", flush=True)
        total = len(df)
        for cat in desired_order:
            if cat in vc:
                print(f"  {cat}: {int(vc[cat])} ({int(vc[cat])}/{total})", flush=True)
            else:
                print(f"  {cat}: 0 (0/{total})", flush=True)
        # If any unexpected categories exist, show them too.
        extra = [c for c in vc.index.tolist() if c not in set(desired_order)]
        for cat in extra:
            print(f"  {cat}: {int(vc[cat])} ({int(vc[cat])}/{total})", flush=True)

    print(f"Done. Wrote: {args.output_csv} | judge_model={model} | scored={scored} | skipped={skipped}")
    print(
        f"TOTAL elapsed: {time.time() - start_all:.2f}s | avg per row: {(time.time() - start_all)/max(total,1):.2f}s | judge_model={model}",
        flush=True,
    )


if __name__ == "__main__":
    main()

