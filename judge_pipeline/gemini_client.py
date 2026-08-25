from __future__ import annotations

import mimetypes
import json
import re
import os
import time
import urllib.request
import urllib.parse
from typing import Any, Optional

from google import genai
from google.genai import types

PROMPT_4_FAILURE_CATEGORIES = (
    "RAG_WRONG_INFO",
    "INPUT_MISREAD",
    "RAG_NON_ADHERENCE",
    "RAG_OVERFITTING",
    "REASONING_ERROR",
    "HALLUCINATION",
    "UNCERTAIN",
)


def judge_failure_categories(prompt_name: str = "prompt_4") -> tuple[str, ...]:
    if prompt_name == "prompt_4":
        return PROMPT_4_FAILURE_CATEGORIES
    return PROMPT_4_FAILURE_CATEGORIES


def build_judge_response_json_schema(failure_categories: tuple[str, ...] | list[str]):
    return {
        "type": "object",
        "properties": {
            "failure_category": {
                "type": "string",
                "enum": list(failure_categories),
            },
            "diagnostic": {"type": "string"},
        },
        "required": ["failure_category", "diagnostic"],
        "additionalProperties": False,
    }


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first JSON object found in `text` by brace matching.
    Returns the substring containing the JSON object, or None.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_judge_json(raw_text: str) -> dict[str, Any]:
    """
    Parse the judge output into JSON.
    The prompt asks for JSON only; in practice Gemini sometimes returns
    (a) truncated JSON, (b) JSON with unescaped backslashes from LaTeX (e.g. \\pm),
    or (c) extra text around the JSON.

    So we do best-effort extraction and lightweight repairs, and if we still
    can't decode, we return `_parse_error` plus any fields we can regex-extract.
    """
    if not raw_text:
        return {"_parse_error": "empty_response", "raw_text": raw_text}

    # 1) Prefer extracting inside ```json ... ``` fences.
    fence_match = re.search(r"```json\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1).strip() if fence_match else raw_text.strip()

    # 2) Extract the outermost {...} block using first "{" and last "}".
    # This avoids "brace matching" getting confused by braces inside LaTeX text.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        # No complete JSON found; still try regex extraction.
        parsed = _regex_extract_fields(raw_text)
        parsed["_parse_error"] = "no_json_object_found"
        parsed["raw_text"] = raw_text
        return parsed

    json_str = candidate[start : end + 1].strip()

    def _fix_json_for_common_gemini_issues(s: str) -> str:
        # Escape invalid backslash escapes in JSON strings.
        # Keep valid JSON escapes: \" \\ \/ b f n r t u
        s = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)

        # Quote unquoted object keys: { failure_category: "rag_issue", ... }
        s = re.sub(
            r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):",
            r'\1"\2"\3:',
            s,
        )

        # Remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    # 3) Try strict JSON first.
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e_strict:
        # 4) Try repaired JSON.
        try:
            repaired = _fix_json_for_common_gemini_issues(json_str)
            return json.loads(repaired)
        except json.JSONDecodeError as e_repaired:
            parsed = _regex_extract_fields(raw_text)
            parsed["_parse_error"] = f"json_decode_error: {e_repaired}"
            parsed["raw_text"] = raw_text
            return parsed


def _regex_extract_fields(text: str) -> dict[str, Any]:
    """
    Extract key/value fields even when full JSON decoding fails.
    This is intentionally forgiving and best-effort.
    """
    out: dict[str, Any] = {}

    def _find(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, flags=re.DOTALL)
        return m.group(1).strip() if m else None

    # Safer JSON-string extraction:
    # matches: "key" : "...." where the string body contains valid JSON
    # escapes (\" \\ \n etc). This avoids early termination when the value
    # itself includes quotes.
    def _find_json_string(key: str) -> Optional[str]:
        pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
        return _find(pattern)

    out["failure_category"] = _find_json_string("failure_category") or _find(
        r"failure_category\s*:\s*\"([^\"]+)\""
    )

    # Prompt_4 uses "diagnostic"; older prompts sometimes used
    # "evidence"/"judge_reason"/"reasoning".
    out["diagnostic"] = _find_json_string("diagnostic")
    if out["diagnostic"] is None:
        # Truncated JSON: no closing quote for diagnostic.
        out["diagnostic"] = _find(r'"diagnostic"\s*:\s*"([\s\S]*)$')
    out["evidence"] = _find_json_string("evidence")
    if out["evidence"] is None:
        out["evidence"] = _find(r'"evidence"\s*:\s*"([\s\S]*)$')

    out["judge_reason"] = _find_json_string("judge_reason")
    if out["judge_reason"] is None:
        out["judge_reason"] = _find(r'"judge_reason"\s*:\s*"([\s\S]*)$')

    out["reasoning"] = _find_json_string("reasoning")
    if out["reasoning"] is None:
        out["reasoning"] = _find(r'"reasoning"\s*:\s*"([\s\S]*)$')

    # Older prompts may include this field; keep best-effort extraction.
    alt = _find_json_string("alternative_considered")
    if alt is not None:
        out["alternative_considered"] = alt

    # Clean Nones
    out = {k: v for k, v in out.items() if v is not None}
    return out


def _is_blank_url(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if not s:
        return True
    return s.lower() in {"nan", "none", "n/a"}


def _download_image_bytes(image_url: str, *, timeout_s: int = 30, max_bytes: int = 10 * 1024 * 1024) -> tuple[bytes, str]:
    """
    Download an image from a URL and return (image_bytes, mime_type).
    Used to attach a real multimodal image part to Gemini.
    """
    req = urllib.request.Request(
        image_url.strip(),
        headers={"User-Agent": "Mozilla/5.0 (compatible; llm-eval-framework)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        mime_type = resp.headers.get_content_type()
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            # Truncate rather than hard-fail; avoids total crash for oversized images.
            data = data[:max_bytes]

    def _url_mime_fallback(u: str) -> str | None:
        # `mimetypes.guess_type()` struggles when query params are present, so use the URL path.
        try:
            parsed = urllib.parse.urlparse(u)
            mime_guess = mimetypes.guess_type(parsed.path)[0]
            return mime_guess
        except Exception:
            return None

    def _sniff_mime_from_bytes(b: bytes) -> str | None:
        # Minimal magic-number sniffing to avoid external deps.
        if b.startswith(b"\xFF\xD8\xFF"):
            return "image/jpeg"
        if b.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
            return "image/gif"
        if b.startswith(b"RIFF") and len(b) > 12 and b[8:12] == b"WEBP":
            return "image/webp"
        if b.startswith(b"BM"):
            return "image/bmp"
        if b.startswith(b"II*\x00") or b.startswith(b"MM\x00*"):
            return "image/tiff"
        return None

    mime_lower = (mime_type or "").strip().lower()

    # Gemini is strict: it rejects non-image MIME types (e.g. `binary/octet-stream`).
    # When the server returns a generic/octet-stream type, sanitize it using URL extension
    # and finally by sniffing the image bytes.
    if (
        not mime_lower
        or mime_lower in {"application/octet-stream", "binary/octet-stream"}
        or not mime_lower.startswith("image/")
    ):
        mime_type = _url_mime_fallback(image_url) or ""

    mime_lower = (mime_type or "").strip().lower()
    if not mime_lower.startswith("image/"):
        mime_type = _sniff_mime_from_bytes(data) or "image/jpeg"

    return data, mime_type


def call_gemini_judge(
    prompt_text: str,
    *,
    gemini_api_key: str,
    model: str,
    image_url: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = 4096,
    thinking_level: str = "MEDIUM",
    max_retries: int = 2,
    retry_backoff_seconds: float = 2.0,
    failure_categories: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Call Gemini and return (raw_text, parsed_json_dict).
    """
    client = genai.Client(api_key=gemini_api_key)

    parts: list[types.Part] = [types.Part(text=prompt_text)]
    if image_url and not _is_blank_url(image_url):
        try:
            img_bytes, mime_type = _download_image_bytes(str(image_url))
            # Gemini expects the image as an inline blob for multimodal inputs.
            parts.append(
                types.Part(
                    inline_data=types.Blob(data=img_bytes, mime_type=mime_type)
                )
            )
        except Exception:
            # If image download fails, we still run the judge with text-only prompt.
            pass

    contents = [types.Content(role="user", parts=parts)]

    thinking_level_enum = getattr(types.ThinkingLevel, str(thinking_level).upper(), None)
    if thinking_level_enum is None:
        thinking_level_enum = types.ThinkingLevel.MEDIUM

    categories = tuple(failure_categories or PROMPT_4_FAILURE_CATEGORIES)
    judge_response_json_schema = build_judge_response_json_schema(categories)

    config_with_thinking = types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=["TEXT"],
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level_enum),
        response_mime_type="application/json",
        response_json_schema=judge_response_json_schema,
    )
    config_without_thinking = types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=["TEXT"],
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_json_schema=judge_response_json_schema,
    )

    last_raw: str = ""
    for attempt in range(1, max_retries + 2):
        try:
            config = config_with_thinking if attempt == 1 else config_without_thinking
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            raw_text = resp.text if hasattr(resp, "text") else str(resp)
            parsed = parse_judge_json(raw_text)
            return raw_text, parsed
        except Exception as e:
            last_raw = str(e)
            if attempt <= max_retries:
                time.sleep(retry_backoff_seconds * attempt)
            else:
                return last_raw, {"_call_error": str(e)}

    return last_raw, {"_unreachable": True}

