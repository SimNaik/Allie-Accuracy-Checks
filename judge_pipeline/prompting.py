from __future__ import annotations

import re
from typing import Any

from config import JudgeConfig


def build_judge_prompt(row: dict[str, Any], cfg: JudgeConfig, prompt_text: str) -> str:
    """
    Render Prompt_2 using the placeholders in Prompt_store.py.

    Your prompt_2 template uses conditional placeholder syntax like:
      {user_input if not null, else "See OCR below"}
    which is not valid Python .format syntax, so we implement targeted replacements.
    """

    def _norm(x: Any) -> str:
        if x is None:
            return ""
        s = str(x)
        if s.lower() in {"nan", "none"}:
            return ""
        return s

    user_input = _norm(row.get(cfg.col_user_input, ""))
    ocr_text = _norm(row.get("current_input_image_ocr", ""))

    qa_context = _norm(row.get(cfg.col_qa_context, ""))
    theory_context = _norm(row.get(cfg.col_theory_context, ""))

    # Solver's response to diagnose: column name is often `reference_response`
    # but can also be `solver_response` depending on your CSV.
    reference_response = _norm(row.get(cfg.col_reference_response, "")) or _norm(
        row.get("reference_response", "")
    )

    frontier_model_response = _norm(row.get(cfg.col_gemini_response, "")) or _norm(
        row.get("frontier_model_response", "")
    )

    # Conditional placeholders from your prompt_2:
    # (we replace the whole expression, not just inner tokens)
    prompt_text = prompt_text.replace(
        '{user_input if not null, else "See OCR below"}',
        user_input if user_input else 'See OCR below',
    )
    prompt_text = prompt_text.replace(
        '{current_input_image_ocr if not null, else "N/A"}',
        ocr_text if ocr_text else "N/A",
    )
    prompt_text = prompt_text.replace(
        '{qa_complete_context if not null, else "Not available"}',
        qa_context if qa_context else "Not available",
    )
    prompt_text = prompt_text.replace(
        '{theory_complete_context if not null, else "Not available"}',
        theory_context if theory_context else "Not available",
    )

    # Replace remaining plain placeholders of the form {token}
    replacements: dict[str, str] = {
        "user_input": user_input,
        "current_input_image_ocr": ocr_text,
        "reference_response": reference_response,
        "qa_complete_context": qa_context,
        "theory_complete_context": theory_context,
        "frontier_model_response": frontier_model_response,
    }

    for token, value in replacements.items():
        # Use plain string replacement (not regex) because `value` can
        # contain backslashes from LaTeX, which breaks regex replacement
        # escaping.
        prompt_text = prompt_text.replace("{" + token + "}", value)

    return prompt_text

