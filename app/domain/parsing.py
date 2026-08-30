"""Shared raw-JSON extraction + validation helpers for LLM engine outputs.

All three engines prompt for a single JSON object and validate it against a
Pydantic model. This centralises the "strip fences, find the object, load,
validate, raise a clear error" logic they all had copy-pasted.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

ModelT = TypeVar("ModelT", bound=BaseModel)


class RawOutputError(RuntimeError):
    """The raw text did not contain valid JSON matching the target schema."""


def parse_json_object(raw_text: str, model: type[ModelT]) -> ModelT:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise RawOutputError(
            f"LLM response did not contain a JSON object. Raw response:\n{raw_text[:2000]}"
        )

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RawOutputError(
            f"LLM response was not valid JSON: {exc}. Raw response:\n{raw_text[:2000]}"
        ) from exc

    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise RawOutputError(
            f"LLM response JSON did not match the expected schema: {exc}. "
            f"Raw response:\n{raw_text[:2000]}"
        ) from exc
