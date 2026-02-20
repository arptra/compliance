from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


FORBIDDEN = {"client", "operator", "chatbot", "agent", "support", "num", "<phone>", "<email>", "<url>"}


class NormalizeTicket(BaseModel):
    client_first_message: str
    short_summary: str
    is_complaint: bool
    complaint_category: str
    complaint_subcategory: str | None = None
    product_area: str | None = None
    severity: Literal["low", "medium", "high"]
    keywords: list[str] = Field(min_length=3, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None

    @field_validator("keywords")
    @classmethod
    def no_forbidden_keywords(cls, v: list[str]) -> list[str]:
        bad = [k for k in v if k.lower() in FORBIDDEN]
        if bad:
            raise ValueError(f"Forbidden keywords: {bad}")
        return v


def schema_dict() -> dict:
    return NormalizeTicket.model_json_schema()
