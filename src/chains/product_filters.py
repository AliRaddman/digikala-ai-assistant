"""Persian product-query filter extraction.

Owner: Benyamin. Includes an LLM implementation and a free offline baseline.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.normalize import normalize
from src.llm.client import CachedLLMClient
from src.retrieval.base import RetrievalFilters

PROMPT_VERSION = "product-filter-v1"

SYSTEM_PROMPT = """You extract search constraints from Persian shopping queries.
Return only the fields in the response schema.

Rules:
- Never invent a numeric constraint that the user did not state.
- All output prices are in Iranian rial. One toman equals ten rial.
- Product rate is on a 0 to 100 scale.
- Keep vague preferences such as cheap, durable, or popular in search_query;
  do not turn them into arbitrary numeric thresholds.
- Remove only explicit structured constraints from search_query; keep the core
  product need and descriptive features.
- Use empty lists and nulls when a constraint is absent.
"""


class ProductFilterPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_query: str = Field(min_length=1)
    price_min_rial: float | None = Field(default=None, ge=0)
    price_max_rial: float | None = Field(default=None, ge=0)
    brands: list[str] = Field(default_factory=list)
    cat1: list[str] = Field(default_factory=list)
    sub_cat: list[str] = Field(default_factory=list)
    min_rate: float | None = Field(default=None, ge=0, le=100)
    min_rate_count: int | None = Field(default=None, ge=0)
    exclude_fake: bool = False

    @field_validator("search_query")
    @classmethod
    def clean_search_query(cls, value: str) -> str:
        value = normalize(value)
        if not value:
            raise ValueError("search_query cannot be empty")
        return value

    @field_validator("brands", "cat1", "sub_cat")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        cleaned = [normalize(value) for value in values]
        return list(dict.fromkeys(value for value in cleaned if value))

    @model_validator(mode="after")
    def validate_price_range(self) -> "ProductFilterPlan":
        if (
            self.price_min_rial is not None
            and self.price_max_rial is not None
            and self.price_min_rial > self.price_max_rial
        ):
            raise ValueError("price_min_rial cannot exceed price_max_rial")
        return self

    def to_retrieval_filters(self) -> RetrievalFilters:
        return RetrievalFilters(
            price_min=self.price_min_rial,
            price_max=self.price_max_rial,
            brands=self.brands or None,
            cat1=self.cat1 or None,
            sub_cat=self.sub_cat or None,
            min_rate=self.min_rate,
            min_rate_count=self.min_rate_count,
            exclude_fake=self.exclude_fake,
        )


class FilterExtractor(Protocol):
    def extract(self, query: str) -> ProductFilterPlan: ...


class LLMFilterExtractor:
    def __init__(self, client: CachedLLMClient) -> None:
        self.client = client

    def extract(self, query: str) -> ProductFilterPlan:
        clean_query = normalize(query)
        result = self.client.generate_structured(
            operation="extract_product_filters",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clean_query},
            ],
            response_model=ProductFilterPlan,
            cache_namespace=PROMPT_VERSION,
        )
        return ProductFilterPlan.model_validate(result.data)


_NUMBER_WORD_VALUES = {
    "صفر": 0,
    "نیم": 0.5,
    "یک": 1,
    "یه": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج": 5,
    "شش": 6,
    "هفت": 7,
    "هشت": 8,
    "نه": 9,
    "ده": 10,
    "یازده": 11,
    "دوازده": 12,
    "سیزده": 13,
    "چهارده": 14,
    "پانزده": 15,
    "شانزده": 16,
    "هفده": 17,
    "هجده": 18,
    "نوزده": 19,
    "بیست": 20,
    "سی": 30,
    "چهل": 40,
    "پنجاه": 50,
    "شصت": 60,
    "هفتاد": 70,
    "هشتاد": 80,
    "نود": 90,
    "صد": 100,
    "یکصد": 100,
    "دویست": 200,
    "سیصد": 300,
    "چهارصد": 400,
    "پانصد": 500,
    "ششصد": 600,
    "هفتصد": 700,
    "هشتصد": 800,
    "نهصد": 900,
}
_NUMBER_WORD = "(?:" + "|".join(
    sorted(map(re.escape, _NUMBER_WORD_VALUES), key=len, reverse=True)
) + ")"
_NUMBER_WORD_SEQUENCE = rf"{_NUMBER_WORD}(?:\s+(?:و\s+)?{_NUMBER_WORD})*"
_NUMBER = rf"(?P<number>\d[\d,]*(?:\.\d+)?|{_NUMBER_WORD_SEQUENCE})"
_SCALE = r"(?P<scale>میلیارد|میلیون|هزار)?"
_CURRENCY = r"(?P<currency>تومان|تومن|ریال)"
_MAX_PRICE_RE = re.compile(
    rf"(?:زیر|کمتر از|حداکثر|تا)\s*{_NUMBER}\s*{_SCALE}\s*{_CURRENCY}"
)
_MIN_PRICE_RE = re.compile(
    rf"(?:بالای|بیشتر از|حداقل)\s*{_NUMBER}\s*{_SCALE}\s*{_CURRENCY}"
)


def _parse_number(value: str) -> float:
    if value[0].isdigit():
        if "," not in value:
            return float(value)
        groups = value.split(",")
        if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]):
            return float("".join(groups))
        return float(value.replace(",", "."))

    tokens = [token for token in value.split() if token != "و"]
    return sum(_NUMBER_WORD_VALUES[token] for token in tokens)


def _price_to_rial(match: re.Match[str]) -> float:
    number = _parse_number(match.group("number"))
    scale = match.group("scale")
    if scale == "هزار":
        number *= 1_000
    elif scale == "میلیون":
        number *= 1_000_000
    elif scale == "میلیارد":
        number *= 1_000_000_000
    if match.group("currency") in {"تومان", "تومن"}:
        number *= 10
    return number


class RuleBasedFilterExtractor:
    """Zero-cost baseline; intentionally conservative rather than clever."""

    def extract(self, query: str) -> ProductFilterPlan:
        clean_query = normalize(query)
        max_match = _MAX_PRICE_RE.search(clean_query)
        min_match = _MIN_PRICE_RE.search(clean_query)
        search_query = _MAX_PRICE_RE.sub(" ", clean_query)
        search_query = _MIN_PRICE_RE.sub(" ", search_query)
        search_query = normalize(search_query) or clean_query
        return ProductFilterPlan(
            search_query=search_query,
            price_min_rial=_price_to_rial(min_match) if min_match else None,
            price_max_rial=_price_to_rial(max_match) if max_match else None,
            exclude_fake=bool(
                re.search(r"(?:اصل|اورجینال|فیک نباش|غیراصل نباش)", clean_query)
            ),
        )
