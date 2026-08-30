"""Reproducible offline benchmark for the semantic-cache control flow.

This benchmark intentionally uses a deterministic encoder and a delayed fake
provider: it measures cache lookup/accounting without a network key or model
download. It must not be presented as a live model-quality A/B. The generated
report separately includes a projection based on this repository's recorded
real API and local query-encoding p50 values.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from src.llm.cache import SQLiteLLMCache
from src.llm.client import CachedLLMClient, ProviderResult
from src.llm.semantic_cache import SemanticCacheRequest
from src.llm.types import TokenUsage
from src.llm.usage import SQLiteUsageLedger


class BenchmarkAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str


_CASES = [
    ("کیف ارزان برای مدرسه", "school_bag", [1.0, 0.0, 0.0, 0.0]),
    ("برای مدرسه کیف کم قیمت می خواهم", "school_bag", [1.0, 0.0, 0.0, 0.0]),
    ("آیا زیپ این کوله خراب می شود؟", "zip_quality", [0.0, 1.0, 0.0, 0.0]),
    ("کاربرها از خرابی زیپ گفته اند؟", "zip_quality", [0.0, 1.0, 0.0, 0.0]),
    ("این دو هدفون را مقایسه کن", "compare", [0.0, 0.0, 1.0, 0.0]),
    ("فرق این دو هدفون چیست؟", "compare", [0.0, 0.0, 1.0, 0.0]),
    ("شکایت اصلی خریداران چیست؟", "complaints", [0.0, 0.0, 0.0, 1.0]),
    ("مهم ترین ایراد از نظر کاربران چیست؟", "complaints", [0.0, 0.0, 0.0, 1.0]),
]


class DeterministicBenchmarkEncoder:
    """Known vectors keep the CI benchmark reproducible and offline."""

    model_id = "deterministic-benchmark-v1"

    def __init__(self) -> None:
        self._vectors = {text: vector for text, _, vector in _CASES}

    def encode(self, text: str) -> Sequence[float]:
        return self._vectors[text]


class DelayedBenchmarkProvider:
    def __init__(self, delay_ms: float) -> None:
        self.delay_seconds = delay_ms / 1_000
        self.calls = 0
        self._intents = {text: intent for text, intent, _ in _CASES}

    def generate_structured(self, *, model, messages, response_model):
        self.calls += 1
        time.sleep(self.delay_seconds)
        query = messages[-1]["content"]
        return ProviderResult(
            data={"intent": self._intents[query]},
            model=model,
            request_id=f"benchmark_{self.calls}",
            usage=TokenUsage(input_tokens=1_000, output_tokens=100),
        )


def _run(*, semantic: bool, delay_ms: float, root: Path) -> dict[str, object]:
    provider = DelayedBenchmarkProvider(delay_ms)
    ledger = SQLiteUsageLedger(root / "usage.sqlite3")
    client = CachedLLMClient(
        provider=provider,
        model="gpt-4o-mini",
        cache=SQLiteLLMCache(root / "cache.sqlite3"),
        ledger=ledger,
        semantic_encoder=DeterministicBenchmarkEncoder() if semantic else None,
        semantic_threshold=0.96,
    )

    results = []
    started = time.perf_counter()
    for query, expected_intent, _ in _CASES:
        result = client.generate_structured(
            operation="semantic_cache_benchmark",
            messages=[{"role": "user", "content": query}],
            response_model=BenchmarkAnswer,
            cache_namespace="semantic-cache-benchmark-v1",
            semantic_cache=SemanticCacheRequest(
                text=query,
                guard={"system_prompt": "classify the shopping intent"},
            ),
        )
        answer = BenchmarkAnswer.model_validate(result.data)
        if answer.intent != expected_intent:
            raise AssertionError(
                f"unsafe cache reuse: expected {expected_intent}, got {answer.intent}"
            )
        results.append(result)
    wall_ms = (time.perf_counter() - started) * 1_000
    summary = ledger.summary()
    semantic_latencies = [
        result.latency_ms for result in results if result.cache_type == "semantic"
    ]
    return {
        "wall_latency_ms": wall_ms,
        "mean_request_latency_ms": summary["mean_latency_ms"],
        "mean_semantic_hit_latency_ms": (
            sum(semantic_latencies) / len(semantic_latencies)
            if semantic_latencies
            else None
        ),
        "logical_requests": summary["logical_requests"],
        "api_calls": summary["api_calls"],
        "exact_cache_hits": summary["exact_cache_hits"],
        "semantic_cache_hits": summary["semantic_cache_hits"],
        "cost_usd": summary["cost_usd"],
        "saved_cost_usd": summary["saved_cost_usd"],
    }


def build_report(delay_ms: float = 50.0) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = _run(
            semantic=False,
            delay_ms=delay_ms,
            root=root / "exact_only",
        )
        semantic = _run(
            semantic=True,
            delay_ms=delay_ms,
            root=root / "semantic",
        )

    baseline_cost = float(baseline["cost_usd"])
    semantic_cost = float(semantic["cost_usd"])
    baseline_wall = float(baseline["wall_latency_ms"])
    semantic_wall = float(semantic["wall_latency_ms"])
    baseline_mean = float(baseline["mean_request_latency_ms"])
    hit_mean = float(semantic["mean_semantic_hit_latency_ms"] or 0.0)

    live_qa_p50_ms = 3_113.0
    local_encode_p50_ms = 18.9
    return {
        "benchmark": "semantic-cache-offline-v1",
        "method": {
            "provider": "fake provider with fixed per-call delay",
            "encoder": "deterministic vectors for four Persian paraphrase pairs",
            "threshold": 0.96,
            "provider_delay_ms": delay_ms,
            "warning": (
                "Infrastructure benchmark only; it does not measure live "
                "embedding false-hit/false-miss quality."
            ),
        },
        "baseline_exact_only": baseline,
        "semantic_cache": semantic,
        "measured_reduction": {
            "api_calls_avoided": int(baseline["api_calls"])
            - int(semantic["api_calls"]),
            "cost_reduction_pct": 100 * (baseline_cost - semantic_cost) / baseline_cost,
            "wall_latency_reduction_pct": 100
            * (baseline_wall - semantic_wall)
            / baseline_wall,
            "semantic_hit_latency_reduction_vs_baseline_mean_pct": 100
            * (baseline_mean - hit_mean)
            / baseline_mean,
        },
        "production_projection_from_recorded_metrics": {
            "historical_product_qa_api_p50_ms": live_qa_p50_ms,
            "historical_local_query_encode_p50_ms": local_encode_p50_ms,
            "projected_per_hit_latency_reduction_pct": 100
            * (live_qa_p50_ms - local_encode_p50_ms)
            / live_qa_p50_ms,
            "historical_product_qa_mean_cost_per_api_call_usd": 0.005416 / 11,
            "local_embedding_cost_per_hit_usd": 0.0,
            "warning": (
                "Projection combines previously recorded components; rerun "
                "with the configured embedding model and live traffic before "
                "claiming a production A/B result."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.delay_ms)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
