"""Stage 4 of the live evaluation: section 2 against the real model.

Owner: Ali, 2026-08-30. Runs ProductQAChain over a fixed question/product set
on the real comment index, records the Persian answers, audits every citation
against the supplied evidence, optionally scores each answer with the
grounding judge, and writes one JSON run file.

--dry-run prints the prompt token estimate and spends nothing.
--limit N runs only the first N questions, for the single-question smoke test
that has to pass before the whole set is worth paying for.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.chains.product_qa import ProductQAChain
from src.eval.grounding import LLMGroundingJudge, audit_citations
from src.llm.client import build_openai_client
from src.llm.config import LLMSettings
from src.llm.usage import SQLiteUsageLedger
from src.retrieval.base import build_retriever

# 5 products, each capped at 50 reviews in the index, spread over five cat1
# values and deliberately including products with many not_recommended
# reviews so a "does it deserve buying" question has something to weigh.
PRODUCTS = {
    "262958": "کرم ضد جوش نوروا Actipur Light Tinted (مراقبت پوست)",
    "2526859": "مکعب روبیک مدل AL23 (اسباب بازی)",
    "151716": "روان نویس یونی-بال Eye UB-157 (نوشت افزار)",
    "963758": "تی شرت زنانه طرح شازده کوچولو (لباس زنانه)",
    "803867": "فوم ماینوکسیدیل ۵٪ ماینو کسیدیل (شامپو و مراقبت مو)",
}

# All four example questions from section 2 of docs/PROJECT_BRIEF.md, spread
# so every product gets two and every question type is asked at least twice.
Q_COMPLAINTS = "ایرادهای پرتکرار این محصول چیست؟"
Q_WORTH = "آیا با توجه به تجربه‌ی کاربران ارزش خرید دارد؟"
Q_SATISFIED = "مردم بیشتر از چه چیزی در این محصول راضی بودند؟"
Q_QUALITY = "خریداران درباره‌ی کیفیت این محصول چه گفته‌اند؟"

CASES = [
    ("qa01", "262958", Q_COMPLAINTS),
    ("qa02", "262958", Q_WORTH),
    ("qa03", "2526859", Q_SATISFIED),
    ("qa04", "2526859", Q_QUALITY),
    ("qa05", "151716", Q_COMPLAINTS),
    ("qa06", "151716", Q_WORTH),
    ("qa07", "963758", Q_QUALITY),
    ("qa08", "963758", Q_SATISFIED),
    ("qa09", "803867", Q_WORTH),
    ("qa10", "803867", Q_COMPLAINTS),
]

# Control: present in the product catalogue, absent from the comment index.
# The chain must answer without any API call at all.
NO_REVIEW_PRODUCT = None  # resolved at runtime, see _pick_no_review_product


def _pick_no_review_product() -> str | None:
    import pandas as pd

    map_path = Path("data/indexes/comments_product_map_v1.json")
    meta_path = Path("data/indexes/products_meta_v1.parquet")
    if not (map_path.exists() and meta_path.exists()):
        return None
    with map_path.open(encoding="utf-8") as handle:
        with_reviews = set(json.load(handle))
    ids = pd.read_parquet(meta_path, columns=["product_id"])["product_id"].astype(str)
    for product_id in ids:
        if product_id not in with_reviews:
            return product_id
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/eval/runs/product_qa_real_v1.json"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--retriever-mode", default="real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = CASES[: args.limit] if args.limit else CASES
    retriever = build_retriever("comment", mode=args.retriever_mode)

    if args.dry_run:
        _dry_run(retriever, cases, args.top_k)
        return

    settings = LLMSettings.from_env()
    client = build_openai_client(settings)
    ledger = SQLiteUsageLedger(settings.usage_path)
    checkpoint = ledger.checkpoint()
    chain = ProductQAChain(retriever=retriever, client=client, max_evidence=args.top_k)
    judge = LLMGroundingJudge(client) if args.judge else None

    items = []
    for case_id, product_id, question in cases:
        started = time.perf_counter()
        try:
            result = chain.run(question, product_id)
        except Exception as exc:  # noqa: BLE001 -- recorded, never retried blindly
            items.append(
                {
                    "case_id": case_id,
                    "product_id": product_id,
                    "product": PRODUCTS.get(product_id),
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        latency_ms = (time.perf_counter() - started) * 1000
        rendered = result.render_fa()
        audit = audit_citations(rendered, result.evidence)
        item = {
            "case_id": case_id,
            "product_id": product_id,
            "product": PRODUCTS.get(product_id),
            "question": question,
            "evidence_count": len(result.evidence),
            "answer_fa": result.answer.answer_fa,
            "sufficient_evidence": result.answer.sufficient_evidence,
            "claims": [claim.model_dump(mode="json") for claim in result.answer.claims],
            "rendered_fa": rendered,
            "citation_audit": audit.model_dump(mode="json"),
            "citation_hallucination": result.citation_hallucination.model_dump(mode="json"),
            "hallucination_warning_fa": result.hallucination_warning_fa(),
            "latency_ms": latency_ms,
            "evidence": [item.as_dict() for item in result.evidence],
        }
        if judge is not None:
            try:
                run = judge.evaluate(
                    question=question,
                    answer=rendered,
                    evidence=result.evidence,
                )
                item["judgment"] = run.as_dict()
            except Exception as exc:  # noqa: BLE001
                item["judge_error"] = f"{type(exc).__name__}: {exc}"
        items.append(item)

    control = _no_review_control(retriever, chain, ledger)
    usage = ledger.summary(after_id=checkpoint)

    scored = [i for i in items if "judgment" in i]
    halluc = [i["citation_hallucination"] for i in items if "citation_hallucination" in i]
    generated_ids = sum(h["generated_ids"] for h in halluc)
    invented_ids = sum(len(h["invented_ids"]) for h in halluc)
    answers_with_invented = sum(1 for h in halluc if h["invented_ids"])
    audited = [i for i in items if i.get("citation_audit", {}).get("integrity_score") is not None]
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "configuration": {
            "prompt_version": __import__("src.chains.product_qa", fromlist=["x"]).PROMPT_VERSION,
            "model": settings.model,
            "base_url": settings.base_url,
            "top_k": args.top_k,
            "retriever_mode": args.retriever_mode,
            "judge": bool(judge),
        },
        "summary": {
            "cases": len(items),
            "errored": sum("error" in i for i in items),
            "insufficient_evidence": sum(i.get("sufficient_evidence") is False for i in items),
            "citation_integrity_rate": (
                sum(i["citation_audit"]["integrity_score"] for i in audited) / len(audited)
                if audited else None
            ),
            "claims_total": sum(len(i.get("claims", [])) for i in items),
            # Grounding, measured on the citations themselves rather than
            # inferred from the judge. Both rates answer the section-4
            # grounding requirement of docs/PROJECT_BRIEF.md directly.
            "generated_comment_ids": generated_ids,
            "invented_comment_ids": invented_ids,
            "invented_id_rate": (invented_ids / generated_ids) if generated_ids else None,
            "answers_with_any_invented_id": answers_with_invented,
            "answer_hallucination_rate": (
                answers_with_invented / len(halluc) if halluc else None
            ),
            "claims_dropped_unsupported": sum(len(h["dropped_claims"]) for h in halluc),
            "judged": len(scored),
            "judge_errors": sum("judge_error" in i for i in items),
            "mean_grounding_score": (
                sum(i["judgment"]["judgment"]["grounding_score"] for i in scored) / len(scored)
                if scored else None
            ),
            "mean_relevance_score": (
                sum(i["judgment"]["judgment"]["relevance_score"] for i in scored) / len(scored)
                if scored else None
            ),
        },
        "no_review_control": control,
        "llm_usage": usage,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("summary", "no_review_control", "llm_usage")},
                     ensure_ascii=False, indent=2))


def _no_review_control(retriever, chain, ledger) -> dict[str, object]:
    """A product with no reviews must be answered with zero API calls."""
    product_id = _pick_no_review_product()
    if product_id is None:
        return {"skipped": "could not resolve a product without reviews"}
    before = ledger.checkpoint()
    result = chain.run(Q_COMPLAINTS, product_id)
    after = ledger.summary(after_id=before)
    return {
        "product_id": product_id,
        "evidence_count": len(result.evidence),
        "answer_fa": result.answer.answer_fa,
        "requests_recorded": after.get("logical_requests"),
        "api_calls": after.get("api_calls"),
        "cost_usd": after.get("cost_usd"),
    }


def _dry_run(retriever, cases, top_k: int) -> None:
    from src.chains.product_qa import SYSTEM_PROMPT, _evidence_block
    from src.retrieval.base import RetrievalFilters

    total_chars = 0
    for case_id, product_id, question in cases:
        evidence = retriever.retrieve(
            question, top_k=top_k, filters=RetrievalFilters(product_ids=[product_id])
        )
        block = "\n\n".join(_evidence_block(item) for item in evidence)
        prompt = SYSTEM_PROMPT + f"Question: {question}\n\nEvidence:\n{block}"
        total_chars += len(prompt)
        print(f"{case_id} {product_id} evidence={len(evidence):2d} prompt_chars={len(prompt):6d}")
    print(f"\ntotal prompt chars: {total_chars}")


if __name__ == "__main__":
    main()
