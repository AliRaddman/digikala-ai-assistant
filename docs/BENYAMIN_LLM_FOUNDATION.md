# Benyamin — LLM foundation

Owner: Benyamin.

This document tracks Benyamin's LLM, evaluation and orchestration work as it
exists on the integrated system, not only the original foundation checkpoint:

- OpenAI Responses API adapter with Structured Outputs
- SQLite exact-match cache plus an opt-in guarded semantic cache
- per-request token, latency, cost and cache-savings ledger
- conservative zero-cost Persian filter-extraction baseline
- LLM Persian filter extractor
- product-discovery chain over the shared `Retriever` contract
- reproducible discovery evaluation over JSONL queries
- chain and end-to-end latency reports with mean, p50, p95 and max
- checkpoint-scoped API call, cache, token and USD accounting
- citation-integrity audit plus an optional structured grounding judge
- default orchestration of all four mandatory capabilities
- real comment retrieval shared by Product QA and Product Comparison
- live Product Discovery and Product QA evaluation with recorded API usage
- a 25-item human-labeling handoff for validating the LLM judge

## Offline demo

```bash
python -m src.chains.product_discovery \
  "شلوار جین مردانه راحت زیر ۵۰۰ هزار تومن" --top-k 5
```

The default uses `MockRetriever` and the free rule-based filter baseline. It
does not need an API key.

## Live structured extraction

Set `LLM_API_KEY` (or `OPENAI_API_KEY`) and optionally the variables already
documented in `.env.example`, then run:

```bash
python -m src.chains.product_discovery \
  "یه کیف دوشی چرم که خریدارها ازش راضی باشن" --use-llm
```

The default cache and usage databases are ignored by the repository's existing
`/data/*` rule:

- `data/cache/llm_cache.sqlite3`
- `data/cache/llm_usage.sqlite3`

No prompt or response text is stored in the usage ledger. The exact-response
cache necessarily stores the validated structured JSON result; its key is a
SHA-256 hash of the model, messages, schema and prompt version.

Semantic lookup is disabled by default. Enable it with:

```bash
export LLM_SEMANTIC_CACHE_ENABLED=true
export LLM_SEMANTIC_CACHE_MODEL=intfloat/multilingual-e5-base
export LLM_SEMANTIC_CACHE_THRESHOLD=0.96
```

The encoder loads lazily. Exact lookup runs first and pays no embedding cost.
On an exact miss, eligible callers embed only the user text; model, prompt
namespace, response schema and caller-supplied guard must match exactly. The
guard carries evidence and product context for QA/comparison, answer and
evidence for the grounding judge. This prevents a similar question from
reusing an answer across products or evidence snapshots. LLM filter extraction
is deliberately excluded: embedding similarity is not safe for numeric
constraints such as "under 500k" versus "under 600k".

`cache_type` distinguishes `none`, `exact` and `semantic`; semantic hits also
record cosine similarity. The ledger summary exposes exact and semantic hit
counts separately. SQLite stores embeddings as normalized float32 blobs and
never puts prompt text in the usage ledger.

The reproducible offline benchmark is:

```bash
python -m scripts.benchmark_semantic_cache \
  --output data/eval/semantic_cache_offline_benchmark_v1.json
```

With four paraphrase hits among eight unique requests, it avoided 4 API calls,
reduced estimated cost by 50%, wall latency by 47.3%, and per-hit latency by
97.9%. It uses a deterministic test encoder and fixed-delay provider, so it is
an infrastructure benchmark, not a live semantic-quality claim. Combining the
previously measured warm local query-encoding p50 (18.9 ms) with Product QA API
p50 (3,113 ms) projects 99.4% latency reduction per warm hit; a live A/B is
still required before reporting a production result.

Pricing is currently configured for `gpt-4o-mini` and its snapshots. Selecting
an unknown model raises an error instead of silently recording a false zero
cost. Add reviewed pricing before enabling another model.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests are entirely offline and do not spend API credit.

## Offline evaluation

Run all 36 current product-discovery queries without an API key:

```bash
python -m src.eval.harness \
  --input data/eval/queries_v1.jsonl \
  --retriever-mode mock \
  --top-k 5 \
  --output data/eval/runs/benyamin_discovery_mock_v1.json
```

The console prints the compact summary and `--output` stores the full report,
including each query's filter plan, retrieved IDs, answer, citations, latency,
constraint audit and any failure. One broken query is recorded as an error and
does not discard the rest of the run.

The report separates:

- `chain_latency_ms`: filter extraction + retrieval + answer rendering
- `end_to_end_latency_ms`: chain work + audits + optional semantic judge
- `llm_usage`: only LLM ledger rows created after this run's checkpoint

Percentiles use linear interpolation over per-query or per-request latency.
The LLM usage report keeps logical requests, actual API calls, cache hits,
input/cached-input/output tokens, billed USD, saved USD, and p50/p95 latency.

## Live filter extraction and grounding judge

After setting `LLM_API_KEY`, both model-backed stages can be enabled explicitly:

```bash
python -m src.eval.harness \
  --input data/eval/queries_v1.jsonl \
  --qrels data/eval/qrels_d50_v2_labeled.csv \
  --retriever-mode real \
  --use-llm-filters \
  --judge-grounding \
  --output data/eval/runs/benyamin_discovery_real_judged_v1.json
```

`--judge-grounding` makes one cached structured-output request per unique
question/answer/evidence bundle. The prompt is versioned as
`grounding-judge-v1`, treats retrieved text as untrusted data, forbids outside
knowledge, and asks for separate relevance and grounding scores from 1 to 5.
The implementation rejects any judge output that cites an evidence ID not
present in the supplied context.

Citation integrity and semantic grounding are intentionally separate:

- citation integrity checks that citation IDs exist in retrieved evidence;
- the LLM judge checks whether the cited evidence actually supports each claim.

An existing citation is not counted as proof of entailment. The report's
`fully_supported_claim_rate` requires the judge to mark a claim `supported` and
attach at least one supplied evidence ID.

## Retrieval labels

The query JSONL contains query text, intent and expected broad subcategory but
keeps gold judgments in Ali's separate graded qrels CSV. The harness accepts
that file through `--qrels`, attaches relevant product IDs by `query_id`, and
uses the 0/1/2 relevance grades for nDCG. `--min-relevance` controls which
grades count as relevant for Recall and MRR and defaults to 1.

Qrels are pooled judgments, not complete catalogue labels. Every report also
includes `judgement_coverage_at_k` and the count of queries with no judged
returned result. When a new run retrieves products outside the original pool,
the harness warns that treating those unjudged items as non-relevant can make
Recall, MRR and nDCG strongly pessimistic. Add the new run to the judging pool
before presenting those metrics as a fair system comparison.

Inline gold IDs remain supported for small or standalone evaluation sets:

```json
{"query_id":"q001","query":"...","intent":"simple","sub_cat":"clothe","relevant_product_ids":["3901234","7712045"]}
```

The LLM judge still needs validation against the team's independent human
labels. `data/eval/human/labels_v1.csv` contains 25 stratified cases, but its
`grounding_1_5` and `relevance_1_5` columns are currently empty. After an
independent teammate fills both columns, run
`scripts/compare_human_vs_judge.py` to report Cohen's kappa, correlations and
disagreement cases. Judge scores alone are not a substitute for human
evaluation.

## Runtime artifacts and integrated status

Product retrieval supports BM25, dense FAISS and hybrid RRF through
`build_retriever("product", mode="real")`; hybrid is the configured default.
Comment retrieval is also implemented through
`build_retriever("comment", mode="real")`. Product QA and Product Comparison
share the same CommentRetriever instance so the 10.6 GB embedding memmap is
not loaded twice.

Real mode still requires the versioned product/comment metadata and index
artifacts documented in `README.md`. These large artifacts live outside Git
and must be copied into `data/indexes/`; code, evaluation inputs and recorded
reports remain versioned in the repository.

Live API evaluation was completed on 2026-08-30 and is recorded under
`data/eval/runs/`. The current aggregate ledger reports 169 logical requests,
93 API calls, 76 cache hits, 153,405 input tokens, 30,811 output tokens and an
estimated cost of $0.041497. Dollar figures use the configured OpenAI rates;
the requests went through the Metis gateway, whose actual billing rate was not
available to the team.

The final 36-query Discovery judge run completed without judge errors and
reported mean grounding 4.31/5 and relevance 4.78/5. The 10-case Product QA
run reported grounding 4.30/5 and relevance 5.00/5. These are model-judge
scores and must remain separate from the pending independent human agreement
measurement described above.

That Product QA report is also the historical v3 citation baseline: 5 of 123
generated comment IDs were absent from evidence (4.1%), affecting 3 of 10
answers. Product QA now uses the
`product-qa-v4-evidence-bound-citations` schema, whose per-request enum permits
only IDs supplied in that request's evidence. The existing quarantine remains
as defence in depth. Offline schema and regression tests verify that fabricated
IDs cannot cross the structured-output boundary; a new live v4 run is still
required before reporting a new raw model hallucination rate.

## Real BM25 product-discovery checkpoint

The first real end-to-end run was completed on 2026-08-26 with Ali's full
product metadata and BM25 artifacts from Drive:

- metadata: 948,352 rows and 10 columns
- BM25 matrix: 948,352 x 372,199
- vocabulary: 372,199 terms
- row and vocabulary dimensions matched exactly
- backend: `bm25`
- 36 queries, `top_k=10`, rule-based filter extractor, no LLM calls

Measured operational results are versioned in
`data/eval/benyamin_discovery_bm25_real_v1.csv`:

- successful queries: 36/36
- empty-result queries: 0
- warm mean chain latency after the first query: 8.22 ms
- p50 chain latency: 7.33 ms
- p95 chain latency: 18.24 ms
- first-query cold load: 2.94 s
- expected-subcategory match among returned products: 82.22%
- explicit filter pass rate: 100%
- citation integrity: 100%

The complete per-query report, including answers, retrieved product IDs,
filter plans, latency and audits, is stored at
`data/eval/runs/benyamin_discovery_bm25_real_v1.json`.

The separate depth-50 qrels covered only 20 of the 360 returned top-10 items
(5.56%), and 20 queries had no judged returned result. Recall, MRR and nDCG
from this run are therefore not reportable: the new full-index BM25 run must be
added to the candidate pool and its previously unseen products judged first.

This run also exposed a sampling artefact in the earlier failure analysis.
Query `q016` (a Dostoevsky book from Nashr-e Cheshmeh) had no exact answer in
the 50k evaluation pool, but the full index returned product `10888139`, an
exact Dostoevsky/Cheshmeh match, at rank 1. "No answer in the pool" must not be
presented as "no answer in the full catalogue."

## Orchestrator checkpoint

`src/orchestrator.py` owns intent routing and chain invocation through the
shared chain and retriever contracts. Its stable intents are:

- `product_discovery`
- `product_qa`
- `product_comparison`
- `category_analytics`

The free rule-based router is the auditable baseline. Explicit discovery
language takes precedence over satisfaction words, so a request such as
"find me a bag buyers liked" is not mistaken for Q&A about one product.
Category analytics and comparisons have stronger dedicated signals.

All four handlers are registered in the default build:

- Product Discovery uses the shared product retriever and the rule-based
  filter baseline.
- Product QA uses product-scoped comment evidence and requires a model only
  when an answer must be generated.
- Product Comparison combines exact product lookup, product-scoped reviews
  and an optional model inference. Facts and evidence still render without an
  API key.
- Category Analytics computes its tables directly from the cleaned Parquet
  data and uses the model only for optional narration.

Missing product IDs return `needs_input`. The generic
`dependency_unavailable` path remains part of the stable contract for custom
or partial orchestrator builds whose handler map omits a route. Handler
exceptions are isolated as structured `error` results rather than crashing
the whole assistant.

Offline demo:

```bash
python -m src.orchestrator \
  "یک کیف روزمره زیر ۲۰۰ هزار تومان معرفی کن" \
  --retriever-mode mock --top-k 3
```

Product Q&A routing:

```bash
python -m src.orchestrator \
  "ایرادهای پرتکرار این محصول چیست؟" \
  --product-id 3901234
```

End-to-end smoke coverage for all four routes:

```bash
# Mock retrievers; a configured LLM key is needed for Product QA to answer.
python scripts/smoke_all_capabilities.py --answers

# Real product/comment artifacts and all four integrated handlers.
python scripts/smoke_all_capabilities.py --retriever-mode real --answers
```
