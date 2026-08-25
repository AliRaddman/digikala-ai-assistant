# Benyamin — LLM foundation

Owner: Benyamin.

This checkpoint implements Benyamin's LLM and evaluation foundation:

- OpenAI Responses API adapter with Structured Outputs
- SQLite exact-match cache
- per-request token, latency, cost and cache-savings ledger
- conservative zero-cost Persian filter-extraction baseline
- LLM Persian filter extractor
- product-discovery chain over the shared `Retriever` contract
- reproducible discovery evaluation over JSONL queries
- chain and end-to-end latency reports with mean, p50, p95 and max
- checkpoint-scoped API call, cache, token and USD accounting
- citation-integrity audit plus an optional structured grounding judge

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

Inline gold IDs remain supported for small or standalone evaluation sets:

```json
{"query_id":"q001","query":"...","intent":"simple","sub_cat":"clothe","relevant_product_ids":["3901234","7712045"]}
```

The LLM judge still needs validation against the team's independent human
labels. The prompt, rubric, disagreement cases and human agreement must be
reported; judge scores alone are not a substitute for human evaluation.

## Remaining dependency

Ali's `ali/retrieval` branch has now been integrated locally. Product retrieval
supports BM25, dense FAISS and hybrid RRF through
`build_retriever("product", mode="real")`. A real run still needs the index
artifacts from Drive and the retrieval dependencies installed locally.

The comment index is not implemented yet, so
`build_retriever("comment", mode="real")` still raises `NotImplementedError`.

No live API request was made while implementing or testing this checkpoint.
