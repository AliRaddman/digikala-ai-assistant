# Benyamin — LLM foundation

Owner: Benyamin.

This checkpoint implements the first independent Benyamin task:

- OpenAI Responses API adapter with Structured Outputs
- SQLite exact-match cache
- per-request token, latency, cost and cache-savings ledger
- conservative zero-cost Persian filter-extraction baseline
- LLM Persian filter extractor
- product-discovery chain over the shared `Retriever` contract

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

## Known dependency

`src.retrieval.base.build_retriever(mode="real")` still raises
`NotImplementedError`. The chain is ready for the real retriever once its
implementation is published without changing the chain contract.
