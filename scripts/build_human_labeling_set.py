"""Stage 5a: build the sheet Ali fills in by hand, and hide the judge's scores.

Owner: Ali, 2026-08-30. Produces three files under data/eval/human/:

  labels_v1.csv        the sheet to fill in -- six columns, two of them empty
  RUBRIC.md            the judge's own rubric, copied verbatim from
                       src/eval/grounding.SYSTEM_PROMPT
  judge_scores_v1.json the judge's scores, kept out of the sheet on purpose

The judge's scores are deliberately absent from the CSV. Seeing them would
anchor the human labels, and an agreement number computed against an anchored
label measures nothing. scripts/compare_human_vs_judge.py joins the two back
together after the sheet is filled in.

Sampling is stratified rather than uniform, with a fixed seed: all 10 section-2
QA answers plus 15 of the 36 discovery answers. A uniform draw from the 46
would have been ~4/5 QA, and the discovery answers are mechanical renderings of
retrieved rows whose grounding is near-trivially 5 -- an agreement statistic
dominated by them would look strong while saying nothing about the part of the
system where grounding is actually at risk.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

SEED = 20260830
DISCOVERY_SAMPLE = 15
EVIDENCE_CHAR_CAP = 400

OUT_DIR = Path("data/eval/human")


def _rubric_text() -> str:
    from src.eval.grounding import SYSTEM_PROMPT

    start = SYSTEM_PROMPT.index("Grounding score rubric:")
    end = SYSTEM_PROMPT.index("Verdict must be")
    return SYSTEM_PROMPT[start:end].strip()


def _clip(text: str, cap: int = EVIDENCE_CHAR_CAP) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _discovery_rows(path: Path) -> list[dict[str, object]]:
    import pandas as pd

    report = json.loads(path.read_text(encoding="utf-8"))
    meta = pd.read_parquet(
        "data/indexes/products_meta_v1.parquet",
        columns=["product_id", "title", "brand", "price", "rate", "sub_cat"],
    )
    meta["product_id"] = meta["product_id"].astype(str)
    lookup = meta.set_index("product_id").to_dict("index")

    rows = []
    for item in report["items"]:
        if not item.get("grounding_judgment"):
            continue
        blocks = []
        for product_id in item["retrieved_product_ids"]:
            record = lookup.get(str(product_id))
            if record is None:
                blocks.append(f"[product:{product_id}] (رکورد یافت نشد)")
                continue
            price = record["price"]
            rate = record["rate"]
            parts = [
                f"[product:{product_id}] {_clip(record['title'], 120)}",
                f"برند: {record['brand']}",
                f"قیمت: {int(price):,} ریال" if price == price else "قیمت: نامشخص",
                f"امتیاز: {rate:g}/100" if rate == rate else "امتیاز: ندارد",
                f"دسته: {record['sub_cat']}",
            ]
            blocks.append(" | ".join(parts))
        rows.append(
            {
                "id": item["query_id"],
                "source": "discovery",
                "question": item["query"],
                "answer": item["answer"],
                "evidence": "\n".join(blocks),
                "judge": item["grounding_judgment"]["judgment"],
            }
        )
    return rows


def _qa_rows(path: Path) -> list[dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in report["items"]:
        if "judgment" not in item:
            continue
        blocks = []
        for evidence in item["evidence"]:
            meta = evidence.get("meta") or {}
            facts = []
            if meta.get("recommendation_status"):
                facts.append(str(meta["recommendation_status"]))
            if meta.get("rate") is not None:
                facts.append(f"امتیاز {meta['rate']:g}/5")
            title = _clip(evidence.get("title") or "", 80)
            head = f"[comment:{evidence['id']}]"
            if title:
                head += f" {title}"
            if facts:
                head += f" ({' | '.join(facts)})"
            blocks.append(f"{head}\n{_clip(evidence.get('text') or '')}")
        rows.append(
            {
                "id": item["case_id"],
                "source": "product_qa",
                "question": f"{item['question']}  [{item['product']}]",
                "answer": item["rendered_fa"],
                "evidence": "\n\n".join(blocks),
                "judge": item["judgment"]["judgment"],
            }
        )
    return rows


def _refuse_to_clobber_labels(sheet: Path, force: bool) -> None:
    """Never silently overwrite labelling work.

    The seed is fixed, so regenerating produces an identical sheet -- identical
    except for the two columns a person spent an hour filling in. Rebuilding is
    a normal thing to do while the docs are being written, and the sheet is
    normally open in an editor at the same time, so the cost of getting this
    wrong is somebody's afternoon.
    """
    if force or not sheet.exists():
        return
    with sheet.open(encoding="utf-8-sig") as handle:
        filled = sum(
            bool((row.get("grounding_1_5") or "").strip())
            or bool((row.get("relevance_1_5") or "").strip())
            for row in csv.DictReader(handle)
        )
    if filled:
        raise SystemExit(
            f"{sheet} already has {filled} labelled row(s); refusing to overwrite "
            "them. Move the file aside, or pass --force if you really mean it."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path,
                        default=Path("data/eval/runs/discovery_real_judged_v2.json"))
    parser.add_argument("--qa", type=Path,
                        default=Path("data/eval/runs/product_qa_real_v1.json"))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if the sheet already has scores in it")
    args = parser.parse_args()

    discovery = _discovery_rows(args.discovery)
    qa = _qa_rows(args.qa)

    rng = random.Random(SEED)
    sampled = qa + rng.sample(discovery, min(DISCOVERY_SAMPLE, len(discovery)))
    rng.shuffle(sampled)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sheet = args.out_dir / "labels_v1.csv"
    _refuse_to_clobber_labels(sheet, args.force)
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question", "answer", "evidence",
                         "grounding_1_5", "relevance_1_5"])
        for row in sampled:
            writer.writerow([row["id"], row["question"], row["answer"],
                             row["evidence"], "", ""])

    hidden = args.out_dir / "judge_scores_v1.json"
    hidden.write_text(
        json.dumps(
            {
                "seed": SEED,
                "note": "Not to be opened before labels_v1.csv is filled in.",
                "scores": {
                    row["id"]: {
                        "source": row["source"],
                        "grounding": row["judge"]["grounding_score"],
                        "relevance": row["judge"]["relevance_score"],
                        "rationale": row["judge"]["rationale"],
                    }
                    for row in sampled
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    rubric = args.out_dir / "RUBRIC.md"
    rubric.write_text(_RUBRIC_TEMPLATE.format(rubric=_rubric_text()), encoding="utf-8")

    print(f"{sheet}  ({len(sampled)} rows: "
          f"{sum(r['source'] == 'product_qa' for r in sampled)} product_qa, "
          f"{sum(r['source'] == 'discovery' for r in sampled)} discovery)")
    print(f"{rubric}")
    print(f"{hidden}  (judge scores, kept out of the sheet)")


_RUBRIC_TEMPLATE = """# راهنمای نمره‌دهی — دقیقاً همان چیزی که به داور مدل داده شد

این متن کلمه‌به‌کلمه از `SYSTEM_PROMPT` در `src/eval/grounding.py` برداشته شده.
اگر با معیار دیگری نمره داده شود، اختلاف با داور مدل معنای «عدم توافق» نمی‌دهد،
فقط یعنی دو نفر به دو سؤال متفاوت جواب داده‌اند.

```
{rubric}
```

## ترجمه‌ی عملی

**grounding_1_5 — آیا هر ادعای پاسخ از شواهد درمی‌آید؟**

| نمره | یعنی |
|---|---|
| ۵ | هر ادعای واقعی مستقیماً از شواهد پشتیبانی می‌شود |
| ۴ | در کل پشتیبانی‌شده، فقط یک جزئیات جزئی بدون پشتوانه |
| ۳ | ترکیبی معنادار از ادعاهای پشتیبانی‌شده و نشده |
| ۲ | بیشتر ادعاهای مهم بدون پشتوانه‌اند یا از شواهد فراتر می‌روند |
| ۱ | پاسخ بی‌پشتوانه یا متناقض با شواهد است، یا شاهد قابل استفاده‌ای ندارد |

**relevance_1_5 — آیا اصلاً به سؤال جواب می‌دهد؟**

| نمره | یعنی |
|---|---|
| ۵ | مستقیم و مفید به سؤال جواب می‌دهد |
| ۳ | نیمه‌جواب، یا حجم زیادی مطلب بی‌ربط دارد |
| ۱ | به سؤال جواب نمی‌دهد |

نمره‌ی ۲ و ۴ در بعد relevance تعریف صریح ندارند؛ داور مدل هم همین rubric را
داشت، پس همان‌طور بین‌آبی استفاده کنید.

## دو نکته درباره‌ی خود فایل

- **فقط بر اساس شواهد همان سطر نمره بدهید،** نه دانش بیرونی درباره‌ی محصول.
  داور مدل هم صریحاً همین محدودیت را داشت.
- متن هر شاهد در ستون `evidence` تا {cap} کاراکتر بریده شده تا سطر خوانا بماند.
  میانه‌ی طول نظرات در این مجموعه ۵۳ کاراکتر است، پس این برش عملاً روی تعداد
  کمی از شواهد اثر می‌گذارد. متن کامل در فایل‌های `data/eval/runs/` هست.
""".replace("{cap}", str(EVIDENCE_CHAR_CAP))


if __name__ == "__main__":
    main()
