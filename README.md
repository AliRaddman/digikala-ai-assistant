<div dir="rtl">

# دستیار هوشمند خرید و تحلیل محصولات دیجی‌کالا

پروژه سوم بوت‌کمپ هوش مصنوعی کوئرا — بهار ۱۴۰۵

**اعضا:** علی، مهیا، فاطمه، بنیامین

**مهلت تحویل:** جمعه ۶ شهریور، ساعت ۲۳:۵۹

**ارائه:** دوشنبه و سه‌شنبه ۹ و ۱۰ شهریور

---

## راه‌اندازی

</div>

```bash
git clone <repo-url> && cd digikala-ai-assistant
python -m venv .venv
source .venv/bin/activate          # ویندوز: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # کلید API داخلش
```

<div dir="rtl">

**`.env` را هیچ کدی به‌صورت خودکار نمی‌خواند** — `python-dotenv` جایی در پروژه
استفاده نشده و `LLMSettings.from_env()` مستقیم `os.getenv` می‌زند. پس پیش از هر
دستوری که به LLM وصل می‌شود، فایل را خودتان در محیط بار کنید:

</div>

```bash
set -a; . ./.env; set +a           # بدون این، LLM_API_KEY خالی می‌ماند
unset ALL_PROXY all_proxy          # پروکسی socks باعث کرش httpx می‌شود
```

<div dir="rtl">

اگر `LLM_BASE_URL` خالی بماند، درخواست‌ها به `api.openai.com` می‌روند و کلید
متیس آن‌جا `401` می‌گیرد. مقدار درست:
`LLM_BASE_URL=https://api.metisai.ir/openai/v1`

Semantic Cache اختیاری و در حالت پیش‌فرض خاموش است. برای فعال‌کردنش:

</div>

```bash
export LLM_SEMANTIC_CACHE_ENABLED=true
export LLM_SEMANTIC_CACHE_MODEL=intfloat/multilingual-e5-base
export LLM_SEMANTIC_CACHE_THRESHOLD=0.96
```

<div dir="rtl">

مدل embedding در اولین درخواست واجد شرایط به‌صورت lazy لود می‌شود. Cache فقط
وقتی پاسخ معنایی را reuse می‌کند که نسخه‌ی prompt، مدل، JSON schema و Context
حفاظتی (Evidence/جدول/محصول) دقیقاً یکسان باشند؛ شباهت سؤال به‌تنهایی کافی
نیست. Ledger از این پس `exact_cache_hits` و `semantic_cache_hits` را جدا ثبت
می‌کند.

`torch` با همان `pip install -r requirements.txt` نصب می‌شود (چون `sentence-transformers` وابستگی سختی به `torch>=2.2` دارد و به‌هرحال آن را می‌کشد). نسخه‌ای که از PyPI می‌آید عمومی است؛ **اگر GPU دارید**، بعد از نصب، build مخصوص CUDA خودتان را از [pytorch.org](https://pytorch.org/get-started/locally/) دوباره نصب کنید — GPUهای جدید (Blackwell / sm_120) به build تازه‌تری از پیش‌فرض PyPI نیاز دارند. نسخه CPU برای کوئری زدن به ایندکس‌های آماده کافی است، ولی برای ساخت ایندکس روی کل کاتالوگ عملی نیست.

برای اجرای تست‌ها `requirements-dev.txt` را نصب کنید (شامل `pytest`).

دیتاست از نسخه ثابت زیر:

</div>

```
https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/89c3133b169c8d3793db8834f56f32fee33d9db0
```

<div dir="rtl">

### گرفتن artifactها

فایل‌های سنگین روی درایو مشترک‌اند و در گیت نیستند. برای اجرای سیستم بدون ساخت مجدد، این‌ها را از `DigikalaProject/` بردارید:

حجم‌ها از فایل‌های واقعی روی دیسک خوانده شده‌اند.

| فایل درایو | مقصد | حجم | لازم برای |
|---|---|---|---|
| `processed/products_clean_v1.parquet` | `data/processed/` | ۳۷ MB | بخش ۱ و ۴ |
| `indexes/products_meta_v1.parquet` | `data/indexes/` | ۲۰ MB | بخش ۱ |
| `indexes/products_bm25_v1.npz` | `data/indexes/` | ۳۲ MB | بخش ۱ |
| `indexes/products_bm25_vocab_v1.json` | `data/indexes/` | ۷ MB | بخش ۱ |
| `indexes/products_e5base_ivfsq8_v1.faiss` | `data/indexes/` | ۷۴۹ MB | بخش ۱ |
| `processed/comments_clean_v1.parquet` | `data/processed/` | ۴۱۴ MB | بخش ۲ و ۴ |
| `indexes/comments_meta_v1.parquet` | `data/indexes/` | ۱۸۸ MB | بخش ۲ |
| `indexes/comments_product_map_v1.json` | `data/indexes/` | ۳۴ MB | بخش ۲ |
| `indexes/comments_bm25_v1.npz` | `data/indexes/` | ۱۴۹ MB | بخش ۲ |
| `indexes/comments_bm25_vocab_v1.json` | `data/indexes/` | ۸ MB | بخش ۲ |
| `indexes/comments_emb_e5base_v1.npy` | **ساخت محلی** | ۱۰.۶ GB | بخش ۲ |

**فایل امبدینگ نظرات (۱۰.۶ GB) روی درایو نیست** — آپلودش با سرعت ۴۰۰ کیلوبایت بر ثانیه حدود ۷ ساعت طول می‌کشد. هر کس خودش با دستور بخش «لایه نظرات» می‌سازد؛ با GPU حدود ۱۲ دقیقه است. بقیه‌ی فایل‌های نظرات روی درایو هستند.

بخش ۴ (تحلیل دسته) فقط به `comments_clean_v1.parquet` نیاز دارد، نه به هیچ ایندکسی.

اگر جای دیگری گذاشتید، `INDEX_DIR` را در `.env` تنظیم کنید.

---

## ساختار

</div>

```
src/
├── orchestrator.py      روتر intent و اتصال chainها
├── data/
│   ├── normalize.py                  نرمال‌ساز فارسی مشترک — قفل
│   ├── products.py                   پاک‌سازی محصولات
│   ├── comments.py                   پاک‌سازی نظرات — مسیر بازیابی
│   ├── comments_cleaning.py          پاک‌سازی نظرات — مسیر کلاسیفایر
│   ├── build_recommendation_splits.py  split گروهی بدون نشت
│   └── sampling.py                   نمونه‌گیری طبقاتی
├── retrieval/
│   ├── base.py          Evidence، RetrievalFilters، Retriever، MockRetriever
│   ├── products.py      BM25Retriever و DenseRetriever
│   ├── comments.py      CommentRetriever — بازیابی دقیق per-product
│   └── hybrid.py        ترکیب با RRF
├── eval/
│   ├── retrieval_metrics.py   Recall@k، nDCG@k، MRR@k
│   ├── grounding.py           ممیزی استناد + LLM-as-a-Judge
│   ├── harness.py             harness ارزیابی بخش ۱
│   └── product_comparison.py  ارزیابی ساختار، استناد و grounding مقایسه
├── llm/                 کلاینت، کش، شمارش توکن و هزینه
├── chains/
│   ├── product_discovery.py    بخش ۱: جست‌وجو و کشف محصول
│   ├── product_filters.py      استخراج فیلتر از متن فارسی
│   ├── product_qa.py           بخش ۲: پرسش‌وپاسخ مستند به comment_id
│   ├── product_comparison.py   مقایسه facts / evidence / inference
│   └── category_analytics.py   بخش ۴: تحلیل سطح دسته (تجمیع، نه بازیابی)
└── classifier/          خالی — کد طبقه‌بند فعلاً فقط در notebooks/fatemeh/

scripts/                 اسکریپت‌های اجرایی (ساخت ایندکس، بنچمارک، ارزیابی)
│   ├── run_product_qa_eval.py        بخش ۲ روی مدل واقعی + داور + نرخ توهم استناد
│   ├── eval_product_comparison.py    بخش مقایسه؛ retrieval-only یا LLM+judge
│   ├── build_human_labeling_set.py   برگه برچسب‌گذاری انسانی (نمرات داور جدا نگه داشته می‌شود)
│   └── compare_human_vs_judge.py     کاپای کوهن، همبستگی، موارد اختلاف
data/eval/human/         اعتبارسنجی انسانی: برگه، rubric، نمرات داور — در گیت هست
tests/                   تست آفلاین — بدون تماس API
data/eval/               مجموعه ارزیابی و نتایج — در گیت هست
data/eval_d50/           اجرای عمق ۵۰ — در گیت هست (بدون emb_cache)
notebooks/<نام>/         هر کس فقط پوشه خودش
docs/                    تحلیل schema دو جدول، DECISIONS.md، FAILURES.md
```

<div dir="rtl">

---

## استفاده از بازیابی

</div>

```python
from src.retrieval.base import build_retriever, RetrievalFilters

retriever = build_retriever("product")
evidence = retriever.retrieve(
    "یه کیف چرم برای کار",
    top_k=10,
    filters=RetrievalFilters(price_max=1_000_000, exclude_fake=True),
)

for ev in evidence:
    print(ev.score, ev.citation(), ev.title)
```

<div dir="rtl">

خروجی `list[Evidence]` است. هر آیتم `.id`, `.text`, `.score`, `.meta` و `.citation()` دارد.

| متغیر محیطی | مقادیر | پیش‌فرض |
|---|---|---|
| `RETRIEVER_MODE` | `mock` / `real` | `mock` |
| `RETRIEVER_BACKEND` | `dense` / `bm25` / `hybrid` | `dense` |
| `INDEX_DIR` | مسیر | `data/indexes` |
| `INDEX_TYPE` | `ivfsq8` / `flat` | `ivfsq8` |
| `HYBRID_FETCH_DEPTH` | عدد | `50` — **کمترش نکنید**، زیر ۵۰ ترکیب بی‌اثر می‌شود |

**نکات مهم:**

- `product_id` رشته است. سمت مقابل هر join را هم `astype(str)` کنید.
- قیمت‌ها **ریال**اند. کاربر تومان می‌گوید، پس موقع استخراج فیلتر ضربدر ۱۰ کنید.
- `rate` از ۱۰۰ است نه ۵، و برای محصول بدون امتیاز `null` است. پس `min_rate` گذاشتن یعنی محصولات بی‌امتیاز خودکار حذف می‌شوند.
- فیلترها **قید سخت**اند نه امتیاز اضافه: محصول خارج از فیلتر برنمی‌گردد حتی اگر شبیه‌تر باشد.
- `kind="comment"` در حالت `real` به `CommentRetriever` وصل است — بازیابی همیشه به `RetrievalFilters.product_ids` محدود می‌شود؛ بدون آن کل ایندکس نظرات را جاروب می‌کند (کندتر، ولی کار می‌کند).

---

## بازتولید نتایج بازیابی

</div>

```bash
# پاک‌سازی محصولات: ۱.۲۸ میلیون ردیف خام ← ۹۴۸ هزار محصول یکتا
python -m src.data.products \
  --raw data/raw/products_raw.parquet \
  --out data/processed/products_clean_v1.parquet

# نمونه ۵۰ هزارتایی طبقاتی برای بنچمارک
python -m src.data.sampling \
  --clean data/processed/products_clean_v1.parquet \
  --out data/processed/products_sample_50k_v1.parquet

# اجرای مدل‌ها و ساخت pool برچسب‌گذاری
python -m scripts.build_pool \
  --sample data/processed/products_sample_50k_v1.parquet \
  --queries data/eval/queries_v1.jsonl --out-dir data/eval

# متریک‌های بازیابی
python -m src.eval.retrieval_metrics \
  --qrels data/eval/qrels_v1_labeled.csv \
  --runs data/eval/runs --queries data/eval/queries_v1.jsonl

# ایندکس کامل روی ۹۴۸ هزار محصول
python -m scripts.build_index \
  --clean data/processed/products_clean_v1.parquet \
  --out-dir data/indexes --index-type ivfsq8

# ارزیابی ترکیبی و آزمون معناداری
python -m scripts.eval_hybrid \
  --runs data/eval_d50/runs --qrels data/eval/qrels_d50_v2_labeled.csv \
  --out data/eval/hybrid_d50_v2.csv
python -m scripts.test_significance \
  --runs data/eval_d50/runs --qrels data/eval/qrels_d50_v2_labeled.csv \
  --rrf-k 60 --w-dense 0.7 \
  --out data/eval/significance_d50_v2.csv

# تأخیر
python -m scripts.measure_latency --queries data/eval/queries_v1.jsonl
```

<div dir="rtl">

### نتایج

روی ۳۶ کوئری فارسی با برچسب دستی. دلیل هر تصمیم در `docs/DECISIONS.md`.

| مدل embedding | nDCG@10 | Recall@10 | زمان انکد ۵۰k |
|---|---|---|---|
| bge-m3 | ۰.۷۰۷ | ۰.۴۳۹ | ۱۷۸ ثانیه |
| **e5-base** ← انتخاب شد | ۰.۶۹۲ | ۰.۴۳۷ | ۹۳ ثانیه |
| e5-large | ۰.۶۳۵ | ۰.۳۸۹ | ۱۳۷ ثانیه |
| BM25 | ۰.۶۱۱ | ۰.۳۷۲ | ۰.۲ ثانیه |

| نوع ایندکس | recall@10 نسبت به دقیق | حجم | جست‌وجو |
|---|---|---|---|
| Flat (دقیق) | ۱.۰۰ | ۲۹۱۳ MB | ۱۰.۳ ms |
| **IVF-SQ8** ← انتخاب شد | ۰.۸۶۷ | ۷۴۹ MB | ۰.۱۶ ms |
| IVF-PQ96 | ۰.۳۹۲ | ۱۱۲ MB | ۰.۱۴ ms |

| بازیابی (عمق ۵۰) | nDCG@10 | Recall@10 | MRR@10 | p50 | p95 |
|---|---|---|---|---|---|
| **Hybrid (RRF، k=60، w=0.7)** | ۰.۷۷۷۸ | ۰.۵۸۵۰ | ۰.۹۳۴۰ | ۵۳.۸ ms | ۷۲.۹ ms |
| dense تنها | ۰.۷۳۲۹ | ۰.۵۶۷۳ | ۰.۸۹۷۸ | ۳۷.۴ ms | ۴۵.۰ ms |
| BM25 تنها | ۰.۶۳۸۹ | ۰.۴۷۴۰ | ۰.۸۴۸۹ | ۱۸.۵ ms | ۱۹.۷ ms |

برتری hybrid نسبت به dense تنها به آستانه معناداری نمی‌رسد (p = ۰.۰۵۸، ۲۱ کوئری بهتر و ۸ بدتر). نسبت به BM25 معنادار است (p = ۰.۰۲). cold start ایندکس dense حدود ۱۷.۰ ثانیه است و یک بار کش می‌شود.

### تأخیر به تفکیک سناریو

از `data/eval/latency_v1.csv` — بازتولید با `python -m scripts.measure_latency --queries data/eval/queries_v1.jsonl`:

| سناریو | p50 | p95 |
|---|---|---|
| فقط encode کوئری | ۱۸.۹ ms | ۲۵.۸ ms |
| BM25 | ۱۸.۵ ms | ۱۹.۷ ms |
| dense | ۳۷.۴ ms | ۴۵.۰ ms |
| dense + فیلتر | ۳۹.۹ ms | ۶۲.۴ ms |
| hybrid | ۵۳.۸ ms | ۷۲.۹ ms |
| hybrid + فیلتر | ۶۳.۸ ms | ۸۴.۹ ms |
| hybrid با top_k=50 | ۵۵.۴ ms | ۶۳.۷ ms |

cold start: dense ۱۷.۰ ثانیه، BM25 ۰.۶ ثانیه.

**گلوگاه encode کوئری است، نه جست‌وجو.** encode یک جمله به‌تنهایی ۱۸.۹ میلی‌ثانیه است — بیش از نیم کل زمان `dense` و ۹۳٪ زمان بازیابی نظرات. پس جای درست بهینه‌سازی **کش کردن بردار کوئری** است، نه ایندکس سریع‌تر: کوئری‌های تکراری در یک فروشگاه واقعی فراوان‌اند و یک کش رشته‌ی کوئری به بردار، بخش غالب هزینه را حذف می‌کند.

> **یادداشت بازتولیدپذیری:** اعداد تأخیر این پروژه به وضعیت GPU و page cache حساس‌اند؛ یک جدول قبلی با همین اسکریپت بازتولید نشد. همه‌ی اعداد تأخیر فعلی (این جدول و جدول لایه نظرات) در **یک نشست پیوسته روی سیستم بی‌کار** گرفته شده‌اند. مقایسه‌ی نسبی بین سناریوها معتبر است؛ عدد مطلق را روی سخت‌افزار دیگر انتظار نداشته باشید.

---

## لایه نظرات: بخش ۲ و ۴

</div>

```bash
# پاک‌سازی نظرات: ۶.۱۶ میلیون ردیف خام ← ۵.۴۱ میلیون نظر
python -m src.data.comments \
  --raw data/raw/comments_raw.parquet \
  --out data/processed/comments_clean_v1.parquet

# ایندکس نظرات: هر محصول حداکثر ۵۰ نظر، دقیق نه تقریبی (نه FAISS)
# روی GPU حدود ۱۲ دقیقه (encode واقعی: ۷۳۱ ثانیه برای ۳.۴ میلیون بردار)
# دلیل معماری در docs/DECISIONS.md
python -m scripts.build_comment_index \
  --clean data/processed/comments_clean_v1.parquet \
  --out-dir data/indexes --max-per-product 50
```

<div dir="rtl">

خروجی در `data/indexes/`: `comments_meta_v1.parquet`، `comments_product_map_v1.json`، `comments_bm25_v1.npz` (+`vocab`) و `comments_emb_e5base_v1.npy`.

**فایل امبدینگ ۱۰.۶ گیگابایتی روی درایو نمی‌رود** و باید محلی ساخته شود (همان دستور بالا)؛ بقیه روی درایو مشترک هستند. اجرا قابل ازسرگیری است: اگر وسط encode قطع شد، دفعه‌ی بعد از همان ردیف ادامه می‌دهد (`comments_emb_e5base_v1.progress.json`).

**بخش ۲ (پرسش‌وپاسخ مستند به نظرات):**

</div>

```bash
# 4835951 = میکروفن کندانسر وسکات BM800 — یک محصول واقعی با ۵۰ نظر در ایندکس
python -m src.chains.product_qa "ایرادهای پرتکرار این محصول چیست؟" \
  --product-id 4835951 --retriever-mode real
```

```python
from src.chains.product_qa import ProductQAChain
from src.llm.client import build_openai_client
from src.retrieval.base import build_retriever

chain = ProductQAChain(
    retriever=build_retriever("comment", mode="real"),
    client=build_openai_client(),
    max_evidence=20,
)
result = chain.run("آیا با توجه به تجربه کاربران ارزش خرید دارد؟", product_id="4835951")

print(result.render_fa())          # متن فارسی با تگ [comment:...] کنار هر ادعا
result.answer.sufficient_evidence  # bool
result.evidence                    # list[Evidence] — شواهد خامی که مدل دید
result.as_dict()                   # ساختار کامل برای harness
```

<div dir="rtl">

پاسخ به ازای هر ادعا حداقل یک `[comment:...]` دارد. در نسخه‌ی
`product-qa-v4-evidence-bound-citations`، schema هر درخواست علاوه بر
`min_length=1`، فقط شناسه‌های همان Evidence را به‌صورت `enum` مجاز می‌کند؛
مدل از نظر ساختاری نمی‌تواند شناسه‌ی تازه بسازد. پس از دریافت نیز قرنطینه‌ی
قبلی به‌عنوان دفاع دوم برای خروجی‌های قدیمی یا Provider ناسازگار باقی مانده
است.

**دو خروجی مجاز که خطا نیستند:**

- شواهد ناکافی → «نظرات کافی برای پاسخ به این سؤال وجود ندارد»
- محصول بدون هیچ نظر → «برای این محصول نظری ثبت نشده است»

حالت دوم نادر نیست: **۶۲۷,۱۹۲ محصول از ۹۴۸,۳۵۲ محصول کاتالوگ (۶۶.۱٪) هیچ نظری ندارند**، پس بخش ۲ عملاً روی حدود یک‌سوم کاتالوگ قابل استفاده است. این مسیر قبل از encode کردن کوئری برمی‌گردد، پس نه تماس API دارد نه هزینه‌ی محاسباتی (حدود یک میکروثانیه). تحلیل کامل در `docs/FAILURES.md`، شکست ۶.

**در دموی نهایی این عدد را صریح بگویید** — اگر دمو فقط روی محصولات پرنظر اجرا شود، تصویری از پوشش سیستم می‌دهد که دو برابر واقعیت است.

**بخش ۴ (تحلیل سطح دسته):** روی `comments_clean_v1.parquet` کامل کار می‌کند، نه ایندکس — پس بدون ساختن ایندکس بالا هم قابل اجراست:

</div>

```bash
python -m src.chains.category_analytics --cat1 "اسباب بازی"
# یا بدون تماس با LLM، فقط جدول‌های تجمیعی:
python -m src.chains.category_analytics --cat1 "اسباب بازی" --no-summary
```

```python
from src.chains.category_analytics import CategoryAnalyticsChain, CategoryScope

chain = CategoryAnalyticsChain(client=None)      # client=None یعنی فقط تجمیع، بدون LLM
report = chain.run(CategoryScope(cat1="کفش زنانه"))

report.top_complaints                    # ۱) پرتکرارترین شکایت‌ها
report.dissatisfied_feature_complaints   # ۲) شکایت‌ها فقط از نظرات not_recommended
report.high_volume_low_recommend         # ۳) پرنظر ولی نرخ پیشنهاد پایین
report.brand_feedback                    # ۴) مقایسه بازخورد برندها
report.no_complaint_mentions             # نظراتی که صریحاً گفته‌اند ایرادی ندارد
print(report.render_fa())
```

<div dir="rtl">

هر چهار خروجی `DataFrame`اند. هر عددی که در خلاصه‌ی فارسی می‌آید از یکی از همین چهار جدول کپی شده — مدل زبانی فقط جدول را به متن تبدیل می‌کند، عدد تولید نمی‌کند؛ اگر عددی بسازد که در جدول نیست، اجرا با خطا متوقف می‌شود (`_validate_insight_values`).

بدون کلید API هم کار می‌کند: در آن حالت جدول‌ها ساخته می‌شوند و فقط خلاصه‌ی فارسی مدل حذف می‌شود. بخش ۲ عمداً این‌طور نیست — آنجا خودِ مدل پاسخ است، پس نبود کلید صریح خطا می‌دهد.

### تأخیر لایه نظرات

روی ایندکس واقعی (۳,۴۳۴,۷۵۵ بردار)، `top_k=20`، روی GPU، ۱۰۰ فراخوانی بعد از پنج فراخوانی گرم‌کننده:

| حالت | mean | p50 | p95 |
|---|---|---|---|
| محصول پرنظر (۵۰ نظر، به سقف خورده) | ۱۹.۹۰ ms | ۱۹.۶۴ ms | ۲۱.۳۷ ms |
| محصول کم‌نظر (۴ نظر) | ۱۹.۶۰ ms | ۱۹.۷۵ ms | ۲۱.۷۲ ms |
| محصول بدون نظر | ۱.۲۲ µs | ۱.۰۶ µs | ۱.۱۵ µs |
| encode کوئری تنها (مرجع) | ۱۸.۴۳ ms | ۱۷.۷۸ ms | ۲۰.۳۷ ms |
| بدون فیلتر (fallback) — کش سرد / گرم | ۱۴,۱۲۰ / ۴۸۴ ms | — | — |

**۹۳٪ زمان صرف encode کوئری می‌شود** (۱۸.۴۳ از ۱۹.۹۰ ms)، نه جست‌وجو — به همین دلیل تعداد نظرات محصول عملاً روی تأخیر اثری ندارد، و به همین دلیل سقف ۵۰ نظر هیچ‌وقت گلوگاه تأخیر نبود. cold start حدود ۱۷.۵ ثانیه است (مدل + meta + نگاشت) و یک بار کش می‌شود. تجمیع بخش ۴ برای دسته کفش زنانه (۶۰ هزار نظر) ۱.۳ ثانیه طول می‌کشد.

---

## قوانین کار تیمی

**۱. هر فایل یک صاحب دارد.**

اسم صاحب فایل در docstring بالای آن نوشته می‌شود. اگر به تغییر در فایل کس دیگری نیاز داشتی، پیام بده — خودت دست نزن. با این قاعده چهار نفر می‌توانند همزمان کار کنند.

**۲. نوتبوک فقط در پوشه خودت.**

نوتبوک در گیت merge conflict وحشتناک درست می‌کند. هیچ‌کس پوشه دیگری را دست نمی‌زند.

**۳. داده در گیت نیست.**

فایل‌های سنگین روی درایو مشترک. اسم فایل‌ها نسخه‌دار (`comments_clean_v2.parquet`) و هیچ‌وقت overwrite نمی‌شود — درایو نه merge دارد نه history، اگر دو نفر همزمان بنویسند یکی بی‌صدا پاک می‌شود.

استثنا: `data/eval/` در گیت است، چون مجموعه ارزیابی باید قابل بازتولید و قابل نشان دادن در ارائه باشد و حجمش چند صد کیلوبایت است.

| پوشه درایو | محتوا | نویسنده |
|---|---|---|
| `raw/` | فایل‌های خام | علی |
| `processed/` | parquet تمیز | products: علی · comments: مهیا |
| `indexes/` | ایندکس‌ها | علی |
| `models/` | چک‌پوینت کلاسیفایر | فاطمه |
| `eval/` | فایل جدا به ازای هر نفر | همه |

**۴. نرمال‌ساز مشترک است و قفل.**

همه از `src/data/normalize.py` استفاده می‌کنند. اگر هر کس نرمال‌ساز خودش را بنویسد، ایندکس و کلاسیفایر روی متن‌های متفاوتی کار می‌کنند و نتایج قابل مقایسه نیست. تغییر فقط با توافق جمعی.

**۵. Mock تا آخر پروژه زنده می‌ماند.**

`MockRetriever` که خروجی ثابت و ساختگی می‌دهد حذف نمی‌شود. هر کس روی chain کار می‌کند با آن شروع می‌کند و بعداً به retriever واقعی سوییچ می‌کند. یعنی هیچ‌کس منتظر آماده شدن کار کس دیگری نمی‌ماند. ضمناً تنها راه تست یک chain بدون لود کردن ۷۵۰ مگابایت ایندکس همین است.

**۶. هر شب ساعت ۲۳ همگام‌سازی.**

push کن، artifact را روی درایو بگذار — حتی اگر ناقص است — و در گروه بنویس چه چیزی آماده است و چه چیزی نه. کار نصفه منتشرشده از کار کامل منتشرنشده مفیدتر است، چون فردا کسی به آن نیاز دارد.

**۷. تسک ذخیره.**

اگر به کار کسی گیر کردی، بیکار نمان. سراغ تسک ذخیره‌ات برو:

علی → نوشتن eval set · مهیا → برچسب‌گذاری دستی · فاطمه → تحلیل خطا · بنیامین → لاگ هزینه

**۸. برنچ جدا.**

هر کس روی `<نام>/<موضوع>` کار می‌کند و با Pull Request به `main` می‌رود. چهار نفر روی یک برنچ یعنی push rejected مدام.

---

## برنامه هفته

هفت روز، شنبه ۳۱ مرداد تا جمعه ۶ شهریور. اجباری‌ها تا چهارشنبه، پنجشنبه امتیازی، جمعه جمع‌بندی.

### شنبه ۳۱ مرداد

قرارداد داده بر عهده مهیا است. او قبل از هر چیز داده خام را بررسی می‌کند — توزیع مقادیر، نوع ستون‌ها، مقادیر گمشده، تکراری‌ها — و بر اساس همان تحلیل نام و نوع ستون‌های خروجی هر دو جدول را تعیین و منتشر می‌کند. بقیه بدون جلسه از همان استفاده می‌کنند.

خروجی این کار در دو سند منتشر شد: [`docs/products_dataset_analysis_schema.md`](docs/products_dataset_analysis_schema.md) و [`docs/comments_dataset_analysis_schema.md`](docs/comments_dataset_analysis_schema.md). بعد از انتشار، schema قفل است؛ تغییرش فقط با اعلام در گروه و ذکر دلیل در `docs/DECISIONS.md`.

| نفر | تسک |
|---|---|
| **علی** (صبح) | repo، نرمال‌ساز فارسی، Mock، دانلود دیتاست، تبدیل به Parquet، آپلود درایو |
| **علی** (بعدازظهر) | مقایسه سه مدل embedding روی داده خودمان + ساخت ایندکس محصولات |
| **مهیا** | **انتشار SCHEMA تا ظهر** + پاک‌سازی نظرات، تحلیل توزیع، نمونه‌گیری طبقاتی مستدل |
| **فاطمه** | تقسیم داده بدون leakage + baseline TF-IDF + اولین Macro F1 |
| **بنیامین** | کلاینت LLM با کش و شمارش توکن + پرامپت استخراج فیلتر روی Mock |

### یکشنبه ۱ شهریور

| نفر | تسک |
|---|---|
| **علی** | تابع `retrieve()` نهایی + ایندکس نظرات — **انتشار artifact تا ظهر** |
| **مهیا** | جدول تجمیعی سطح دسته: نرخ پیشنهاد، شمارش نظر، آمار برند |
| **فاطمه** | فاین‌تیون ParsBERT، Macro F1، مقایسه با baseline |
| **بنیامین** | بخش ۱ سیستم: جست‌وجو و کشف محصول با retriever واقعی |

انتشار artifact علی تا ظهر مهم است — تنها گلوگاه واقعی وسط هفته همین‌جاست.

### دوشنبه ۲ شهریور

| نفر | تسک |
|---|---|
| **علی** | بخش ۲: پرسش‌وپاسخ مستند با استناد به comment_id |
| **مهیا** | بخش ۴: شکایات پرتکرار، مقایسه برند، محصولات با نارضایتی بالا |
| **فاطمه** | بخش ۳: مقایسه محصولات با سه لایه جدا facts / evidence / inference |
| **بنیامین** | یکپارچه‌سازی chainها + شروع harness ارزیابی |

پایان امروز: هر چهار قابلیت اجباری end-to-end جواب می‌دهند.

### سه‌شنبه ۳ شهریور

| نفر | تسک |
|---|---|
| **علی** | eval set بازیابی + Recall@10 و nDCG@10 |
| **مهیا** | eval set تحلیلی + بررسی درستی خروجی بخش ۴ |
| **فاطمه** | eval بخش مقایسه + confusion matrix و تحلیل خطای کلاسیفایر |
| **بنیامین** | LLM-as-a-Judge برای grounding + لاگ latency و هزینه، گزارش p50/p95 و مصرف توکن |

### چهارشنبه ۴ شهریور — قفل فیچر

هیچ فیچر جدیدی اضافه نمی‌شود. فقط اندازه‌گیری.

- هر نفر ۲۵ نمونه دستی برچسب می‌زند (جمعاً ۱۰۰) — همین اعتبارسنجی انسانی judge است
- **هیچ‌کس خروجی بخشی که خودش ساخته را ارزیابی نمی‌کند.** سوگیری دارد و در ارائه قابل دفاع نیست
- بنیامین: یکپارچه‌سازی و اجرای کامل ارزیابی
- هر نفر ۳ شکست واقعی از بخش خودش با تحلیل علت

### پنجشنبه ۵ شهریور — امتیازی

| نفر | تسک | نمره |
|---|---|---|
| **علی** | Hybrid Search + Reranking با شواهد کمی | ۴ |
| **بنیامین** | Router + کش معنایی، با عدد کاهش هزینه | ۳ |
| **مهیا** | داشبورد Gradio | ۳ |
| **فاطمه** | LoRA روی مدل طبقه‌بندی یا اعتبارسنجی انسانی judge با Cohen's kappa | ۴ یا ۳ |

جمعاً ۱۳ تا ۱۴ نمره، همه در امتداد کار روزهای قبل.

### جمعه ۶ شهریور

- صبح: README نهایی، جدول نتایج، failure analysis
- بعدازظهر: اسلاید و تمرین ارائه
- **مرور متقابل:** هر نفر بخش یکی دیگر را توضیح دهد. صورت پروژه گفته همه اعضا باید درک مناسبی از کل سیستم داشته باشند و ممکن است در ارائه از مهیا درباره router بپرسند
- شب: تحویل

---

## ارزیابی

| بعد | متریک | مسئول | وضعیت | عدد ثبت‌شده |
|---|---|---|---|---|
| Retrieval | Recall@10، nDCG@10، MRR | علی | ✅ | nDCG@10 = ۰.۷۷۷۸ + آزمون معناداری — `data/eval/hybrid_d50_v2.csv` |
| Latency | p50 / p95 به تفکیک سناریو | علی | ✅ | هفت سناریو بازیابی + چهار سناریو نظرات — `data/eval/latency_v1.csv` |
| Failure Analysis | نمونه‌های واقعی شکست، علت و تلاش برای بهبود | همه | ✅ | **دوازده شکست** با اندازه‌گیری کمّی — `docs/FAILURES.md` |
| Grounding | نسبت claim های دارای شاهد معتبر | بنیامین / علی | ✅ | `citation_integrity = ۱.۰`؛ **نرخ توهم شناسه ۴.۱٪، نرخ توهم پاسخ ۳۰٪** |
| کیفیت پاسخ | judge ۱–۵ | بنیامین / علی | ✅ | بخش ۱: grounding ۴.۳۱ / relevance ۴.۷۸ روی ۳۶ کوئری · بخش ۲: **۴.۳۰ / ۵.۰۰** روی ۱۰ سوال |
| مقایسه محصول | completeness + citation integrity + judge اختیاری | فاطمه / بنیامین | 🟡 زیرساخت کامل | اجرای تاریخی ۲۴/۲۴ فقط ساختار retrieval را سنجیده؛ `inference = 0/24` و کیفیت متن هنوز عدد ندارد |
| Cost | تعداد فراخوانی، توکن ورودی/خروجی، دلار | بنیامین / علی | ✅ | ۱۶۹ درخواست، ۹۳ تماس، **$0.041497** — جدول پایین |
| برچسب انسانی | کاپای کوهن در برابر judge | علی | 🟡 در جریان | ۲۵ مورد آماده‌ی برچسب‌گذاری — `data/eval/human/labels_v1.csv` |
| طبقه‌بندی | **Macro F1** — متریک اصلی بخش سوم | فاطمه | 🟡 نصفه | baseline TF-IDF روی test = **۰.۶۸۳۳**؛ ParsBERT فقط validation = ۰.۷۱۳۱، عدد test ندارد |

خلاصه‌ی قابل‌ردیابی اجرای تاریخی مقایسه در
`data/eval/runs/product_comparison_retrieval_historical_v1.json` ثبت شده است.
این اجرا با `llm_client=None` انجام شده؛ بنابراین ۲۴/۲۴ فقط یعنی دو محصول و
reviewهای scoped آن‌ها بازیابی شده‌اند. Harness جدید latency مورد اول را جدا از
حالت گرم گزارش می‌کند و کیفیت معنایی را فقط وقتی inference و judge واقعاً حاضر
باشند اندازه‌گیری‌شده می‌نامد.

زیرساخت `src/llm` و `src/eval` کار بنیامین است؛ اجرای زنده، رفع باگ‌های
اعتبارسنجی و اعداد این جدول کار علی در روز آخر.

</div>

### هزینه و مصرف توکن

<div dir="rtl">

| | |
|---|---|
| درخواست منطقی | ۱۶۹ |
| تماس واقعی API | ۹۳ |
| cache hit | ۷۶ (**۴۵.۰٪**) |
| توکن ورودی / خروجی | ۱۵۳,۴۰۵ / ۳۰,۸۱۱ |
| **هزینه** | **$0.041497** |
| صرفه‌جویی کش | **$0.049071** |
| نسبت به بودجه‌ی $5 گروه | **۰.۸۳٪** |

| عملیات | درخواست | API | cache | توکن ورودی | توکن خروجی | هزینه | p50 تماس |
|---|---|---|---|---|---|---|---|
| `judge_grounding` | ۱۱۴ | ۴۶ | ۶۸ | ۱۱۳,۸۲۶ | ۲۶,۴۵۵ | **$0.032947** | ۶,۱۵۹ms |
| `answer_product_qa` | ۱۳ | ۱۱ | ۲ | ۲۷,۰۹۸ | ۲,۲۵۲ | $0.005416 | ۳,۱۱۳ms |
| `extract_product_filters` | ۴۲ | ۳۶ | ۶ | ۱۲,۴۸۱ | ۲,۱۰۴ | $0.003135 | ۱,۷۱۱ms |

سه نکته که بدون آن‌ها این جدول گمراه‌کننده است:

۱. **اعداد دلاری بر پایه‌ی نرخ رسمی OpenAI محاسبه شده‌اند.** درخواست‌ها از
gateway متیس رفته‌اند و نرخ واقعی متیس در دست ما نیست. شمارش توکن و تماس دقیق
است؛ تبدیل به دلار تخمینی است. جزئیات در `docs/DECISIONS.md`.

۲. **صرفه‌جویی کش از کل هزینه بیشتر است.** `CachedLLMClient` پیش از اعتبارسنجی
در کش می‌نویسد، پس پاسخی که یک اعتبارسنج ردش کرده هم می‌ماند و اجرای دوباره پس
از اصلاح رایگان است. همین $0.0237 را از یک باگ نجات داد.

۳. **از این $0.0415، مبلغ $0.0237 برای صفر نتیجه خرج شد** — ۵۷٪ کل هزینه، صرف
۳۶ فراخوانی داور که دو باگ اعتبارسنجی همه‌شان را دور ریختند. شکست ۹ در
`docs/FAILURES.md`.

### چیزهایی که اعداد بالا نمی‌گویند

- **`constraint_pass_rate = ۱.۰` صحت استخراج را ثابت نمی‌کند.** این متریک
  می‌سنجد نتایج قیدِ اعلام‌شده را رعایت کرده‌اند، نه اینکه خود قید درست بوده.
  در اجرای تاریخی `q026` هر دو مسیر ۱.۰ گرفتند و هر دو غلط بودند؛ parser عدد
  حروفی baseline بعداً رفع شد، ولی ضعف مفهومی متریک باقی است. شکست ۷.
- **استخراج فیلتر با LLM از رجکس بدتر بود** — ۳ کوئری بدتر، ۰ بهتر، ۹۰ برابر
  کندتر. به همین دلیل خاموش است. شکست ۸.
- **دو عدد توهم استناد اجرای تاریخی v3 را با هم بخوانید.** ۴.۱٪ شناسه‌ها
  ساختگی بودند، ولی ۳۰٪ پاسخ‌ها دست‌کم یکی داشتند. در v4 شناسه‌های مجاز به
  Evidence همان درخواست محدود شده‌اند و Regression test، شناسه‌ی ساختگی را
  پیش از قرنطینه رد می‌کند. نرخ خام مدل بعد از این تغییر هنوز به اجرای زنده‌ی
  جدید نیاز دارد و نباید بدون آن صفر گزارش شود. شکست ۱۰.
- **کاپای انسانی هنوز محاسبه نشده** و وقتی شد، روی نمونه‌ی **طبقاتی** خواهد
  بود نه یکنواخت — بخش ۲ عمداً ۴۰٪ وزن دارد در حالی که سهم واقعی‌اش ۲۲٪ است.
  دلیلش و پیامدش در `docs/FAILURES.md` بخش «محدودیت‌های خودِ روش ارزیابی».
- **rubric بعد `relevance` فقط ۵ و ۳ و ۱ را تعریف کرده**؛ ۲ و ۴ مقدار مجازند
  ولی بی‌تعریف. هر اختلافی روی آن دو نمره، نقص rubric است نه اختلاف قضاوت.

**بودجه:** سقف ۵ دلار برای کل گروه شامل توسعه و تست. embedding و مدل‌های محلی
از بودجه کم نمی‌کنند — و همین یک تصمیم، بین $2.05 و $3.08 صرفه‌جویی کرد
(`docs/DECISIONS.md`).

### Semantic Cache

Semantic Cache در مسیرهای Product QA، Comparison و داور Grounding وصل شده،
اما opt-in است. برای QA و Comparison، Evidence و مشخصات
محصول داخل guard دقیق قرار دارند؛ پس سؤال مشابه درباره‌ی Context متفاوت cache
hit نمی‌شود. Exact cache همیشه قبل از encode بررسی می‌شود و سریع‌ترین مسیر
باقی می‌ماند.

مسیر استخراج فیلتر عمداً semantic نمی‌شود: دو کوئری «زیر ۵۰۰ هزار» و «زیر
۶۰۰ هزار» ممکن است embedding بسیار نزدیک داشته باشند، اما reuse کردن خروجی
اول برای دومی قید عددی غلط می‌سازد. این مسیر فقط Exact Cache دارد.

Benchmark آفلاین ثبت‌شده در
`data/eval/semantic_cache_offline_benchmark_v1.json` روی ۸ درخواست یکتا و ۴
جفت paraphrase این نتیجه را داد:

| حالت | API call | semantic hit | هزینه‌ی تخمینی | wall latency |
|---|---:|---:|---:|---:|
| فقط Exact Cache | ۸ | ۰ | $0.001680 | 414.6 ms |
| Semantic Cache | ۴ | ۴ | $0.000840 | 218.4 ms |

در این workload با hit rate پنجاه‌درصدی، هزینه **۵۰٪** و wall latency
**۴۷.۳٪** کم شد؛ latency خود semantic hit نسبت به میانگین baseline **۹۷.۹٪**
کمتر بود. Provider دارای delay ثابت ۵۰ms و encoder بردارهای deterministic دارد
تا تست بدون API و دانلود مدل بازتولید شود؛ بنابراین این اعداد کیفیت hit مدل
واقعی را ثابت نمی‌کنند.

بر پایه‌ی دو عدد واقعی قبلی پروژه — p50 تماس Product QA برابر ۳,۱۱۳ms و p50
encode محلی کوئری برابر ۱۸.۹ms — کاهش latency هر hit در حالت گرم حدود **۹۹.۴٪**
برآورد می‌شود. این ترکیب دو اندازه‌گیری قبلی است، نه A/B زنده؛ پیش از ادعای
production باید با مدل تنظیم‌شده و ترافیک واقعی دوباره سنجیده شود.

</div>

```bash
python -m scripts.benchmark_semantic_cache \
  --output data/eval/semantic_cache_offline_benchmark_v1.json
```

<div dir="rtl">

</div>

### بازتولید

<div dir="rtl">

</div>

```bash
set -a; . ./.env; set +a; unset ALL_PROXY all_proxy

# بخش ۱ + داور معنایی — پیکربندی نهایی سیستم (فیلتر قاعده‌محور)
python -m src.eval.harness --input data/eval/queries_v1.jsonl \
  --qrels data/eval/qrels_d50_v2_labeled.csv --retriever-mode real \
  --judge-grounding --output data/eval/runs/discovery_real_judged_v2.json

# بخش ۲ روی ۱۰ سوال و ۵ محصول واقعی، با داور و schema جدید Citation
python scripts/run_product_qa_eval.py --judge \
  --output data/eval/runs/product_qa_real_v2_citation_enum.json

# مقایسه محصول روی ۲۴ case واقعی؛ بدون تماس LLM و بدون هزینه API
python -m scripts.eval_product_comparison --retriever-mode real \
  --output data/eval/runs/product_comparison_retrieval_v1.json

# فقط با تأیید هزینه: تولید inference و داوری grounding
python -m scripts.eval_product_comparison --retriever-mode real \
  --with-llm --judge-grounding \
  --output data/eval/runs/product_comparison_llm_judged_v1.json

# ساخت برگه‌ی برچسب‌گذاری انسانی (بدون تماس API)
python scripts/build_human_labeling_set.py

# پس از پر کردن labels_v1.csv: کاپا، همبستگی و موارد اختلاف
python scripts/compare_human_vs_judge.py --show-text

# اجرای کاملاً آفلاین، بدون کلید API
python -m src.eval.harness --retriever-mode mock --top-k 5
```

<div dir="rtl">

هر اجرای دوباره‌ی یک نسخه‌ی یکسان از دستورهای بالا از کش دیسکی می‌خواند و
**صفر دلار** هزینه دارد، مادام که پرامپت، نام مدل و JSON schema پاسخ عوض نشده
باشند — هر سه در کلید کش هستند. تغییر v3 به schema پویای v4 عمداً cache miss
می‌دهد و یک اجرای زنده‌ی تازه لازم دارد. جزئیات خروجی و نحوه‌ی اتصال qrels در
`docs/BENYAMIN_LLM_FOUNDATION.md` آمده است.

---

## تصمیم‌های اصلی

هر تصمیم مهندسی در `docs/DECISIONS.md` ثبت می‌شود. این فایل جمعه مستقیم به اسلاید تبدیل می‌شود، چون صورت پروژه گفته هر تصمیم باید قابل توضیح و مبتنی بر تحلیل داده باشد.

| تصمیم | چرا |
|---|---|
| embedding محلی | صورت پروژه گفته ابزار محلی از بودجه کم نمی‌کند، و هزینه کمتر با کیفیت مشابه امتیاز مثبت دارد |
| FAISS به‌جای vector DB | ایندکس یک فایل است و ساده روی درایو رد و بدل می‌شود؛ هیچ سرویسی بالا نمی‌آید |
| معماری دو مدلی | کارهای پرتکرار روی مدل ارزان، پاسخ نهایی روی مدل قوی — همین Router بخش امتیازی است |
| تقسیم گروهی بر اساس product_id | نظرات یک محصول شبیه هم‌اند؛ split تصادفی یعنی مدل محصول را حفظ می‌کند نه الگوی زبانی را، و Macro F1 مصنوعاً بالا می‌رود |
| انتخاب embedding با آزمایش | طبق بنچمارک FaMTEB مدل‌های اختصاصی فارسی مثل ParsBERT در retrieval نمره تک‌رقمی دارند و مدل‌های چندزبانه بالای ۴۰؛ ولی داده ما محاوره‌ای و پر از فینگلیش است پس روی داده خودمان می‌سنجیم |
| IVF-SQ8 به‌جای IVF-PQ | PQ فقط ۳۹٪ نتایج ایندکس دقیق را برمی‌گرداند، با هیچ تنظیمی بهتر نشد. SQ8 به ۸۷٪ می‌رسد با همان سرعت |
| ترکیب با RRF | مقیاس امتیاز BM25 و کسینوس با هم نمی‌خواند؛ RRF فقط رتبه را می‌بیند و در آزمایش از جمع وزنی بهتر بود |

</div>
