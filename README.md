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

`torch` جداگانه نصب می‌شود، چون نسخه‌اش به CUDA سیستم بستگی دارد. دستور درست را از [pytorch.org](https://pytorch.org/get-started/locally/) بردارید. نسخه CPU برای همه چیز کار می‌کند جز ساخت ایندکس روی کل کاتالوگ.

دیتاست از نسخه ثابت زیر:

</div>

```
https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/89c3133b169c8d3793db8834f56f32fee33d9db0
```

<div dir="rtl">

### گرفتن artifactها

فایل‌های سنگین روی درایو مشترک‌اند و در گیت نیستند. برای اجرای سیستم بدون ساخت مجدد، این‌ها را از `DigikalaProject/` بردارید:

| فایل درایو | مقصد | حجم |
|---|---|---|
| `processed/products_clean_v1.parquet` | `data/processed/` | ۳۶ MB |
| `indexes/products_meta_v1.parquet` | `data/indexes/` | ۳۵ MB |
| `indexes/products_bm25_v1.npz` | `data/indexes/` | ۶۰ MB |
| `indexes/products_bm25_vocab_v1.json` | `data/indexes/` | ۱۰ MB |
| `indexes/products_e5base_ivfsq8_v1.faiss` | `data/indexes/` | ۷۴۹ MB |

برای بخش ۲ (پرسش‌وپاسخ بر پایه نظرات) این‌ها هم لازم‌اند:

| فایل درایو | مقصد | حجم |
|---|---|---|
| `processed/comments_clean_v1.parquet` | `data/processed/` | ۴۱۵ MB |
| `indexes/comments_meta_v1.parquet` | `data/indexes/` | ۱۸۸ MB |
| `indexes/comments_product_map_v1.json` | `data/indexes/` | ۳۴ MB |
| `indexes/comments_bm25_v1.npz` | `data/indexes/` | ۱۴۹ MB |
| `indexes/comments_bm25_vocab_v1.json` | `data/indexes/` | ۸ MB |
| `indexes/comments_emb_e5base_v1.npy` | `data/indexes/` | ۱۰.۵ GB |

فایل امبدینگ ۱۰.۵ گیگابایتی روی درایو نمی‌رود (آپلودش با سرعت ۴۰۰ کیلوبایت بر ثانیه حدود ۷ ساعت طول می‌کشد). به‌جایش هر کس با دستور بخش «لایه نظرات» خودش می‌سازد — با GPU حدود ۱۲ دقیقه است. بقیه‌ی فایل‌های نظرات روی درایو هستند.

بخش ۴ (تحلیل دسته) فقط به `comments_clean_v1.parquet` نیاز دارد، نه به هیچ ایندکسی.

اگر جای دیگری گذاشتید، `INDEX_DIR` را در `.env` تنظیم کنید.

---

## ساختار

</div>

```
src/
├── data/
│   ├── normalize.py     نرمال‌ساز فارسی مشترک — قفل
│   ├── products.py      پاک‌سازی محصولات
│   ├── comments.py      پاک‌سازی نظرات
│   └── sampling.py      نمونه‌گیری طبقاتی
├── retrieval/
│   ├── base.py          Evidence، RetrievalFilters، Retriever، MockRetriever
│   ├── products.py      BM25Retriever و DenseRetriever
│   ├── comments.py      CommentRetriever — بازیابی دقیق per-product
│   └── hybrid.py        ترکیب با RRF
├── eval/
│   └── retrieval_metrics.py   Recall@k، nDCG@k، MRR@k
├── llm/                 کلاینت، کش، router، پرامپت‌ها
├── chains/
│   ├── product_discovery.py    بخش ۱: جست‌وجو و کشف محصول
│   ├── product_qa.py           بخش ۲: پرسش‌وپاسخ مستند به comment_id
│   └── category_analytics.py   بخش ۴: تحلیل سطح دسته (تجمیع، نه بازیابی)
└── classifier/          پیش‌بینی recommendation_status

scripts/                 اسکریپت‌های اجرایی (ساخت ایندکس، بنچمارک، ارزیابی)
data/eval/               مجموعه ارزیابی و نتایج — در گیت هست
notebooks/<نام>/         هر کس فقط پوشه خودش
docs/                    SCHEMA.md و DECISIONS.md
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
  --runs data/eval_d50/runs --qrels data/eval/qrels_d50_v2_labeled.csv
python -m scripts.test_significance \
  --runs data/eval_d50/runs --qrels data/eval/qrels_d50_v2_labeled.csv \
  --rrf-k 60 --w-dense 0.7

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

| بازیابی (عمق ۵۰) | nDCG@10 | Recall@10 | MRR@10 | p50 |
|---|---|---|---|---|
| **Hybrid (RRF، k=60، w=0.7)** | ۰.۷۷۷۸ | ۰.۵۸۵۰ | ۰.۹۳۴۰ | ۲۶.۵ ms |
| dense تنها | ۰.۷۳۲۹ | ۰.۵۶۷۳ | ۰.۸۹۷۸ | ۱۴.۱ ms |
| BM25 تنها | ۰.۶۳۸۹ | ۰.۴۷۴۰ | ۰.۸۴۸۹ | ۱۱.۴ ms |

برتری hybrid نسبت به dense تنها به آستانه معناداری نمی‌رسد (p = ۰.۰۵۸، ۲۱ کوئری بهتر و ۸ بدتر). نسبت به BM25 معنادار است (p = ۰.۰۲). cold start ایندکس dense حدود ۱۴.۵ ثانیه است و یک بار کش می‌شود.

---

## لایه نظرات: بخش ۲ و ۴

</div>

```bash
# پاک‌سازی نظرات: ۶.۱۶ میلیون ردیف خام ← ۵.۴۱ میلیون نظر
python -m src.data.comments \
  --raw data/raw/comments_raw.parquet \
  --out data/processed/comments_clean_v1.parquet

# ایندکس نظرات: هر محصول حداکثر ۵۰ نظر، دقیق نه تقریبی (نه FAISS)
# با GPU حدود ۱ ساعت طول می‌کشد — دلیل معماری در docs/DECISIONS.md
python -m scripts.build_comment_index \
  --clean data/processed/comments_clean_v1.parquet \
  --out-dir data/indexes --max-per-product 50
```

<div dir="rtl">

خروجی در `data/indexes/`: `comments_meta_v1.parquet`، `comments_emb_e5base_v1.npy`، `comments_product_map_v1.json`، `comments_bm25_v1.npz` (+`vocab`). این چهار فایل هم روی درایو مشترک می‌روند، مثل ایندکس محصولات.

**بخش ۲ (پرسش‌وپاسخ مستند به نظرات):**

</div>

```bash
python -m src.chains.product_qa "ایرادهای پرتکرار این محصول چیست؟" \
  --product-id 3901234 --retriever-mode real
```

<div dir="rtl">

پاسخ به ازای هر ادعا حداقل یک `[comment:...]` دارد؛ اگر نظری برای آن محصول نبود یا شواهد کافی نبود، جمله‌ی صریح «نظرات کافی برای پاسخ به این سؤال وجود ندارد» برمی‌گردد — این یک خروجی مجاز است، نه خطا.

**بخش ۴ (تحلیل سطح دسته):** روی `comments_clean_v1.parquet` کامل کار می‌کند، نه ایندکس — پس بدون ساختن ایندکس بالا هم قابل اجراست:

</div>

```bash
python -m src.chains.category_analytics --cat1 "اسباب بازی"
# یا بدون تماس با LLM، فقط جدول‌های تجمیعی:
python -m src.chains.category_analytics --cat1 "اسباب بازی" --no-summary
```

<div dir="rtl">

هر عددی که در خلاصه‌ی فارسی می‌آید از یکی از چهار جدول تجمیع‌شده کپی شده — مدل زبانی فقط جدول را به متن تبدیل می‌کند، عدد تولید نمی‌کند؛ اگر عددی بسازد که در جدول نیست، اجرا با خطا متوقف می‌شود (`_validate_insight_values`).

### تأخیر لایه نظرات

روی ایندکس واقعی (۳,۴۳۴,۷۵۵ بردار)، `top_k=20`، بعد از سه فراخوانی گرم‌کننده:

| حالت | mean | p50 | p95 |
|---|---|---|---|
| محصول پرنظر (۵۰ نظر، به سقف خورده) | ۴.۲ ms | ۴.۱ ms | ۴.۲ ms |
| محصول کم‌نظر (۴ نظر) | ۳.۳ ms | ۳.۳ ms | ۳.۳ ms |
| محصول بدون نظر | ۰.۶ µs | ۰.۵ µs | ۰.۶ µs |
| بدون فیلتر محصول (fallback، اسکن کامل) | ۴۰۷ ms | — | — |

cold start حدود ۱۶.۸ ثانیه است (مدل + meta + نگاشت محصول) و یک بار کش می‌شود. تجمیع بخش ۴ برای دسته اسباب‌بازی (۳۷۱ هزار نظر) ۱.۸ ثانیه طول می‌کشد.

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

قرارداد داده (`docs/SCHEMA.md`) بر عهده مهیا است. او قبل از هر چیز داده خام را بررسی می‌کند — توزیع مقادیر، نوع ستون‌ها، مقادیر گمشده، تکراری‌ها — و بر اساس همان تحلیل نام و نوع ستون‌های خروجی هر دو جدول را تعیین و منتشر می‌کند. بقیه بدون جلسه از همان استفاده می‌کنند.

بعد از انتشار، SCHEMA قفل است. تغییرش فقط با اعلام در گروه و ذکر دلیل در `docs/DECISIONS.md`.

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

| بعد | متریک | مسئول | وضعیت |
|---|---|---|---|
| کیفیت پاسخ | judge ۱–۵ + برچسب انسانی | بنیامین | — |
| Grounding | نسبت claim های دارای شاهد معتبر | بنیامین | — |
| Retrieval | Recall@10، nDCG@10، MRR | علی | ✅ |
| طبقه‌بندی | **Macro F1** — متریک اصلی بخش سوم | فاطمه | در جریان |
| Latency | p50 / p95 به تفکیک سناریو | بنیامین | ✅ بازیابی |
| Cost | تعداد فراخوانی، توکن ورودی/خروجی، دلار | بنیامین | — |
| Failure Analysis | حداقل ۱۲ نمونه واقعی با تحلیل | همه | — |

**بودجه:** سقف ۵ دلار برای کل گروه شامل توسعه و تست. کش دیسکی از روز اول فعال باشد — بیشتر هزینه در فاز دیباگ می‌سوزد نه در اجرای نهایی. embedding و مدل‌های محلی از بودجه کم نمی‌کنند.

اجرای آفلاین harness بنیامین روی مجموعه‌ی فعلی ۳۶ کوئری:

</div>

```bash
python -m src.eval.harness --retriever-mode mock --top-k 5
```

<div dir="rtl">

برای اجرای کامل با retriever واقعی و judge، پس از تنظیم کلید API از گزینه‌های
`--retriever-mode real --use-llm-filters --judge-grounding` استفاده کنید. جزئیات
خروجی و نحوه‌ی اتصال qrels در `docs/BENYAMIN_LLM_FOUNDATION.md` آمده است.

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
