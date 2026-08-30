# راهنمای نمره‌دهی — دقیقاً همان چیزی که به داور مدل داده شد

این متن کلمه‌به‌کلمه از `SYSTEM_PROMPT` در `src/eval/grounding.py` برداشته شده.
اگر با معیار دیگری نمره داده شود، اختلاف با داور مدل معنای «عدم توافق» نمی‌دهد،
فقط یعنی دو نفر به دو سؤال متفاوت جواب داده‌اند.

```
Grounding score rubric:
5 = every substantive factual claim is directly supported by supplied evidence
4 = supported overall, with only a minor unsupported detail
3 = a material mix of supported and unsupported claims
2 = most material claims are unsupported or overstate the evidence
1 = the answer is unsupported, contradicted, or has no usable evidence

Relevance score rubric:
5 = directly and usefully answers the question
3 = partially answers it or includes substantial irrelevant material
1 = does not answer the question
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
- متن هر شاهد در ستون `evidence` تا 400 کاراکتر بریده شده تا سطر خوانا بماند.
  میانه‌ی طول نظرات در این مجموعه ۵۳ کاراکتر است، پس این برش عملاً روی تعداد
  کمی از شواهد اثر می‌گذارد. متن کامل در فایل‌های `data/eval/runs/` هست.
