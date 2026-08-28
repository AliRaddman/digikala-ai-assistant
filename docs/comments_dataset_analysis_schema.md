# Comments Dataset Analysis Report

## Project: Digikala AI Assistant

## 1. Dataset Overview

The comments dataset contains user reviews collected from Digikala products.

- Total comments: 6,156,289
- Total columns: 15
- Dataset type: User Reviews Dataset
- Main usage: User experience analysis, recommendation-status prediction, category-level analysis, and RAG knowledge retrieval

The comments dataset is connected to products through:

```text
product_id
```

---

# 2. Comments Dataset Schema

| Field | Type | Description | Usage |
|---|---|---|---|
| id | integer | Unique comment identifier | Comment tracking and evidence citation |
| title | string | Review title | Short review summary |
| body | string | Main user review text | Main RAG and classification content |
| created_at | string | Review creation date | Time analysis |
| rate | float | User rating score | Auxiliary structured signal |
| recommendation_status | string | Recommendation status | Three-class prediction target |
| is_buyer | boolean | Verified buyer flag | Review reliability signal |
| product_id | integer | Related product ID | Product-comment relation and leakage-free split |
| advantages | string | Product advantages | Positive feature extraction |
| disadvantages | string | Product disadvantages | Complaint and issue extraction |
| likes | integer | Helpful votes | Review importance |
| dislikes | integer | Negative votes | Review validation |
| seller_title | string | Seller name | Seller analysis |
| seller_code | string | Seller identifier | Seller connection |
| true_to_size_rate | string | Size matching information | Clothing-related analysis |

---

# 3. Data Quality Analysis

## Title

Missing values:

```text
2,865,062
```

Approximately 46% of reviews do not have a title.

Decision:

The title is optional because the main information usually exists in the `body` field. Missing titles should not cause a review to be removed.

---

## Body

Missing values:

```text
637
```

This is a very small percentage of the dataset.

Decision:

Reviews without usable body text can be removed from text-based modeling and retrieval, unless useful information exists in `advantages` or `disadvantages`.

---

## Recommendation Status

Missing values:

```text
894,426
```

The project requires prediction of `recommendation_status` in exactly three classes:

```text
recommended
not_recommended
no_idea
```

Important distinction:

```text
recommended      = explicit recommendation
not_recommended  = explicit rejection
no_idea          = explicit "no opinion" class
NaN               = recommendation status was not recorded
```

Decision:

- Keep all three real classes unchanged.
- Do **not** map `no_idea` to missing values.
- Do **not** map missing (`NaN`) values to `no_idea`.
- For the supervised recommendation-status classifier, rows with missing `recommendation_status` should be excluded from the labeled training/evaluation set.
- `rate` may be retained as an auxiliary analysis signal, but it should not be used to overwrite or infer the official target label.

---

## Advantages

Missing values:

```text
5,448,084
```

Most users do not fill this field.

Decision:

Keep the field separately. It is sparse, but useful for:

- Positive feature extraction
- Product comparison
- Category-level analysis
- Grounded user-experience summaries

Do not discard it after creating `review_text`.

---

## Disadvantages

Missing values:

```text
5,744,017
```

Most users do not fill this field.

Decision:

Keep the field separately. It is especially important for:

- Repeated complaint analysis
- Product weaknesses
- Category-manager questions
- Product comparison
- Grounded Q&A

Do not discard it after creating `review_text`.

---

## True To Size Rate

Missing values:

```text
6,065,516
```

This field is mostly empty.

Decision:

Keep it as optional metadata and use it only for relevant product categories such as clothing and footwear.

---

# 4. Duplicate Analysis

Duplicate comments:

```text
1,318
```

Duplicate rate is very low.

Recommended cleaning step:

```python
df = df.drop_duplicates(subset=["id"])
```

The duplicate-removal decision should be applied before indexing or model splitting.

---

# 5. Important Fields for the Project

## 1. body

Main user experience text.

Example:

```text
کیفیت خوبه ولی کمی ظریفه
```

Primary use:

- Recommendation-status classification
- Review retrieval
- RAG evidence
- User-experience analysis

---

## 2. disadvantages

Useful for finding product problems and repeated complaints.

Example:

```text
بسته بندی ضعیف بود
```

Primary use:

- Complaint analysis
- Category-manager analytics
- Product weaknesses
- Product comparison

---

## 3. advantages

Useful for extracting positive product aspects.

Example:

```text
کیفیت ساخت عالی
```

Primary use:

- Positive feature extraction
- Product comparison
- User satisfaction summaries

---

## 4. recommendation_status

This field is the official three-class prediction target.

Valid labeled classes:

```text
recommended
not_recommended
no_idea
```

It must remain separate from any optional sentiment field.

---

## 5. product_id

This field connects comments to products and should also be used when creating leakage-resistant train/validation/test splits.

Comments belonging to the same product should not be distributed across training and evaluation sets if the team uses product-level grouping.

---

# 6. RAG Processing Strategy

Raw review example:

```text
خیلی خوبه ولی باتری زود خالی میشه
```

A processed representation may contain:

```text
Comment ID:
12345

Product ID:
252058

User Review:
خیلی خوبه ولی باتری زود خالی میشه

Recommendation Status:
recommended
```

Important:

RAG answers about user experience must remain grounded in real comments. Therefore, `comment_id` and `product_id` must be preserved so that retrieved evidence can be cited by the final system.

---

# 7. Cleaning Pipeline

Before indexing or modeling:

## Step 1: Remove invalid records

Recommended checks:

- Remove duplicated comment IDs
- Remove rows with no usable text for text-based tasks
- Detect extremely short or uninformative reviews instead of deleting them blindly
- Preserve `product_id` and `id`

---

## Step 2: Persian Text Normalization

Use the project's shared normalizer instead of creating a separate normalization implementation.

Example import:

```python
from src.data.normalize import normalize
```

Normalize text fields such as:

```text
title
body
advantages
disadvantages
```

Typical normalization includes character unification such as:

```text
ي -> ی
ك -> ک
```

The exact behavior should follow the shared `normalize` implementation used by the repository.

---

## Step 3: Preserve Cleaned Fields Separately

Recommended cleaned fields:

```text
title_clean
body_clean
advantages_clean
disadvantages_clean
```

Keeping these fields separate is important because different downstream tasks use them differently.

For example:

- `body_clean` → classifier and general retrieval
- `advantages_clean` → positive aspect analysis
- `disadvantages_clean` → complaint analysis

---

## Step 4: Create Combined Review Text

Create an additional field:

```text
review_text
```

It may combine:

```text
title_clean
body_clean
advantages_clean
disadvantages_clean
```

Example:

```text
عنوان: ...
متن نظر: ...
مزایا: ...
معایب: ...
```

Important:

`review_text` is an additional representation. It must **not** replace or delete the separate cleaned fields.

---

## Step 5: Preserve Recommendation Target

Do not convert the target into only positive/negative sentiment.

Correct target:

```text
recommended
not_recommended
no_idea
```

Missing labels remain missing:

```text
NaN
```

Optional sentiment features may be created separately if needed, but they are not a replacement for `recommendation_status`.

---

# 8. Recommended Processed Dataset Schema

The processed comments dataset should preserve the information needed by classification, RAG, product comparison, and category analytics.

| Field | Description |
|---|---|
| comment_id | Unique comment ID |
| product_id | Related product ID |
| title_clean | Normalized title |
| body_clean | Normalized body |
| advantages_clean | Normalized positive points |
| disadvantages_clean | Normalized negative points |
| review_text | Combined normalized review representation |
| recommendation_status | Original three-class target or NaN |
| rate | User rating |
| is_buyer | Buyer verification |
| created_at | Review date |
| likes | Helpful votes |
| dislikes | Negative votes |
| seller_title | Seller name |
| seller_code | Seller identifier |
| true_to_size_rate | Optional size-related feedback |

Optional additional fields may be added, but these core fields should not be removed without a downstream justification.

---

# 9. Recommendation-Status Modeling Dataset

The supervised classification dataset should be derived from the processed dataset.

Use only rows whose label is one of:

```text
recommended
not_recommended
no_idea
```

Do not include:

```text
recommendation_status = NaN
```

The project evaluation metric is:

```text
Macro F1
```

Therefore all three classes must be preserved and evaluated separately.

To reduce leakage risk, splitting should consider grouping by `product_id` rather than randomly distributing comments from the same product across train and test sets.

---

# 10. Vector Database Structure

Example review evidence:

```json
{
  "comment_id": 12345,
  "product_id": 252058,
  "text": "بسته بندی ضعیف بود",
  "metadata": {
    "rate": 1,
    "recommendation_status": "not_recommended",
    "is_buyer": true
  }
}
```

The vector index should preserve `comment_id` because the final assistant must be able to show which real reviews support its answer.

---

# 11. Final Architecture

```text
Products Dataset
        |
        | product_id
        |
        v
Comments Dataset
        |
        v
Cleaning + Shared Persian Normalization
        |
        v
Processed Comments
        |
        +--------------------+
        |                    |
        v                    v
Recommendation          Comment Retrieval
Classifier              / Embeddings / Index
(3 classes)                  |
        |                     v
        |                RAG Evidence
        |                     |
        +----------+----------+
                   |
                   v
          Shopping Assistant
                   |
        +----------+----------+
        |                     |
        v                     v
 Product Comparison    Category Analytics
```

---

# Conclusion

The comments dataset is suitable for building the Persian Digikala shopping assistant, but the processed representation must support several downstream tasks at the same time.

Main preparation decisions:

- Normalize Persian text using the repository's shared normalizer.
- Remove invalid and duplicate records carefully.
- Preserve `comment_id` and `product_id`.
- Preserve `advantages` and `disadvantages` as separate cleaned fields.
- Create `review_text` as an additional combined field.
- Preserve `recommendation_status` as the official three-class target:
  - `recommended`
  - `not_recommended`
  - `no_idea`
- Keep missing `recommendation_status` values (`NaN`) distinct from `no_idea`.
- Use labeled rows only for supervised recommendation prediction.
- Keep evidence identifiers available for grounded RAG answers and evaluation.
