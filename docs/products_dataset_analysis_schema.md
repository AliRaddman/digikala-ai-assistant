# Products Dataset Analysis Report

## Project: Digikala AI Assistant

## 1. Dataset Overview

The products dataset contains raw product information collected from
Digikala.

-   Total records: 1,283,496 products
-   Total columns: 12
-   Data type: Product catalog dataset
-   Main usage: Product retrieval, recommendation, and RAG knowledge
    base

------------------------------------------------------------------------

# 2. Product Schema

  ----------------------------------------------------------------------------
  Field                  Type              Description       Usage in RAG
  ---------------------- ----------------- ----------------- -----------------
  id                     integer           Unique product    Connect product
                                           identifier        data with
                                                             comments

  title_fa               string            Persian product   Main semantic
                                           title             search field

  Rate                   integer           Product rating    Ranking and
                                           score             recommendation

  Rate_cnt               integer           Number of ratings Confidence of
                                                             rating

  Category1              string            Main category     Category
                                                             filtering

  Category2              string            Secondary         More accurate
                                           category          retrieval

  Brand                  string            Product brand     Brand comparison

  Price                  integer           Current product   Budget filtering
                                           price             

  Seller                 string            Product seller    Seller
                                                             information

  Is_Fake                boolean           Fake product      Quality filtering
                                           indicator         

  min_price_last_month   integer           Minimum price in  Price analysis
                                           previous month    

  sub_category           string            Detailed product  Domain filtering
                                           category          
  ----------------------------------------------------------------------------

------------------------------------------------------------------------

# 3. Data Quality Analysis

## Missing Values

### Category2

-   Total products: 1,283,496
-   Available values: 1,073,388
-   Missing values: 210,108

Recommendation:

Replace missing values with:

    Unknown

------------------------------------------------------------------------

### Seller

-   Missing values: 223

Recommendation:

Replace missing sellers with:

    Unknown Seller

------------------------------------------------------------------------

# 4. Duplicate Analysis

Duplicate product IDs:

    335,144

Approximately:

    26.1% of dataset

Recommendation:

Remove duplicated products before creating embeddings.

Cleaning step:

``` python
df.drop_duplicates(subset=["id"])
```

------------------------------------------------------------------------

# 5. Category Analysis

Top categories:

  Category                    Count
  ------------------------ --------
  اکسسوری زنانه و مردانه     95,999
  لباس زنانه                 93,392
  اسباب بازی                 92,466
  لباس مردانه                68,178
  اکسسوری زنانه              64,470

The dataset covers multiple marketplace categories.

------------------------------------------------------------------------

# 6. Brand Analysis

Number of unique brands:

    8,961

Brand information can be used for:

-   Brand comparison
-   Recommendation ranking
-   User queries about specific brands

------------------------------------------------------------------------

# 7. Price Analysis

Statistics:

  Metric              Value
  --------- ---------------
  Mean            8,491,273
  Median          1,560,000
  Maximum     8,499,990,000

Observations:

-   Price distribution is highly skewed.
-   Some extreme values exist.
-   Price normalization is required before ranking.

Recommended checks:

-   Remove zero prices.
-   Detect abnormal prices.
-   Normalize currency format.

------------------------------------------------------------------------

# 8. RAG Data Preparation Strategy

## Vector Embedding Fields

These fields should be converted into searchable text:

-   title_fa
-   Category1
-   Category2
-   sub_category
-   Brand
-   Seller

Example:

    محصول:
    گوشی موبایل سامسونگ A54

    برند:
    Samsung

    دسته:
    موبایل

    قیمت:
    15000000

------------------------------------------------------------------------

## Metadata Fields

These fields should remain as filters:

-   id
-   Price
-   Rate
-   Rate_cnt
-   Is_Fake

Example:

User query:

"گوشی سامسونگ زیر 20 میلیون"

Process:

1.  Semantic search: Samsung phone products

2.  Metadata filtering: Price \< 20,000,000

------------------------------------------------------------------------

# 9. Recommended Cleaning Pipeline

Steps before indexing:

1.  Remove duplicate products
2.  Normalize Persian text
3.  Handle missing values
4.  Validate prices
5.  Remove invalid records
6.  Generate embedding text
7.  Store processed dataset

------------------------------------------------------------------------

# 10. Final Schema Decision

The final product entity:

    Product
    |
    |-- id
    |-- title_fa
    |-- category
    |-- brand
    |-- price
    |-- rating
    |-- seller
    |-- fake_status
    |-- price_history

This schema is ready for the next stages: - Comment processing -
Embedding generation - Vector database indexing - RAG retrieval
