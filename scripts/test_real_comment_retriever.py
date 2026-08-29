from pathlib import Path

from src.retrieval.comments import CommentRetriever
from src.retrieval.base import RetrievalFilters


INDEX_DIR = Path("data/indexes/comparison_eval")

retriever = CommentRetriever(index_dir=INDEX_DIR)

product_id = "114286"

results = retriever.retrieve(
    "کیفیت محصول، نقاط قوت و ضعف و ارزش خرید",
    top_k=5,
    filters=RetrievalFilters(
        product_ids=[product_id]
    ),
)

print("Retrieved:", len(results))

for i, ev in enumerate(results, 1):
    print("\n", "=" * 60)
    print("Rank:", i)
    print("Score:", round(ev.score, 4))
    print("Comment ID:", ev.id)
    print("Product ID:", ev.product_id)
    print("Rate:", ev.meta.get("rate"))
    print("Recommendation:", ev.meta.get("recommendation_status"))
    print("Text:", ev.text)