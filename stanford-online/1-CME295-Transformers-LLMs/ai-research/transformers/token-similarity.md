# Is Cosine Similarity the only algorithm to use? 
The right choice depends on what your embeddings represent and how they were trained. Let's go through the main ones.

Here's the shorthand/common abbreviations used in code, papers, and documentation:

| Full Name | Short Name / Abbreviation |
|---|---|
| Cosine Similarity | **Cosine** / **Cos Sim** |
| Euclidean Distance | **L2 distance** / **L2 norm** |
| Dot Product | **Dot Product** / **Inner Product (IP)** |
| Manhattan Distance | **L1 distance** / **L1 norm** / **Taxicab distance** |
| Minkowski Distance | **Minkowski** (no common shorthand — often just called "Lp distance") |
| Jaccard Similarity | **Jaccard** / **Jaccard Index** |
| Mahalanobis Distance | **Mahalanobis** (no shorter form in common use) |
| KL Divergence | **KL Div** / **KLD** |
| Wasserstein Distance | **Wasserstein** / **EMD** (Earth Mover's Distance) |

**Where you'll see these in practice** (e.g., in vector databases like FAISS, Pinecone, Weaviate, or Milvus), the config option is usually just called one of:
- `cosine`
- `l2` (Euclidean)
- `ip` or `dot` (dot product)
- `l1` (Manhattan)

So if you're setting up a vector search index and see a `metric_type` or `distance` parameter, it'll almost always be one of these four short codes (`cosine`, `l2`, `ip`, `l1`) — the more exotic ones (Mahalanobis, Wasserstein, KL) are rarely built into vector DB defaults and usually require custom implementation.

 
## 1. Cosine Similarity (recap)

$$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

Measures the **angle** between two vectors, ignoring magnitude. Ranges from -1 to 1 (1 = identical direction, 0 = orthogonal/unrelated, -1 = opposite).

**Best for:** Text embeddings (word2vec, sentence transformers, most LLM embeddings), where the *direction* of the vector encodes meaning and magnitude is often just an artifact of word frequency or vector length.

## 2. Euclidean Distance (L2 Norm)

$$d(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}$$

Measures **straight-line distance** between two points in space. Unlike cosine, it *does* care about magnitude — two vectors pointing the same direction but different lengths will have nonzero Euclidean distance.

**Best for:** Image embeddings, or any case where absolute position in the vector space (not just direction) carries meaning — e.g., face recognition embeddings (FaceNet), where distance directly correlates to "how similar do these faces look."

**Movie example:** If a movie is embedded as `[action_score, romance_score, comedy_score]`, Euclidean distance tells you how far apart two movies are on all three axes simultaneously.

## 3. Dot Product (Inner Product)

$$A \cdot B = \sum_{i=1}^{n} A_i B_i$$

Similar to cosine similarity, but **not normalized** — so it's influenced by both angle *and* magnitude. A vector with larger magnitude will score higher even at the same angle.

**Best for:** Recommendation systems (e.g., matrix factorization-based collaborative filtering), where magnitude often encodes something meaningful, like "popularity" or "confidence" of an item embedding. Many vector databases (like FAISS) default to dot product for retrieval when magnitude is informative.

**Gotcha:** If your embeddings aren't normalized, dot product and cosine similarity can rank things very differently — worth checking which one your embedding model was actually trained/optimized for.

## 4. Manhattan Distance (L1 Norm / "Taxicab Distance")

$$d(A, B) = \sum_{i=1}^{n} |A_i - B_i|$$

Sums the *absolute* differences per dimension, rather than squaring them (like a taxi driving along a city grid instead of flying straight through buildings).

**Best for:** High-dimensional sparse data, or when you want to reduce the influence of large outlier differences in any single dimension (since it doesn't square the differences like Euclidean does).

## 5. Minkowski Distance (the generalization)

$$d(A, B) = \left(\sum_{i=1}^{n} |A_i - B_i|^p\right)^{1/p}$$

This is a general formula where:
- p = 1 → Manhattan distance
- p = 2 → Euclidean distance
- p → ∞ → Chebyshev distance (max absolute difference across any single dimension)

Useful when you want a tunable middle ground between L1 and L2 behavior.

## 6. Jaccard Similarity

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Doesn't operate on dense embedding vectors at all — instead compares **sets** (e.g., sets of words, tags, or genres). Measures overlap relative to total combined size.

**Movie example:** If Movie A has genre tags `{Action, Sci-Fi, Thriller}` and Movie B has `{Action, Sci-Fi, Comedy}`, Jaccard similarity = 2/4 = 0.5.

**Best for:** Categorical/tag-based similarity rather than learned dense embeddings — common in recommendation systems working off metadata rather than neural embeddings.

## 7. Mahalanobis Distance

$$d(A, B) = \sqrt{(A-B)^T S^{-1} (A-B)}$$

Like Euclidean distance, but accounts for **correlations between dimensions** and **scales each dimension by its variance** (using the covariance matrix S). Two points that look far apart in raw Euclidean terms might be "close" in Mahalanobis terms if that direction of difference is common/expected in the data.

**Best for:** Anomaly/outlier detection in embedding space, or when your embedding dimensions are correlated and you don't want that correlation to distort distance measurements.

## 8. KL Divergence / Wasserstein Distance (for distributions, not points)

If your "embeddings" are actually probability distributions (e.g., topic distributions from LDA, or softmax outputs), you'd use:
- **KL Divergence:** Measures how one probability distribution diverges from a reference one (asymmetric — KL(A‖B) ≠ KL(B‖A)).
- **Wasserstein Distance (Earth Mover's Distance):** Measures the "cost" of transforming one distribution into another — popular in GANs (Wasserstein GAN) and comparing document topic distributions.

**Movie example:** If you represent a movie as a distribution over genres (e.g., 60% action, 30% thriller, 10% comedy) rather than a single point, these metrics compare distributions rather than points.

## Quick Comparison Table

| Metric | Cares about magnitude? | Typical use case |
|---|---|---|
| **Cosine Similarity** | No | Text/sentence embeddings, semantic search |
| **Euclidean (L2)** | Yes | Image embeddings, clustering (k-means) |
| **Dot Product** | Yes | Recommender systems, unnormalized retrieval (FAISS) |
| **Manhattan (L1)** | Yes | High-dim sparse data, outlier-robust comparisons |
| **Minkowski** | Yes (tunable) | General framework (generalizes L1/L2) |
| **Jaccard** | N/A (set-based) | Tag/genre overlap, categorical similarity |
| **Mahalanobis** | Yes (scale-aware) | Anomaly detection, correlated dimensions |
| **KL Divergence / Wasserstein** | N/A (distribution-based) | Comparing probability distributions, topic models |

## Practical Note

For most modern embedding models (sentence transformers, OpenAI/Anthropic-style embeddings, word2vec), **cosine similarity is the default choice** because these models are typically trained with cosine similarity (or a closely related contrastive loss) as the training objective — so the vector *direction* is where the meaning lives, and using a different metric at inference time than what the model was trained with often gives noticeably worse results. Always check your embedding model's documentation to see which metric it was optimized for — using the wrong one is a surprisingly common source of poor retrieval quality in RAG systems and vector search pipelines.