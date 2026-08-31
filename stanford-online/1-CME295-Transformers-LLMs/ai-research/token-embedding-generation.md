One-Hot Encoding (OHE) is actually the most primitive way to represent categorical data/words as vectors, and it has well-known problems (huge sparse vectors, no notion of similarity between categories, doesn't scale to large vocabularies). 
Here's the landscape of alternatives, roughly in order of increasing sophistication.

## Quick Recap: Why OHE Falls Short

If you one-hot encode movie genres `{Action, Comedy, Drama}`:
- Action = `[1, 0, 0]`, Comedy = `[0, 1, 0]`, Drama = `[0, 0, 1]`
- Every pair of vectors is **equidistant** — the encoding has no idea that "Action" and "Thriller" are more similar to each other than "Action" and "Romance." It also blows up in size with large vocabularies (10,000 words = 10,000-dimensional sparse vectors, mostly zeros).

## 1. Frequency/Count-Based Methods

### Bag of Words (BoW)
Represents a document as a vector of word counts, ignoring order.
**Example:** "I loved the movie" → count of each word's occurrence across a fixed vocabulary.
- **Limitation:** Still sparse, still no semantic relationships between words, and loses word order entirely.

### TF-IDF (Term Frequency–Inverse Document Frequency)
Weighs words by how important they are to a specific document relative to the whole corpus — common words like "the" get down-weighted, rare/distinctive words get up-weighted.
- **Better than BoW** for tasks like search/retrieval, but still doesn't capture semantic meaning (synonyms like "great" and "excellent" are still totally unrelated vectors).

## 2. Dense, Learned Embeddings (the real alternative to OHE)

These are the modern replacement — instead of sparse, hand-built vectors, you learn a **dense, low-dimensional vector** for each word/category, where similar things end up close together in the vector space.

### Word2Vec
Learns embeddings by predicting a word from its context (**CBOW**) or predicting context from a word (**Skip-gram**). Famous for capturing analogies like `king - man + woman ≈ queen`.
- Produces ~100-300 dimensional dense vectors instead of 10,000+ dimensional sparse ones.

### GloVe (Global Vectors)
Similar goal to Word2Vec, but trained differently — instead of predicting context, it directly factorizes a **word co-occurrence matrix** (how often word pairs appear together across the whole corpus) into dense vectors.

### FastText
Extension of Word2Vec that represents words as a **bag of character n-grams** rather than whole words. This lets it generate reasonable embeddings even for words it's never seen before (great for misspellings, rare words, or morphologically rich languages).
- Example: "loved" might be broken into `<lo, lov, ove, ved, ed>` internally, so even "lovedd" (a typo) gets a sensible embedding.

### Contextual Embeddings (BERT, ELMo, GPT-style)
Unlike Word2Vec/GloVe (which give a word the *same* vector no matter the context), these generate a **different embedding for the same word depending on its sentence context**.
- Example: "I went to the **bank** to withdraw cash" vs. "I sat by the river**bank**" — the word "bank" gets two completely different embeddings depending on surrounding words.
- This is the standard now for anything built on transformer architectures.

## 3. Entity/Categorical Embeddings (for non-text categorical data)

For structured/tabular data (like movie genres, user IDs, product categories), you can learn embeddings the same way neural networks learn word embeddings:

### Learned Embedding Layers
Instead of one-hot encoding a categorical feature (like `genre` or `user_id`) and feeding it into a model, you pass it through a **trainable embedding layer** that maps each category to a dense vector, learned jointly with the rest of the model during training.
- Very common in recommendation systems — e.g., Netflix/Spotify-style models learn dense embeddings for each user and each movie/song, where proximity in that space reflects taste similarity.

### Entity Embeddings (structured data specifically)
A specific technique (popularized by a well-known Kaggle competition write-up) that trains embeddings for categorical variables in tabular data (like `store_id`, `day_of_week`) as part of a neural network, often outperforming OHE + traditional models like decision trees.

## 4. Dimensionality-Reduction-Based Encodings

### PCA / SVD on OHE vectors
Rather than learning embeddings from scratch, you can take your original sparse OHE (or count-based) vectors and compress them into dense, lower-dimensional vectors using **matrix factorization** techniques.
- Latent Semantic Analysis (LSA) — applies SVD to a term-document matrix (like TF-IDF) to find latent topics/dimensions.

### Autoencoders
Neural networks trained to compress input into a small "bottleneck" layer and then reconstruct it — the bottleneck layer becomes a dense embedding of the original (often sparse) input.

## 5. Ordinal / Target-Based Encodings (simpler, non-neural options)

For tabular ML (not deep learning), there are lighter-weight alternatives to OHE that don't involve "embeddings" in the neural sense, but solve similar problems:

| Method | What it does |
|---|---|
| **Label/Ordinal Encoding** | Assigns each category an integer (e.g., Action=0, Comedy=1) — works only if there's a meaningful order, otherwise misleads models into thinking categories are numerically related |
| **Target Encoding (Mean Encoding)** | Replaces each category with the average target value for that category (e.g., replace "Action" with the average rating of Action movies) — powerful but risks data leakage if not done carefully with cross-validation |
| **Frequency Encoding** | Replaces each category with how often it appears in the dataset |
| **Hashing Trick (Feature Hashing)** | Maps categories to a fixed-size vector via a hash function — avoids the need to know vocabulary size in advance, common in large-scale/streaming ML systems |

## Quick Comparison Table

| Method | Dense or Sparse? | Captures Semantic Similarity? | Common Use Case |
|---|---|---|---|
| One-Hot Encoding | Sparse | No | Baseline, small vocab/categories |
| Bag of Words / TF-IDF | Sparse | No | Classic NLP, search/retrieval |
| Word2Vec / GloVe | Dense | Yes (static) | Pre-transformer NLP |
| FastText | Dense | Yes (subword-aware) | Handling rare words/typos |
| BERT/GPT-style embeddings | Dense | Yes (contextual) | Modern NLP, LLMs |
| Learned Embedding Layers | Dense | Yes (task-specific) | Recommender systems, tabular deep learning |
| PCA/SVD/Autoencoders | Dense | Somewhat (linear/non-linear compression) | Dimensionality reduction |
| Target/Frequency Encoding | Scalar (1D) | No (but injects useful signal) | Gradient-boosted trees (XGBoost, LightGBM) |

**Bottom line:** OHE is really only still used for **low-cardinality categorical features** (e.g., a "genre" column with 10 categories) or as a baseline. The moment you have high-cardinality data (words, user IDs, product SKUs) or want to capture semantic relationships, dense learned embeddings (Word2Vec-style for text, embedding layers for tabular data) are the standard replacement.