**Tokenization** is the process of breaking down raw text into smaller units — called **tokens** — that a machine learning model can actually process. Since ML models work with numbers, not raw text, tokenization is the first step in converting a sentence into a numerical format (each token gets mapped to an ID, which is then converted to embeddings).

Let's use this sentence throughout: **"I loved the movie's plot!"**

## Why Tokenization Matters

- Models can't understand raw strings — they need discrete, countable units to build a vocabulary and embeddings from.
- The *way* you tokenize affects vocabulary size, how well the model handles rare/unknown words, and how much sequence length you end up with.
- It's a trade-off: fewer, coarser tokens keep sequences short but blow up vocabulary size (and struggle with new/rare words); more, finer tokens shrink the vocabulary but produce longer sequences.

## 1. Word-Level Tokenization

Splits text on whitespace and punctuation, treating each word as a token.

**Example:** `["I", "loved", "the", "movie's", "plot", "!"]`

- **Pros:** Intuitive, preserves whole-word meaning.
- **Cons:** Vocabulary explodes with every new word form (e.g., "loved," "loving," "loves" are all separate tokens). Struggles badly with typos, rare words, or words never seen during training (**out-of-vocabulary / OOV problem**) — the model just sees an `<UNK>` token and loses all information.

## 2. Character-Level Tokenization

Splits text into individual characters.

**Example:** `["I", " ", "l", "o", "v", "e", "d", " ", "t", "h", "e", ...]`

- **Pros:** Tiny, fixed vocabulary (just the alphabet + punctuation + digits). No OOV problem — any word can be built from characters.
- **Cons:** Sequences become very long (a 5-word sentence becomes 25+ tokens), making it harder for the model to learn long-range meaning, and it's computationally expensive since transformers scale poorly with sequence length.

## 3. Subword Tokenization (the modern standard)

A middle ground: common words stay whole, but rare/complex words get split into meaningful chunks. This is what GPT, BERT, LLaMA, and Claude-style models actually use.

**Example:** `["I", "loved", "the", "movie", "'s", "plot", "!"]` or, if "loved" were rarer: `["I", "lov", "##ed", "the", ...]`

There are a few popular algorithms for this:

### a) Byte-Pair Encoding (BPE)
Starts with individual characters, then iteratively **merges the most frequently co-occurring pairs** into new tokens, repeating until a target vocabulary size is reached.
- Example: if "lo" and "ve" frequently appear together in training data, they get merged into "love" as one token, then further merges might combine it with "d" to form "loved" — but if "loved" isn't frequent enough, it might stay split as "lov" + "ed."
- **Used by:** GPT-2, GPT-3, RoBERTa.

### b) WordPiece
Very similar to BPE, but instead of merging the *most frequent* pair, it merges the pair that **maximizes the likelihood of the training data** (a slightly more statistically-driven merge criterion). Produces tokens like `movie` + `##'s` (the `##` prefix indicates "this continues the previous token" — no space before it).
- **Used by:** BERT, DistilBERT.

### c) Unigram Language Model (used in SentencePiece)
Works backward from BPE/WordPiece: starts with a large vocabulary of candidate subwords and **iteratively removes tokens** that least hurt the overall likelihood of the training corpus, until the target vocab size is reached. This gives a probabilistic way to pick the "best" segmentation of a sentence, and it can even produce multiple valid tokenizations for the same word.
- **Used by:** T5, ALBERT, XLNet, and many multilingual models (since it handles languages without spaces, like Japanese/Chinese, more gracefully).

### d) SentencePiece (a framework, not an algorithm itself)
Treats the input as a raw stream of Unicode characters (including spaces) rather than pre-splitting on whitespace first. This makes it language-agnostic — it works the same way on English, Chinese, or Hindi without needing separate word-boundary rules. It typically implements either BPE or Unigram internally.

## Quick Comparison

| Method | Vocab Size | Sequence Length | Handles OOV? | Used By |
|---|---|---|---|---|
| Word-level | Very large | Short | Poorly | Older NLP (Word2Vec-era) |
| Character-level | Tiny | Very long | Perfectly | Rare in modern LLMs |
| BPE | Medium | Medium | Well | GPT-2/3, RoBERTa |
| WordPiece | Medium | Medium | Well | BERT |
| Unigram/SentencePiece | Medium | Medium | Well | T5, multilingual models |

**Bottom line:** Subword tokenization (especially BPE and its variants) won out in modern LLMs because it strikes the best balance — it keeps vocabulary size manageable, keeps sequences reasonably short, and gracefully handles words it's never seen by breaking them into familiar pieces (e.g., an unfamiliar word like "tokenization" might become `["token", "ization"]`, letting the model infer meaning from known sub-parts).