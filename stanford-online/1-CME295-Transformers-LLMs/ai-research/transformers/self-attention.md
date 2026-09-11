# Synopsis
What problem it solves, through the mechanics, to a worked numerical example, and finally how it's assembled into the full transformer block.

## 1. The problem self-attention solves

Before transformers, sequence models (RNNs, LSTMs) processed tokens **one at a time, in order**, carrying a "hidden state" forward. This created two structural problems:

1. **Long-range dependencies decay.** Information about a word from 50 tokens ago has to survive being repeatedly compressed and passed through 50 sequential updates before it can influence the current word. In practice, this information degrades or vanishes.
2. **No parallelism.** Since each step depends on the previous one, you can't compute hidden states for token 5 until you've computed tokens 1–4. This makes training painfully slow on long sequences.

**Self-attention's core idea:** instead of passing information step-by-step through a chain, let **every token directly look at every other token in the sequence, all at once**, and decide how much to "pay attention to" each one. Distance in the sequence no longer matters — token 1 can attend to token 50 just as easily as to token 2. And since every token's attention computation is independent of the others, it's fully parallelizable.

## 2. The core intuition: Query, Key, Value

The mechanism is often explained with a librarian analogy, and it's genuinely useful:

- **Query (Q)** — "What am I looking for?" Each token generates a query representing what kind of information it wants from the rest of the sequence.
- **Key (K)** — "What do I have to offer?" Each token also generates a key, representing what it advertises about itself, so other tokens can decide if it's relevant to them.
- **Value (V)** — "Here's my actual content." Once a token has been identified as relevant, its value is what actually gets passed along.

The word "self" in **self**-attention means all three of Q, K, and V are derived from the **same input sequence** — every token is simultaneously asking questions (Q), advertising itself (K), and offering content (V) to every other token in the same sentence. This is in contrast to **cross-attention** (used in encoder-decoder models like the original translation transformer), where the queries come from one sequence (say, the decoder) and the keys/values come from a different sequence (the encoder's output).

## 3. The mechanics, step by step

Say your input is a sequence of `n` token embeddings, each of dimension `d_model`, stacked into a matrix `X` of shape `(n, d_model)`.

**Step 1 — Project into Q, K, V.**
You learn three weight matrices, `W_Q`, `W_K`, `W_V`, each of shape `(d_model, d_k)`, and multiply:

```
Q = X · W_Q     shape: (n, d_k)
K = X · W_K     shape: (n, d_k)
V = X · W_V     shape: (n, d_v)
```

Each row of `Q`, `K`, `V` now corresponds to one token's query, key, and value vectors. These weight matrices are **learned during training** — the model figures out, through gradient descent, what kinds of questions are useful to ask and what kinds of self-descriptions are useful to advertise.

**Step 2 — Compute attention scores via dot product.**
This is exactly the dot product concept from earlier in our conversation, applied directly: to find out how much token `i`'s query "matches" token `j`'s key, you take their dot product.

```
scores = Q · Kᵀ     shape: (n, n)
```

Every entry `scores[i][j]` is the dot product of token `i`'s query with token `j`'s key — a raw measure of how relevant token `j` is to token `i`, exactly as we discussed: large positive dot product means the vectors point in a similar direction (high relevance), near zero means unrelated.

**Step 3 — Scale.**
```
scaled_scores = scores / sqrt(d_k)
```
Why divide by `sqrt(d_k)`? As the dimensionality `d_k` grows, dot products tend to grow larger in magnitude just from having more terms summed together — even for random vectors. Very large values pushed into softmax (next step) produce extremely peaked, near-one-hot distributions with tiny gradients, which stalls learning. Scaling by `sqrt(d_k)` keeps the values in a numerically well-behaved range regardless of dimension size.

**Step 4 — Softmax, to get attention weights.**
```
attention_weights = softmax(scaled_scores, axis=-1)     shape: (n, n)
```
This converts each row into a proper probability distribution — the weights for token `i` across all tokens `j` sum to 1. Now each entry `attention_weights[i][j]` genuinely means "how much should token `i` attend to token `j`," on a 0-to-1 scale.

**Step 5 — Weighted sum of values.**
```
output = attention_weights · V     shape: (n, d_v)
```
Each token's output representation is now a **weighted blend of every token's value vector**, weighted by how relevant that token was determined to be. A token that was highly relevant (high attention weight) contributes strongly to the output; an irrelevant token contributes almost nothing.

**The full formula, all together** (from "Attention Is All You Need"):

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) V
```

## 4. Let's make this concrete with real numbersWalking through what just happened, for the sentence "The cat sat" (3 tokens, toy 4-dim embeddings):

- **Q, K, V** are each `(3, 3)` — one row per token, projected from the original 4-dim embeddings into a 3-dim space via the learned weight matrices.
- **Raw scores** `Q·Kᵀ` is a `(3,3)` matrix — entry `[1][0] = 0.58` means "cat"'s query has a fairly strong dot-product match with "The"'s key.
- **After scaling and softmax**, look at row 1 (the word "cat"): its attention weights are `[0.384, 0.250, 0.366]` — meaning "cat" attends 38.4% to "The", 25.0% to itself, and 36.6% to "sat". These sum to exactly 1.0, as required of a probability distribution.
- **The final output row for "cat"** — `[-0.722, 0.838, -1.101]` — is literally `0.384 × V[0] + 0.250 × V[1] + 0.366 × V[2]`, i.e., "cat"'s new representation is a blend of everyone's value vectors, weighted by relevance. This blended vector is what actually moves forward into the rest of the network — it's no longer *just* "cat," it's "cat, informed by its context."

## 5. Why one attention "head" isn't enough — Multi-Head Attention

A single attention computation forces the model to squeeze *all* kinds of relationships (syntactic role, coreference, semantic similarity, positional proximity, etc.) into one shared Q/K/V projection. That's restrictive — real language has many simultaneous types of relationships between words.

**The fix:** run several independent attention computations ("heads") in parallel, each with its **own** learned `W_Q`, `W_K`, `W_V` matrices, each projecting into a smaller dimension `d_k = d_model / num_heads`. Then concatenate all the heads' outputs and pass through one final linear projection:

```
head_i = Attention(X·W_Q_i, X·W_K_i, X·W_V_i)     for i = 1 ... h
MultiHead(X) = Concat(head_1, ..., head_h) · W_O
```

In practice, different heads often specialize — empirically, some heads learn to track syntactic dependencies (like subject-verb agreement), others track positional adjacency, others track coreference (pronoun-to-noun links) — all discovered automatically through training, never explicitly programmed.

## 6. Masking — controlling what a token is allowed to see

Two important variants:

**Causal (look-ahead) mask** — used in decoder-only models like GPT. When generating text token-by-token, a token must **not** be allowed to attend to future tokens (that would be cheating — peeking at the answer). This is enforced by setting the scores for all "future" positions to `-∞` before the softmax step, so their attention weight becomes exactly 0:

```
scores[i][j] = -∞   for all j > i
```

**Padding mask** — when batching sequences of different lengths together, shorter sequences get padded with dummy tokens. You mask those padding positions the same way, so real tokens never attend to meaningless padding.

## 7. Where self-attention sits inside a full Transformer block

Self-attention is one component inside a larger repeating block. A standard transformer layer looks like this:

```
input
  │
  ├──► Multi-Head Self-Attention
  │         │
  └───(+)───┘   ← residual/skip connection: add input back to attention output
        │
   Layer Normalization
        │
        ├──► Feed-Forward Network (two linear layers + activation, e.g. ReLU/GELU)
        │         │
        └───(+)───┘   ← another residual connection
              │
        Layer Normalization
              │
            output → fed into the next stacked layer
```

**Why the residual connections matter:** they let gradients flow directly backward through the network without being forced through every transformation, which is critical for training very deep stacks (modern LLMs stack dozens to over a hundred of these blocks).

**Why the feed-forward network still matters:** self-attention is good at mixing information *across* tokens, but it's a linear operation with respect to V (a weighted sum). The feed-forward network applies a per-token nonlinear transformation, which is where a lot of the model's actual "reasoning" capacity gets applied to each token's blended representation.

## 8. Tying back to positional encoding

Notice: nowhere in the self-attention formula does token **order** appear. `Attention(Q, K, V)` would produce the exact same result if you shuffled the tokens and shuffled the corresponding rows of the output — attention is fundamentally **permutation-invariant**. That's precisely why, as we discussed earlier, positional encodings have to be added to the token embeddings *before* they ever reach the attention layer — it's the only place order information gets injected into an otherwise order-blind mechanism.

## 9. Computational cost — the elephant in the room

The score matrix `Q·Kᵀ` has shape `(n, n)` — computing and storing it costs **O(n² · d)** time and **O(n²)** memory, where `n` is sequence length. This quadratic scaling in sequence length is *the* central bottleneck of transformer architectures — doubling your context window quadruples your attention compute/memory cost. This is exactly why techniques like sparse attention, sliding-window attention, and various "linear attention" approximations exist — they're all attempts to avoid paying full quadratic cost for very long sequences, at some tradeoff in modeling capacity.

## Summary, one sentence each

- **Self-attention:** every token computes a query, key, and value from itself, then updates its representation as a weighted blend of every token's value, where weights come from query-key dot-product similarity.
- **Multi-head:** run several attention computations in parallel with different learned projections, so the model can capture multiple types of relationships simultaneously.
- **Masking:** controls which tokens are allowed to attend to which others (causal for autoregressive generation, padding for batching).
- **Transformer block:** self-attention + residual connection + layer norm + feed-forward network + another residual connection + layer norm, stacked many times deep.

Want to go deeper on any piece next — e.g., a worked multi-head example with actual head-splitting numbers, why layer normalization specifically (vs batch norm) is used, or how KV-caching works at inference time to avoid recomputing attention for every previously-generated token?
