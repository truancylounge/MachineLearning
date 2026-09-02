# The Transformer Encoder-Decoder Architecture
### A Complete Walkthrough, from Embeddings to Output Probabilities

**Worked example sentence:** `[BOS] A cute teddy bear is reading. [EOS]`

Tokenized as 8 tokens: `[BOS], A, cute, teddy, bear, is, reading, [EOS]`

---

## 0. Setup: What We Start With

Before any "Transformer magic" happens, every token is converted into a raw embedding vector via a lookup table (learned jointly with the rest of the model, similar in spirit to Word2Vec but trained end-to-end). Assume our model uses embedding dimension **d_model = 4** (real models use 512, 768, 4096+; we use 4 so we can do the math by hand).

| Token | Raw Embedding (before position info) |
|---|---|
| `[BOS]` | [0.1, 0.0, 0.0, 0.1] |
| `A` | [0.2, 0.1, 0.0, 0.0] |
| `cute` | [0.6, 0.8, 0.1, 0.3] |
| `teddy` | [0.7, 0.6, 0.2, 0.4] |
| `bear` | [0.5, 0.7, 0.3, 0.5] |
| `is` | [0.1, 0.2, 0.5, 0.1] |
| `reading` | [0.3, 0.1, 0.6, 0.7] |
| `[EOS]` | [0.0, 0.0, 0.1, 0.1] |

These are made-up illustrative numbers — not from a real trained model — chosen so the arithmetic stays clean while still showing you exactly how the mechanics work.

---

## 1. Position-Aware Embeddings — What and Why

### What it is
A Transformer processes **all tokens simultaneously** — there's no sequential loop like in an RNN/LSTM that naturally "remembers" order. If you feed the raw embeddings straight into the attention mechanism, the model would treat the sentence as a **bag of tokens** — `"teddy bear cute a is reading"` would look identical to `"a cute teddy bear is reading"`. That's a disaster for language, where order carries meaning.

**Positional encoding** solves this by injecting information about *where* each token sits in the sequence directly into its embedding, before anything else happens.

### How it's computed
The original Transformer paper uses fixed sine/cosine functions of position:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where `pos` is the token's position (0, 1, 2, ... 7 for our 8 tokens) and `i` indexes the embedding dimension. This produces a unique wave-pattern "fingerprint" per position that the model can learn to interpret. (Many modern models — GPT, BERT — instead use a **learned** positional embedding table, exactly like the word embedding table, and just add it in. Either way, the *purpose* is identical.)

### Why it's done *after* the embedding lookup (not before, not instead of)
1. The embedding table's job is to encode **what the token means** (its identity/semantics). The positional encoding's job is to encode **where it is**. These are two orthogonal pieces of information, and addition lets the model carry both simultaneously in the same vector without needing a separate pathway.
2. It has to happen *after* embedding lookup because positional encoding is added **elementwise** to the embedding vector — you need the embedding vector to already exist before you can add anything to it:

$$x_i = \text{Embedding}(token_i) + PE(pos_i)$$

3. It must happen *before* attention, because self-attention has **no innate sense of order** — it only sees a set of vectors and compares them pairwise. If positional info isn't baked into the vectors *before* they enter attention, the order information is permanently lost and can never be recovered downstream.

**For our sentence:** `bear` is the 5th token (position 4, zero-indexed). Its final input vector becomes:

$$x_{bear} = [0.5, 0.7, 0.3, 0.5] + PE(4) = [0.5, 0.7, 0.3, 0.5] \text{ (we'll treat this as already including position info for simplicity going forward)}$$

---

## 2. Projecting into Query, Key, Value Spaces — The Core Idea

This is the part that confuses almost everyone the first time, so let's slow down.

### The intuition first
Self-attention is fundamentally about answering: **"For each word, which other words in the sentence should I pull information from, and how much?"**

To make that computable, every token's vector gets projected into **three different learned "roles"**:

| Role | Question it answers | Analogy |
|---|---|---|
| **Query (Q)** | "What am I looking for?" | The search term you type into a search engine |
| **Key (K)** | "What do I have to offer, as a label?" | The title/tags on a document in a search index |
| **Value (V)** | "What's the actual content I'll hand over if picked?" | The full content of that document |

Every token gets a Query, a Key, *and* a Value — because every token simultaneously **asks** a question (as Query) and **can be asked about** (as Key/Value) by every other token, including itself.

### How the projection actually works
This isn't three different embeddings pulled from different tables — it's the **same input vector** `x`, multiplied by three different learned weight matrices:

$$Q = XW^Q \qquad K = XW^K \qquad V = XW^V$$

Where:
- `X` is the matrix of all 8 token vectors stacked together (shape: 8 × d_model, i.e., 8 × 4 in our toy example)
- `W^Q`, `W^K`, `W^V` are learned weight matrices (shape: d_model × d_k, e.g., 4 × 2), **different from each other**, and shared across all tokens and positions
- The outputs `Q`, `K`, `V` each have shape 8 × d_k (one row per token)

**Why "projecting onto 3 spaces"?** Think of it geometrically: `X` lives in one 4-dimensional space (the "meaning" space from the embedding). Multiplying by `W^Q` doesn't just resize the vector — it **rotates and reshapes** it into a *new* space specifically optimized for "being compared as a question." `W^K` reshapes the same original vector into a *different* space optimized for "being compared as an answer/label." `W^V` reshapes it into yet another space optimized for "being the actual content passed along." Since `W^Q`, `W^K`, `W^V` are learned independently, the network can carve out three specialized geometries from one input — none of these three "views" of the token look alike numerically, even though they all originated from the same `x`.

### Worked numbers
Let `d_k = 2`. Take these toy weight matrices (learned in a real model; invented here for demonstration):

$$W^Q = \begin{bmatrix}1&0\\0&1\\1&1\\0&0\end{bmatrix} \qquad W^K = \begin{bmatrix}0&1\\1&0\\1&1\\1&0\end{bmatrix} \qquad W^V = \begin{bmatrix}1&1\\0&1\\1&0\\0&1\end{bmatrix}$$

For `bear` = [0.5, 0.7, 0.3, 0.5], matrix multiplication (dot the row vector against each column of `W^Q`):

$$q_{bear} = [0.5, 0.7, 0.3, 0.5] \cdot W^Q = [\,(0.5{\cdot}1{+}0.7{\cdot}0{+}0.3{\cdot}1{+}0.5{\cdot}0),\ (0.5{\cdot}0{+}0.7{\cdot}1{+}0.3{\cdot}1{+}0.5{\cdot}0)\,] = [0.8,\ 1.0]$$

Doing the same with `W^K` and `W^V` for every token gives us full Q, K, V matrices. For reference, here are the Key and Value vectors for all 8 tokens (computed the same way):

| Token | Key (K) | Value (V) |
|---|---|---|
| `[BOS]` | [0.1, 0.1] | [0.1, 0.2] |
| `A` | [0.1, 0.2] | [0.2, 0.3] |
| `cute` | [1.2, 0.7] | [0.7, 1.7] |
| `teddy` | [1.2, 0.9] | [0.9, 1.7] |
| `bear` | [1.5, 0.8] | [0.8, 1.7] |
| `is` | [0.8, 0.6] | [0.6, 0.4] |
| `reading` | [1.4, 0.9] | [0.9, 1.1] |
| `[EOS]` | [0.2, 0.1] | [0.1, 0.1] |

---

## 3. The Self-Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In words, four steps:
1. **Score** every Query against every Key (matrix multiply `QK^T`)
2. **Scale** the scores down by `√d_k`
3. **Normalize** the scores into a probability distribution per row (softmax)
4. **Aggregate**: use those probabilities as weights to blend the Value vectors

Let's run `bear`'s query through all four steps.

### Step 1 — Score (QKᵀ)
Take `q_bear = [0.8, 1.0]` and dot it against every token's Key vector:

| Against | Dot Product Calculation | Raw Score |
|---|---|---|
| `[BOS]` | 0.8(0.1) + 1.0(0.1) | 0.18 |
| `A` | 0.8(0.1) + 1.0(0.2) | 0.28 |
| `cute` | 0.8(1.2) + 1.0(0.7) | 1.66 |
| `teddy` | 0.8(1.2) + 1.0(0.9) | 1.86 |
| `bear` | 0.8(1.5) + 1.0(0.8) | 2.00 |
| `is` | 0.8(0.8) + 1.0(0.6) | 1.24 |
| `reading` | 0.8(1.4) + 1.0(0.9) | 2.02 |
| `[EOS]` | 0.8(0.2) + 1.0(0.1) | 0.26 |

### Step 2 — Scale by √d_k
`d_k = 2`, so `√d_k ≈ 1.414`. Divide every score above by 1.414:

`[0.13, 0.20, 1.17, 1.32, 1.41, 0.88, 1.43, 0.18]`

### Step 3 — Softmax
Softmax exponentiates each score and normalizes so all 8 weights sum to 1:

| Token | Attention Weight |
|---|---|
| `[BOS]` | 0.054 |
| `A` | 0.058 |
| `cute` | 0.153 |
| `teddy` | 0.176 |
| `bear` | 0.194 |
| `is` | 0.113 |
| `reading` | 0.197 |
| `[EOS]` | 0.057 |

(Sums to ~1.0.) Notice `bear` ends up attending heavily to `teddy`, `cute`, itself, and `reading` — and barely at all to `[BOS]`, `A`, `[EOS]`. In a *real trained* model, these weights would reflect learned linguistic structure (e.g., "bear" attending strongly to its describing adjective "cute" and its head-noun partner "teddy"); here they're an artifact of our made-up weight matrices, but the mechanics are identical.

### Step 4 — Weighted sum of Values
Multiply each attention weight by its corresponding Value vector and sum:

$$z_{bear} = 0.054[0.1,0.2] + 0.058[0.2,0.3] + 0.153[0.7,1.7] + 0.176[0.9,1.7] + 0.194[0.8,1.7] + 0.113[0.6,0.4] + 0.197[0.9,1.1] + 0.057[0.1,0.1]$$

$$z_{bear} \approx [0.69,\ 1.18]$$

**This is the payoff:** `bear`'s new representation, `z_bear`, is no longer just "bear" in isolation — it's a blend, weighted by relevance, of `bear`, `teddy`, `cute`, and `reading`'s *Value* vectors. The model has literally computed "bear, understood in the context of being a cute teddy that's being read to/about."

---

## 4. The Math, Explained Properly

### Why matrix multiplication for QKᵀ?
The dot product between two vectors is a direct measure of **how aligned/similar their directions are** in vector space. If `q_bear` and `k_teddy` point in similar directions (learned, over training, to happen when two tokens are semantically/syntactically related), their dot product is large. If they point in unrelated directions, the dot product is small or negative.

Computing `QKᵀ` as one matrix multiplication is just a highly efficient way of computing **every pairwise dot product between every Query and every Key simultaneously** — instead of writing a nested loop `for each query: for each key: dot(q,k)`, one matrix multiply produces the entire 8×8 grid of scores in a single GPU-parallelizable operation. This is a huge part of why Transformers train faster than RNNs — RNNs are inherently sequential, while this whole score matrix is computed in parallel.

### Why divide by √d_k specifically?
This is subtle but important. Assume, for a moment, that the individual components of `q` and `k` are roughly independent random values with mean 0 and variance 1 (a reasonable approximation early in training, before weights specialize). The dot product `q·k = Σᵢ qᵢkᵢ` is a sum of `d_k` such terms. By basic probability, the **variance of a sum of d_k independent terms is d_k times the variance of one term** — so:

$$\text{Var}(q \cdot k) = d_k$$

This means as `d_k` grows (larger embedding dimensions — real models use 64, 128 per head), the raw dot products swing to much larger magnitudes purely as a side effect of dimensionality, **not because the tokens are more related.** Large-magnitude scores, when passed through softmax, push the output toward an extremely peaked (near one-hot) distribution — almost all probability mass on the single highest score, everything else near zero.

**Why that's bad:** softmax's gradient is largest near the middle of its range and shrinks toward zero as the distribution saturates toward one-hot. If the pre-softmax scores are too large, the gradients backpropagated through softmax become **vanishingly small**, and the model essentially stops learning from that attention computation — a classic vanishing gradient problem.

Dividing by `√d_k` rescales the variance back down to 1 (since `Var(q·k / √d_k) = d_k / d_k = 1`), keeping the scores in a well-behaved range regardless of how large `d_k` is, which keeps softmax's gradients healthy and training stable. It's not an arbitrary choice — `√d_k` is precisely the factor that cancels out the dimensionality-driven variance growth.

---

## 5. Multi-Head Attention & Gradient Descent

### Multi-head attention: doing this in parallel, multiple times
Instead of computing *one* set of Q, K, V projections with `d_k = d_model`, the Transformer **splits** the projection into `h` parallel "heads," each with its own independently-learned `W^Q_i, W^K_i, W^V_i` and a smaller dimension `d_k = d_model / h`.

For example, with `d_model = 512` and `h = 8` heads, each head works in `d_k = 64` dimensions. Each head:
1. Independently projects `X` into its own `Q_i, K_i, V_i`
2. Independently runs the full `softmax(QᵢKᵢᵀ/√d_k)Vᵢ` attention computation
3. Produces its own output `z_i` (shape: 8 tokens × 64 dims)

Then all `h` outputs are **concatenated** back together (8 × 64 concatenated 8 times → 8 × 512) and passed through one more learned matrix `W^O` to blend them back into the model's working dimension:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(z_1, z_2, ..., z_h)W^O$$

### Why bother — what's the benefit?
A single attention head can only learn **one** notion of "relatedness" at a time — its softmax distribution is a single weighted average, so it has to compromise if a word needs to attend to different things for different reasons. Multiple heads let the model learn **several relationship types in parallel**, e.g.:
- Head 1 might specialize in **syntactic** relationships (subject "bear" ↔ verb "is")
- Head 2 might specialize in **descriptive/modifier** relationships ("teddy" ↔ "cute")
- Head 3 might track **positional proximity** (nearby words in general)
- Head 4 might capture **long-range dependencies** (e.g., linking a pronoun back to its noun several words earlier)

This is empirically shown to matter enormously — models with multi-head attention consistently outperform single-head models of equivalent total size, because each head becomes a specialized "lens" on the sentence, and the concatenation + `W^O` lets the model combine all these lenses into one rich representation.

### Gradient Descent (how any of these weights ever get learned)
None of `W^Q, W^K, W^V, W^O` (or the embedding table, or the feed-forward weights) are hand-designed — they all start as small random numbers and are learned through training:

1. **Forward pass:** run the sentence through the whole network (as above) to get an output prediction (e.g., a probability distribution over the next word).
2. **Compute loss:** compare the prediction to the true answer using a loss function — typically **cross-entropy loss** for language tasks:
   $$\text{Loss} = -\log(P(\text{correct token}))$$
   This penalizes the model heavily when it assigns low probability to the correct answer.
3. **Backpropagation:** using the chain rule, compute the **gradient** of the loss with respect to *every single weight* in the network — i.e., "if I nudge this specific number in `W^Q` up slightly, does the loss go up or down, and by how much?" This gradient signal flows backward from the output layer, through the decoder, through cross-attention, through the encoder, all the way to the embedding table.
4. **Update weights:**
   $$W_{new} = W_{old} - \eta \cdot \frac{\partial \text{Loss}}{\partial W}$$
   Where `η` (the learning rate) controls step size. In practice, adaptive optimizers like **Adam** adjust each weight's effective step size individually based on the history of its recent gradients, which speeds up and stabilizes convergence versus plain gradient descent.
5. **Repeat** over millions of sentences, many epochs — gradually, `W^Q` "learns" to project tokens such that grammatically/semantically related tokens produce high dot products with each other's `W^K` projections, purely because doing so reduces the cross-entropy loss on the training data. Nobody designs these matrices by hand; they emerge entirely from this iterative error-correction process.

---

## 6. The Self-Attention Layer (as a full sublayer)

Sections 2–5 described the *internal mechanics* of attention. But inside an actual encoder block, "the self-attention layer" refers to the whole sublayer, which wraps multi-head attention with two more critical ingredients:

$$\text{output} = \text{LayerNorm}(X + \text{MultiHead}(Q,K,V))$$

- **Residual (skip) connection** — the original input `X` is added back to the attention output. This matters enormously for deep networks (Transformers commonly stack 12–96+ layers): it gives gradients a direct, unimpeded path back to earlier layers during backpropagation, preventing them from vanishing as they pass through many layers, and it means each layer only needs to learn a "correction" or "refinement" to add on top of its input, rather than reconstructing everything from scratch.
- **Layer Normalization** — rescales each token's vector (across its own dimensions) to have stable mean/variance. This keeps activations in a consistent numeric range as they flow through dozens of stacked layers, which dramatically stabilizes and speeds up training.

So "the self-attention layer" = Multi-Head Attention + Add (residual) + Normalize, all together, as one building block that gets stacked N times in the encoder.

---

## 7. The Feed-Forward Network (FFN)

After the self-attention sublayer, each token's vector independently passes through a small, fully-connected neural network — applied **identically and independently to every token position** (no mixing across tokens happens here; that's exactly what attention already did):

$$\text{FFN}(z) = \max(0,\ zW_1 + b_1)W_2 + b_2$$

- First linear layer **expands** the dimension (e.g., from `d_model = 512` up to `d_ff = 2048`)
- A non-linearity (**ReLU** in the original paper, often **GELU** in modern models) is applied
- Second linear layer **projects back down** to `d_model`

This block, too, is wrapped in a residual connection + LayerNorm, just like attention:

$$\text{output} = \text{LayerNorm}(z + \text{FFN}(z))$$

**Why it's needed:** self-attention is fundamentally a **linear** weighted-averaging operation (softmax weights times a weighted sum of Values) — on its own, it can only *mix* information across tokens, not perform complex non-linear transformations of that information. The FFN is where the actual non-linear "thinking" happens — it lets the model apply a much richer, learned function to each token's now-contextualized representation. You can think of attention as "gathering the right information from the room" and the FFN as "processing that information privately, in depth."

---

## 8. The End Result of the Encoder

The encoder block (self-attention sublayer → FFN sublayer) is stacked `N` times (e.g., 6 layers in the original Transformer paper, more in larger models). After all `N` layers, we're left with a final matrix of shape **8 tokens × d_model** — one richly contextualized vector per input token.

**What is this used for?**

1. **As Keys and Values for the decoder's cross-attention** (see Section 10) — in the classic encoder-decoder Transformer (used for translation, summarization, etc.), these final encoder vectors are handed to *every* decoder layer, letting the decoder "look back" at the full input sentence while generating output.
2. **As input to a task-specific head**, in encoder-only models (like BERT): e.g., the final vector for a special `[CLS]` token gets passed to a classification layer for tasks like sentiment analysis; or every token's final vector feeds into a token-tagging head for tasks like named entity recognition.
3. **As a general-purpose sentence/document embedding** — pooling (e.g., averaging) all token vectors together produces a single dense vector representing the whole input, usable for semantic search, clustering, or retrieval (this is exactly how sentence-embedding models like Sentence-BERT work).

For our example, the encoder's output is essentially a matrix where the vector for `bear` now *encodes* "cute-teddy-bear-being-read," the vector for `reading` encodes its relationship back to "bear," and so on — every token has absorbed relevant context from the whole sentence.

---

## 9. The Decoder Process

The decoder's job (in a sequence-to-sequence setting — e.g., machine translation, or generating a caption/continuation) is to **generate the output sequence one token at a time**, using both (a) what it's generated so far, and (b) the encoder's understanding of the input.

Say we're translating our sentence into French, or continuing it. At each generation step, the decoder receives the tokens generated so far (starting with a `[BOS]` token), and its stack of decoder layers does the following, per layer:

### 9a. Masked Multi-Head Self-Attention
Same mechanism as encoder self-attention (Q, K, V projections, `softmax(QKᵀ/√d_k)V`) — but with one critical difference: a **causal mask** is applied to the score matrix *before* softmax, setting all scores for "future" positions to `-∞`. This forces token `i` to only attend to tokens `1...i`, never anything ahead of it.

**Why this matters:** during training, the decoder is fed the *entire* target sequence at once (for parallelism/speed), but it must learn to predict each token using **only** what would actually be available at generation time — i.e., it can't be allowed to "cheat" by peeking at the answer it's trying to predict. The mask enforces this.

### 9b. Cross-Attention (see Section 10 below)

### 9c. Feed-Forward Network
Identical structure to the encoder's FFN (Section 7) — expand, non-linearity, project back down, wrapped in residual + LayerNorm.

This three-part block (masked self-attention → cross-attention → FFN) is stacked `N` times, just like the encoder.

---

## 10. Cross-Attention — and Does the Decoder Have an FFN?

### What cross-attention is
Cross-attention is structurally *identical* to self-attention (same formula: `softmax(QKᵀ/√d_k)V`) — the only difference is **where Q, K, V come from**:

| | Self-Attention (encoder or masked decoder) | Cross-Attention (decoder) |
|---|---|---|
| **Query (Q)** | From the same sequence | From the **decoder's** current representations |
| **Key (K)** | From the same sequence | From the **encoder's final output** |
| **Value (V)** | From the same sequence | From the **encoder's final output** |

So in cross-attention, the decoder generates a Query ("what am I trying to figure out right now, to produce the next token?"), and that Query is compared against the **Keys of every token in the source sentence** (produced once by the encoder, reused at every decoder layer/step). The resulting attention weights determine how much of each *source* token's Value the decoder pulls in.

**Concretely:** while generating the French word for "bear" (`ours`), the decoder's Query at that step would produce high attention scores against the encoder's Key vector for `bear` (and likely `teddy`, `cute` too) — letting the decoder "look directly at" the relevant part of the input sentence at exactly the moment it needs that information, rather than relying on a single compressed summary of the whole sentence (which was the major bottleneck of older RNN-based encoder-decoder models).

### Does the decoder have a Feed-Forward Network?
**Yes.** Every decoder layer has its own FFN, structurally identical to the encoder's (Section 7) — expand → non-linear activation → project down, wrapped in a residual connection and LayerNorm. It's applied *after* cross-attention, giving the decoder the same "private processing" capability the encoder has, now operating on representations that have absorbed both (a) what's been generated so far and (b) relevant context pulled from the source sentence.

**Full decoder layer, in order:**
$$\text{Masked Self-Attention} \rightarrow \text{Add and Norm} \rightarrow \text{Cross-Attention} \rightarrow \text{Add and Norm} \rightarrow \text{FFN} \rightarrow \text{Add and Norm}$$

---

## 11. The Final Output: Linear Projection + Softmax

After the last decoder layer, we have one final vector per output position, shape `d_model` (e.g., 512-dimensional) — but we need a **probability distribution over the entire vocabulary** (which might contain 30,000–100,000+ possible tokens) to actually decide what word comes next.

Two final steps:

### Step 1 — Linear Projection
A learned weight matrix `W_{vocab}` (shape: `d_model × |vocabulary|`) projects the final decoder vector up into vocabulary-sized space, producing raw, unbounded scores called **logits**:

$$\text{logits} = z_{final} \cdot W_{vocab} + b$$

If the vocabulary has 50,000 tokens, this produces 50,000 raw numbers — one per possible next word — where larger numbers indicate the model thinks that word is more likely.

### Step 2 — Softmax
Softmax converts these raw logits into a proper probability distribution (all values between 0 and 1, summing to 1):

$$P(\text{token}_i) = \frac{e^{\text{logit}_i}}{\sum_{j} e^{\text{logit}_j}}$$

**Example output**, continuing our sentence after "A cute teddy bear is":

| Candidate next token | Probability |
|---|---|
| reading | 0.35 |
| sitting | 0.20 |
| sleeping | 0.15 |
| on | 0.08 |
| ... (thousands more, tiny probabilities) | ... |

**Choosing the actual output token** from this distribution is a separate decision (not part of the architecture itself): **greedy decoding** always picks the single highest-probability token; **sampling** draws randomly according to the probabilities (adding variety/creativity); **beam search** explores several likely sequences in parallel and keeps the overall highest-scoring one. This chosen token is then fed back into the decoder as input for generating the *next* token, and the whole process repeats until an `[EOS]` token is generated.

---

## Full Picture: Encoder-Decoder Data Flow

```
INPUT: [BOS] A cute teddy bear is reading [EOS]
   │
   ▼
Embedding Lookup + Positional Encoding  (Section 1)
   │
   ▼
┌─────────────────────────────────────┐
│  ENCODER  (repeated × N layers)      │
│  ┌─────────────────────────────┐    │
│  │ Multi-Head Self-Attention    │◄───┼── Q,K,V from same sequence (Sections 2-6)
│  │  Add & Norm                  │    │
│  ├─────────────────────────────┤    │
│  │ Feed-Forward Network         │    │  (Section 7)
│  │  Add & Norm                  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
   │
   │  Final Encoder Output (Section 8)
   │  → becomes K, V for every decoder layer's cross-attention
   ▼
┌─────────────────────────────────────┐
│  DECODER  (repeated × N layers)      │  ← fed target/generated tokens so far
│  ┌─────────────────────────────┐    │
│  │ Masked Multi-Head Self-Attn  │    │  (Section 9a)
│  │  Add & Norm                  │    │
│  ├─────────────────────────────┤    │
│  │ Cross-Attention              │◄───┼── Q from decoder, K,V from ENCODER
│  │  Add & Norm                  │    │   (Section 10)
│  ├─────────────────────────────┤    │
│  │ Feed-Forward Network         │    │  (Section 10)
│  │  Add & Norm                  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
   │
   ▼
Linear Projection → Vocabulary Logits  (Section 11)
   │
   ▼
Softmax → Probability Distribution
   │
   ▼
Next Token Selected (greedy / sampling / beam search)
   │
   ▼
[Fed back into decoder — repeat until [EOS] is generated]
```

Training all of this happens exactly as described in Section 5: forward pass → cross-entropy loss against the true next token → backpropagation of gradients through every weight matrix in this entire diagram (embedding tables, every `W^Q/W^K/W^V/W^O`, every FFN weight, `W_{vocab}`) → gradient descent updates → repeat, millions of times, across a massive training corpus, until the network reliably produces coherent, contextually-appropriate output.