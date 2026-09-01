# UseCase: Teaching Word2Vec & Transformers via "A cute teddy bear is reading"

Alright, settle in — let's use one running example sentence for everything: **"A cute teddy bear is reading."**

We'll tokenize it as: `[A, cute, teddy, bear, is, reading]` — 6 tokens, positions 1 through 6.

---

# Part 1: Word2Vec — Learning What "teddy" Means from Its Neighbors

Word2Vec's core idea: **"You shall know a word by the company it keeps."** It doesn't know what a teddy bear *is* — it just learns that the token "teddy" tends to appear near "bear," "cute," "toy," "stuffed," etc., across millions of sentences. Over time, "teddy" ends up living close to those words in vector space.

There are two training setups:

### Skip-Gram: Predict the neighbors from the center word
Pick "teddy" as the center word, with a context window of size 2. The model is trained to predict:
```
teddy → A (probably)
teddy → cute (probably)
teddy → bear (probably)
teddy → is (probably)
```
Each time it's wrong, it nudges "teddy's" vector slightly.

### CBOW (Continuous Bag of Words): Predict the center word from neighbors
Reverse direction — given `[A, cute, ___, bear, is]`, predict the blank is "teddy."

**What comes out the other end:** a dense vector for "teddy," maybe 100-300 numbers long, e.g. `[0.21, -0.05, 0.88, ...]`. This vector isn't interpretable by itself, but its *position relative to other word vectors* is meaningful — "teddy" ends up near "stuffed," "plush," "toy," and far from "mortgage" or "spreadsheet."

**Key limitation to flag for your students:** Word2Vec gives "teddy" **one fixed vector**, no matter the sentence. "Teddy" the bear and "Teddy" the person's nickname get the *same* embedding. That's the exact problem Transformers were built to solve.

---

# Part 2: Transformer Architecture — Understanding "teddy" *in context*

Now let's walk our sentence through a Transformer, step by step, the way you'd walk a class through it on a whiteboard.

## Step 1: Tokenization + Input Embedding
Each token gets converted to an initial vector (often via a learned embedding table, conceptually similar to Word2Vec's output):
```
A       → [0.1, 0.4, ...]
cute    → [0.9, 0.2, ...]
teddy   → [0.3, 0.7, ...]
bear    → [0.5, 0.5, ...]
is      → [0.2, 0.1, ...]
reading → [0.6, 0.8, ...]
```

## Step 2: Positional Encoding
Transformers process all tokens **in parallel** (unlike RNNs, which go word-by-word), so they have no built-in sense of order. We inject a positional signal (via sine/cosine functions or learned position vectors) into each token's embedding, so the model knows "teddy" is token #3, not #5.
```
teddy_embedding + position_3_encoding = teddy's final input vector
```

## Step 3: Self-Attention — "Who should 'bear' pay attention to?"
This is the heart of the Transformer. For every token, we compute three vectors via learned weight matrices:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I actually offer if picked?"

For the word **"bear,"** its Query vector gets compared (via dot product) against every other token's Key vector:
```
bear's Query · teddy's Key  → high score (strongly related)
bear's Query · cute's Key   → medium score
bear's Query · reading's Key → lower score
bear's Query · A's Key      → very low score
```

These scores get passed through a **softmax** (turning them into a probability distribution that sums to 1) — this is called the **attention weights**. Then we take a weighted sum of all the Value vectors using these weights:
```
bear's new representation = 0.6×(teddy's Value) + 0.25×(cute's Value) + 0.1×(reading's Value) + 0.05×(A's Value)
```

**This is the magic:** "bear" is no longer just "bear" — it's now a vector that's been *contextually blended* with "teddy" and "cute," so the model represents it as "a cute teddy-bear," not a wild grizzly. This directly fixes Word2Vec's fixed-vector limitation.

*(In practice, this happens with **Multi-Head Attention** — multiple sets of Q/K/V run in parallel, each learning to focus on different relationships: one head might track grammar (subject-verb agreement between "bear" and "is"), another might track descriptive relationships ("cute" → "teddy"), etc.)*

## Step 4: Feed-Forward Network
After attention, each token's new vector passes through a small neural network (same weights applied independently to each position) that adds further non-linear transformation — think of this as the model "thinking a bit more" about each token individually after it's gathered context.

## Step 5: Add & Normalize + Stack Layers
Residual connections (add the input back to the output) and layer normalization stabilize training. This whole block (attention → feed-forward → normalize) is stacked **N times** (e.g., 12, 24, 96 layers in real models) — each layer refines the representation further, building progressively richer understanding.

---

# Part 3: Training the Model — Forward Pass → Loss → Backprop → Update

Now let's cover exactly what you asked about — the training loop. Let's say we're training the model to do **next-word prediction**: given `"A cute teddy bear is"`, predict `"reading."`

### Step 1: Forward Pass
The sentence flows through all the steps above (embeddings → positional encoding → attention layers → feed-forward layers) and, at the final layer, produces a probability distribution over the **entire vocabulary** for "what comes next":
```
P(reading) = 0.35
P(sitting) = 0.20
P(sleeping) = 0.15
P(eating)  = 0.05
... (thousands of other words, tiny probabilities)
```

### Step 2: Compute Loss (Cross-Entropy)
We compare this predicted distribution against the **true answer** ("reading" should be 1.0, everything else 0.0 — a one-hot target). **Cross-entropy loss** measures how far off the predicted distribution is from this truth:

$$\text{Loss} = -\log(P(\text{correct word}))$$

In our case: $-\log(0.35) \approx 1.05$

**Intuition:** If the model had assigned "reading" a probability of 0.35, the loss is moderate — it wasn't totally wrong, but it wasn't confident either. If it had predicted "reading" with 0.9 probability, loss would be much smaller ($-\log(0.9) \approx 0.10$). If it had assigned "reading" nearly 0 probability, loss shoots toward infinity — cross-entropy **punishes confident wrong answers very harshly**, which is exactly the behavior you want during training.

### Step 3: Backpropagation
Now we need to figure out: **which weights, in which direction, caused this loss?** Backpropagation computes the **gradient** of the loss with respect to every single weight in the network — that is, "if I nudge this specific weight up or down slightly, how much does the loss increase or decrease?"

This is done via the **chain rule** of calculus, propagating the error signal *backward* through the network — from the final output layer, back through each Transformer layer, back through attention weights, all the way to the input embeddings. Every weight matrix (Q, K, V projections, feed-forward weights, embedding table) gets its own gradient.

### Step 4: Weight Update (Gradient Descent)
Once we know the gradient (direction of steepest *increase* in loss) for every weight, we update each weight by moving it a small step in the **opposite** direction (to *decrease* loss):

$$w_{\text{new}} = w_{\text{old}} - \eta \times \frac{\partial \text{Loss}}{\partial w}$$

Where $\eta$ (eta) is the **learning rate** — how big a step to take. Too large, and training becomes unstable (overshoots the target); too small, and training crawls forward glacially. In practice, modern Transformers use adaptive optimizers like **Adam**, which adjust the effective step size per-weight based on recent gradient history.

### Step 5: Repeat — Thousands / Millions of Times
This whole cycle (forward pass → loss → backward pass → weight update) is one **training step**. It's repeated across every sentence in the training corpus, for many passes (**epochs**), gradually nudging every weight in the network — the embedding table, every attention head's Q/K/V matrices, every feed-forward layer — until the model reliably predicts "reading" (and millions of other next-words) with high confidence.

---

## The Full Picture, Tied Together

| Stage | What Happens | Analogy |
|---|---|---|
| **Word2Vec** | Learn static word meaning from neighboring words | Teddy bear gets a fixed "identity card" based on who it usually stands next to |
| **Transformer Embedding + Position** | Convert tokens to vectors + inject order | Give each word a name tag *and* a position number in line |
| **Self-Attention** | Let each token dynamically gather context from others | "Bear" turns to "teddy" and "cute" and says, "help me understand who I am *in this sentence*" |
| **Feed-Forward** | Each token processes its gathered context further | Quiet individual reflection after group discussion |
| **Forward Pass → Loss** | Model guesses the next word, we measure how wrong it was | Professor grades the exam, cross-entropy is the harshness of the grading curve |
| **Backprop** | Trace the error backward to find each weight's contribution to the mistake | Post-mortem: "Which of your assumptions led you astray, and by how much?" |
| **Weight Update** | Nudge every weight slightly to reduce that error next time | Study the mistakes, adjust your approach before the next exam |

**One more thing worth flagging for your students:** static embeddings (Word2Vec) are typically trained *once*, then used as a fixed lookup table. Transformer weights (including their internal embeddings) are trained *end-to-end*, jointly with everything else — the embedding table itself gets updated by backprop too, meaning the model's very notion of "what a word means" evolves as training progresses, shaped directly by how well it's predicting the next word.