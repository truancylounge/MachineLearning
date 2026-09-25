Let's compute this as full matrices rather than one row at a time — this is what actually happens inside the model (all 8 tokens processed simultaneously in one matrix multiply).

## Step 1: The Full Q and K Matrices

Using the same toy weights from before (`W^Q`, `W^K` applied to each token's embedding), here's every token's Query and Key vector:

| Token | Q | K |
|---|---|---|
| BOS | [0.1, 0.0] | [0.1, 0.1] |
| A | [0.2, 0.1] | [0.1, 0.2] |
| cute | [0.7, 0.9] | [1.2, 0.7] |
| teddy | [0.9, 0.8] | [1.2, 0.9] |
| bear | [0.8, 1.0] | [1.5, 0.8] |
| is | [0.6, 0.7] | [0.8, 0.6] |
| reading | [0.9, 0.7] | [1.4, 0.9] |
| EOS | [0.1, 0.1] | [0.2, 0.1] |

**Shapes:** `Q` is 8×2 (8 tokens, d_k=2), `K` is 8×2, so `Kᵀ` is 2×8. Multiplying `Q × Kᵀ` gives an **8×8** matrix — every token's query compared against every token's key, all at once.

## Step 2: Two Cells Computed by Hand (to show the pattern)

**Cell (teddy, bear)** — how much does "teddy" attend to "bear"?
$$q_{teddy} \cdot k_{bear} = (0.9)(1.5) + (0.8)(0.8) = 1.35 + 0.64 = 1.99$$

**Cell (reading, teddy)** — how much does "reading" attend to "teddy"?
$$q_{reading} \cdot k_{teddy} = (0.9)(1.2) + (0.7)(0.9) = 1.08 + 0.63 = 1.71$$

Every one of the 64 cells in the 8×8 grid is computed exactly this way — row `i` of `Q` dotted with row `j` of `K`.

## Step 3: Full Raw Score Matrix (QKᵀ, 8×8)

| Q\K | BOS | A | cute | teddy | bear | is | read | EOS |
|---|---|---|---|---|---|---|---|---|
| **BOS** | 0.01 | 0.01 | 0.12 | 0.12 | 0.15 | 0.08 | 0.14 | 0.02 |
| **A** | 0.03 | 0.04 | 0.31 | 0.33 | 0.38 | 0.22 | 0.37 | 0.05 |
| **cute** | 0.16 | 0.25 | 1.47 | 1.65 | 1.77 | 1.10 | 1.79 | 0.23 |
| **teddy** | 0.17 | 0.25 | 1.64 | 1.80 | **1.99** | 1.20 | 1.98 | 0.26 |
| **bear** | 0.18 | 0.28 | 1.66 | 1.86 | 2.00 | 1.24 | 2.02 | 0.26 |
| **is** | 0.13 | 0.20 | 1.21 | 1.35 | 1.46 | 0.90 | 1.47 | 0.19 |
| **reading** | 0.16 | 0.23 | 1.57 | **1.71** | 1.91 | 1.14 | 1.89 | 0.25 |
| **EOS** | 0.02 | 0.03 | 0.19 | 0.21 | 0.23 | 0.14 | 0.23 | 0.03 |

(Notice `teddy→bear = 1.99` and `reading→teddy = 1.71` match the hand calculations above — the matrix just holds all 64 of these simultaneously.)

## Step 4: Scale by √d_k (÷1.414)

| Q\K | BOS | A | cute | teddy | bear | is | read | EOS |
|---|---|---|---|---|---|---|---|---|
| **BOS** | 0.007 | 0.007 | 0.085 | 0.085 | 0.106 | 0.057 | 0.099 | 0.014 |
| **A** | 0.021 | 0.028 | 0.219 | 0.233 | 0.269 | 0.156 | 0.262 | 0.035 |
| **cute** | 0.113 | 0.177 | 1.040 | 1.167 | 1.252 | 0.778 | 1.266 | 0.163 |
| **teddy** | 0.120 | 0.177 | 1.160 | 1.273 | 1.407 | 0.849 | 1.400 | 0.184 |
| **bear** | 0.127 | 0.198 | 1.174 | 1.315 | 1.414 | 0.877 | 1.429 | 0.184 |
| **is** | 0.092 | 0.141 | 0.856 | 0.955 | 1.033 | 0.636 | 1.040 | 0.134 |
| **reading** | 0.113 | 0.163 | 1.110 | 1.209 | 1.351 | 0.806 | 1.336 | 0.177 |
| **EOS** | 0.014 | 0.021 | 0.134 | 0.148 | 0.163 | 0.099 | 0.163 | 0.021 |

## Step 5: Softmax Each Row → Attention Weight Matrix

Each row is independently normalized into a probability distribution (sums to 1 across each row):

| Q\K | BOS | A | cute | teddy | bear | is | read | EOS |
|---|---|---|---|---|---|---|---|---|
| **BOS** | 0.119 | 0.119 | 0.128 | 0.128 | 0.131 | 0.125 | 0.130 | 0.120 |
| **A** | 0.109 | 0.110 | 0.133 | 0.135 | 0.140 | 0.125 | 0.139 | 0.111 |
| **cute** | 0.060 | 0.064 | 0.151 | 0.171 | 0.187 | 0.116 | 0.189 | 0.063 |
| **teddy** | 0.054 | 0.058 | 0.154 | 0.172 | 0.197 | 0.113 | 0.195 | 0.058 |
| **bear** | 0.054 | 0.058 | 0.153 | 0.176 | 0.194 | 0.113 | 0.197 | 0.057 |
| **is** | 0.069 | 0.073 | 0.148 | 0.164 | 0.177 | 0.119 | 0.178 | 0.072 |
| **reading** | 0.057 | 0.060 | 0.153 | 0.169 | 0.195 | 0.113 | 0.192 | 0.060 |
| **EOS** | 0.115 | 0.116 | 0.130 | 0.132 | 0.134 | 0.125 | 0.134 | 0.116 |

Notice `BOS` and `EOS` end up with fairly **flat/uniform** distributions — their Query vectors are small and don't strongly discriminate between tokens. Content words (`cute`, `teddy`, `bear`, `reading`) all show a clear preference for each other, since their Key/Query vectors were larger and more differentiated.

## Step 6: Multiply by V — Final Output Matrix Z

$$Z = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This is another matrix multiplication: the 8×8 attention-weight matrix times the 8×2 `V` matrix produces the final **8×2 output** — one new, context-blended vector per token:

| Token | Z (new contextualized vector) |
|---|---|
| BOS | [0.549, 0.922] |
| A | [0.571, 0.962] |
| cute | [0.674, 1.157] |
| teddy | [0.686, 1.183] |
| **bear** | **[0.688, 1.183]** ✓ matches earlier lecture |
| is | [0.653, 1.118] |
| reading | [0.680, 1.171] |
| EOS | [0.559, 0.940] |

`bear`'s row confirms the exact value from the original walkthrough — this is the same computation, just now shown as it actually runs: **one matrix multiply produces all 8 rows of Z simultaneously**, rather than looping token-by-token. This 8×2 matrix is what gets passed forward into the residual connection + LayerNorm (Section 6) and then into the Feed-Forward Network (Section 7).

# [Notes] What is Matrix Transpose?
**Transpose** just means flipping a matrix so rows become columns and columns become rows. `Kᵀ` ("K transpose") is literally `K` turned on its side — no values change, only their arrangement.

## Why K starts as 8×2

`K` has one row per token, one column per dimension (`d_k = 2`):

$$K = \begin{bmatrix} 0.1 & 0.1 \\ 0.1 & 0.2 \\ 1.2 & 0.7 \\ 1.2 & 0.9 \\ 1.5 & 0.8 \\ 0.8 & 0.6 \\ 1.4 & 0.9 \\ 0.2 & 0.1 \end{bmatrix} \begin{matrix} \leftarrow \text{BOS}\\ \leftarrow \text{A}\\ \leftarrow \text{cute}\\ \leftarrow \text{teddy}\\ \leftarrow \text{bear}\\ \leftarrow \text{is}\\ \leftarrow \text{reading}\\ \leftarrow \text{EOS} \end{matrix}$$

8 rows (tokens) × 2 columns (dimensions) → shape **(8, 2)**.

## Transposing it → 2×8

$$K^T = \begin{bmatrix} 0.1 & 0.1 & 1.2 & 1.2 & 1.5 & 0.8 & 1.4 & 0.2 \\ 0.1 & 0.2 & 0.7 & 0.9 & 0.8 & 0.6 & 0.9 & 0.1 \end{bmatrix}$$

Each **row** of `K` (a token's key vector) becomes a **column** of `Kᵀ`. So column 1 of `Kᵀ` is BOS's key `[0.1, 0.1]`, column 5 is bear's key `[1.5, 0.8]`, and so on. Shape flips to **(2, 8)**.

## Why we need this specific shape — it's a matrix multiplication requirement

The rule for multiplying two matrices `A × B`: the number of **columns in A** must equal the number of **rows in B**. The result takes the outer dimensions: `(rows of A) × (columns of B)`.

| Attempt | Shapes | Valid? | Why |
|---|---|---|---|
| `Q × K` | (8,2) × (8,2) | ❌ | Q has 2 columns, K has 8 rows — don't match |
| `Q × Kᵀ` | (8,2) × (2,8) | ✅ | Q has 2 columns, Kᵀ has 2 rows — match! → result is (8,8) |

That "match" isn't just bookkeeping — it's what makes the dot-product interpretation work. **Column `j` of `Kᵀ` is exactly row `j` of the original `K`** (bear's key vector, teddy's key vector, etc.), unchanged. So computing "row `i` of Q times column `j` of Kᵀ" is mathematically identical to "row `i` of Q dotted with row `j` of K" — which is exactly the pairwise query-vs-key comparison we want. The transpose is just the mechanical trick that lets standard matrix multiplication produce that 8×8 grid of pairwise dot products in one operation, rather than us writing a manual double loop.

# [Notes] Is Cosine Similarity in Embeddings same as Dot Product?
They're related but **not the same** — cosine similarity is a *normalized* version of dot product. The formula makes this explicit:

$$\text{cosine similarity}(A, B) = \frac{A \cdot B}{\|A\| \, \|B\|} = \frac{\text{dot product}}{\text{magnitude of A} \times \text{magnitude of B}}$$

So cosine similarity = dot product, **divided by** the lengths (magnitudes) of both vectors. This division is what strips out magnitude and leaves only the *angle* between the vectors.

## The key consequence: dot product cares about vector length, cosine doesn't

**Worked example.** Take two vectors:

$$A = [3, 4] \qquad B = [4, 3]$$

$$\text{dot}(A,B) = (3)(4) + (4)(3) = 24 \qquad \|A\| = \sqrt{3^2+4^2} = 5 \qquad \|B\| = \sqrt{4^2+3^2} = 5$$

$$\text{cosine}(A,B) = \frac{24}{5 \times 5} = 0.96$$

Now scale `A` by a factor of 2 (same *direction*, double the *length*) → `C = [6, 8]`:

| | dot(·, B) | cosine(·, B) |
|---|---|---|
| **A = [3,4]** | 24 | 0.96 |
| **C = [6,8]** (same direction, 2× length) | **48** | **0.96** |

The dot product **doubled** just because the vector got longer — even though it points in exactly the same direction as before. Cosine similarity stayed **identical**, because it only measures the angle, completely ignoring how long the vector is.

## When are they actually equal?

$$\text{If } \|A\| = \|B\| = 1 \text{ (unit-normalized vectors)}, \text{ then cosine}(A,B) = \text{dot}(A,B)$$

If both vectors are pre-normalized to length 1 before comparison, the denominator becomes `1 × 1 = 1`, and cosine similarity collapses to being exactly the dot product. This is why some embedding pipelines (e.g., many sentence-embedding models) normalize every vector to unit length right after generating it — doing so means you can use the cheaper, simpler dot product at search/retrieval time and get results mathematically identical to cosine similarity.

## Tying this back to the `QKᵀ` attention example

In self-attention, we used the **raw dot product** (`QKᵀ/√d_k`) — not cosine similarity. This is a deliberate design choice: magnitude is allowed to matter. A token whose Query/Key vector has learned to be "louder" (larger magnitude) can produce a stronger attention signal than one of equal *direction* but smaller magnitude. If Transformers used cosine similarity instead, two tokens pointing the same direction would always score identically regardless of how confidently/strongly the model represents them — you'd lose that extra degree of expressiveness. (A few specialized architectures *do* experiment with cosine-based attention for training stability, but the standard Transformer intentionally keeps raw, magnitude-sensitive dot products.)