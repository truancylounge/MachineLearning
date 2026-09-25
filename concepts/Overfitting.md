**Overfitting** in the context of evaluation metrics means: your model's metric score looks great on the data it was trained/tuned on, but the score drops sharply on data it hasn't seen before. The model hasn't learned the underlying pattern — it's memorized the specifics (including noise/quirks) of the training set, which don't generalize.

## How it shows up: the train vs. test gap

Going back to our movie review classifier — suppose you track accuracy on both the **training set** (data the model learned from) and a **held-out test set** (data it's never seen) across training epochs:

| Epoch | Train Accuracy | Test Accuracy | Gap |
|---|---|---|---|
| 1 | 68% | 66% | 2 |
| 5 | 82% | 79% | 3 |
| 10 | 91% | 83% | 8 |
| 20 | 98% | 81% | 17 |
| 30 | 99.5% | 77% | 22.5 |

Notice: training accuracy keeps climbing toward near-perfect, but test accuracy **peaks around epoch 10, then actually gets worse** even as training accuracy keeps improving. That widening gap *is* overfitting — the model is increasingly memorizing specific training reviews (exact phrasings, even noise like typos) rather than learning generalizable sentiment patterns.The point where the two lines start diverging (epoch 10 here) is exactly where you'd want to stop training — this technique is called **early stopping**, and it's one of the most direct practical uses of tracking metrics on a held-out set.

## Why it happens across different metrics, not just accuracy

The same divergence pattern shows up in **precision, recall, and F1** individually, and it can happen unevenly:

| Metric | Train | Test | What this tells you |
|---|---|---|---|
| Precision | 97% | 91% | Small gap — model's "positive" calls stay fairly trustworthy |
| Recall | 99% | 68% | Large gap — model has memorized exactly which *training* reviews were positive, but misses genuinely positive *new* reviews it hasn't seen phrased that way before |
| F1 | 98% | 78% | Reflects the imbalance above |

This is useful diagnostically: a big recall gap specifically (like above) suggests the model latched onto very specific phrasings in the positive training examples rather than the general concept of positive sentiment.

## A related but distinct problem: overfitting to the *validation set itself*

This connects to the ablation discussion from earlier. Even a proper train/validation/test split isn't fully immune: if you (or an automated hyperparameter search) run **hundreds of experiments**, all evaluated against the same validation set, and keep picking whichever configuration scores highest on it, you're implicitly fitting to that validation set's specific quirks — not general performance. This is why:
- A **final, untouched test set** exists — checked only once, at the very end, never used to make any decisions during development.
- In LLM research specifically, this is called **benchmark contamination/overfitting** — a model can end up performing suspiciously well on a public benchmark (like MMLU) partly because that benchmark's questions leaked into training data, or because a research team iterated so many times against the same benchmark that they effectively tuned for it rather than for genuine capability.

## Quick reference: how to detect vs. prevent overfitting

| | Detect it | Prevent/reduce it |
|---|---|---|
| **What to do** | Compare train metric vs. test/validation metric — a widening gap = overfitting | Regularization (dropout, weight decay), early stopping, more training data, simpler model, cross-validation |

**Bottom line:** overfitting isn't a metric itself — it's a *pattern you observe across metrics*, specifically a growing gap between how a model scores on data it learned from versus data it didn't. Any single metric (accuracy, F1, etc.) reported alone, without its train/test comparison, can't tell you whether overfitting occurred — you always need both numbers side by side.