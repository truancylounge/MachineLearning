# High Level Concepts
- NLP Overview
- Tokenization
- Word representation
- RNNs
- Self attention mechanism
- Transformer Architecture

## Natural Language Processing (NLP)
- It's a field around manipulating text or computing things with text
- NLP tasks can be broadly classified in 3 buckets:
  - **Classification** i.e. using some text pass it to model and predict output i.e sentiment or intent etc
    - **Sentiment extraction**, i.e. movie reviews or product reviews or Tweets
    - **Intent detection**, semantic detection of intent of user
    - Language detection
    - Topic modeling
  - **Multi Classification**, we still have text as input, but we try to predict more than one thing 
    - **Named Entity Recognition (NER)**, given an input text label some words which indicate location, time, PII etc
    - Part of speech tagging, figuring out which words are Nouns etc. (No longer trendy)
    - **Dependency Parsing and Constituency Parsing**, these are interesting techniques used by models to parse a sentence and figure out relationship between different words
  - **Generation**, this is what GPT works on i.e. text as input and text as output
    - **Machine translation**, convert text english to german.
    - **Question Answering**
    - **Summarization**
    - **Text generation**
- **Evaluation Metrics** for Sentiment Extraction & also NER (at token level like PII detection)
  - **Accuracy**, % of observations that were correctly predicted. If test data is skewed we can't just depend on Accuracy, need other metrics
  - **Precision**, % of predicted positives that were correct
  - **Recall**, % of active positives that are correct
  - **F1 score**, harmonic mean of precision and recall
  - Example: Suppose a dataset is imbalanced - 95 negative reviews and only 5 positive reviews (common in complaint-heavy product),
              A lazy model that predicts **negative** for **every single review** would score 95% Accuracy, looks great on first glance but the model never correctly identified a positive review.
              This is why Accuracy alone is very dangerous on imbalanced data.
## Tokenization
- Models don't really understand text they understand numbers. So we need to process text and make it quantifiable for models to consume.
- More info about Tokenization [GitHub Pages](./ai-research/1-Tokenization.md).
         