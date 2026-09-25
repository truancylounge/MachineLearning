# Intro to PreTraining datasets
- Huggingface **Fineweb** - https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
  - Great tutorial on how to use web crawed data and pre trained datasets to train a small or LLM
- Free opensource repository of web crawled data
    - **Common Crawl**, https://commoncrawl.org/
- **Datatrove**, an open-source data processing library that allowed us to seamlessly scale our filtering and deduplication setup to thousands of CPU cores.
  - https://github.com/huggingface/datatrove
  - https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py
- Lighteval, supports 1000+ evaluation tasks across multiple domains and languages.
  - https://github.com/huggingface/lighteval/

## Key Concepts
1. Deduplication & Deduplication parameters
2. Quality Filtering
3. WebCrawled data formats, [WebCrawled Data formats](concepts/Webcrawled-data-formats.md)


### Deduplicating
- The web has many aggregators, mirrors, templated pages or just otherwise repeated content spread over different domains and webpages. 
- Sometimes, these duplicated pages can even be introduced by the crawler itself, when different links point to the same page.
- Removing these duplicates (deduplicating) has been correlated with improvements in model performance and a reduction in memorization of pretraining data , which might allow for better generalization.
- Additionally, the performance uplift obtained through deduplication can be equated to increased training efficiency: by removing duplicated content, a model can reach the same performance level with fewer training iterations – or equivalently, for a given number of training tokens, a model will have seen more diverse data.
- Techniques
  - Common approaches rely on hashing techniques to speed up the process, or on building efficient data structures to index the data (like suffix arrays)
  - Methods can also be “fuzzy”, by using some similarity metric to mark documents as duplicates, or “exact” by checking for exact matches between two documents (or lines, paragraphs, or whatever other granularity level being used)
- 

### Overfitting
Overfitting in the context of evaluation metrics means: your model's metric score looks great on the data it was trained/tuned on, but the score drops sharply on data it hasn't seen before. 
The model hasn't learned the underlying pattern — it's memorized the specifics (including noise/quirks) of the training set, which don't generalize.
More Info [Overfitting](concepts/Overfitting.md).


