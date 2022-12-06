# GPT-2

title: Language Models are Unsupervised Multitask Learners

url: <https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>

## Introduction

The authors demonstrate that language models begin to learn different tasks without any explicit supervision (*zero-shot*) when trained on a new dataset of millions of webpages called WebText.

The capacity of the language model is essential to the success of zero-shot task transfer and increasing it improves performance in a log-linear fashion across tasks.

## Methods

### Language model

Training a language model in a probabilistic framework as estimating a conditional distribution of the output given the input and the task information (multi-task/meta-learning),
$$p(output|input,task)$$

The language model is *auto-regressive*, i.e., it predict the next word given the previous words.

#### Task conditioning

1. architectural level, e.g., task specific encoders and decoders
2. algorithmic level, e.g., the inner and outer loop optimization framework of MAML
3. **language provides a flexible way to specify tasks (prompt)**

#### Speculation

A language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement. If a language model is able to do this it will be, in effect, performing unsupervised multitask learning. We test whether this is the case by analyzing the performance of language models in a zero-shot setting on a wide variety of tasks.

### Training dataset

Although web scrapes such as Common Crawl are many orders of magnitude larger than current language
modeling datasets, they have significant data quality issues.

Instead, the authors created a new web scrape which emphasizes document quality.

1. Only scraped web pages which have been curated/filtered by humans: scraped all outbound links from Reddit which received at least 3 karma. This can be thought of as a heuristic indicator for whether other users found the link interesting, educational, or just funny.
2. Extract the text from HTML responses
3. de-duplication and some heuristic based cleaning
4. removed all Wikipedia documents from WebText since it is a common data source for other datasets and could complicate analysis due to over-lapping training data with test evaluation tasks.

Results in over 8 million documents for a total of 40 GB of text.

### Input Representation

A general language model (LM) should be able to compute the probability of (and also generate) any string.

While processing Unicode strings as a sequence of UTF-8 bytes elegantly fulfills this requirement, current byte-level LMs are not competitive with word-level LMs on large scale datasets.

#### Byte Pair Encoding (BPE)

A practical middle ground between character and word level language modeling which effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences.

BPE on Unicode code points: The size of the base vocabulary is too large (> 130,000) compared to the 32,000 to 64,000 token vocabularies often used with BPE.

#### BPE on byte level

1. A base vocabulary of size 256

2. Naive BPE results in suboptimal merges due to the greedy strategy. To avoid this, the authors prevent BPE from merging across character categories, with an exception for spaces.

3. Enable the model to assign a probability to any Unicode string.

### Model

The model largely follows the details of the GPT model with a few modification.

1. Layer normalization was moved to the input of each sub-block, similar to a pre-activation residual network and an additional layer normalization was added after the final self-attention block.
2. A modified initialization which accounts for the accumulation on the residual path with model depth is used. Scale the weights of residual layers at initialization by a factor of $1/\sqrt{N}$, where $N$ is the number of residual layers.
3. The vocabulary is expanded to 50,257.
4. We also increase the context size from 512 to 1024 tokens,
5. A larger batchsize of 512 is used.

## Experiments

### Language Modeling

This is the primary task the models are trained for.

Evaluating the log-probability of different datasets according to a WebText LM.

### LAMBADA

The LAMBADA dataset tests the ability of systems to model long-range dependencies in text. The task is to predict the final word of sentences which require at least 50 tokens of context for a human to
successfully predict.

### Reading Comprehension

The Conversation Question Answering dataset (CoQA) consists of documents from 7 different domains paired with natural language dialogues between a question asker and a question answerer about the document. CoQA tests reading comprehension capabilities and also the ability of models to answer questions that depend on conversation history (such as “Why?”).

Use a greedy decoding from GPT-2 conditioned on a document, the history of the associated conversation, and a final token.

### Summarization

To induce summarization behavior we add the text `TL;DR:` after the article and generate 100 tokens with Top-$k$ random sampling with $k = 2$ which reduces repetition and encourages more abstractive summaries than greedy decoding. We use the first 3 generated sentences in these 100 tokens as the summary.

### Translation

In order to help it infer that this is the desired task, the author condition the language model
on a context of example pairs of the format `english sentence = french sentence` and then after a final prompt of `english sentence =` they sample from the model with greedy decoding and use the first generated sentence as the translation.

### Question Answering

Similar to translation, the context of the language model is seeded with example question answer pairs which helps the model infer the short answer style of the dataset.
