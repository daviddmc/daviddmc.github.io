# GPT-3

title: Language Models are Few-Shot Learners

url: <https://arxiv.org/pdf/2005.14165.pdf>

## Introduction

Although recent large pre-training models are *task-agnostic in architecture*, they still require *task-specific fine-tuning* datasets of thousands or tens of thousands of examples.

This work shows that scaling up language models greatly improves *task-agnostic*, *few-shot* performance, sometimes even reaching competitiveness with prior state-of-the-art finetuning approaches.

The author train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks,GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model (prompt).

## Methods

### Different Settings

#### Fine-Tuning (FT)

- A pre-trained model by training on a supervised dataset specific to the desired task. Typically thousands to hundreds of thousands of labeled examples are used.
- The main advantage: strong performance on many benchmarks.
- The main disadvantages: the need for a new large dataset for every task, the potential for poor generalization out-of-distribution, and the potential to exploit spurious features of the training data, potentially resulting in an unfair comparison with human performance.
- In this work the authors do not fine-tune GPT-3.

#### Few-Shot (FS)

- Refer to the setting where the model is given a few demonstrations of the task at inference time as conditioning, but no weight updates are allowed.
- The number of samples $K$ is in the range of 10 to 100 as this is how many examples can fit in the model’s context window (`nctx = 2048`).
- The main advantages: A major reduction in the need for task-specific data and reduced potential to learn an overly narrow distribution from a large but narrow fine-tuning dataset.
- The main disadvantage: Results from this method have so far been much worse than SOTA fine-tuned models. Also, a small amount of task specific data is still required.

#### One-Shot (1S)

- It is the same as few-shot except that only one demonstration is allowed, in addition to a natural
language description of the task.
- The reason to distinguish one-shot from few-shot and zero-shot (below) is that it most closely matches the way in which some tasks are communicated to humans.

#### Zero-Shot (0S)

- No demonstrations are allowed, and the model is only given a natural language instruction describing the task.
- This method provides maximum convenience, potential for robustness, and avoidance of spurious correlations.
- But it is also the most challenging setting.

### Model

The same model and architecture as GPT-2, with the exception that

1. GPT-3 use alternating dense and locally banded sparse attention patterns in the layers of the transformer.
2. To study the dependence of ML performance on model size, 8 different sizes of model were trained, ranging over three orders of magnitude from 125 million parameters to 175 billion parameters, with the last being the model called GPT-3.

### Training Dataset

Uunfiltered or lightly filtered versions of Common Crawl tend to have lower quality than more curated datasets. Therefore, the authors took 3 steps to improve the average quality of the datasets:

1. Downloaded and filtered a version of CommonCrawl based on similarity to a range of high-quality reference corpora;
2. Performed fuzzy deduplication at the document level, within and across datasets, to prevent redundancy and preserve the integrity of held-out validation set as an accurate measure of overfitting;
3. Added known high-quality reference corpora to the training mix to augment CommonCrawl and increase its diversity.

## Experiments

### Evaluation

For few-shot learning, the authors evaluate each example in the evaluation set by randomly drawing K examples from that task’s training set as conditioning

#### multiple-choice problems

Provide K examples of context plus correct completion, followed by one example of context only, and compare the LM likelihood of each completion. For most tasks they compare the per-token likelihood (to normalize for length). However, sometime it might be beneficial to normalize by the unconditional probability of each completion, by computing $$P(completion|context)/P(completion|answercontext),$$ where answer context is the string `"Answer: "` or `"A: "` and is used to prompt that the completion should be an answer.

#### binary classification

Give the options more semantically meaningful names (e.g. `"True"` or `"False"` rather than 0 or 1) and then treat the task like multiple choice.

#### Free-form completion

Use beam search with a beam width of 4 and a length penalty of $\alpha$ = 0.6.

### LAMBADA

Although the completion in LAMBADA is always the last word in a sentence, a standard language model has no way of knowing this detail. It thus assigns probability not only to the correct ending but also to other valid continuations of the paragraph. This problem has been partially addressed in the past with stop-word filters (which ban “continuation” words). The few-shot setting can "frame" the task as a cloze-test and allows the language model to infer from examples that a completion of exactly one word is desired, as shown in following fill-in-the-blank format:

```
Alice was friends with Bob. Alice went to visit her friend ___. → Bob
George bought some baseball equipment, a ball, a glove, and a ___. →
```

### Translation

For GPT-2 a filter was used on a multilingual collection of documents to produce an English only dataset due to capacity concerns. Since the capacity increases by over two orders of magnitude from GPT-2 to GPT-3, the scope of the training dataset is also expanded to include more representation of other languages. the majority of the data is derived from raw Common Crawl with only quality-based filtering. Although GPT-3’s training data is still primarily English (93% by word count), it also includes 7% of text in other languages.

Zero-shot GPT-3, which only receives on a natural language description of the task, still underperforms recent unsupervised NMT results. However, providing only a single example demonstration for each translation task improves performance by over 7 BLEU and nears competitive performance with prior work.
