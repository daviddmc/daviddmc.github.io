---
layout: paper-note
title: "InstructGPT"
description: Training language models to follow instructions with human feedback
date: 2022-01-27

paper_type: arXiv
paper_url: https://arxiv.org/pdf/2203.02155.pdf

bibliography: paper-notes.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Takeaways
  - name: Introduction
  - name: Methods
    subsections:
    - name: High-Level Methodology
    - name: Dataset
    - name: Models
  - name: Experiments
    subsections:
    - name: Evaluation
    - name: Results on the API Distribution
    - name: Results on Public NLP Datasets
    - name: Qualitative results
  - name: Discussion
    subsections:
    - name: Implications for Alignment Research
    - name: Who are We Aligning to?
    - name: Limitations
    - name: Open Questions

---

## Takeaways

- Making LMs bigger does not inherently make them better at following a user's intent.
- Reinforcement learning from human feedback (**RLHF**) is a promising direction for aligning LM with user intent.
- Outputs from the 1.3B InstructGPT model are preferred by humans to outputs from the 175B GPT-3, despite having 100x fewer parameters.
- InstructGPT shows improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets.

## Introduction

Making LMs bigger does not inherently make them better at following a user's intent, i.e., not *aligned* with their users.

The model may generate outputs that are untruthful, toxic, or not helpful.

This is because the LM objective used for many recent large LMs, i.e., predicting the next token on a webpage from the internet, is different from the objective "follow the user’s instructions helpfully and safely".

Ideas:

- Aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. 
- Use reinforcement learning from human feedback (**RLHF**) to fine-tune GPT-3<d-cite key="GPT-3"></d-cite> to follow a broad class of written instructions. 
- This technique uses human preferences as a reward signal to fine-tune the models.

Main Findings:

- Labelers significantly prefer InstructGPT outputs over outputs from GPT-3
- InstructGPT generalizes to the preferences of "held-out" labelers.
- Public NLP datasets are not reflective of how our language models are used.
- InstructGPT models show improvements in truthfulness over GPT-3.
- InstructGPT shows small improvements in toxicity over GPT-3, but not bias.
- The performance regressions on public NLP datasets can be minimized by modifying the RLHF fine-tuning procedure.
- InstructGPT shows promising generalization to instructions outside of the RLHF finetuning distribution.
- InstructGPT still makes simple mistakes.

## Methods

### High-Level Methodology

1. Collect demonstration data, and train a supervised policy (supervised fine-tune, **SFT**).
2. Collect comparison data, and train a reward model (**RM**).
3. Optimize a policy against the reward model using **PPO**.

Steps 2 and 3 can be iterated continuously.

<div class="l-page" style="text-align:center;">
  <img src="https://cdn.openai.com/instruction-following/draft-20220126f/methods.svg" width="100%" style="margin-bottom: 12px; background-color: white;">
  <p></p>
</div>

### Dataset

#### Prompt Dataset

- Use text prompts submitted to the OpenAI API, specifically those using *an earlier version of the InstructGPT models*.
- Deduplicate prompts by checking for prompts that share a long common prefix. 
- Limit the number of prompts to 200 per user ID.
- Create the train, validation, and test splits based on *user ID*.
- Filter all prompts in the training split for personally identifiable information.

For each prompt, the task can be

- a natural language instruction (e.g. “Write a story about a wise frog”), 
- few-shot examples (e.g. giving two examples of frog stories, and prompting the model to generate a new one)
- an implicit continuation (e.g. providing the start of a story about a frog). 

#### Train the Very First InstructGPT

To train the very first InstructGPT models, the authors asked labelers to *write prompts themselves*. This is because an initial source of instruction-like prompts is needed to bootstrap the process and these kinds of prompts weren't often submitted to the regular GPT-3 models on the API. 

Three kinds of prompts were written by the labelers:
- Plain: an arbitrary task, while ensuring the tasks had sufficient diversity.
- Few-shot: an instruction, and multiple query/response pairs for that instruction.
- User-based: a prompt corresponding to the use cases stated in waitlist applications to the OpenAI API.

#### Datasets for Fine-Tuning

Three different datasets used in the fine-tuning procedure are built from the prompt dataset.

1. **SFT:** A prompt is sampled from the prompt dataset, and a labeler writes an answer to this prompt, supervised learning (13k prompts)
2. **RM:** A prompt and several model outputs are sampled, and a labeler ranks the outputs from the best to worst. This data is used to train the reward model. (33k prompts)
3. **PPO:** Another prompt dataset from the API. This data is used to train PPO with the RM. (31k prompts)

#### Human Data Collection

Hired a team of about 40 labelers.

During training and evaluation, the alignment criteria may come into conflict.

- During training helpfulness to the user is prioritized
- In the final evaluation truthfulness and harmlessness are prioritized.  

### Models

#### SFT

- Fine-tune GPT-3 on the labeler demonstrations using supervised learning.
- Select the final SFT model based on the RM score on the validation set.

#### RM

- Start from the SFT model with the final unembedding layer removed.
- Train a model to take in a prompt and response, and output a scalar reward.

Model size: The 6B RMs are used because they save computation and the training of 175B RM could be unstable.

Loss function:

$$
L(\theta)=-\frac{1}{K \choose 2}\mathbb{E}_{(x,y_w,y_l)\sim D}[\log(\sigma(r_\theta(x,y_w)- r_\theta(x,y_l)))],
$$

where $$r_\theta(x,y)$$ is the score outputed by the RM for prompt $$x$$ and completion $$y$$ with parameters $$\theta$$, $$y_w$$ is the preferred completion out of the pair of $$y_w$$ and $$y_l$$, and $$D$$ is the dataset of human comparisons.

#### RL

Fine-tune the SFT model using PPO.

The environment is a bandit environment that presents a random customer prompt and expects a response to the prompt. Given the prompt and response, it produces a reward determined by the reward model and ends the episode.

In addition, a per-token KL penalty from the SFT model is added at each token to mitigate overoptimization of the reward model.

The value function is initialized from the RM.

An improved algorithm: **PPO-ptx**

- Mixing the pretraining gradients into the PPO gradients.
- Maximize the following combined objective function in RL training

  $$
  \mathbb{E}_{(x,y)\sim D_{\pi_{\phi}^\text{RL}}}[r_{\theta}(x,y)-\beta\log(\pi_\phi^\text{RL}(y|x)/\pi^\text{SFT}(y|x))]+\gamma \mathbb{E}_{x\sim D_\text{pretrain}}[\log(\pi_\phi^\text{RL}(x))],
  $$

  where $$\pi_\phi^\text{RL}$$ is the learned RL policy, $$\pi^\text{SFT}$$ is the supervised trained model, and $$D_\text{pretrain}$$ is the pretraining distribution.
- Falling back into PPO when $$\gamma=0$$.

#### Baselines

- GPT-3
- GPT-3-prompted: provide a few-shot prefix to "prompt" GPT-3 into an instruction-following mode
- GPT-3-FT: fine-tune GPT-3 on a variety of NLP tasks

## Experiments

### Evaluation

#### Definition of Alignment

- **Helpfulness:**
  - The model should follow instructions, but also infer intention from a few-shot prompt or another interpretable pattern.
  - The main metric is labeler preference ratings. 
  - However, since the labelers are not the users who generated the prompts, there could be a divergence between what a user actually intended and what the labeler thought was intended from only reading the prompt.
- **Truthfulness (honesty):**
  - Whether the model’s statements about the world are true.
  - Evaluate the model’s tendency to make up information on closed-domain tasks ("hallucinations")
  - Use the TruthfulQA dataset.
- **Harmlessness:**
  - The harms from language models depend on how their outputs are used in the real world.
  - The labelers evaluate whether the output is inappropriate in the context of a customer assistant, denigrates a protected class, or contains.
  - Benchmark the models on datasets intended to measure bias and toxicity. 

#### Evaluations on API Distribution

The main metric is *human preference ratings* on a held-out set of prompts from the same source as the training distribution. 

- How often the model's outputs are preferred to a baseline policy (175B SFT model)?
- The overall quality of each response on a 1-7 Likert scale (given by the labelers).

#### Evaluations on Public NLP Datasets

- Datasets that capture an aspect of LM safety, particularly truthfulness, toxicity, and bias.
- Datasets that capture zero-shot performance on traditional NLP tasks, e.g., 
  - question answering, 
  - reading comprehension, 
  - summarization. 
- The RealToxicityPrompts dataset for human evaluations of toxicity.

### Results on the API Distribution

- **Labelers significantly prefer InstructGPT outputs over outputs from GPT-3.**
  - Preference order: PPO-ptx ~ PPO > SFT > GPT-3 (prompted) > GPT-3
  - Outputs from the 1.3B InstructGPT are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters.
  - Results do not change significantly when evaluated on prompts submitted to GPT-3 models on the API.
  - InstructGPT outperforms GPT along several more concrete axes:
    1. "Attempts correct instruction"
    2. "Follows explicit constraints"
    3. "Hallucinations": making up facts
    4. "Uses language appropriate for customer assistant" 
- **InstructGPT generalizes to the preferences of "held-out" labelers.** 
  - Held-out labelers (who did not produce any training data) have similar ranking preferences as workers who produce training data.
- **Public NLP datasets are not reflective of how our language models are used.**
  - Fine-tune GPT-3 on a variety of NLP tasks (GPT-3-FT)
  - Likert scores ranking: PPO-ptx > SFT > GPT-3 (prompted) ~ GPT-3-FT > GPT-3
  - InstructGPT model outperforms GPT-3-FT for two reasons.
    1. Public NLP datasets are designed to capture tasks that are easy to evaluate with automatic metrics, whereas more than half of the prompts in the API distribution are open-ended generation and brainstorming.
    2. It can be difficult for public NLP datasets to obtain a very high diversity of inputs. 

### Results on Public NLP Datasets

- **InstructGPT models show improvements in truthfulness over GPT-3.**
  - InstructGPT generates truthful and informative answers more often than GPT-3.
  - InstructGPT does not have to be specifically instructed to tell the truth to exhibit improved truthfulness.
  - InstructGPT hallucinate (i.e. fabricate information) less often than GPT-3 on closed-domain tasks from our API distribution.
- **InstructGPT shows small improvements in toxicity over GPT-3, but not bias.**
  - Obtain automatic toxicity scores of models outputs using  [the Perspective API](https://www.perspectiveapi.com)
  - Human evaluation: absolute toxicity, toxicity relative to the prompt, continuity, and overall output preference.
  - When instructed to produce a safe and respectful output, InstructGPT models generate less toxic outputs than those from GPT-3. This advantage disappears when the respectful prompt is removed.
  - When explicitly prompted to produce a toxic output, InstructGPT outputs are much more toxic than those from GPT-3.
  - In terms of the propensity to generate biased speech, InstructGPT is *not* less biased than GPT-3. But when instructed to act respectfully InstructGPT exhibits higher bias.
- **The performance regressions on public NLP datasets can be minimized by modifying the RLHF fine-tuning procedure.**
  - InstructGPT (PPO model trained on the API distribution) suffers from performance regressions on several public NLP datasets (alignment tax).
  - Adding pretraining updates (PPO-ptx) mitigates these performance regressions on all datasets.
  - Mixing in pretraining updates performs better than the simpler solution of increasing the KL coefficient. 

### Qualitative results

- **InstructGPT shows promising generalization to instructions outside of the RLHF finetuning distribution.**
  - InstructGPT shows the ability to follow instructions in non-English languages, and perform summarization and question-answering for code, although non-English languages and code form a tiny minority of the fine-tuning data.
- **InstructGPT still makes simple mistakes.**
  - When given an instruction with a false premise, the model sometimes incorrectly assumes the premise is true.
  - The model can overly hedge; when given a simple question, it can sometimes say that there is no one answer to the question and give multiple possible answers, even when there is one fairly clear answer from the context.
  - The model’s performance degrades when instructions contain multiple explicit constraints or when constraints can be challenging for language models.

## Discussion

### Implications for Alignment Research

This research is part of a broader research program to align AI systems with human intentions.

Lessons for alignment research more generally:

- The cost of increasing model alignment is modest relative to pretraining.
- There is some evidence that InstructGPT generalizes "following instructions" to settings that people don’t supervise it in.
- Most of the performance degradations introduced by the fine-tuning can be mitigated.
- This work has validated alignment techniques from research in the real world.

### Who are We Aligning to?

Factors that influence the fine-tuning data that ultimately determine what and who the models are aligning to.

- Aligning to demonstrations and preferences provided by our training labelers.
- Aligning to the preferences of the researchers who designed this study.
- Aligning to what customers think is valuable and what their end-users think is valuable to currently use the API for.

### Limitations

#### Limitations of Methodology

- The behavior of InstructGPT models is determined in part by the human feedback obtained from the labelers.
- Some of the labeling tasks rely on value judgments that may be impacted by the identity of the labelers.
- Most comparisons are only labeled by 1 labeler for cost reasons.

#### Limitations of Models

- The models are neither fully aligned nor fully safe: still generate toxic or biased outputs, make up facts, and generate sexual and violent content without explicit prompting.
- The model also fails to generate reasonable outputs on some inputs.
- In most cases, the model follows the user’s instruction, even if that could lead to harm in the real world. 

### Open Questions

- Further decrease the models' propensity to generate toxic, biased, or otherwise harmful outputs. 
- Training the model to be harmless despite user instructions is important but is also difficult because whether an output is harmful depends on the context in which it's deployed.
- A promising future path is combining RLHF with other methods of steerability.
- In addition to RLHF, there are many other algorithms that could be used to train policies on the demonstration and comparison data to get even better results.
- Comparisons are not necessarily the most efficient way of providing an alignment signal.
- The proposal for mitigating the alignment tax, by incorporating pretraining data into RLHF finetuning (PPO-ptx), does not completely mitigate performance regressions and may make certain undesirable behaviors more likely for some tasks.
- A principle-based approach to alignment, i,e, identifying "fair principles" for alignment.
