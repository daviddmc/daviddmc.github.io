# CLIP: Contrastive Language-Image Pre-training

title: Learning Transferable Visual Models From Natural Language Supervision

url: <https://arxiv.org/pdf/2103.00020.pdf>

## Introduction

The authors demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch

After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks.

## Methods

### Natural Language Supervision

At the core of CLIP is the idea of learning perception from supervision contained in natural language

### Dataset

A new dataset of 400M image-text pairs (WebImageText)

### Selecting an Efficient Pre-Training Method

#### Exploration

**Initial approach:** Jointly train an image CNN and text transformer from scratch to predict the caption of an image.

**Problem:** Such approach learns to recognize ImageNet classes three times slower than a much simpler baseline that predicts a bag-ofwords encoding of the same text.

**Reason:** They try to predict the exact words of the text accompanying each image. This is a difficult task due to the wide variety of descriptions, comments, and related text that co-occur with images.

**Solution:** Contrastive representation learning.

#### CLIP

Given a batch of $N$ (image, text) pairs, CLIP is trained to predict which of the $N\times N$ possible (image, text) pairings across a batch actually occurred.

To do this, CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the $N$ real pairs in the batch while minimizing the cosine similarity of the embeddings of the $N^2 − N$ incorrect pairings (see the pseudocode below).

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

CLIP was trained from scratch without initializing the image encoder with ImageNet weights or the text encoder with pre-trained weights.

Only used linear projection to map image/text representation to the multi-model embedding space (instead of non-linear projection)

#### Model

**Image encoder:** ResNet / ViT
**Text encoder:** Transformer

## Experiments

### Zero-Shot Transfer

**zero-shot classification:** using class name as input text and predict the most probable pair (maximizing cosine similarity)

**Prompt engineering:** The input texts during training are sentences while the label of an image is just a word. To help bridge this distribution gap, we found that using the prompt template "A photo of a {label}." Zero-shot performance can be significantly improved by customizing the prompt text to each task.

### Representation Learning

How to evaluate the quality of learned representation?

1. Fitting a linear classifier on the representation extracted from the model and measuring its performance on various datasets.
2. Fine-tunning end-to-end

The author use the first method, since fine-tunning end-to-end would change the representation and potentially mask some failures.

### Robustness to Natural Distribution Shift

DL models are exceedingly adept at finding correlations and patterns which hold across their training dataset and thus improve in-distribution performance. However many of these correlations and patterns are actually spurious and do not hold for other distributions and result in large drops in performance on other datasets.

**Effective robustness** measures improvements in accuracy under distribution shift above what is predicted by the documented relationship between in-distribution and out-of-distribution accuracy.

**Relative robustness** captures any improvement in out-of-distribution accuracy.

#### Zero-shot

Intuitively, a zero-shot model should not be able to exploit spurious correlations or patterns that hold only on a specific distribution, since it is not trained on that distribution.

Results show that zero-shot models can be much more robust, however, they do not necessarily mean that supervised learning on ImageNet causes a robustness gap. Other details of CLIP, such as its large and diverse pre-training dataset or use of natural language supervision could also result in much more robust models regardless of whether they are zero-shot or fine-tuned.

#### Fine-tune on ImageNet

Although adapting CLIP to the ImageNet distribution increases its ImageNet
accuracy by 9.2% to 85.4% overall, and ties the accuracy of the 2018 SOTA from Mahajan et al. (2018), *average accuracy under distribution shift slightly decreases*.

#### Flexible classes

The target classes across the transfer datasets are not always perfectly aligned with those of ImageNet. With CLIP we can instead generate a custom zero-shot classifier for each dataset directly based on its class names. This improves average effective robustness by 5% but is concentrated in large improvements on only a few datasets.

#### Few-shot

Few-shot CLIP also increases effective robustness compared to existing ImageNet models but is less robust than zero-shot CLIP. Minimizing the amount of ImageNet training data used for adaption increases effective robustness at the cost of decreasing relative robustness.
