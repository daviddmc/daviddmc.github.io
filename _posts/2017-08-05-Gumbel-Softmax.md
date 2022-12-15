---
layout: paper-note
title: "Gumbel-Softmax"
description: Categorical Reparameterization with Gumbel-Softmax
date: 2017-08-05

paper_type: arXiv
paper_url: https://arxiv.org/pdf/1611.01144.pdf

bibliography: paper-notes.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Takeaways
  - name: The Gumbel Distribution
  - name: The Gumbel-Max Trick
  - name: The Gumbel-Softmax Distribution
  - name: The Gumbel-Softmax Estimator
  - name: The Straight-Through Gumbel-Softmax Estimator

---

## Takeaways

- Categorical latent variables can not backpropagate through samples.
- Replace the non-differentiable sample from a categorical distribution with a differentiable sample from a **Gumbel-Softmax** distribution. 
- The Gumbel-Softmax distribution has the essential property that it can be **smoothly annealed into** a categorical distribution.

## The Gumbel Distribution

Notation: $$X\sim\text{Gumbel}(\mu, \beta)$$, where $$\mu\in\mathbb{R}$$ is the location parameter and $$\beta>0$$ is the scale parameter.

PDF:

$$
f_X(x)=\frac{1}{\beta}e^{-(z+e^{-z})}, \text{ where } z=\frac{x-\mu}{\beta}.
$$

CDF:

$$
F_X(x)=e^{-e^{-z}}, \text{ where } z=\frac{x-\mu}{\beta}.
$$

See [Wiki](https://en.wikipedia.org/wiki/Gumbel_distribution) for more details.

## The Gumbel-Max Trick

Let $$\pi=(\pi_1,\dots,\pi_k)$$ be $$k$$-d nonnegative vector, where not all elements are zero, and let $$g_1,\dots,g_k$$ be $$k$$ iid samples from $$\text{Gumbel}(0,1)$$. Then

$$
\arg\max_i(g_i+\log\pi_i)\sim\text{Categorical}\left(\frac{\pi_j}{\sum_i\pi_i}\right)_j
$$

Proof:

Let $$I = \arg\max_i\{G_i + \log\pi_i\}$$ and $$M = \max_i\{G_i + \log\pi_i\}$$.

$$
\begin{aligned}
\mathbb{P}(I=i)&=\mathbb{P}(G_i + \log\pi_i < M, \forall j\neq i) \\
& = \int_{-\infty}^\infty f_{G_i}(m-\log\pi_i) \prod_{j\neq i} F_{G_j}(m-\log\pi_j) dm \\
& = \int_{-\infty}^\infty \exp(\log\pi_i-m-\exp(\log\pi_i-m)) \prod_{j\neq i} \exp(-\exp(\log\pi_j-m)) dm \\
& = \int_{-\infty}^\infty \exp(\log\pi_i-m)\exp(-\exp(\log\pi_i-m)) \prod_{j\neq i} \exp(-\exp(\log\pi_j-m)) dm \\
& = \int_{-\infty}^\infty \exp(\log\pi_i-m) \prod_{j} \exp(-\exp(\log\pi_j-m)) dm \\
& = \int_{-\infty}^\infty \exp(\log\pi_i-m) \exp(-\sum_{j}\exp(\log\pi_j-m)) dm \\
& = \int_{-\infty}^\infty \exp(\log\pi_i)\exp(-m) \exp(-\exp(-m)\sum_{j}\exp(\log\pi_j)) dm \\
& = \int_{-\infty}^\infty \pi_i\exp(-m) \exp(-\exp(-m)\sum_{j}\pi_j) dm \\
& = \int_{0}^\infty \pi_i \exp(-x\sum_{j}\pi_j) dx \\
& = \frac{\pi_i}{\sum_j\pi_j}
\end{aligned}
$$

## The Gumbel-Softmax Distribution

Relax the Gumbel-Max trick by replacing argmax with softmax (continuous, differentiable) and generate $$k$$-d sample vectors

$$
y_i = \frac{\exp((\log(\pi_i)+g_i)/\tau)}{\sum_{j=1}^k\exp((\log(\pi_j)+g_j)/\tau)}.
$$

PDF:

$$
f_{Y_1,\dots,Y_k}(y_1,\dots,y_k;\pi,\tau)=\Gamma(k)\tau^{k-1}\left( \sum_{i=1}^k \pi_i/y_i^\tau \right)^{-k}\prod_{i=1}^k(\pi_i/y_i^{\tau+1}).
$$

- The Gumbel-Softmax distribution interpolates between discrete one-hot-encoded categorical distributions and continuous categorical densities.
- For low temperatures, the expected value of a Gumbel-Softmax random variable approaches the expected value of a categorical random variable with the same logits.
- As the temperature increases, the expected value converges to a uniform distribution over the categories.
- Samples from GumbelSoftmax distributions are identical to samples from a categorical distribution as $$\tau\rightarrow 0$$.
- At higher temperatures, Gumbel-Softmax samples are no longer one-hot, and become uniform as $$\tau\rightarrow\infty$$.

## The Gumbel-Softmax Estimator

The Gumbel-Softmax distribution is smooth for $$\tau > 0$$, and therefore has a well-defined gradient $$\partial y/\partial \pi$$ with respect to the parameters $$\pi$$. Thus, by replacing categorical samples with Gumbel-Softmax samples we can use backpropagation to compute gradients.

Denote the procedure of replacing non-differentiable categorical samples with a differentiable approximation during training as the **Gumbel-Softmax estimator**.

A tradeoff between small and large temperatures:

- Small $$\tau$$: Close to one-hot but the variance of the gradients is large
- Large $$\tau$$: Samples are smooth but the variance of the gradients is small. 

In practice

- Start at a high temperature and anneal to a small but non-zero temperature.
- Or let $$\tau$$ be a trainable parameter (can be interpreted as entropy regularization).

## The Straight-Through Gumbel-Softmax Estimator

For scenarios that are constrained to sampling discrete values

- Discretize $$y$$ using argmax.
- But use the continuous approximation in the backward pass.

Call this Straight-Through (ST) Gumbel-Softmax Estimator.