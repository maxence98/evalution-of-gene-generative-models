# Evaluation of gene generative models


## Abstract
Generative models have immense potential for application in the field of synthetic gene design. Preliminary explorations of this application have made progress in the literature, proposing many generative model designs for different gene datasets and gene property targets. However, a significant gap exists between this field and others in terms of the metrics used to evaluate model performance. Without a comprehensive evaluation method, it is challenging to thoroughly compare the generative capabilities of different models. In this work, we conduct a critical review of the existing evaluation methods used in gene generation, as well as the common methods from other fields. We reveal that currently used feature spaces suffer from capturing high-level features. To develop a more comprehensive method, we propose to apply a pre-trained BERT model for feature extraction and measure the feature distribution similarity with Fréchet distance and MMD. A systematic experiment is designed to compare the sensitivity of both the existing methods and the proposed new methods. Results show that the proposed methods have better performance than the existing methods in terms of differing the abilities of generative models for gene sequences, while the currently used methods are not capable enough to distinguish the simple baselines from complex models.

## Installation
You can install the dependencies by create a conda environment with

```bash
conda env create -f environment.yml
```
or
```bash
conda create --name <env> --file requirements.txt
```

## Usage
The evaluation methods are implemented in `Evaluation.ipynb` in scripts .

For model training, use the `train.py`. Models are implemented in `models`.
To train and generate samples with Markov model, use `markov.ipynb`.