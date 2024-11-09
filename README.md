# [EMNLP 2024] A Comparison of Language Modeling and Translation as Multilingual Pretraining Objectives

## Overview
This repository contains the code and data for the paper "A Comparison of Language Modeling and Translation as Multilingual Pretraining Objectives". The paper investigates the impact of different pretraining objectives on multilingual language models and compares their performance on various downstream tasks.

## Abstract
Pretrained language models (PLMs) have shown impressive performance and garnered significant attention in the NLP community. This paper compares multilingual pretraining objectives under controlled conditions, focusing on two main observations: (1) the architecture dictates the optimal pretraining objective, and (2) multilingual translation can be a highly effective pretraining objective.

## Features
- **Controlled Evaluation**: Ensures comparability by using consistent training data and model architectures.
- **Multilingual Focus**: Evaluates performance across six languages.
- **Pretraining Models**: Compares BART architecture with machine translation objective (2-MT),  BART architecture with denoising objective (2-LM), masked language modeling (MLM), causal language modeling (CLM), and translation language modeling (TLM).
- **Downstream Tasks**: Includes sentiment analysis (SA), named entity recognition (NER), part-of-speech (POS) tagging, and natural language inference (NLI).

## Usage
This repo has git-lfs enabled as we also host fairseq model weights (`./model_weights`) in this repo. You can skip those LFS by  
`GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Helsinki-NLP/lm-vs-mt.git`

Release the processed dataset for pretraining here https://a3s.fi/mickusti-2005099-pub/lm-vs-mt_data.tar.gz 

## Citation

```
@inproceedings{li-etal-2024-comparison,
    title = "A Comparison of Language Modeling and Translation as Multilingual Pretraining Objectives",
    author = {Li, Zihao  and
      Ji, Shaoxiong  and
      Mickus, Timothee  and
      Segonne, Vincent  and
      Tiedemann, J{\"o}rg},
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.888",
    pages = "15882--15894",
    abstract = "Pretrained language models (PLMs) display impressive performances and have captured the attention of the NLP community.Establishing best practices in pretraining has, therefore, become a major focus of NLP research, especially since insights gained from monolingual English models may not necessarily apply to more complex multilingual models.One significant caveat of the current state of the art is that different works are rarely comparable: they often discuss different parameter counts, training data, and evaluation methodology.This paper proposes a comparison of multilingual pretraining objectives in a controlled methodological environment. We ensure that training data and model architectures are comparable, and discuss the downstream performances across 6 languages that we observe in probing and fine-tuning scenarios.We make two key observations: (1) the architecture dictates which pretraining objective is optimal; (2) multilingual translation is a very effective pretraining objective under the right conditions.We make our code, data, and model weights available at https://github.com/Helsinki-NLP/lm-vs-mt.",
}
```
