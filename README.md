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
@inproceedings{li2024stacksbetteronecomparison,
      title={A Comparison of Language Modeling and Translation as Multilingual Pretraining Objectives}, 
      author={Zihao Li and Shaoxiong Ji and Timothee Mickus and Vincent Segonne and Jörg Tiedemann},
      year={2024},
      booktitle={Proceedings of EMNLP},
      url={https://arxiv.org/abs/2407.15489}, 
}
```
