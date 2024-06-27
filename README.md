# Two Stacks Are Better Than One: A Comparison of Language Modeling and Translation as Multilingual Pretraining Objectives

## Overview
This repository contains the code and data for the paper "Two Stacks Are Better Than One: A Comparison of Language Modeling and Translation as Multilingual Pretraining Objectives". The paper investigates the impact of different pretraining objectives on multilingual language models and compares their performance on various downstream tasks.

## Abstract
Pretrained language models (PLMs) have shown impressive performance and garnered significant attention in the NLP community. This paper compares multilingual pretraining objectives under controlled conditions, focusing on two main observations: (1) the architecture dictates the optimal pretraining objective, and (2) multilingual translation can be a highly effective pretraining objective.

## Features
- **Controlled Evaluation**: Ensures comparability by using consistent training data and model architectures.
- **Multilingual Focus**: Evaluates performance across six languages.
- **Pretraining Models**: Compares BART architecture with machine translation objective (2-MT),  BART architecture with denoising objective (2-LM), masked language modeling (MLM), causal language modeling (CLM), and translation language modeling (TLM).
- **Downstream Tasks**: Includes sentiment analysis (SA), named entity recognition (NER), and part-of-speech (POS) tagging.

## Usage

## Citation
