# Natural Language Inference Transformer Models Code

## Overview
This repository contains the code used in our paper:
“**The Geometry of Meaning: Evaluating Sentence Embeddings from Diverse Transformer-Based Models for Natural Language Inference**.”

The goal of this work is to explore how pretrained transformer-based models encode sentence-level meaning for NLI tasks using simple geometric comparisons, specifically pooling techniques and vector norms, without any supervised fine-tuning.
This code implements transformer-based models (BERT, GPT, RoBERTa, and XLNet) for Natural Language Inference (NLI) using various pooling strategies (Max, Min, and Mean) and norm calculations (L1, L2, L-inf).

## Prerequisites
The code requires Python 3 and the following libraries:
- Hugging Face Transformers
- PyTorch
- numpy
