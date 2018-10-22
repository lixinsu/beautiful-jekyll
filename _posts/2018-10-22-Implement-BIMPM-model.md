---
layout: post
title: Implementation Details about BIMPM for NLI
subtitle: Four match function for NLI
tags: [deep learning, textual entailment]
---

## Introduction
The BIMPM model is one of the state of the art models in SNLI dataset, 
whichs need the models to classify a pair of sentences, premise and hypothesis, to neural, contrdiction, and entailment.
The model mainly consiste of embedidng, LSTM, four matching functions, aggregration LSTM and fc layers.
The difficulty of implement the [BIMPM model](https://arxiv.org/abs/1702.03814) lies in the four matching function. 
The other part are just usual LSTM , char-level mebedding and pre-trained word embeddding.
In this blog, I describe the details about the calculation setps in four functions. 


## Four funciton for matching
In fact, these four funcitons are essentially used to calculate the attention weight, somehow like the multi-head attention. 
The term `perspective` semantically equals to `head` in the **Attention is all you need** paper. 
Let's start.

### function I( full mathcing )

