---
layout: post
title: Implementation Details about BIMPM for NLI
subtitle: Four match function for NLI
tags:
  - deep learning
  - textual entailment
published: true
---

## Introduction
The BIMPM model is one of the state of the art models in SNLI dataset, 
whichs ask the models to classify a pair of sentences premise and hypothesis to neural, contrdiction, and entailment.  
The model mainly consiste of embedidng, contextual LSTM, four matching operations, aggregration LSTM and fc layers.  
The difficulty of implement the [BIMPM model](https://arxiv.org/abs/1702.03814) lies in the four matching function. 
In this blog, I will describe the details about the calculation setps in four operations. I will give the input and output of each operation.  

### Preliminary
P denotes premise.  
H denotes hypothese.  
b denotes the batch_size.  
p denotes the lengths of premise.     
h denotes the length of hypothesis.
Before four matching funciton, the input word indics are first converted to embedding space, and passed through LSTM.

P: [b, p]
H: [b, h]
p_emb = word_char_Embedding(P)
h_emb = word_char_Embedding(H)
p_emb: [b,p,350]
h_emb: [b,h,350]

LSTM(in=300, out=100, bidirectional=True)   
p_con = LSTM(p_emb)  
h_con = LSTM(h_emb)  

p_con: [b,p,200]  
h_con: [b,p,200]    
Note: con means contextual

split the forward(fw) and backward(bw)
con_p_fw: [b,p,100]
con_p_bw: [b,p,100]
con_h_fw: [b,p,100]
con_h_bw: [b,p,100]

Then we are ready for matching operations. 



### Operation One



con_p_fw torch.Size([64, 31, 100])  
con_p_bw torch.Size([64, 31, 100])  
con_h_fw torch.Size([64, 20, 100])  
con_h_bw torch.Size([64, 20, 100])  

mv_p_full_fw = Maching Layer I (con_p_fw, con_h_fw[:, -1, :], mp_w1)    
mv_p_full_bw = Maching Layer I (con_p_bw, con_h_bw[:, -1, :], mp_w2)    
mv_h_full_fw = Maching Layer I (con_h_fw, con_p_fw[:, -1, :], mp_w1)    
mv_h_full_bw = Maching Layer I (con_h_bw, con_p_bw[:, -1, :], mp_w2)   

mp_w1 torch.Size([20, 100])  
mp_w2 torch.Size([20, 100])  

mv_max_fw = Maching Layer II (con_p_fw, con_h_fw)  
mv_max_bw = Maching Layer II (con_p_bw, con_h_bw)  

mv_max_fw torch.Size([64, 31, 20, 20])  
mv_max_bw torch.Size([64, 31, 20, 20])  

mv_h_max_fw = max(mv_max_fw, dim=1)   torch.Size([64, 20, 20])  
mv_p_max_fw = max(mv_max_fw, dim=2)   torch.Size([64, 31, 20])  
mv_h_max_bw = max(mv_max_bw, dim=1)   torch.Size([64, 20, 20])  
mv_p_max_bw = max(mv_max_bw, dim=2)   torch.Size([64, 31, 20])  


Layer attentive matching  
First, we calculate two attention weight matrices   
att_fw = attention(con_p_fw, con_h_fw)    torch.Size([64, 31, 20])  
att_bw = attention(con_p_bw, con_h_bw)    torch.Size([64, 31, 20])  

Next， we multiply the attention weight with premise and hypothesis context vectors separately.  
att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)  
att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)  

att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)  
att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)  

**Note** This is not attention vecor, it doesn't do the dot product and keep the dimension for later use (weighted sum or max)  

att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))  
att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))  
att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))  
att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))  

Then do the same thing as Matching I  

Layer max attentive matching  
att_max_h_fw, _ = att_h_fw.max(dim=2)  
att_max_h_bw, _ = att_h_bw.max(dim=2)  
att_max_p_fw, _ = att_p_fw.max(dim=1)  
att_max_p_bw, _ = att_p_bw.max(dim=1)  

Then do the same thing as Matching I   

