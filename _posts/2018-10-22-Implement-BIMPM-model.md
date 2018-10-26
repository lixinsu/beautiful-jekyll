---
layout: post
title: Implementation Details about BIMPM for NLI
subtitle: Four matching operations for NLI
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

The diagram is as folloes:
![o1.png]({{site.baseurl}}/img/o1.png)  

The base cosine matching function fm is define as follows in original paper:
![base_function.png]({{site.baseurl}}/img/base_function.png)  

con_p_fw: [b,p,100]
con_p_bw: [b,p,100]
con_h_fw: [b,p,100]
con_h_bw: [b,p,100] 

mv_p_full_fw = fm(con_p_fw, con_h_fw[:, -1, :], W1)    
mv_p_full_bw = fm(con_p_bw, con_h_bw[:, -1, :], W2)    
mv_h_full_fw = fm(con_h_fw, con_p_fw[:, -1, :], W1)    
mv_h_full_bw = fm(con_h_bw, con_p_bw[:, -1, :], W2)   
**where W1, W2 is of size [20, 100], as depicted in the paper, the model best performance when number of perspectives quals 20.**
mv_p_full_fw: [b, p, 20]
mv_p_full_bw: [b, p, 20]
mv_h_full_fw: [b, h, 20]
mv_h_full_bw: [b, h, 20]

### Operation Two
![o2.png]({{site.baseurl}}/img/o2.png)  
con_p_fw: [b,p,100]
con_p_bw: [b,p,100]
con_h_fw: [b,p,100]
con_h_bw: [b,p,100] 

mv_common_fw = parewise_fm(con_p_fw, con_h_fw)    
mv_common_bw = pairwise_fm(con_p_bw, con_h_bw)    
Inside parewise_fm, we compute fm(con_p_fw[:,i,:], con_h_fw[:,j,:]) for each i = 1 to p , j= 1 to h .

mv_common_fw: [b, p, h, 20]  
mv_common_bw: [b, p, h, 20]  

mv_h_max_fw = max(mv_common_fw, dim=1)   
mv_p_max_fw = max(mv_common_fw, dim=2)   
mv_h_max_bw = max(mv_common_bw, dim=1)    
mv_p_max_bw = max(mv_common_bw, dim=2)  

mv_h_max_fw: [b, h, 20]  
mv_p_max_fw: [b, p, 20]  
mv_h_max_bw: [b, h, 20]   
mv_p_max_bw: [b, p, 20]

### Operation Three  and Four
![o3.png]({{site.baseurl}}/img/o3.png)  
![o4.png]({{site.baseurl}}/img/o4.png)  
The difficlty lies in the computation of the attentive matrix.  
First we calculate the attention weigths, in practice each attention weight is a cosine similirity between two vectors.  

con_p_fw: [b,p,100]
con_p_bw: [b,p,100]
con_h_fw: [b,p,100]
con_h_bw: [b,p,100] 

att_fw = attention(con_p_fw, con_h_fw)    [b, p, h]  
att_bw = attention(con_p_bw, con_h_bw)    [b, p, h]      
**Inside we calculate cosne(con_p_fw[b, i, :], con_h_fw[b, j, :])**

att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)  
att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)  
att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)  
att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)  
**con_p_fw.unsqueeze(1) [b,p,1,100]**
**att_fw.unsqueeze(3) [b, p, h, 1]**
**con_h_bw.unsqueeze(1) [b, 1, h, 100]**
**For i-th vector in premise, denoted as con_p_fw[0,i,:], The above multiplication is substantially generate a squence weighted vectors(of size [h, 100]) for it**
att_h_fw [b,p,h,100]
att_h_bw [b,p,h,100]
att_p_fw [b,p,h,100]
att_p_bw [b,p,h,100]

**这一步计算出的四个矩阵为matching 操作3 操作4服务，其中操作3是对 p or h 维度求和， 操作4是求max。**

操作3 (其中除法是为了归一化)
att_mean_h_fw = att_h_fw.sum(dim=2) / att_fw.sum(dim=2, keepdim=True)  
att_mean_h_bw = att_h_bw.sum(dim=2) / att_bw.sum(dim=2, keepdim=True)  
att_mean_p_fw = att_p_fw.sum(dim=1) / att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1)  
att_mean_p_bw = att_p_bw.sum(dim=1) / att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1)  


操作4(求最大)  
att_max_h_fw, _ = att_h_fw.max(dim=2)  
att_max_h_bw, _ = att_h_bw.max(dim=2)   
att_max_p_fw, _ = att_p_fw.max(dim=1)  
att_max_p_bw, _ = att_p_bw.max(dim=1)    

讲过以上操作 操作3，4要进行匹配的向量求出，那么就自然可以用fm进行匹配。  
mv_p_att_mean_fw = fm(con_p_fw, att_mean_h_fw, W5)  
mv_p_att_mean_bw = fm(con_p_bw, att_mean_h_bw, W6)  
mv_h_att_mean_fw = fm(con_h_fw, att_mean_p_fw, W5)  
mv_h_att_mean_bw = fm(con_h_bw, att_mean_p_bw, W6)  

mv_p_att_max_fw = fm(con_p_fw, att_max_h_fw, W7)  
mv_p_att_max_bw = fm(con_p_bw, att_max_h_bw, W8)  
mv_h_att_max_fw = fm(con_h_fw, att_max_p_fw, W7)  
mv_h_att_max_bw = fm(con_h_bw, att_max_p_bw, W8)  


### Last 

mv_p = torch.cat([mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)  
 mv_h = torch.cat([mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)  
 agg_p_last = LSTM(mv_p)
 agg_h_last = LSTM(mv_h)
pred = fc(cat(agg_p_last,agg_h_last ))

DONE~
总结： BIMPM模型和multihead attention比较像，但是比较其操作3,4中的attention vector算的比较贵，归一化用的不是softmax，不能保证权重为正。负的权重应该是不对的，这回负的权重乘到负的vecor向量上得到了正的值，anyway，想不通有点怪。
