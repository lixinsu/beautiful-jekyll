---
layout: post
title: Implementation Details about BIMPM for NLI
subtitle: Four matching operations for NLI
tags:
  - deep learning
  - textual entailment
published: true
date: '2018-10-26'
---

## Introduction
The BIMPM model is one of the state of the art models on SNLI dataset,
which classify a pair of sentences, premise and hypothesis, to neural, contradiction, and entailment.   
The model mainly consiste of embedidng, contextual encoding, four matching operations, aggregration and fc layers.  
The difficulty of implement the [BIMPM model](https://arxiv.org/abs/1702.03814) lies in the four matching operations. 
In this blog, I will describe the details about the calculation setps. I will give the input and output of each operation, including the size and the calculation in the operation. This blog can be viewed as an explaination of [code repo](https://github.com/galsang/BIMPM-pytorch).

### Preliminary
- P denotes premise.  
- H denotes hypothese.  
- b denotes the batch_size.  
- p denotes the lengths of premise.     
- h denotes the length of hypothesis.  
- x:[a,b] Tensor x is of size a x b.  


Before four matching funciton, the input word indics are first converted to embedding space, and passed through LSTM.

P: [b, p] (size)  
H: [b, h]  
p_emb = word_char_Embedding(P)  
h_emb = word_char_Embedding(H)  
p_emb: [b,p,350]  
h_emb: [b,h,350]  

LSTM(in=300, out=100, bidirectional=True)    
p_con = LSTM(p_emb) -> [b,p,200]  
h_con = LSTM(h_emb) -> [b,p,200]   
Note: con means contextual, we get the contextual representaitons for the p and h.

We split the forward(fw) and backward(bw) contextual representations as follows:
con_p_fw: [b,p,100]  
con_p_bw: [b,p,100]  
con_h_fw: [b,p,100]  
con_h_bw: [b,p,100]  

Now, we are ready for matching operations. 

### Operation One

The diagram is as folloes:
![o1.png]({{site.baseurl}}/img/o1.png)  

The base cosine matching function fm is define as follows in original paper:
![base_function.png]({{site.baseurl}}/img/base_function.png)  

con_p_fw: [b,p,100]   
con_p_bw: [b,p,100]     
con_h_fw: [b,p,100]    
con_h_bw: [b,p,100]   

We conduct the basic fm operation as defined in the above picture to get the matching vectors.  
Note that, the **function fm** conduct the basic matching bwteen sequence of vectors with a single vectors or a sequence of vectors with a sequence of vectors.  
For example:  C =   fm(A, B, W)  
A: [b,p, d] B:[b,d] then C:[b,p 20]  
A: [b,p, d] B:[b,p,d] then C:[b,p 20]  

mv_p_full_fw = fm(con_p_fw, con_h_fw[:, -1, :], W1)-> [b, p, 20]     
mv_p_full_bw = fm(con_p_bw, con_h_bw[:, 0, :], W2) ->[b, p, 20]     
mv_h_full_fw = fm(con_h_fw, con_p_fw[:, -1, :], W1) ->[b, h, 20]     
mv_h_full_bw = fm(con_h_bw, con_p_bw[:, 0, :], W2) ->[b, h, 20]     
20 is number of perspectives.   
Note: operation fm is just like the multi-head attention, which has multiple attention weights for each pair of vectors.  



### Operation Two
![o2.png]({{site.baseurl}}/img/o2.png)  
con_p_fw: [b,p,100]  
con_p_bw: [b,p,100]  
con_h_fw: [b,p,100]  
con_h_bw: [b,p,100]   

mv_common_fw = pairwise_fm(con_p_fw, con_h_fw)   ->  [b, p, h, 20]   
mv_common_bw = paireise_fm(con_p_bw, con_h_bw)   ->  [b, p, h, 20]   

Inside fm, we compute fm(con_p_fw[b_index,i,:], con_h_fw[b_index,j,:]), where i = 1 to p , j= 1 to h .

We then compute the maximum for to get the matching vectors.     
mv_h_max_fw = max(mv_common_fw, dim=1) -> [b, h, 20]     
mv_p_max_fw = max(mv_common_fw, dim=2) ->[b, p, 20]   
mv_h_max_bw = max(mv_common_bw, dim=1)  ->[b, h, 20]    
mv_p_max_bw = max(mv_common_bw, dim=2)  ->[b, p, 20]  


### Operation Three  and Four
![o3.png]({{site.baseurl}}/img/o3.png)
![o4.png]({{site.baseurl}}/img/o4.png)  
The difficlty lies in the computation of the attentive vector.  
First we calculate the attention weigths, specifically, each attention weight is a cosine similirity between two vectors.  

con_p_fw: [b,p,100]  
con_p_bw: [b,p,100]  
con_h_fw: [b,p,100]  
con_h_bw: [b,p,100]   

att_fw = attention(con_p_fw, con_h_fw) -> [b, p, h]  
att_bw = attention(con_p_bw, con_h_bw)  ->  [b, p, h]    

**Inside we calculate cosine similirity between two vectors con_p_fw[b_index, i, :] and con_h_fw[b_index, j, :])**

att_h_fw = con_h_fw.unsqueeze(1) \* att_fw.unsqueeze(3)  -> [b,p,h,100]  
att_h_bw = con_h_bw.unsqueeze(1) \* att_bw.unsqueeze(3)  -> [b,p,h,100]  
att_p_fw = con_p_fw.unsqueeze(2) \* att_fw.unsqueeze(3)  -> [b,p,h,100]  
att_p_bw = con_p_bw.unsqueeze(2) \* att_bw.unsqueeze(3)  -> [b,p,h,100]

con_p_fw.unsqueeze(1): [b,p,1,100]   
att_fw.unsqueeze(3): [b, p, h, 1]  
con_h_bw.unsqueeze(1): [b, 1, h, 100]

**For i-th vector con\_p\_fw[b\_index,i,:] in premise corresponds to a matrix of size [h,100] composed by the hypothesis representations.   
In operation 3, we calulate the weighted sum of of the matrix [h, 100] to vector of size 100.   
In operation 4, we max([h,100]) to get a vector of size 100**   

**In operation3, attentive vector is the weighted sum of each hidden dimension.**     
att_mean_h_fw = att_h_fw.sum(dim=2) / att_fw.sum(dim=2, keepdim=True)     ->[b, p, 100]  
att_mean_h_bw = att_h_bw.sum(dim=2) / att_bw.sum(dim=2, keepdim=True)     ->[b, p, 100]  
att_mean_p_fw = att_p_fw.sum(dim=1) / att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1)    ->[b, h, 100]  
att_mean_p_bw = att_p_bw.sum(dim=1) / att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1)    -> [b, h, 100]   


**In operation 4, attentive vector is maximum of each hidden dimension)**    
att_max_h_fw, _ = att_h_fw.max(dim=2)   ->[b, p, 100]  
att_max_h_bw, _ = att_h_bw.max(dim=2)    ->[b, p, 100]  
att_max_p_fw, _ = att_p_fw.max(dim=1)   ->[b, h, 100]  
att_max_p_bw, _ = att_p_bw.max(dim=1)    ->[b, h, 100]  

**Consequently, we can use the base operation fm to matching the contextual representations (e.g.con\_[p|h]\_[f|b]w) with attentive vectors(e.g.att\_[max|mean]\_[h|p]\_[b|f]w)**     
mv_p_att_mean_fw = fm(con_p_fw, att_mean_h_fw, W5)  -> [b,p,20]  
mv_p_att_mean_bw = fm(con_p_bw, att_mean_h_bw, W6) -> [b,p,20]  
mv_h_att_mean_fw = fm(con_h_fw, att_mean_p_fw, W5)  -> [b,h,20]    
mv_h_att_mean_bw = fm(con_h_bw, att_mean_p_bw, W6)  -> [b,h,20]  

mv_p_att_max_fw = fm(con_p_fw, att_max_h_fw, W7)  -> [b,p,20]   
mv_p_att_max_bw = fm(con_p_bw, att_max_h_bw, W8)  -> [b,p,20]  
mv_h_att_max_fw = fm(con_h_fw, att_max_p_fw, W7)  -> [b,h,20]  
mv_h_att_max_bw = fm(con_h_bw, att_max_p_bw, W8)  -> [b,h,20]  


### Last 

mv_p = torch.cat([mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)  
 mv_h = torch.cat([mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)  
 agg_p_last = LSTM(mv_p)
 agg_h_last = LSTM(mv_h)
pred = fc(cat(agg_p_last,agg_h_last ))

DONE~
总结： BIMPM模型和multihead attention比较像，但是比较其操作3,4中的attention vector算的比较贵，归一化用的不是softmax，不能保证权重为正。负的权重应该是不对的，这回负的权重乘到负的vecor向量上得到了正的值，anyway，想不通有点怪。
