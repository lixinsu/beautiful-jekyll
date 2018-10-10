---
layout: post
title: Paper Reading
subtitle: ELMo
tags: [deep learning]
---
## 《Deep Contextualized Word Representation--NAACL 2018
### 前言
这篇文章中，我们介绍来自Allen AI 的 Deep Contextualized Word Representation。 该文章提出一种词向量 ELMo（Embeddings from Language Models）。 该词项量与Word2Vec ，GLoVe和CoVe相比取得更好效果。   
**论文贡献如下**
 - 从大规模语言模型学习得到的词向量能有效用于半监督迁移学习中
 - 加ELMo 到基本任何NLPtask（分类， 问答，文本蕴含，语义角色标注，命名实体识别等）都会有显著性能提升

### EMLo 如何工作的 
ELMo 结构首先训练一个相当复杂的神经网络语言模型，模型结构来自于前人[工作](https://arxiv.org/abs/1602.02410). 
语言模型的Wikipedia定义为 	`A statistical language model is a probability distribution over sequences of words.` 。可以将句子的语言模型分解为根据已有的词预测下一个词。有了训练好的语言模型，我们更容易知道`I am going to write with a` 下一个词是 `pencil` 的概率大于`frog`。

对于ELMo，该模型的主要部分是2层的双向的LSTM网络。如下图所示  
![Alt text](/img/1539137783760.png)

ELMo实现中，还加入残差结构。  
![Alt text](/img/1539137809702.png)

此外，ELMo还做了更加复杂但是有价值的操作，对于每个词的embedding其不是简单的在embedding matrix 里查找对应词的embedding，其首先将每个词拆分为字符，获得每个字符的embedding， 将词的字符embedding表示通过卷积层（多种filter）+ max-pool 层，再将该表示通过 2-layer highway network。通过以上基层获得每个词的向量表示，将该向量输入到LSTM。细节可以参见关于[字符感知的语言模型](https://arxiv.org/pdf/1508.06615.pdf)
下图截取自该论文。  

![Alt text](/img/1539132499103.png)  

该过程的缩略表示图如下  
![Alt text](/img/1539132801190.png)  

**这些转换的优势在于以下几点**
 - 字符embedding可以捕捉的语素信息,对袋外词有合理的表示
 - 卷积可以捕捉ngram信息，highway可以对输入信息进行转换
 
以上是关于其结构的介绍，
 ELMo的亮点在于其如何使用预训练的语言模型，而其语言模型结构基本同[论文](https://arxiv.org/abs/1602.02410)
 假定我们想要寻找第k个词的embedding，使用上述语言模型对整个句子进行运算得到两个不同层的LSTM输出和上下文无关向量输出（即通过CNN—>Highway后的输出）。我们对上述三者进行加权，如下图所示
 ![Alt text](/img/1539138806514.png)  

具体公式如下
![Alt text](/img/1539134755347.png)  
其中$S_i$表示softmax-normalized权重，是任务特定的可学习参数，$\gamma_k$表示一个超参数缩放因子。
将该加权表示代替或者加入到NLP任务的最低层，即代替Glove 或者拼接到其上，即可见效，具体任务使用细节有待调试，比如加入到RNN的输入处，还是RNN的输入输出都加，此处RNN指NLP task本来模型中最低层的RNN。 另外加L2正则也可以提升模型效果。
### 实验结果和分析

| Task  | Previous SOTA  |  ELMo Results |
|---|---|---|
| SQuAD (question/answering)	| 84.4 |	85.8 |
| SNLI (textual entailment)	| 88.6	| 88.7 |
| Semantic Role Labelling |	81.7 |	84.6 |
| Coref Resolution | 	67.2 |	70.4 |
| NER |	91.93 |	92.22 |
| SST-5 (sentiment analysis) |	53.7 |	54.7 |

 ELMo 在超越了6个任务的SOTA，且该向量可以被用在多种其他任务上。  
 进一步的实验结果分析可以说明该2-layer 双向LSTM模型的第二层捕捉更加长期的上下文信息，该层的输出适合于做词义消歧等语义任务，第一层捕捉较小的上下文信息

### Contextual embedding 发展脉络浅谈
ELMo 借助大规模语言模型和CoVE(NIPS 17)启发，[paper](https://arxiv.org/abs/1602.02410)大规模语言模型如何在大语料上训练语言模型，[paper](https://github.com/salesforce/cove)研究使用翻译模型中的encoder编码输出作为contextual 词向量。ELMo借助大规模语言模型的结构，利用无穷量的单语语料，超越CoVe。

该工作的前奏是作者在ACL17年发表的[TagLM](https://arxiv.org/abs/1705.00108)， 其利用语言模型结构基本同ELMo，但是只用到其LSTM最上层输出，ELMo用了多层的输出，对多层输出进行了依据特定任务的加权，且在6个NLP task进行实验。

该工作后续的发展，作者在[EMNLP18发表新的论文](https://arxiv.org/abs/1808.08949)对不同结构语言模型效果进行探讨，弥补ELMo直接就无理由的上了LSTM结构的草率。

[OpenAI有工作](https://blog.openai.com/language-unsupervised/)利用transformer训练大规模语言模型，且迁移学习的野心更加庞大，不再是利用语言模型产生词向量，而是语言模型直接用到任务上，也就是语言模型的输出直接就接任务特定的输出层，例如，对于蕴含任务其直接将两段文本拼接，输入语言模型做分类，居然取得STOA。大规模无标注语料的力量特别强大，预训练的语言模型真的有点imagenet上预训练的卷积层的味道。


### 实现资源
作者提供的Tensorflow版本code,支持训练和使用 [biLM](https://github.com/allenai/bilm-tf)  
 AllenNLP toolkit(PyTorch实现) 里包含预测部分代码,[ELMo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) ,其中包含英语的预训练模型  
HIT SCIR 的zhengbo从AllenNLP中抽取并加入训练code的 PyTorch版本代码[ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs) ,其中包含了多种语言的预训练模型

`向开发者的劳动致敬`	
`文中的图截取自 https://www.mihaileric.com/posts/deep-contextualized-word-representations-elmo/ ,向作者的劳动致敬`。
