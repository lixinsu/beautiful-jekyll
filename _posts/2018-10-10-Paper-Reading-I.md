---
layout: post
title: Paper Reading
subtitle: ELMo
tags: [deep learning]
---
## 《Deep Contextualized Word Representations》--NAACL 2018
### 前言
这篇文章中，我们介绍来自Allen AI 的 《Deep Contextualized Word Representation》。 该文章提出一种新的词向量 ELMo（Embeddings from Language Models）。 该词项量与Word2Vec ，GLoVe和CoVe相比取得更好效果。   
论文贡献如下：  
 - 从大规模语言模型学习得到的词向量能有效用于半监督迁移学习中；
 - 添加ELMo到多个NLP任务（例如：分类， 问答，文本蕴含，语义角色标注，命名实体识别等）都能显著提升任务的性能。

### EMLo 如何工作的 
ELMo 结构首先训练一个相对复杂的神经网络语言模型，模型结构来自于[大规模语言模型工作](https://arxiv.org/abs/1602.02410). 
语言模型在Wikipedia定义为 	`A statistical language model is a probability distribution over sequences of words.` 。可以将句子的概率分解为根据已有的词预测下一个词。有了训练好的语言模型，我们更容易知道`I am going to write with a` 下一个词是 `pencil` 的概率大于`frog`。

对于ELMo，该模型的主要部分是2层的双向LSTM网络。如下图所示  
![Alt text](/img/1539137783760.png)

在ELMo的具体实现中，作者还引入了残差结构，来克服参数量太大带来的训练困难。  
![Alt text](/img/1539137809702.png)

此外，ELMo中采用字符感知的语言模型，具体来说每个词的embedding并不是简单的在embedding matrix 里查找对应的embedding，它首先将每个词拆分为字符序列，获得每个字符的embedding， 将词的字符embedding表示通过卷积层（多种filter）+池化层；再将该表示通过两层的 highway network。通过以上几层获得词的向量表示，将该表示输入到LSTM。细节可以参见关于[字符感知的语言模型](https://arxiv.org/pdf/1508.06615.pdf)
下图截取自该论文。  

![Alt text](/img/1539132499103.png)  

ELMo中每个词的embedding的计算过程如下  
![Alt text](/img/1539132801190.png)  

**这些转换的优势在于以下几点**
 - 字符embedding经过多种尺寸的卷积核可以捕捉语素信息,使得对未登陆词也有合理的embedding表示
 - highway network可以对输入信息进行转换
 
以上是关于其结构的介绍，ELMo的亮点在于其如何使用预训练的语言模型的输出结果。其对语言模型的不同层的输出进行加权，且权重根据任务去学习。具体过程如下，
假定我们想要寻找第k个词的ELMo embeddings，我们将整个句子通过预训练的语言模型，收集上下文无关的词embedding和上下文相关的两层LSTM的输出，然后对上述三者进行加权，得到最终的ELMo embedding，过程如下图所示  
 ![Alt text](/img/1539138806514.png)  
具体公式如下  
![Alt text](/img/1539134755347.png)  
其中$S_i$表示softmax-normalized权重，是任务特定的可学习参数，$\gamma_k$表示一个超参数缩放因子。  
使用时，将ELMo embedding代替或者加入到NLP任务的最底层，即代替Glove 或者拼接到其上。具体任务使用细节有待调试，比如把ELMo加入到RNN的输入处，还是RNN的输入输出都加，此处RNN是指NLP任务中的模型中最底层的RNN层。 另外为模型参数添加L2正则也可以提升模型效果。

### 实验结果和分析
| Task  | Previous SOTA  |  ELMo Results |
|---|---|---|
| SQuAD (question/answering)	| 84.4 |	85.8 |
| SNLI (textual entailment)	| 88.6	| 88.7 |
| Semantic Role Labelling |	81.7 |	84.6 |
| Coref Resolution | 	67.2 |	70.4 |
| NER |	91.93 |	92.22 |
| SST-5 (sentiment analysis) |	53.7 |	54.7 |

 ELMo 在上述6个任务上的SOTA指标上都取得了更好都性能，且该向量可以被用在多种其他任务上。  
 进一步的实验结果分析可以说明该2-layer 双向LSTM模型的第二层捕捉更加长期的上下文信息，该层的输出适合于做词义消歧等语义任务，第一层捕捉较小的上下文信息

### Contextual Embedding 发展脉络浅谈
ELMo 借助大规模语言模型受CoVE(NIPS 17)启发，[paper](https://arxiv.org/abs/1602.02410)研究如何在大语料上训练语言模型，[paper](https://github.com/salesforce/cove)研究使用翻译模型中的encoder编码输出作为contextual embedding。ELMo借助了大规模语言模型的结构，利用大量的单语语料和字符感知的语言模型，超越CoVe。

该工作的前身是作者在ACL17年发表的[TagLM](https://arxiv.org/abs/1705.00108)， 其语言模型结构基本同ELMo，但是只用到其LSTM最上层输出，ELMo则使用了多层的输出，对多层输出进行加权，且在6个NLP任务上进行了实验。

作者在[EMNLP18发表新的论文](https://arxiv.org/abs/1808.08949)对不同结构语言模型效果进行探讨，弥补了ELMo直接无解释的使用LSTM结构问题。

[OpenAI有工作](https://blog.openai.com/language-unsupervised/)利用transformer训练大规模语言模型，其目的不再是利用语言模型产生词向量，而是将语言模型直接用到任务上，也就是语言模型的输出直接接上任务特定的输出层，例如，对于蕴含任务其直接将两段文本拼接，输入语言模型做分类，取得SOTA。大规模无标注语料的力量特别强大，预训练的语言模型真的有点imagenet上预训练的卷积层的味道。


### 实现资源
作者提供的Tensorflow版本code,支持训练和预测 [biLM](https://github.com/allenai/bilm-tf)  
 AllenNLP toolkit(PyTorch实现) 里包含预测部分代码,[ELMo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) ,其中包含英语的预训练模型  
HIT SCIR 的zhengbo从AllenNLP中抽取EMLo部分代码并加入训练部分，形成支持训练和预测的PyTorch版本代码[ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs) ,其中包含了汉语的预训练模型

`向开发者的劳动致敬`	
`文中的图截取自 https://www.mihaileric.com/posts/deep-contextualized-word-representations-elmo/ ,向作者的劳动致敬`。
