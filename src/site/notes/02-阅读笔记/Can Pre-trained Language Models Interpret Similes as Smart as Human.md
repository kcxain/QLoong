---
{"dg-publish":true,"permalink":"/02-阅读笔记/Can Pre-trained Language Models Interpret Similes as Smart as Human/","title":"Can Pre-trained Language Models Interpret Similes as Smart as Human","tags":["明喻","ACL2022"]}
---


## 构造负例

- 使用 ConceptNet 的 HasProperty 属性和 COMET 来初步检索出 property 的负例池
- 将负例池中的每个 property 放入原句子中，输入 RoBERTa。把 property 的词向量与表征句子的 \[CLS\] 向量拼接起来，作为其分数。选取与正例 property 作余弦相似度的 top 3 property

![image.png](https://kkcx.oss-cn-beijing.aliyuncs.com/img/20240317163222.png)

## 思路

baseline 选取了如下统计方法

- EMB
- Meta4meaning
- ConScore
- MIUWE

### MLM

用 Bert 和 RoBERTa 做 MLM。文章还分别将本体或喻体替换为 [UNK]，将特征替换为 am/is/are，对比模型效果下降了多少，来确定什么成分影响最大。最终结果是喻体>主体>特征，与常识吻合。

### TransE

[Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)

把 topic 和 vehicle 看作两个实体，它们之间的 property 为属性，这是一个三元组 $(t,p,v)$，然后做 $t+p$ 与 $v$ 的距离。

最后的损失函数是把 TransE 的损失与 MLM 的损失加起来

## 我的实验

### 一、zero shot 中，不设置候选，在整个词表中选择分数最大的词

**结果：** 词表大小为 30522

bert：

![image.png](https://kkcx.oss-cn-beijing.aliyuncs.com/img/20240317163315.png)

roberta：

![image.png](https://kkcx.oss-cn-beijing.aliyuncs.com/img/20240317163338.png)

**问题：** 如果明喻句子中没有 event(the predicate indicating act or state)，那么就不可能选对。例如：

![image.png](https://kkcx.oss-cn-beijing.aliyuncs.com/img/20240317163402.png)

论文为此例构造的候选是：happy, unhappy, interested, sad。在这种情况下是可以选对的。

### 二、将模型改造成 Condenser 的形式

用 bert-base-uncased 权重，和随机初始化最后两层权重。

微调后，最好验证结果为 68.65，相比论文 67.74 提高了

> [!question]
> 为什么提高了？这个点很奇怪，可以想个好故事

### 三、将模型换为 DeBERTa

使用 microsoft/deberta-base，在 General 数据集上跑

~~zero shot: 24.77~~

> [!bug]
> huggingface 的 DebertaForMaskedLM 实现有 bug！这个实验暂时做不了，除非自己修 bug，见：[https://github.com/huggingface/transformers/pull/18674](https://github.com/huggingface/transformers/pull/18674)

### 四、在我们数据集上的实验

问题：我们的数据集有很多是短语形式的词语，这种该如何处理？以及有一些复合词，像 macrophagous，uninteresting 这种不在词表，wordpiece 分词的时候会把它分成多个 token。

![image.png](https://kkcx.oss-cn-beijing.aliyuncs.com/img/20240317163825.png)

目前思路：无论候选是短语还是单个词，都把它们看作短语来做，用 predict 的 top1 去和四个选项做相似度来计算排序

bert-large 模型

解释

zero-shot：56.03

原论文数据微调后：58.04

生成

zero-shot：29.67

注意到生成任务数据的喻体候选大部分都是：冠词 + 名词 或 修饰词 + 名词的形式

所以思路是：取喻体候选的最后一个单词作为候选，计算模型生成该单词的概率

**bert-large** 模型：

zero shot：43.62

**bert-base** 模型：

zero shot：45.10
