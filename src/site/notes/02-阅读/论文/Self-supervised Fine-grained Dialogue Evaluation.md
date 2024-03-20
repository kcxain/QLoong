---
{"dg-publish":true,"permalink":"/02-阅读/论文/Self-supervised Fine-grained Dialogue Evaluation/","title":"Self-supervised Fine-grained Dialogue Evaluation","tags":["对话评价","COLING2022"]}
---


## 多级排序损失

定义语料 $C=\{D_m\}_{m=1}^M$，$D_m$ 指第 $m$ 个对话，它有 $n_m$ 轮，替换其中 $i$ 轮，得到的为 $D_m^i$

### Separation Loss

求不同级别之间的差，它们之间的最大值作为优化目标

### Compactness Loss

缩小同一个级别与参考分数之间的差

### R-drop Loss

同一个 step 里面，对于同一个样本，**前向传播两次，由于 Dropout 的存在，会得到两个不同但差异很小的概率分布**，通过在原来的交叉熵损失中加入这两个分布的 KL 散度损失，来共同进行反向传播，参数更新。
