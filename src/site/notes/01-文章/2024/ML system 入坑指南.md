---
{"dg-publish":true,"permalink":"/01-文章/2024/ML system 入坑指南/","title":"ML system 入坑指南","tags":["MLSys"]}
---


# ML system 入坑指南

> 转载自：[摸黑干活](https://fazziekey.github.io/)

最近 ChatGPT 大火，越来越多人开始关注大模型，但对于大模型落地而言，除了先进的算法，其背后的 MLsystem(机器学习系统)， 从分布式训练到高效推理的完整链路同样重要， 好的基础设施是应用爆发的基础.

作为一个入坑 MLsys 快两年半的练习生， 本文主要围绕自己学习的经历来构筑，会持续更新，希望能给希望入坑的新人一个指引，也给非 Mlsys 背景但感兴趣的其他领域的同学一些启发.

<!--truncate-->

## Course

首先是课程，入坑 MLsys，基本的计算机背景知识比如数据结构就不多聊了，更多讲讲一些更加专业性的进阶课程

### Operating System

#### 南京大学 JYY OS

南京大学 JYY 老师开的操作系统课内容非常硬核， workload 巨大，课程质量比肩四大

#### MIT 6.S081

MIT 经典 OS 课，资料，lab 都非常全

- [课程主页](https://pdos.csail.mit.edu/6.828/2020/schedule.html)
- [MIT 6.S081 中文 Tutorial Book](https://mit-public-courses-cn-translatio.gitbook.io/mit6-s081/)

### Parallel computing

#### CMU15418 Parallel computing

并行计算非常好的入门课，内容硬核，workload 巨大，涉及现代多处理器，CPU 加速比如 SIMD，分布式通讯协议 MPI，GPU 加速 Cuda 编程，异构计算，同步，Cache

- [课程主页](http://15418.courses.cs.cmu.edu/spring2016/)

#### UCB cs267 Applications of Parallel Computers

HPC 祖师爷 Jim Demmel 22 spring 最新版本，

- [课程主页](https://sites.google.com/lbl.gov/cs267-spr2022?pli=1)

### Distributed system

#### MIT6.824 分布式系统

这门课推荐的人也非常多了，用 go 实现，了解传统的分布式系统知识和历史对现代的分布式机器学习系统的学习还是有一定的帮助，不过对于做 MLsys 不是必须.

- [课程主页](https://mit-public-courses-cn-translatio.gitbook.io/mit6-824/)

### MLSystem

#### CMU DL System

陈天奇老师的课，涉及 nn 库实现，自动微分，GPU 加速，模型部署和部分 AI 编译内容，内容除了分布式训练涉及的不够，基础的 MLsys 还是非常全面的.

- [课程主页](https://dlsyscourse.org/)

#### Mini torch

完全用 python 实现的简单 torch 版本，涉及自动微分，张量，GPU 加速.适合新手入门

- [课程主页](https://minitorch.github.io/)

#### 机器学习系统: 设计和实现

华为 Mindspore 团队 (没错，就是我打过工的 Team) 和一群大佬 AP 搞的， 计算图，编译器前后端，分布式训练都有涉及，内容比较全面，比较适合有一定基础的人阅读或者作为工具书.

- [主页](https://openmlsys.github.io/#)

#### System for AI

微软发起的，目前还在快速迭代更新的工具书，舍和补全基础.

- [主页](https://github.com/microsoft/AI-System)

### AI Compilation

#### Machine Learning Compilation

还是陈天奇老师的课，以 TVM 为基础， AI 编译器这样前沿的领域为数不多的课.

- [课程主页](https://mlc.ai/summer22-zh/)

### Large model

对于做 ML system 的同学而言，了解一些最新的算法也是非常必要的，不用过度关系一些 fancy 的技巧，更多关注模型架构，参数，大的范式上的变化即可.

算法的业界进展确实太快了，很难有系统的课，某些顶级大学会用讲座的形式开展，去讲 GPT，PLAM 这样的大模型， 看论文和官方网站，blog 是更好的选择.

可以看李沐老师论文精讲去 follow 一些比较新且影响力巨大的工作， Muli is all you need !!!

- [paper-reading github 主页](https://github.com/mli/paper-reading)

### Large model training & paper

这块目前还没有比较系统的课，大规模的分布式训练开始应用也就这几年的事情，也是 MLsys 领域的最大热点，这里简单总结一下需要掌握的知识点和参考论文

- Data Parallel(数据并行)
- Distributed Data Parallel(分布式数据并行)
  - [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)
- Mix precise Training(混合精度训练)
  - [Mix precise Training](https://arxiv.org/abs/1710.03740)
- Zero Optimizer(零冗余优化器)
  - [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- Tensor Parallel(张量并行)
  - 1D 并行:[Megatron-LM](https://arxiv.org/abs/1909.08053)
- Pipeline Parallel(流水并行)
  - [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- Auto Parallel(自动并行)
  - [Alpa:Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023)
- Sequence Parallel(序列并行)
  - [Sequence Parallelism: Long Sequence Training from
    System Perspective](https://arxiv.org/abs/2105.13120)
- Large batchsize(超大 batch size)
  - [LARS:Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)
- MOE(混合专家模型)
  - [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- Kernal Fusion(算子融合)
  - [Flash attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Optimizer Fusion(优化器融合)
  - [OPTIMIZER FUSION: EFFICIENT TRAINING WITH BETTER LOCALITY AND PARALLELISM](https://arxiv.org/abs/2104.00237)
- Activation checkpoint
  - [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- Fine-tune accelerate(微调加速)
  - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Isomorphic training(异构训练)
  - [PatrickStar: Parallel Training of Pre-trained Models via
    Chunk-based Dynamic Memory Management](https://arxiv.org/abs/2108.05818)
- Asynchronous Distributed Dataflow(异步数据流)
  - [PATHWAYS: ASYNCHRONOUS DISTRIBUTED DATAFLOW FOR ML](https://arxiv.org/abs/2203.12533)
- Distributed Scheduling(分布式调度)
  - [RLlib: Abstractions for Distributed Reinforcement Learning](https://arxiv.org/abs/1712.09381)
- Quant(量化)
  - [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- 大语言模型推理
  - [FlexGen: High-throughput Generative Inference of Large Language Models with a Single GPU](https://github.com/FMInference/FlexGen/blob/main/docs/paper.pdf)

> 每个细分领域的论文还有很多，不一一列举了，对于入坑来说，抓住主线即可.

## Programming Languages

### C++

- [Cmake](https://www.hahack.com/codes/cmake/)
- [现代CPP](https://changkun.de/modern-cpp/zh-cn/01-intro/)

### Python

Python 基础课这个太多了，不作推荐了，做 MLsys 比较需要掌握用 python 调用 C，比如 Cpython，pybind，以及一些 python 高级特性，比如 hook，装饰器

- [pybind](https://github.com/pybind/pybind11)

### Cuda

这个可以参考的也比较多，英伟达的官方手册永远是最好的参考.

- [Cuda programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### OpenCL

对于非 Nvidia 芯片的设备，比如手机 Soc，移动端推理芯片大多不支持 cuda，那么用 OpenCL 来做异构加速就是一个更通用的方案

- [OpenCL异构计算](https://www.bookstack.cn/read/Heterogeneous-Computing-with-OpenCL-2.0/README.md)

# 军火库

这里会简单总结我接触或使用和直接参与开发的 Mlsys 的军火库，会持续更新.

## Framework

对于 MLsys 这样前沿的领域而言，因为很多方面并没有足够的资料，经常被迫去直接学习源码，以实际的 case 作为学习手段也是非常好的方式.这里简单归类一下我遇到过的 MLsys，大多数处于简单了解和使用，有少部分比较深入看过源码.

### Inference

#### TensorRT

英伟达的推理方案， 目前整体上在英伟达 GPU 上做的最好的推理框架，比较是自己的卡.

- [github](https://github.com/NVIDIA/TensorRT)

#### AI Template

FaceBook 刚搞的一个推理库，在很多硬件上速度性能都超过 TensorRT， 还比较新的框架

- [github](https://github.com/facebookincubator/AITemplate)

### Severing

#### triton-inference-server

英伟达的 ML serving 框架，比较成熟

- [github](https://github.com/triton-inference-server/server)

#### clip-as-service

Jina-AI 做的，一家中国 start up， 在 mass(model as service) 上是一个非常不错的落地产品，特别喜欢

- [github](https://github.com/jina-ai/clip-as-service)

### Mobile inference

移动端推理是我比较深入做过的，对其底层了解的比较多

#### Mindsporelite

我有幸参与写过的推理引擎，对于全流程在 mindspore 上做的体验还是不错的.

- [gitee](https://gitee.com/mindspore/mindspore)

#### MNN

阿里达摩院做的，我写 mindsporelite 的遇到问题的时候经常被 mentor 叫去学习一下友商的代码，CPU 的一些 kernel 用汇编写的，这点映像非常深刻，代码质量非常高.

- [github](https://github.com/alibaba/MNN)

#### TensorFlowlite

集成在 Tensorflow 的移动端推引擎，国际上应该是最早做的移动端推理.没错，TFlite 的大哥就是那个从谷歌跑路重回斯坦福读博的皮特·沃登，他写了 TinyML 这本书，对整个移动端推理都是有重要意义的.我也从这学习了不少.google 的代码质量太高了，看注释都能学很多.

- [github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)

#### NCNN

国内做的最早的端侧推理引擎，腾讯搞的，不得不说，很多东西还是需求驱动， 靠各种移动 APP 为主要产品的中国互联网公司，移动推理引擎做的都不错.

- [github](https://github.com/Tencent/ncnn)

## DeepLearning Framework

大的深度学习的框架的介绍太多了，不一一介绍了，大同小异，做 ML 系统多少都会看看各种框架做的一些新特性，Torch2.0 的一些东西可以关注.

- [Torch](https://github.com/pytorch/pytorch)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [Mindspore](https://gitee.com/mindspore/mindspore)
- [Jax](https://github.com/google/jax)
- [oneflow](https://github.com/Oneflow-Inc/oneflow)
- [paddle](https://github.com/PaddlePaddle/Paddle)
- [ivy](https://github.com/unifyai/ivy)

## AI compiler

编译这块我接触的不多，这两个项目可以参考

### TVM

- [github](https://github.com/apache/tvm)

### BladeDISC

- [github](https://github.com/alibaba/BladeDISC)

## Distributed training

对于大模型而言，分布式训练是比不缺的，除了最基本的分布式数据并行，各种混合并行策略，显存优化策略必不可缺，也因此出现了一系列的大模型训练框架.

### ColossalAI

All in 参与的项目，刚刚 star 接近 15k 了，在 torch 生态下支持各种并行和显存优化策略，给自己打个广告，欢迎各位朋友 star，关注

- [github](https://github.com/hpcaitech/ColossalAI)

### Megatron-LM

NVIDIA 做的模型并行库，也是最早开源的模型并行，但对缺乏分布式训练背景的人使用不太友好.

- [Github](https://github.com/NVIDIA/Megatron-LM)

### Deepspeed

微软的大模型训练框架，核心技术是 Zero infinity 相关的一系列 paper，使用 Megatron-LM 作为张量并行的支持

- [github](https://github.com/microsoft/DeepSpeed-MII)

### huggingface accelerate

huggingface 的加速器，对各种不同硬件做了兼容，在 huggingface 生态下非常好用，在分布式上做了比较多的封装，可以直接调用 Deepspeed.

- [github](https://github.com/huggingface/accelerate)

### Bagua

Bagua 在多机通讯上做了非常多的工作，对 allreduce 等分布式通讯做了不少优化.

- [github](https://github.com/BaguaSys/bagua)

### lightning

lightning 集成了各种分布式后端，可以很方便的启动各种分布式策略，lightning 本身是一套更好的 MLflow，设计理念是算法和工程分开，提供了大量自定义 hook，对于大型 AI 项目而言，是个不错的选择，但是学习门槛不低，对团队人员水平要求比较高.

- [github](https://github.com/Lightning-AI/lightning)

## Distributed Communication

在 nccl 之前 MPI 是分布式通讯的主要方式，简单了解一下还是有必要的

### MPI(the Message Passing Interface)

- [MPI Tutorials](https://mpitutorial.com/tutorials/)

## Tips

最后简单给几个 Tips，

- 尽可能参与到业界的项目去，多注重落地，在实验玩一些很 fancy 但不落地的东西意义不大，目前 MLsys 业界比学界跑的快太多了，学界比较有影响力的工作往往也是何大公司合作的，比如前面提到的 [Alpa](https://arxiv.org/abs/2201.12023) 的实验就是在 Google 的 TPU 集群和 AWS 上做的，一般的 lab 更本没有条件.
- 及时关注各种 Blog，新的论文，这个领域每天都有新的东西出来，比如我就比较喜欢每天去 twitter，一些 Discord channel 挖宝，总能带来一些惊喜
- Mlsys 是一个交叉全面的领域，除了软件和算法，懂点基本的硬件，学会做产品也很重要，更多时候易用性大于性能.
- 新的东西肯定是看不完的，抓住主干即可，在自己做的部分保证专注
- 拥抱开源社区，不管是 github 还是 huggingface，你永远想象不出开源社区的老哥的创造力
