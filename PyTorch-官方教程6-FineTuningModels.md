---
title: PyTorch官方教程(六)-FineTuning Torchvision Models
sitemap: true
categories: PyTorch
date: 2018-07-11 19:08:43
tags:
- PyTorch
---

在本篇教程中, 我们将教授你如何 finetune 一个 torchvision models, 这些 models 都已经在 ImageNet 的 1000-class 数据集上进行过预训练. 由于每个模型的结构都不太相同, 因此没有通常的代码模板来适应所有的场景需要. 在这里, 我们将会演示两种类型的迁移学习: finetuning 和 feature extraction. 在 **finetuning** 中, 我们会从一个预训练好的模型开始, 将该模型的参数应用到新的任务上去, 然后重新训练整个模型. 在 **feature extraction** 中, 我们同样会从一个预训练好的模型开始, 但是仅仅更新最后的几层网络, 而将之前的网络层参数固定不变. 通常情况下, 这两种迁移学习都包含以下几步:
- 用预训练模型初始化参数
- 更改最后的一层或几层神经层, 使其输出适应我们的新数据集
- 在 optimization 算法中定义我们希望更新的参数
- 执行训练
