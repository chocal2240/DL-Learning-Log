# BERT 医疗文本分类三种训练策略对比报告

**实验日期：** 2026-06-13  
**设备：** NVIDIA GeForce RTX 4060 Laptop GPU（8GB）  
**任务：** KUAKE-QIC 中文医疗搜索意图 11 分类  
**数据：** 1,955 条有标签文本，分层切分为训练 1,368 / 验证 293 / 测试 294  
**统一设置：** `bert-base-chinese`、最大长度 64、batch size 16、FP16、最多 8 epochs、seed 42

## 核心结果

| 策略 | 测试 Accuracy | Macro-F1 | 训练时间 | 峰值显存 | 可训练参数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 随机初始化 BERT | 68.37% | 45.49% | 74.89 秒 | 1,807.7 MB | 102.28M（100%） |
| 冻结预训练 BERT | 59.52% | 35.32% | **30.42 秒** | **453.2 MB** | 8,459（0.0083%） |
| 预训练 BERT + LoRA | **77.21%** | **56.94%** | 67.55 秒 | 913.0 MB | 303,371（0.2957%） |

训练时间仅统计 `Trainer.train()`，不含模型下载、测试评估和 W&B 上传；显存为 `torch.cuda.max_memory_allocated()` 记录的训练过程峰值。

## 结果分析

**LoRA 综合表现最佳。** 相比随机初始化，LoRA 的 Accuracy 提高 8.84 个百分点、Macro-F1 提高 11.45 个百分点，同时训练时间缩短约 9.8%、峰值显存降低约 49.5%。它只训练 0.30% 参数，却保留了预训练知识并允许注意力层适应医疗文本，是本实验最合理的默认方案。

**冻结微调成本最低，但效果受限。** 它只更新分类头，训练时间比随机初始化少约 59.4%，峰值显存少约 74.9%，但 Accuracy 和 Macro-F1 均最低。这说明固定的通用 BERT 表示不足以完全适应细粒度医疗意图，线性分类头的调整能力有限。

**随机初始化能拟合高频类别，但代价较高。** 全参数训练使其优于冻结策略，但仍落后 LoRA，说明 1,368 条训练文本不足以从零学出稳定语言表示。测试集中部分稀有类别仅有 4 至 7 条，三种策略在若干稀有类别上的 F1 都为 0，因此本实验应同时参考 Macro-F1，不能只看 Accuracy。

## 适用场景

| 策略 | 更适合 | 不适合 |
| --- | --- | --- |
| 随机初始化 | 验证预训练价值、网络结构教学、拥有海量领域语料时从零训练 | 小数据、短周期、有限算力的实际项目 |
| 冻结微调 | 极低显存、快速原型、只需廉价基线、编码器不能修改的部署环境 | 领域差异明显或要求较高准确率的任务 |
| LoRA | 小中型领域数据、多任务 Adapter、有限 GPU 下追求效果与成本平衡 | 必须彻底重塑模型能力且拥有大量数据和算力的场景 |

**结论：** 对当前小规模中文医疗分类任务，优先选择 **LoRA**；冻结微调适合作为最低成本基线，随机初始化主要用于教学对照。

## W&B 记录

- [数据准备与 Dataset Artifact](https://wandb.ai/chocal2240-lanzhou-university/bert-medical-intent-comparison/runs/apyqemzx)
- [随机初始化](https://wandb.ai/chocal2240-lanzhou-university/bert-medical-intent-comparison/runs/r0d5e1rb)
- [冻结微调](https://wandb.ai/chocal2240-lanzhou-university/bert-medical-intent-comparison/runs/xlgghuup)
- [LoRA](https://wandb.ai/chocal2240-lanzhou-university/bert-medical-intent-comparison/runs/1tibjfwv)
- [三种策略汇总](https://wandb.ai/chocal2240-lanzhou-university/bert-medical-intent-comparison/runs/qm5jcgjr)
