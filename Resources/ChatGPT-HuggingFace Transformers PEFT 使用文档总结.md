# Hugging Face Transformers 与 PEFT 使用文档总结

> 目标：面向 BERT 微调、MLM 继续预训练、LoRA 参数高效微调实验，整理 Hugging Face 官方文档中最常用的概念、API 和代码范式。  
> 适用场景：中文/英文文本分类、BERT 迁移学习、冻结微调、全量微调、LoRA 微调、Mini-BERT MLM 预训练。

---

## 目录

1. [整体关系](#1-整体关系)
2. [安装依赖](#2-安装依赖)
3. [Transformers 核心概念](#3-transformers-核心概念)
4. [Transformers 常用 API](#4-transformers-常用-api)
5. [文本分类：BERT 微调标准范式](#5-文本分类bert-微调标准范式)
6. [冻结 BERT 主体：特征提取式迁移学习](#6-冻结-bert-主体特征提取式迁移学习)
7. [MLM 预训练 / 继续预训练标准范式](#7-mlm-预训练--继续预训练标准范式)
8. [Mini-BERT 从零 MLM 预训练范式](#8-mini-bert-从零-mlm-预训练范式)
9. [PEFT 核心概念](#9-peft-核心概念)
10. [LoRA 核心参数解释](#10-lora-核心参数解释)
11. [PEFT + LoRA 文本分类标准范式](#11-peft--lora-文本分类标准范式)
12. [LoRA adapter 保存、加载与合并](#12-lora-adapter-保存加载与合并)
13. [三种 BERT 迁移策略对比代码范式](#13-三种-bert-迁移策略对比代码范式)
14. [推荐实验组织方式](#14-推荐实验组织方式)
15. [常见问题](#15-常见问题)
16. [官方文档链接](#16-官方文档链接)

---

# 1. 整体关系

Hugging Face 生态中，做 BERT / Transformer 实验最常用的库包括：

```text
transformers：加载预训练模型、tokenizer、训练器 Trainer
datasets：加载和处理数据集
evaluate / sklearn：计算评估指标
peft：添加 LoRA、Adapter、Prefix Tuning 等参数高效微调模块
accelerate：辅助分布式、混合精度等训练
wandb：记录实验过程
```

对于 BERT 微调和 LoRA 实验，可以理解为：

```text
Transformers 负责“模型本体 + tokenizer + 训练流程”
PEFT 负责“在模型上挂 LoRA adapter”
```

典型组合：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
```

---

# 2. 安装依赖

## 2.1 基础安装

```bash
pip install transformers datasets evaluate accelerate scikit-learn
```

## 2.2 使用 PEFT / LoRA

```bash
pip install peft
```

## 2.3 使用 W&B 记录实验

```bash
pip install wandb
wandb login
```

---

# 3. Transformers 核心概念

Transformers 官方文档中，最常见的抽象包括：

| 概念 | 作用 |
|---|---|
| `pipeline` | 快速推理接口，适合体验模型 |
| `Tokenizer` | 把文本转成模型输入，如 `input_ids`、`attention_mask` |
| `AutoTokenizer` | 自动根据模型名加载对应 tokenizer |
| `AutoModel` | 加载没有任务头的基础模型 |
| `AutoModelForSequenceClassification` | 加载带文本分类头的模型 |
| `AutoModelForMaskedLM` | 加载 masked language modeling 模型 |
| `TrainingArguments` | 定义训练超参数 |
| `Trainer` | 封装训练、验证、保存、评估流程 |
| `DataCollator` | 将样本组成 batch，并做 padding / mask 等处理 |

---

# 4. Transformers 常用 API

## 4.1 `pipeline`

适合快速测试模型：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)
```

特点：

```text
优点：非常简单，适合快速体验
缺点：不适合做正式训练实验
```

---

## 4.2 `AutoTokenizer`

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
```

常用编码方式：

```python
encoded = tokenizer(
    "这个模型效果很好。",
    truncation=True,
    max_length=128,
    padding="max_length",
    return_tensors="pt"
)
```

输出通常包括：

```text
input_ids：token 对应的编号
attention_mask：哪些位置是真实 token，哪些是 padding
token_type_ids：句子 A / 句子 B 标记，BERT 句子对任务常用
```

---

## 4.3 `AutoModelForSequenceClassification`

用于文本分类：

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=3
)
```

含义：

```text
BERT 主体加载预训练权重
分类头根据 num_labels 新建
分类头通常是随机初始化的
```

---

## 4.4 `AutoModelForMaskedLM`

用于 MLM 预训练或继续预训练：

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
```

用于：

```text
输入包含 [MASK] 的句子
模型预测被 mask 的 token
```

---

## 4.5 `TrainingArguments`

定义训练参数：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs/bert_cls",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none"
)
```

常用参数：

| 参数 | 含义 |
|---|---|
| `output_dir` | 输出目录 |
| `eval_strategy` | 评估频率 |
| `save_strategy` | 保存频率 |
| `learning_rate` | 学习率 |
| `per_device_train_batch_size` | 每张设备上的训练 batch size |
| `num_train_epochs` | 训练轮数 |
| `weight_decay` | 权重衰减 |
| `logging_steps` | 日志记录间隔 |
| `load_best_model_at_end` | 训练结束时加载最佳模型 |
| `metric_for_best_model` | 判断最佳模型的指标 |
| `report_to` | 日志后端，如 `wandb`、`tensorboard`、`none` |

> 注意：较新版本中常用 `eval_strategy`；旧版本可能使用 `evaluation_strategy`。

---

## 4.6 `Trainer`

官方 Trainer 是一个完整的 PyTorch 训练/评估循环封装。

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

常用方法：

| 方法 | 作用 |
|---|---|
| `trainer.train()` | 开始训练 |
| `trainer.evaluate()` | 在验证集/测试集上评估 |
| `trainer.predict()` | 预测 |
| `trainer.save_model()` | 保存模型 |

---

# 5. 文本分类：BERT 微调标准范式

下面是最常见的 BERT 文本分类完整流程。

## 5.1 加载数据

### Hugging Face 数据集

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
```

### 自定义 CSV

假设 CSV 有两列：

```text
text,label
```

代码：

```python
from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files={
        "train": "data/train.csv",
        "validation": "data/valid.csv",
        "test": "data/test.csv",
    }
)
```

---

## 5.2 Tokenize

```python
from transformers import AutoTokenizer

model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
```

---

## 5.3 动态 Padding

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

作用：

```text
同一个 batch 内动态 padding 到该 batch 的最大长度，
避免所有样本都 padding 到固定 max_length，节省计算。
```

---

## 5.4 加载分类模型

```python
from transformers import AutoModelForSequenceClassification

num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)
```

---

## 5.5 定义指标

```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy.compute(
        predictions=preds,
        references=labels
    )["accuracy"]

    macro_f1 = f1.compute(
        predictions=preds,
        references=labels,
        average="macro"
    )["f1"]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }
```

如果类别不平衡，建议重点看：

```text
Macro-F1
Confusion Matrix
Recall
```

---

## 5.6 训练

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="outputs/bert_cls",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

## 5.7 测试与保存

```python
test_metrics = trainer.evaluate(tokenized_dataset["test"])
print(test_metrics)

trainer.save_model("outputs/bert_cls/best_model")
tokenizer.save_pretrained("outputs/bert_cls/best_model")
```

---

## 5.8 推理

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "outputs/bert_cls/best_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

text = "这个模型的效果非常好。"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()

print(pred)
```

---

# 6. 冻结 BERT 主体：特征提取式迁移学习

冻结 BERT 主体，只训练分类头：

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=num_labels
)

for param in model.bert.parameters():
    param.requires_grad = False
```

检查可训练参数：

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
```

通常你应该主要看到：

```text
classifier.weight
classifier.bias
```

不同模型的主体名称不同：

| 模型 | 主体名称可能是 |
|---|---|
| BERT | `model.bert` |
| RoBERTa | `model.roberta` |
| DistilBERT | `model.distilbert` |
| DeBERTa | `model.deberta` |

建议先打印结构：

```python
print(model)
```

冻结微调适合：

```text
数据少
快速 baseline
显存有限
不想破坏预训练模型主体
```

---

# 7. MLM 预训练 / 继续预训练标准范式

MLM 即 Masked Language Modeling，是 BERT 的核心预训练任务之一。

流程：

```text
无标签文本
↓
tokenizer
↓
随机 mask 一部分 token
↓
BertForMaskedLM 预测被 mask 的 token
```

## 7.1 加载模型

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "bert-base-chinese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
```

如果是继续预训练，这里用 `from_pretrained` 加载已有模型。

---

## 7.2 Tokenize

```python
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )

tokenized_dataset = raw_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=raw_dataset["train"].column_names,
)
```

---

## 7.3 MLM Data Collator

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)
```

作用：

```text
动态随机 mask 部分 token
生成 MLM 训练所需的 labels
```

`mlm_probability=0.15` 是 BERT 常见 mask 比例。

---

## 7.4 训练

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="outputs/bert_mlm",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

---

## 7.5 保存继续预训练模型

```python
trainer.save_model("outputs/bert_domain_mlm")
tokenizer.save_pretrained("outputs/bert_domain_mlm")
```

后续可以这样用于下游分类：

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "outputs/bert_domain_mlm",
    num_labels=num_labels
)
```

---

# 8. Mini-BERT 从零 MLM 预训练范式

如果不是继续预训练，而是自己创建一个小 BERT：

```python
from transformers import BertConfig, BertForMaskedLM

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
    max_position_embeddings=128,
)

model = BertForMaskedLM(config)
```

区别：

| 写法 | 含义 |
|---|---|
| `AutoModelForMaskedLM.from_pretrained(...)` | 加载已有预训练权重 |
| `BertForMaskedLM(config)` | 按配置随机初始化模型 |

Mini-BERT 适合用来理解：

```text
MLM 数据如何构造
预训练 loss 如何下降
预训练模型如何保存
预训练后如何迁移到分类任务
```

---

# 9. PEFT 核心概念

PEFT 是 Parameter-Efficient Fine-Tuning，即参数高效微调。

它的核心思想是：

```text
不更新全部预训练模型参数，
而是只训练少量新增参数或少数可训练模块。
```

常见 PEFT 方法：

```text
LoRA
AdaLoRA
IA3
Prefix Tuning
Prompt Tuning
Adapter Tuning
```

对你当前最重要的是：

```text
LoRA
```

PEFT 常用 API：

| API | 作用 |
|---|---|
| `LoraConfig` | 定义 LoRA 配置 |
| `get_peft_model` | 给基础模型添加 LoRA adapter |
| `PeftModel` | 加载已训练 adapter |
| `TaskType` | 指定任务类型 |
| `print_trainable_parameters()` | 打印可训练参数比例 |
| `save_pretrained()` | 保存 adapter |
| `from_pretrained()` | 加载 adapter |

---

# 10. LoRA 核心参数解释

LoRA 的核心形式：

```text
原始线性层：y = Wx
加入 LoRA：y = Wx + scaling × BAx
```

其中：

```text
W：原模型权重，冻结
A、B：LoRA 低秩矩阵，可训练
scaling = lora_alpha / r
```

---

## 10.1 `r`

```python
r=8
```

LoRA 的秩，决定 adapter 容量。

| r | 特点 |
|---|---|
| 小，如 4 | 参数少、显存低、表达能力有限 |
| 中，如 8/16 | 常用折中 |
| 大，如 32/64 | 表达能力强，但参数更多，可能过拟合 |

入门建议：

```text
r = 8
```

---

## 10.2 `lora_alpha`

```python
lora_alpha=16
```

LoRA 分支缩放系数：

```text
scaling = lora_alpha / r
```

常见组合：

```text
r=8, alpha=16
r=16, alpha=32
```

---

## 10.3 `lora_dropout`

```python
lora_dropout=0.1
```

LoRA 分支上的 dropout，用于正则化。

常见值：

```text
0.05
0.1
```

小数据任务可先用：

```text
0.1
```

---

## 10.4 `target_modules`

```python
target_modules=["query", "value"]
```

指定在哪些模块上插入 LoRA。

BERT 中常见：

```text
query
value
```

也可以尝试：

```text
query, key, value
query, value, dense
```

初学建议先用：

```python
target_modules=["query", "value"]
```

---

## 10.5 `bias`

```python
bias="none"
```

控制是否训练 bias。

| 选项 | 含义 |
|---|---|
| `"none"` | 不训练 bias，最常用 |
| `"all"` | 训练所有 bias |
| `"lora_only"` | 只训练 LoRA 相关 bias |

入门建议：

```text
bias="none"
```

---

## 10.6 `task_type`

```python
task_type=TaskType.SEQ_CLS
```

常见任务类型：

| TaskType | 含义 |
|---|---|
| `SEQ_CLS` | 序列分类，如文本分类 |
| `TOKEN_CLS` | token 分类，如 NER |
| `CAUSAL_LM` | GPT 类因果语言模型 |
| `SEQ_2_SEQ_LM` | T5/BART 类序列到序列模型 |

BERT 文本分类用：

```python
TaskType.SEQ_CLS
```

---

## 10.7 `modules_to_save`

```python
modules_to_save=["classifier"]
```

文本分类任务中，除了 LoRA adapter，分类头也需要保存。

如果不保存分类头，可能出现：

```text
adapter 保存了
但分类头没有正确保存
重新加载后效果不对
```

BERT 分类头通常叫：

```text
classifier
```

不确定时：

```python
print(model)
```

---

# 11. PEFT + LoRA 文本分类标准范式

## 11.1 加载基础模型

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=num_labels
)
```

---

## 11.2 添加 LoRA

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
    modules_to_save=["classifier"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

含义：

```text
BERT 原始参数大多冻结
Attention 中 query/value 加 LoRA adapter
训练 LoRA 参数和分类头
```

---

## 11.3 Trainer 训练 LoRA 模型

LoRA 包装后，训练流程和普通 Transformers 模型基本一样：

```python
training_args = TrainingArguments(
    output_dir="outputs/bert_lora_cls",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

经验学习率：

| 方法 | 学习率起点 |
|---|---|
| 全量微调 BERT | `2e-5`、`3e-5`、`5e-5` |
| LoRA 微调 | `1e-4`、`2e-4` |
| 只训练分类头 | `1e-3` 左右可试 |

---

# 12. LoRA adapter 保存、加载与合并

## 12.1 保存 adapter

```python
model.save_pretrained("outputs/bert_lora_adapter")
tokenizer.save_pretrained("outputs/bert_lora_adapter")
```

或：

```python
trainer.save_model("outputs/bert_lora_adapter")
tokenizer.save_pretrained("outputs/bert_lora_adapter")
```

通常会保存：

```text
adapter_config.json
adapter_model.safetensors
tokenizer 文件
分类头参数，前提是 modules_to_save 设置正确
```

---

## 12.2 加载 adapter 推理

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base_model_name = "bert-base-chinese"
adapter_dir = "outputs/bert_lora_adapter"
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=num_labels
)

model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

text = "这个模型的效果非常好。"

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()

print(pred)
```

---

## 12.3 合并 LoRA 权重

训练完可以把 adapter 合并进原模型权重：

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("outputs/bert_lora_merged")
tokenizer.save_pretrained("outputs/bert_lora_merged")
```

合并后：

```text
W_new = W + ΔW
```

优点：

```text
推理时不再需要单独加载 adapter
结构更像普通模型
```

缺点：

```text
保存体积变大
不如 adapter 方式方便切换任务
```

---

# 13. 三种 BERT 迁移策略对比代码范式

## 13.1 全量微调

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=num_labels
)
```

不冻结任何参数。

训练：

```text
BERT 主体 + classifier 全部训练
```

---

## 13.2 冻结 BERT，只训练分类头

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=num_labels
)

for param in model.bert.parameters():
    param.requires_grad = False
```

训练：

```text
只训练 classifier
```

---

## 13.3 LoRA 微调

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=num_labels
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
    modules_to_save=["classifier"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

训练：

```text
LoRA adapter + classifier
```

---

# 14. 推荐实验组织方式

建议用一个统一脚本，通过参数区分不同模式：

```bash
python train.py --mode full
python train.py --mode frozen
python train.py --mode lora
```

代码逻辑：

```python
if mode == "full":
    model = load_full_finetune_model()

elif mode == "frozen":
    model = load_full_finetune_model()
    freeze_backbone(model)

elif mode == "lora":
    model = load_full_finetune_model()
    model = add_lora(model)
```

这样可以保证：

```text
同一个数据集
同一个 tokenizer
同一个评价指标
同一个训练框架
只改变迁移策略
```

这是规范消融实验的关键。

---

## 14.1 推荐记录指标

```text
Accuracy
Macro-F1
训练时间
显存占用
可训练参数量
best_val_metric
test_metric
```

推荐结果表：

| 方法 | 可训练参数量 | Accuracy | Macro-F1 | 训练时间 | 显存占用 |
|---|---:|---:|---:|---:|---:|
| 冻结 BERT | 待填 | 待填 | 待填 | 待填 | 待填 |
| 全量微调 | 待填 | 待填 | 待填 | 待填 | 待填 |
| LoRA | 待填 | 待填 | 待填 | 待填 | 待填 |

---

# 15. 常见问题

## 15.1 为什么分类头是随机初始化的？

因为预训练 BERT 不知道你的下游任务有几类。

例如：

```text
bert-base-chinese 原本没有“医疗文本三分类”的任务头。
```

因此 `AutoModelForSequenceClassification` 会新建一个分类头。

---

## 15.2 冻结后 BERT 还会工作吗？

会。

冻结只是不更新参数。

```text
仍然前向传播
仍然输出特征
只是参数不被训练数据修改
```

---

## 15.3 LoRA 是不是只训练分类头？

不是。

LoRA 会在 Transformer 内部一些线性层旁边加入低秩 adapter。

```text
冻结特征提取：只训练分类头
LoRA：训练 LoRA adapter + 分类头
```

---

## 15.4 为什么 LoRA 要设置 `modules_to_save=["classifier"]`？

因为文本分类任务中，分类头是任务相关模块，也需要保存。

否则 adapter 保存了，分类头可能没有正确保存。

---

## 15.5 `target_modules` 怎么确定？

先打印模型结构：

```python
print(model)
```

BERT 常见：

```python
target_modules=["query", "value"]
```

如果换成其他模型，需要根据模型内部模块名调整。

---

## 15.6 LoRA 和全量微调的学习率一样吗？

不一定。

通常：

```text
全量微调 BERT：2e-5 / 3e-5 / 5e-5
LoRA：1e-4 / 2e-4
只训练分类头：1e-3 左右可试
```

最终以验证集表现为准。

---

# 16. 官方文档链接

## Transformers

- Quickstart  
  https://huggingface.co/docs/transformers/quicktour

- Trainer  
  https://huggingface.co/docs/transformers/en/main_classes/trainer

- Text Classification  
  https://huggingface.co/docs/transformers/en/tasks/sequence_classification

- Masked Language Modeling  
  https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling

- Data Collator  
  https://huggingface.co/docs/transformers/en/main_classes/data_collator

- Transformers + PEFT integration  
  https://huggingface.co/docs/transformers/en/peft

## PEFT

- PEFT 官方文档  
  https://huggingface.co/docs/peft/en/index

- PEFT LoRA Guide  
  https://huggingface.co/docs/peft/en/developer_guides/lora

- PEFT LoRA API Reference  
  https://huggingface.co/docs/peft/en/package_reference/lora

- PeftModel API  
  https://huggingface.co/docs/peft/en/package_reference/peft_model

- PEFT GitHub  
  https://github.com/huggingface/peft

---

# 17. 一句话总结

```text
Transformers 负责加载和训练预训练模型；
PEFT 负责给模型加 LoRA 等参数高效微调模块；
Trainer 负责统一训练流程；
DataCollator 负责 batch 级数据处理；
LoRA 通过少量低秩 adapter 参数适配下游任务。
```

对于 BERT 迁移学习实验，最重要的三种范式是：

```text
全量微调：Transformers
冻结微调：Transformers + requires_grad=False
LoRA 微调：Transformers + PEFT
```
