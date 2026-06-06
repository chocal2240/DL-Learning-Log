# BERT LoRA MLM Demo

这个小项目对应一条完整的 BERT 学习路线：

1. 用 `google-bert/bert-base-chinese` 作为基础模型。
2. 在中文酒店评论上做少量 MLM 继续预训练。
3. 用 LoRA 做中文情感二分类微调。
4. 对比直接 LoRA 微调和先 MLM 继续预训练再 LoRA 微调。

默认数据集是 `lansinuote/ChnSentiCorp`，字段为 `text` 和 `label`，标签含义是负面/正面评论。

## 文件结构

```text
bert_lora_mlm_demo/
├── requirements.txt
├── 01_mlm_continue_pretrain.py
├── 02_lora_finetune_cls.py
├── 03_predict.py
└── README.md
```

## 安装依赖

```bash
cd Code/bert_lora_mlm_demo
pip install -r requirements.txt
```

如果你已经安装过仓库根目录的深度学习依赖，只需要补装：

```bash
pip install transformers datasets evaluate accelerate peft scikit-learn wandb
```

## 1. MLM 继续预训练

这一步不是从零训练 BERT，而是在已有中文 BERT 上继续做遮盖词预测。

```bash
python 01_mlm_continue_pretrain.py
```

默认输出目录：

```text
bert-mlm-continued/
```

显存不够时可以降低 batch size：

```bash
python 01_mlm_continue_pretrain.py --batch_size 8
python 01_mlm_continue_pretrain.py --batch_size 4
```

如果想跑更多数据：

```bash
python 01_mlm_continue_pretrain.py --max_train_samples 8000 --num_train_epochs 2
```

使用全部训练集，并将训练曲线实时记录到 W&B：

```bash
set WANDB_PROJECT=bert-lora-mlm-demo
python 01_mlm_continue_pretrain.py ^
  --max_train_samples 0 ^
  --num_train_epochs 5 ^
  --batch_size 8 ^
  --logging_steps 20 ^
  --report_to wandb ^
  --run_name mlm-full-5ep
```

脚本会在每个 epoch 后计算 validation loss 和 perplexity，并保留 validation loss 最低的 checkpoint。

## 2. LoRA 情感分类微调

默认使用第一步得到的 `./bert-mlm-continued` 作为基础模型：

```bash
python 02_lora_finetune_cls.py
```

默认输出目录：

```text
bert-lora-sentiment/
```

如果想做 baseline，也就是不做 MLM 继续预训练，直接对原始中文 BERT 做 LoRA 微调：

```bash
python 02_lora_finetune_cls.py ^
  --model_name google-bert/bert-base-chinese ^
  --output_dir bert-lora-baseline
```

Linux/macOS 可以把上面的 `^` 换成 `\`。

全量数据、多轮训练并启用 early stopping：

```bash
set WANDB_PROJECT=bert-lora-mlm-demo
python 02_lora_finetune_cls.py ^
  --max_train_samples 0 ^
  --num_train_epochs 10 ^
  --early_stopping_patience 3 ^
  --logging_steps 20 ^
  --report_to wandb ^
  --run_name lora-full-10ep
```

## 3. 预测

使用 MLM 继续预训练后的基础模型和 LoRA adapter：

```bash
python 03_predict.py
```

自定义输入文本：

```bash
python 03_predict.py --text "位置很好，服务也很贴心。" --text "房间很吵，体验很差。"
```

如果要测试 baseline adapter：

```bash
python 03_predict.py ^
  --base_model google-bert/bert-base-chinese ^
  --lora_model bert-lora-baseline
```

## 推荐实验记录

| 实验组 | 做法 | 目的 |
| --- | --- | --- |
| A | `bert-base-chinese` + LoRA 微调 | baseline |
| B | MLM 继续预训练 + LoRA 微调 | 观察领域继续预训练是否提升下游分类 |
| C | 全参数微调，可选 | 和 LoRA 比较训练参数量、显存、时间 |

重点记录：

```text
accuracy
f1
训练时间
显存占用
可训练参数比例
```

LoRA 微调脚本会打印可训练参数比例，可以直接写进实验报告。

## 本次扩展实验结果

本次使用 RTX 4060 Laptop GPU、完整的 9600 条训练集进行实验，并通过 W&B 原生记录训练曲线。

| 实验 | 实际训练 | 最佳验证 F1 | 测试 Accuracy | 测试 F1 |
| --- | ---: | ---: | ---: | ---: |
| MLM 继续预训练 | 4 epochs | perplexity 3.1894 | - | - |
| MLM 后 LoRA | 7 epochs，early stop | 0.9359 | 0.9400 | 0.9395 |
| 原始 BERT LoRA baseline | 8 epochs | 0.9410 | 0.9433 | 0.9431 |

实验说明：

- 扩大数据量和训练轮数后，两个 LoRA 实验都明显高于最初的小规模结果。
- 当前配置下，baseline 略高于 MLM 后 LoRA，测试 Accuracy 高约 0.33 个百分点。
- 这说明本次 MLM 继续预训练没有产生额外增益。可能原因包括训练语料与分类语料相同、MLM 数据规模仍较小，或继续预训练学习率和轮数需要进一步调参。
- `bert-mlm-continued` 和 `bert-lora-sentiment` 已指向本次扩展实验产物；baseline 可通过 `bert-lora-baseline` 使用。

W&B runs：

- [全量 MLM 继续预训练](https://wandb.ai/chocal2240-lanzhou-university/bert-lora-mlm-demo/runs/xymu70wq)
- [全量 MLM 后 LoRA](https://wandb.ai/chocal2240-lanzhou-university/bert-lora-mlm-demo/runs/3mu2ttr9)
- [全量原始 BERT LoRA baseline](https://wandb.ai/chocal2240-lanzhou-university/bert-lora-mlm-demo/runs/wimzj616)
