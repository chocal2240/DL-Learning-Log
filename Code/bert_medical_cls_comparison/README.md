# BERT 医疗文本分类策略对比

本项目在中文医疗搜索意图分类任务上对比三种 BERT 训练方式：

| 策略 | BERT 编码器 | 训练参数 |
| --- | --- | --- |
| `random_init` | 按 BERT 配置随机初始化 | 全部参数 |
| `frozen` | 加载预训练权重并冻结 | 仅分类头 |
| `lora` | 加载预训练权重 | Query/Value LoRA 与分类头 |

随机初始化组只复用 `bert-base-chinese` 的分词器和网络结构，不加载其模型权重。因此它和另外两组的差距能直观体现预训练的价值。

## 数据集

任务使用 CBLUE 的 KUAKE-QIC，它把中文医疗搜索问题分为 11 种意图。官方完整数据包含 6,931 条训练样本和 1,955 条验证样本。本项目默认只使用公开镜像中的 1,955 条有标签验证数据，再按标签分层切分为：

| split | 比例 | 约计样本数 |
| --- | ---: | ---: |
| train | 70% | 1,368 |
| validation | 15% | 293 |
| test | 15% | 294 |

这样做是为了在 RTX 4060 Laptop 上快速完成教学型对比，而不是追求榜单成绩。切分种子固定为 `42`，三组实验使用完全相同的数据。

数据来源：

- [CBLUE 官方仓库](https://github.com/CBLUEbenchmark/CBLUE)
- [KUAKE-QIC 小数据镜像](https://huggingface.co/datasets/wyp/CBlue-KUAKE-QIC)

## 项目结构

```text
bert_medical_cls_comparison/
├── config.yaml             # 数据、训练策略和 W&B 配置
├── common.py               # 公共常量与兼容函数
├── prepare_data.py         # 下载、分层切分、上传 W&B Artifact
├── train.py                # 训练单个策略
├── run_experiments.py      # 串行训练三组并汇总
├── requirements.txt
└── README.md
```

运行后会生成：

```text
D:/DL-Learning-Log-Artifacts/bert_medical_cls_comparison/
├── data/processed/         # 固定切分后的 JSONL 数据
├── outputs/<strategy>/     # 最佳模型或 LoRA Adapter
├── results/*.json          # 指标、耗时、显存、分类报告
├── results/comparison.csv  # 三组对比表
└── wandb/                  # W&B 本地日志
```

默认把大文件写入 D 盘，避免占用系统盘。路径可通过 `config.yaml` 中的 `storage.root_dir` 和 `storage.cache_dir` 修改。

## 环境准备

```powershell
conda activate learning
cd Code/bert_medical_cls_comparison
pip install -r requirements.txt
wandb login
```

默认 W&B 项目名是 `bert-medical-intent-comparison`。如需指定自己的团队，在 `config.yaml` 中填写 `wandb.entity`。

## 只准备数据

以下命令会下载数据、创建固定切分，并把实际训练数据上传为 W&B Dataset Artifact：

```powershell
python prepare_data.py --upload-to-wandb
```

Artifact 名为 `kuake-qic-small`，包含三个 JSONL split 和数据分布元信息。准备数据的 run 还会记录一张 `dataset/all_samples` W&B Table，便于直接查看全部文本、标签和所属 split。

## 完整运行

```powershell
python run_experiments.py
```

脚本会依次执行：

1. 准备数据并上传 W&B Artifact。
2. 训练 `random_init`。
3. 训练 `frozen`。
4. 训练 `lora`。
5. 生成本地 CSV/Markdown 对比表和 W&B 对比图。

预计 RTX 4060 Laptop 总耗时约 15 到 40 分钟，实际时间受显卡功耗、CUDA 版本和首次下载模型的网络速度影响。三组最多都训练 8 个 epoch，使训练成本更便于横向比较；配置采用动态 padding、最大长度 64、FP16 和 early stopping。

## 单独运行

```powershell
python train.py --strategy random_init
python train.py --strategy frozen
python train.py --strategy lora
```

不连接 W&B 的代码冒烟测试可使用：

```powershell
python train.py --strategy frozen --wandb-mode disabled
```

## 对比指标

每个训练 run 都会记录：

- `test_accuracy`
- `test_macro_f1`
- `test_weighted_f1`
- 训练耗时
- GPU 峰值显存
- 可训练参数量和占比
- 混淆矩阵
- 每个类别的 precision、recall 和 F1

KUAKE-QIC 存在类别不平衡，因此模型选择以验证集 `macro_f1` 为准，Accuracy 作为辅助指标。预期现象是随机初始化明显最弱，冻结微调训练最快且占用最小，LoRA 在效果和训练成本之间取得较好的平衡。

## 已完成实验

2026 年 6 月 13 日已在 RTX 4060 Laptop 上完成三组训练。实际结果与适用场景分析见 [REPORT.md](REPORT.md)，在线曲线与数据表见 [W&B 项目](https://wandb.ai/chocal2240-lanzhou-university/bert-medical-intent-comparison)。

## 常用调整

显存不足时，将 `config.yaml` 中的 `train_batch_size` 改为 `8`。想使用官方完整训练集时，可以把天池下载的 JSON 转为当前 `id/text/label/label_name` 格式，或扩展 `prepare_data.py` 的数据源；训练和评估代码无需改动。
