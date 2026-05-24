# Transformer Practice

*项目借鉴B站UP主“炮哥带你学”的课程分享源码*

这是一个用于学习 Transformer 机器翻译模型实现的 PyTorch 练习项目。项目从零实现了经典 Encoder-Decoder Transformer 的核心模块，包括词向量、位置编码、多头注意力、前馈网络、残差连接、LayerNorm、Beam Search 解码、训练循环和 BLEU 评估。

当前版本在保留原有项目结构和学习价值的基础上，对源码做了现代化整理，使代码更易读、更容易调试，也更适合较新的 PyTorch 环境。

## 学习日志迁移说明

当前目录是放入 `DL-Learning-Log` 的源码版，保留模型、训练、验证、翻译、数据处理和分词器训练代码；未包含本地训练数据、分词器产物、模型权重、训练输出和 wandb 日志。

如需重新训练或翻译，请先在本目录补齐 `data/json/` 数据，运行 `python data/get_corpus.py` 生成语料，再运行 `python tokenizer/tokenize.py` 生成 SentencePiece 分词器。

## 项目结构

```text
transformer_practice/
├── beam_decoder.py              # Beam Search 解码
├── config.py                    # 模型、训练、路径和设备配置
├── main.py                      # 训练与验证入口
├── translate.py                 # 单句翻译入口
├── requirements.txt             # Python 依赖
├── data/
│   ├── json/                    # train/dev/test JSON 数据，本源码版未包含
│   ├── get_corpus.py            # 从 JSON 生成平行语料
│   ├── analyze_corpus.py        # 语料统计脚本
│   └── json_unicode_preview.py  # JSON 中文预览脚本
├── model/
│   ├── tf_model.py              # Transformer 模型主体
│   └── train_utils.py           # 损失计算与 Noam 学习率调度
├── tokenizer/
│   ├── tokenize.py              # SentencePiece 分词器训练脚本
│   ├── eng.model / eng.vocab    # 英文分词器，本源码版未包含
│   └── chn.model / chn.vocab    # 中文分词器，本源码版未包含
├── tools/
│   ├── data_loader.py           # Dataset、Batch 和 mask 工具
│   ├── tokenizer_utils.py       # 分词器加载工具
│   └── create_exp_folder.py     # 实验目录创建工具
├── run/                         # 训练输出目录，本源码版未包含
└── weights/                     # 权重文件目录，本源码版未包含
```

## 环境安装

建议使用 Python 3.10 或更新版本。

```bash
pip install -r requirements.txt
```

PyTorch 建议根据自己的 CUDA 版本从官网安装对应构建：

```text
https://pytorch.org/get-started/locally/
```

如果没有可用 GPU，代码会自动回退到 CPU。

## 数据格式

训练、验证和测试数据位于 `data/json/`，格式为：

```json
[
  ["English sentence.", "中文句子。"],
  ["Another English sentence.", "另一句中文。"]
]
```

如需重新生成 SentencePiece 训练语料：

```bash
python data/get_corpus.py
```

如需重新训练英文和中文分词器：

```bash
python tokenizer/tokenize.py
```

## 训练模型

在项目根目录运行：

```bash
python main.py
```

训练过程会：

- 加载 `data/json/train.json` 和 `data/json/dev.json`
- 构建 Transformer 模型
- 使用 Noam 学习率调度训练
- 在验证集上计算 BLEU
- 将最优权重和最后一轮权重保存到 `run/train/exp*/weights/`

训练参数可在 `config.py` 中调整，例如：

- `d_model`
- `n_heads`
- `n_layers`
- `d_ff`
- `dropout`
- `batch_size`
- `epoch_num`
- `beam_size`
- `max_len`

`batch_size` 和 `epoch_num` 也可以用环境变量覆盖，例如在 8GB 显存的 RTX 4060 Laptop 上可使用：

```powershell
$env:BATCH_SIZE="8"
$env:EPOCH_NUM="3"
python main.py
```

如需先跑一小段实验验证流程，可设置 `TRAIN_SUBSET_SIZE` 和 `DEV_SUBSET_SIZE`。

### 使用 wandb 记录实验

本项目已支持 Weights & Biases。默认不会启用，避免普通本地训练要求登录；需要上传实验时用环境变量打开：

```bash
USE_WANDB=1 WANDB_PROJECT=transformer-practice python main.py
```

在 PowerShell 中可写为：

```powershell
$env:USE_WANDB="1"
$env:WANDB_PROJECT="transformer-practice"
python main.py
```

如果没有提前登录 wandb，可临时设置 `WANDB_API_KEY`。训练会记录 batch 级训练 loss/learning rate，以及每轮 train loss、val loss、BLEU、best BLEU 和 CUDA 显存峰值；wandb 会自动生成对应曲线。默认不上传模型权重，如需上传 checkpoint artifact，可设置 `WANDB_LOG_MODEL=1`。

## 单句翻译

确认 `config.py` 中的 `test_model_path` 指向已有权重后，运行：

```bash
python translate.py
```

程序会进入交互模式，输入英文句子后返回中文翻译。

## 本版本整理重点

- 使用 `pathlib.Path` 管理项目路径，减少相对路径带来的问题。
- 移除旧版 PyTorch 中常见的 `Variable`、`.data` 等过时写法。
- 用标准 autograd 流程重写损失计算，避免手动 scatter/gather 多 GPU 张量。
- 保留 `MultiGPULossCompute` 类名，方便旧代码入口继续工作。
- 改进设备逻辑，支持 CUDA 自动检测和 CPU 回退。
- 给 `torch.load` 增加 `map_location`，提高跨设备加载权重的兼容性。
- 分词器加载增加缓存，避免重复读取 `.model` 文件。
- 整理训练、验证、测试和翻译入口，使代码职责更清楚。

## 学习建议

如果你的目标是熟悉 Transformer，建议按下面顺序阅读源码：

1. `tools/data_loader.py`：理解 token、padding mask 和 subsequent mask。
2. `model/tf_model.py`：理解 Transformer 的模块组合。
3. `model/train_utils.py`：理解 Noam 学习率和 token 级 loss。
4. `beam_decoder.py`：理解 Beam Search 如何逐步生成译文。
5. `main.py`：理解训练、验证和保存权重的完整流程。
