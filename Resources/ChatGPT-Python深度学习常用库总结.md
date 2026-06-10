# Python 深度学习常用库总结（增强版）

> 目标：整理深度学习、BERT、Hugging Face、LoRA/PEFT 实验中常见的 Python 标准库与第三方库。  
> 重点补充：`argparse`、`copy`、`__future__`、`collections.abc`、`inspect`、`typing`、`dataclasses`、`functools` 等在源码、教程、BERT 项目中经常出现但容易忽略的库。

---

## 目录

1. [总体分类](#1-总体分类)
2. [路径、文件与系统交互](#2-路径文件与系统交互)
3. [配置、数据格式与缓存](#3-配置数据格式与缓存)
4. [命令行参数与实验配置](#4-命令行参数与实验配置)
5. [复制、对象处理与容器工具](#5-复制对象处理与容器工具)
6. [类型提示与接口抽象](#6-类型提示与接口抽象)
7. [源码检查、函数签名与动态工具](#7-源码检查函数签名与动态工具)
8. [时间、日志与实验记录](#8-时间日志与实验记录)
9. [随机性与复现实验](#9-随机性与复现实验)
10. [数值计算与表格数据处理](#10-数值计算与表格数据处理)
11. [可视化与实验追踪](#11-可视化与实验追踪)
12. [PyTorch 相关库](#12-pytorch-相关库)
13. [BERT / Hugging Face 常用库](#13-bert--hugging-face-常用库)
14. [NLP 数据处理常用库](#14-nlp-数据处理常用库)
15. [图像任务常用库](#15-图像任务常用库)
16. [评估指标与机器学习工具](#16-评估指标与机器学习工具)
17. [深度学习项目典型 import 模板](#17-深度学习项目典型-import-模板)
18. [建议学习顺序](#18-建议学习顺序)

---

# 1. 总体分类

深度学习项目中，Python 库大致可以分为以下几类：

| 类别 | 常用库 | 主要用途 |
|---|---|---|
| 文件路径 | `os`, `pathlib`, `glob`, `shutil` | 管理路径、目录、数据文件、输出文件 |
| 配置与数据格式 | `json`, `yaml`, `csv`, `pickle` | 保存配置、读取数据、缓存对象 |
| 命令行参数 | `argparse` | 通过命令行控制实验参数 |
| 对象复制 | `copy` | 深拷贝/浅拷贝模型配置、字典、对象 |
| 容器工具 | `collections`, `collections.abc` | 计数器、默认字典、抽象容器类型判断 |
| 类型提示 | `typing`, `dataclasses` | 提升代码可读性，定义结构化配置 |
| 函数工具 | `functools`, `itertools` | 函数组合、缓存、迭代器工具 |
| 源码检查 | `inspect` | 查看函数签名、源码、参数 |
| 兼容性 | `__future__` | 使用未来版本 Python 特性 |
| 日志时间 | `logging`, `time`, `datetime` | 日志输出、训练耗时、实验时间戳 |
| 随机性控制 | `random`, `numpy.random`, `torch.manual_seed` | 固定随机种子 |
| 数值计算 | `numpy` | 数组、矩阵、数值运算 |
| 表格数据 | `pandas` | CSV、DataFrame、数据清洗 |
| 可视化 | `matplotlib`, `seaborn` | 曲线图、柱状图、混淆矩阵 |
| 深度学习 | `torch`, `torchvision` | 建模、训练、图像任务 |
| BERT / NLP | `transformers`, `datasets`, `tokenizers`, `peft`, `evaluate` | 预训练模型、数据集、LoRA、指标 |
| 指标评估 | `sklearn.metrics` | Accuracy、F1、分类报告、混淆矩阵 |
| 实验记录 | `wandb`, `tensorboard` | 记录超参数、loss、accuracy、曲线 |

---

# 2. 路径、文件与系统交互

---

## 2.1 `os`

`os` 用于和操作系统交互。

```python
import os

data_dir = "data"
train_path = os.path.join(data_dir, "train.csv")

os.makedirs("outputs/checkpoints", exist_ok=True)

cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
```

常见用途：

```text
创建文件夹
读取环境变量
拼接路径
判断文件是否存在
```

---

## 2.2 `pathlib`

现代 Python 更推荐使用 `pathlib` 管理路径。

```python
from pathlib import Path

root = Path(".")
data_dir = root / "data"
train_path = data_dir / "train.csv"

output_dir = Path("outputs") / "bert_cls"
output_dir.mkdir(parents=True, exist_ok=True)
```

常见操作：

```python
path = Path("data/train.csv")

print(path.exists())   # 是否存在
print(path.name)       # train.csv
print(path.stem)       # train
print(path.suffix)     # .csv
print(path.parent)     # data
```

遍历文件：

```python
for file in Path("data").glob("*.csv"):
    print(file)
```

递归遍历：

```python
for file in Path("data").rglob("*.jsonl"):
    print(file)
```

在深度学习项目中推荐写法：

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
```

---

## 2.3 `glob`

用于按通配符查找文件。

```python
import glob

image_paths = glob.glob("data/images/*.jpg")
print(image_paths[:5])
```

现代写法通常用：

```python
from pathlib import Path

image_paths = list(Path("data/images").glob("*.jpg"))
```

---

## 2.4 `shutil`

用于复制、移动、删除文件和文件夹。

```python
import shutil

shutil.copy("config.json", "outputs/config.json")
shutil.copytree("data", "backup/data", dirs_exist_ok=True)
shutil.rmtree("outputs/temp")
```

常见用途：

```text
备份配置
复制最佳模型
清理临时目录
保存实验快照
```

---

# 3. 配置、数据格式与缓存

---

## 3.1 `json`

`json` 常用于保存实验配置、标签映射、训练结果。

保存配置：

```python
import json

config = {
    "model_name": "bert-base-chinese",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 3,
    "max_length": 128,
}

with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
```

读取配置：

```python
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print(config["learning_rate"])
```

保存标签映射：

```python
label2id = {
    "negative": 0,
    "positive": 1,
}

with open("label2id.json", "w", encoding="utf-8") as f:
    json.dump(label2id, f, ensure_ascii=False, indent=2)
```

注意：

```text
JSON 的 key 会被转成字符串。
例如 {0: "negative"} 保存后再读回来，key 会变成 "0"。
```

---

## 3.2 `.jsonl`

很多 NLP 数据集使用 `.jsonl`，即一行一个 JSON。

示例：

```json
{"text": "这个模型很好", "label": 1}
{"text": "效果很差", "label": 0}
```

读取：

```python
import json

data = []

with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(data[0])
```

写入：

```python
with open("predictions.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

`.jsonl` 适合大规模文本数据，因为可以逐行读取，不必一次加载全部文件。

---

## 3.3 `csv`

标准库 `csv` 用于简单 CSV 读取。

```python
import csv

with open("data/train.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        text = row["text"]
        label = int(row["label"])
```

复杂表格处理更推荐 `pandas`。

---

## 3.4 `pickle`

`pickle` 可以保存 Python 对象。

```python
import pickle

obj = {
    "texts": ["你好", "模型很好"],
    "labels": [0, 1],
}

with open("cache.pkl", "wb") as f:
    pickle.dump(obj, f)

with open("cache.pkl", "rb") as f:
    loaded_obj = pickle.load(f)
```

注意：

```text
不要加载不可信来源的 pickle 文件，因为有安全风险。
```

---

## 3.5 `yaml`

YAML 常用于实验配置，第三方库是 `PyYAML`。

安装：

```bash
pip install pyyaml
```

配置文件 `config.yaml`：

```yaml
model_name: bert-base-chinese
batch_size: 16
learning_rate: 0.00002
num_epochs: 3
max_length: 128
```

读取：

```python
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(config["model_name"])
```

YAML 比 JSON 更适合人工编辑复杂配置。

---

# 4. 命令行参数与实验配置

---

## 4.1 `argparse`

`argparse` 是标准库，常用于通过命令行传入实验参数。

基本写法：

```python
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="bert-base-chinese")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--mode", type=str, choices=["full", "frozen", "lora"], default="full")

args = parser.parse_args()

print(args.model_name)
print(args.mode)
```

运行：

```bash
python train.py --mode lora --learning_rate 2e-4 --batch_size 16
```

在消融实验中非常常见：

```bash
python train.py --mode full
python train.py --mode frozen
python train.py --mode lora
```

推荐写法：

```python
def parse_args():
    parser = argparse.ArgumentParser(description="BERT transfer learning experiment")

    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--mode", type=str, choices=["full", "frozen", "lora"], default="full")

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    print(args)


if __name__ == "__main__":
    main()
```

---

## 4.2 `dataclasses`

`dataclasses` 用于定义结构化配置对象。

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "bert-base-chinese"
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 128
    mode: str = "full"

config = TrainConfig()
print(config.learning_rate)
```

优点：

```text
比普通 dict 更清晰
可以配合类型提示
适合管理训练配置
```

结合 `asdict` 保存 JSON：

```python
from dataclasses import asdict
import json

config_dict = asdict(config)

with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config_dict, f, ensure_ascii=False, indent=2)
```

---

## 4.3 Hugging Face 中的 `HfArgumentParser`

Hugging Face 很多官方示例会用 `HfArgumentParser`，它可以把 dataclass 自动转成命令行参数。

```python
from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass
class ModelArguments:
    model_name_or_path: str = "bert-base-chinese"

@dataclass
class DataArguments:
    train_file: str = "data/train.csv"
    validation_file: str = "data/valid.csv"
    max_length: int = 128

parser = HfArgumentParser((ModelArguments, DataArguments))
model_args, data_args = parser.parse_args_into_dataclasses()

print(model_args.model_name_or_path)
print(data_args.train_file)
```

这种写法在 Transformers 官方训练脚本中很常见。

---

# 5. 复制、对象处理与容器工具

---

## 5.1 `copy`

`copy` 提供浅拷贝和深拷贝。

```python
import copy

a = {"lr": 2e-5, "layers": [1, 2, 3]}

b = copy.copy(a)      # 浅拷贝
c = copy.deepcopy(a)  # 深拷贝
```

区别：

```python
a = {"layers": [1, 2, 3]}

b = copy.copy(a)
c = copy.deepcopy(a)

a["layers"].append(4)

print(b["layers"])  # [1, 2, 3, 4]
print(c["layers"])  # [1, 2, 3]
```

在深度学习中的用途：

```text
复制配置字典
复制模型配置对象
保留一份原始参数
避免修改原对象导致实验污染
```

例如做消融实验：

```python
import copy

base_config = {
    "model_name": "bert-base-chinese",
    "batch_size": 16,
    "learning_rate": 2e-5,
}

lora_config = copy.deepcopy(base_config)
lora_config["mode"] = "lora"
lora_config["learning_rate"] = 2e-4
```

注意：复制 PyTorch 模型时也可以用 `copy.deepcopy(model)`，但大模型会占用大量显存/内存，通常不建议随便复制完整模型。

---

## 5.2 `collections`

`collections` 提供一些常用容器。

### `Counter`

统计标签分布：

```python
from collections import Counter

labels = [0, 1, 1, 0, 1, 2]
counter = Counter(labels)

print(counter)
print(counter.most_common())
```

在数据分析中常用：

```python
from collections import Counter

label_counts = Counter(dataset["train"]["label"])
print(label_counts)
```

### `defaultdict`

自动创建默认值：

```python
from collections import defaultdict

class_to_texts = defaultdict(list)

for text, label in zip(texts, labels):
    class_to_texts[label].append(text)
```

### `OrderedDict`

有序字典。现代 Python 的普通 dict 已经保持插入顺序，但在模型权重、旧代码中仍可能看到。

```python
from collections import OrderedDict

state = OrderedDict()
state["layer1.weight"] = None
state["layer1.bias"] = None
```

PyTorch 的 `state_dict` 在很多情况下看起来类似有序映射。

---

## 5.3 `collections.abc`

`collections.abc` 提供抽象基类，用于判断对象是否像某种容器。

常见类型：

```python
from collections.abc import Mapping, Sequence, Iterable, Callable
```

### `Mapping`

判断是不是类似字典的对象：

```python
from collections.abc import Mapping

def process_config(config):
    if isinstance(config, Mapping):
        print("这是一个字典类配置")
```

### `Sequence`

判断是不是序列：

```python
from collections.abc import Sequence

def is_sequence(x):
    return isinstance(x, Sequence) and not isinstance(x, str)
```

注意字符串也是 Sequence，所以常常要排除 `str`。

### `Iterable`

判断是否可迭代：

```python
from collections.abc import Iterable

def flatten(items):
    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item
```

### 在深度学习代码中的用途

`collections.abc` 常见于库源码，例如：

```text
判断 config 是否是 Mapping
判断输入是否是 Sequence
判断 callback 是否 Callable
判断 dataset 是否 Iterable
```

初学不需要频繁手写，但读源码时经常会遇到。

---

# 6. 类型提示与接口抽象

---

## 6.1 `typing`

`typing` 用于类型提示，提升代码可读性。

```python
from typing import List, Dict, Tuple, Optional, Union

def tokenize_texts(texts: List[str], max_length: int = 128) -> Dict[str, List[int]]:
    pass
```

常见类型：

| 类型 | 含义 |
|---|---|
| `List[str]` | 字符串列表 |
| `Dict[str, int]` | key 为字符串、value 为整数的字典 |
| `Tuple[int, int]` | 二元组 |
| `Optional[str]` | 可以是字符串，也可以是 None |
| `Union[str, Path]` | 可以是 str 或 Path |
| `Any` | 任意类型 |
| `Callable` | 可调用对象 |

示例：

```python
from typing import Optional, Union
from pathlib import Path

def load_config(path: Union[str, Path]) -> dict:
    path = Path(path)
    ...
```

在 BERT 项目中，类型提示常用于：

```text
配置函数
数据处理函数
模型构建函数
训练入口函数
```

---

## 6.2 新版本类型写法

Python 3.9+ 可以写：

```python
def func(texts: list[str]) -> dict[str, int]:
    ...
```

而旧写法是：

```python
from typing import List, Dict

def func(texts: List[str]) -> Dict[str, int]:
    ...
```

---

## 6.3 `__future__`

`__future__` 不是普通库，而是让当前文件启用未来版本 Python 的某些特性。

最常见：

```python
from __future__ import annotations
```

作用：推迟类型注解的求值。

例如：

```python
from __future__ import annotations

class Node:
    def __init__(self, next_node: Node | None = None):
        self.next_node = next_node
```

如果没有未来注解，在一些 Python 版本中，`Node` 在类定义过程中还没完全创建，可能会出问题。

在深度学习项目中，`from __future__ import annotations` 常见于：

```text
Hugging Face 源码
较新的 Python 项目
类型提示较多的项目
```

你看到它时只需要知道：

```text
它通常是为了让类型注解更灵活、更兼容。
不影响核心训练逻辑。
```

---

# 7. 源码检查、函数签名与动态工具

---

## 7.1 `inspect`

`inspect` 可以查看函数签名、源码、参数信息。

### 查看函数签名

```python
import inspect
from transformers import TrainingArguments

print(inspect.signature(TrainingArguments))
```

适合搞清楚某个类/函数有哪些参数。

---

### 查看源码

```python
import inspect
from transformers import Trainer

print(inspect.getsource(Trainer.train))
```

注意：有些函数是 C/C++ 实现或动态生成的，可能无法获取源码。

---

### 判断对象类型

```python
import inspect

def foo(x):
    return x + 1

print(inspect.isfunction(foo))
print(inspect.isclass(str))
```

---

### 在深度学习中的用途

`inspect` 很适合用来：

```text
查看 Hugging Face 某个函数的参数
确认 Trainer 内部调用逻辑
调试第三方库
学习源码
写通用工具函数
```

例如：

```python
import inspect
from transformers import Trainer

sig = inspect.signature(Trainer.__init__)
print(sig)
```

可以看到 `Trainer` 初始化需要哪些参数。

---

## 7.2 `functools`

`functools` 提供函数工具。

### `partial`

固定函数的一部分参数。

```python
from functools import partial

def tokenize_fn(batch, tokenizer, max_length):
    return tokenizer(batch["text"], truncation=True, max_length=max_length)

tokenize_with_config = partial(
    tokenize_fn,
    tokenizer=tokenizer,
    max_length=128
)

dataset = dataset.map(tokenize_with_config, batched=True)
```

在 Hugging Face 数据处理中很常见。

---

### `lru_cache`

缓存函数结果。

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def load_label_name(label_id):
    print("读取标签名")
    return id2label[label_id]
```

适合重复计算成本较高的函数。

---

## 7.3 `itertools`

用于迭代器组合。

```python
import itertools

for lr, batch_size in itertools.product([2e-5, 5e-5], [16, 32]):
    print(lr, batch_size)
```

适合做网格搜索：

```python
for lr, r in itertools.product([1e-4, 2e-4], [4, 8, 16]):
    print(f"LoRA lr={lr}, r={r}")
```

---

# 8. 时间、日志与实验记录

---

## 8.1 `time`

记录训练耗时：

```python
import time

start = time.time()

# train()

end = time.time()
print(f"训练耗时: {end - start:.2f} 秒")
```

---

## 8.2 `datetime`

生成实验时间戳：

```python
from datetime import datetime

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
print(run_name)
```

常用于创建输出目录：

```python
output_dir = Path("outputs") / f"bert_lora_{run_name}"
```

---

## 8.3 `logging`

比 `print()` 更适合正式项目。

```python
import logging

logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("开始训练")
logging.info("epoch=1, loss=0.523")
logging.warning("验证集指标没有提升")
```

同时输出到终端和文件：

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("train.log", encoding="utf-8")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info("训练开始")
```

---

# 9. 随机性与复现实验

---

## 9.1 `random`

Python 标准随机库。

```python
import random

random.seed(42)

items = [1, 2, 3, 4]
random.shuffle(items)
```

---

## 9.2 固定随机种子

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

注意：

```text
完全可复现有时会降低训练速度。
某些 GPU 操作仍可能有非确定性。
```

Hugging Face 也有：

```python
from transformers import set_seed

set_seed(42)
```

---

# 10. 数值计算与表格数据处理

---

## 10.1 `numpy`

常用于模型输出处理、指标计算、数组保存。

```python
import numpy as np

logits = np.array([[1.2, 0.3], [0.1, 2.0]])
preds = np.argmax(logits, axis=-1)

print(preds)
```

常用函数：

| 函数 | 作用 |
|---|---|
| `np.array` | 创建数组 |
| `np.argmax` | 取最大值索引 |
| `np.mean` | 平均值 |
| `np.std` | 标准差 |
| `np.concatenate` | 拼接数组 |
| `np.save` / `np.load` | 保存/读取 `.npy` |

---

## 10.2 `pandas`

用于表格数据处理、结果分析。

```python
import pandas as pd

df = pd.read_csv("data/train.csv")
print(df.head())
print(df["label"].value_counts())
```

保存实验结果：

```python
results = [
    {"method": "full", "accuracy": 0.91, "macro_f1": 0.90},
    {"method": "lora", "accuracy": 0.89, "macro_f1": 0.88},
]

df = pd.DataFrame(results)
df.to_csv("outputs/results.csv", index=False)
```

---

# 11. 可视化与实验追踪

---

## 11.1 `matplotlib`

画 loss / accuracy 曲线：

```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3]
train_loss = [0.8, 0.5, 0.3]
val_loss = [0.9, 0.6, 0.4]

plt.figure()
plt.plot(epochs, train_loss, label="train_loss")
plt.plot(epochs, val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs/loss_curve.png", dpi=300)
plt.show()
```

---

## 11.2 `seaborn`

画混淆矩阵：

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()
```

---

## 11.3 `wandb`

记录实验：

```python
import wandb

wandb.init(
    project="bert-transfer-learning",
    name="bert_lora_r8",
    config={
        "model": "bert-base-chinese",
        "method": "lora",
        "r": 8,
        "learning_rate": 2e-4,
    }
)

wandb.log({
    "train_loss": 0.42,
    "val_accuracy": 0.88,
    "val_macro_f1": 0.86,
})

wandb.finish()
```

Hugging Face Trainer 中：

```python
training_args = TrainingArguments(
    output_dir="outputs/bert_lora",
    report_to="wandb",
    run_name="bert_lora_r8",
)
```

---

## 11.4 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/exp1")

writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Accuracy/val", val_acc, epoch)

writer.close()
```

启动：

```bash
tensorboard --logdir runs
```

---

# 12. PyTorch 相关库

---

## 12.1 `torch`

PyTorch 核心库。

```python
import torch

x = torch.randn(32, 10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 12.2 `torch.nn`

定义模型层和损失函数：

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

loss_fn = nn.CrossEntropyLoss()
```

---

## 12.3 `torch.optim`

优化器：

```python
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)
```

---

## 12.4 `torch.utils.data`

Dataset / DataLoader：

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

---

# 13. BERT / Hugging Face 常用库

---

## 13.1 `transformers`

最常见导入：

```python
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    set_seed,
)
```

常用对象：

| 对象 | 用途 |
|---|---|
| `AutoTokenizer` | 加载 tokenizer |
| `AutoModelForSequenceClassification` | 文本分类 |
| `AutoModelForMaskedLM` | MLM 预训练 |
| `BertConfig` | 自定义 BERT 配置 |
| `BertForMaskedLM` | 创建 Mini-BERT MLM 模型 |
| `TrainingArguments` | 训练参数 |
| `Trainer` | 训练框架 |
| `DataCollatorWithPadding` | 文本分类动态 padding |
| `DataCollatorForLanguageModeling` | MLM 动态 mask |
| `set_seed` | 固定随机种子 |

---

## 13.2 `datasets`

加载和处理数据：

```python
from datasets import load_dataset, Dataset, DatasetDict

dataset = load_dataset("csv", data_files={
    "train": "data/train.csv",
    "validation": "data/valid.csv",
})
```

从 pandas 转换：

```python
from datasets import Dataset

hf_dataset = Dataset.from_pandas(df)
```

常用方法：

```python
dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.shuffle(seed=42)
dataset = dataset.remove_columns(["text"])
dataset = dataset.rename_column("label", "labels")
```

---

## 13.3 `evaluate`

计算指标：

```python
import evaluate

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
```

在 Trainer 中：

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    macro_f1 = f1.compute(predictions=preds, references=labels, average="macro")["f1"]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }
```

---

## 13.4 `peft`

LoRA 参数高效微调：

```python
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
```

典型配置：

```python
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

---

## 13.5 `accelerate`

Hugging Face 的训练加速与分布式工具。

常见于：

```text
多 GPU 训练
混合精度训练
大模型训练
Trainer 底层依赖
```

手写训练循环时可能会看到：

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
```

初学阶段如果用 Trainer，不需要直接深入 `accelerate`。

---

## 13.6 `safetensors`

安全、快速的模型权重格式。

```text
adapter_model.safetensors
model.safetensors
```

Hugging Face 现在很多模型默认使用 `.safetensors`。

常见于：

```text
保存 LoRA adapter
保存预训练模型权重
加载 Hugging Face checkpoint
```

一般不需要手动操作，`transformers` 和 `peft` 会处理。

---

## 13.7 `huggingface_hub`

用于登录、下载、上传模型到 Hugging Face Hub。

```python
from huggingface_hub import login

login()
```

下载文件：

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="bert-base-chinese",
    filename="config.json"
)
```

初学阶段主要通过 `from_pretrained()` 间接使用。

---

# 14. NLP 数据处理常用库

---

## 14.1 `re`

正则表达式，用于文本清洗。

```python
import re

text = "Hello!!!  这是一个测试123。"
text = re.sub(r"\s+", " ", text)
text = re.sub(r"[0-9]+", "<NUM>", text)
```

常见用途：

```text
去除多余空格
替换数字
清理特殊符号
抽取字段
```

---

## 14.2 `unicodedata`

处理 Unicode 文本。

```python
import unicodedata

text = "ＡＢＣ１２３"
normalized = unicodedata.normalize("NFKC", text)

print(normalized)  # ABC123
```

中文 NLP 数据清洗时有用：

```text
全角转半角
统一 Unicode 表示
处理异常字符
```

---

## 14.3 `jieba`

中文分词库。

```bash
pip install jieba
```

```python
import jieba

tokens = jieba.lcut("我正在学习深度学习")
print(tokens)
```

注意：

```text
BERT 使用自己的 WordPiece tokenizer，通常不需要先用 jieba 分词。
但传统机器学习文本分类、词频统计、关键词分析时会用到 jieba。
```

---

## 14.4 `sentencepiece`

很多模型的 tokenizer 使用 SentencePiece。

```bash
pip install sentencepiece
```

常见于：

```text
T5
ALBERT
LLaMA 类模型
多语言模型
```

BERT-base-chinese 通常不需要你直接操作它。

---

# 15. 图像任务常用库

---

## 15.1 `torchvision`

图像数据集、图像增强、模型。

```python
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
```

CIFAR-10：

```python
train_dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
```

图像增强：

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
```

---

## 15.2 `PIL`

读取图像：

```python
from PIL import Image

img = Image.open("image.jpg").convert("RGB")
```

---

## 15.3 `cv2`

OpenCV：

```python
import cv2

img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

普通分类任务里，`PIL + torchvision` 往往已经够用。

---

# 16. 评估指标与机器学习工具

---

## 16.1 `sklearn.metrics`

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")

print(classification_report(y_true, y_pred))
```

BERT 文本分类常用：

```text
accuracy
macro_f1
precision
recall
classification_report
confusion_matrix
```

---

## 16.2 `sklearn.model_selection`

划分数据集：

```python
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)
```

`stratify` 用于保持类别比例。

---

# 17. 深度学习项目典型 import 模板

---

## 17.1 BERT 文本分类 / LoRA 实验模板

```python
from __future__ import annotations

import os
import re
import json
import time
import copy
import random
import logging
import argparse
import inspect
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Optional, Union, Any

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from datasets import load_dataset, Dataset as HFDataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

from tqdm import tqdm
```

---

## 17.2 CIFAR-10 / ResNet 消融实验模板

```python
import os
import time
import copy
import random
import argparse
import logging
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tqdm import tqdm
```

---

# 18. 建议学习顺序

如果库太多，可以按优先级学。

---

## 第一层：所有深度学习项目都常用

```text
pathlib
json / jsonl
argparse
logging
time / datetime
random
numpy
pandas
matplotlib
tqdm
torch
```

---

## 第二层：BERT / Hugging Face 必须掌握

```text
transformers
datasets
evaluate
peft
sklearn.metrics
```

重点对象：

```text
AutoTokenizer
AutoModelForSequenceClassification
AutoModelForMaskedLM
TrainingArguments
Trainer
DataCollatorWithPadding
DataCollatorForLanguageModeling
LoraConfig
get_peft_model
TaskType
PeftModel
```

---

## 第三层：读源码、看官方脚本时经常遇到

```text
copy
collections
collections.abc
typing
dataclasses
inspect
functools
itertools
__future__
```

这些不一定每天手写，但读 Hugging Face、PyTorch、别人项目源码时很常见。

---

## 第四层：文本清洗和高级工具

```text
re
unicodedata
yaml
huggingface_hub
accelerate
safetensors
sentencepiece
jieba
```

---

# 19. 一句话总结

```text
深度学习项目中的 Python 库可以分成三层：
第一层是实验基础设施，比如 pathlib、json、argparse、logging；
第二层是数值计算和训练框架，比如 numpy、pandas、torch；
第三层是任务生态库，比如 transformers、datasets、peft、evaluate。
```

对于当前 BERT / LoRA 学习阶段，最应该优先补的是：

```text
argparse：管理实验参数
pathlib：管理路径
json/jsonl：管理配置和文本数据
logging：记录训练日志
copy：复制配置，避免实验污染
collections / collections.abc：理解数据容器和源码
inspect：查看第三方库函数参数和源码
transformers / datasets / peft / evaluate：完成 BERT 与 LoRA 实验
```
