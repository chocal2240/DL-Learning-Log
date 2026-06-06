# PyTorch 迁移学习完整笔记

### 第7-8周学习记录：
- 

> 基于 [learnpytorch.io 第6章](https://www.learnpytorch.io/06_pytorch_transfer_learning/) 整理，B站视频链接 `https://www.bilibili.com/video/BV1LW411u7pA/` 解析失败，无法获取视频内容。

## 一、迁移学习基础
### 1.1 核心定义
迁移学习是将一个模型在**大规模通用数据集**上学习到的**特征模式（权重）**，迁移到**特定任务**上使用的技术。本质是"站在巨人的肩膀上"，复用已验证有效的模型架构和预训练权重。

### 1.2 为什么使用迁移学习
- **性能优势**：复用在百万级数据上训练的特征提取能力，远优于小数据集从头训练
- **效率优势**：训练速度快（仅需微调少量参数）、数据需求少
- **工业界标准**：研究表明，即使下游任务与预训练任务相关性较弱，迁移学习仍是最优选择

### 1.3 核心思想
- 冻结预训练模型的**特征提取层**（保留通用视觉/语言特征）
- 仅训练**分类头/输出层**（适配特定任务的类别数量）
- 数据量充足时可进一步**微调部分底层特征**

## 二、预训练模型资源库
| 资源平台 | 特点 | 适用场景 |
|---------|------|---------|
| PyTorch 域库 | 官方维护，与PyTorch无缝集成，支持最新API | 基础CV/NLP任务 |
| HuggingFace Hub | 跨模态模型最全，社区活跃，支持一键加载 | 复杂NLP/CV任务 |
| timm (PyTorch Image Models) | 包含几乎所有最新CV模型，功能丰富 | 计算机视觉前沿研究 |
| Paperswithcode | 关联最新论文与代码，提供基准对比 | 复现SOTA模型 |

## 三、PyTorch 迁移学习完整流程
### 3.1 环境与依赖准备
**要求**：PyTorch ≥1.12，torchvision ≥0.13（支持新的多权重API）
```python
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchinfo import summary  # 用于查看模型结构

# 设备无关代码
device = "cuda" if torch.cuda.is_available() else "cpu"

# 导入模块化脚本（来自课程仓库）
from going_modular.going_modular import data_setup, engine
```

### 3.2 数据获取与预处理
#### 3.2.1 数据集准备
以披萨/牛排/寿司三分类任务为例：
```python
import os
import zipfile
from pathlib import Path
import requests

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# 自动下载并解压数据集
if not image_path.is_dir():
    image_path.mkdir(parents=True, exist_ok=True)
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        f.write(request.content)
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        zip_ref.extractall(image_path)
    os.remove(data_path / "pizza_steak_sushi.zip")

train_dir = image_path / "train"
test_dir = image_path / "test"
```

#### 3.2.2 数据变换（两种方式）
**关键原则**：自定义数据必须与预训练模型的训练数据保持**相同的预处理方式**

##### 方式1：手动创建变换（兼容旧版本）
```python
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet数据集均值
                         std=[0.229, 0.224, 0.225])   # ImageNet数据集标准差
])
```

##### 方式2：自动获取变换（torchvision 0.13+推荐）
```python
# 获取预训练权重
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
# 自动获取对应权重的预处理变换
auto_transforms = weights.transforms()
```

**对比**：自动变换确保与预训练完全一致，手动变换可灵活添加数据增强

#### 3.2.3 创建DataLoader
```python
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=auto_transforms,
    batch_size=32
)
```

### 3.3 预训练模型加载与定制
#### 3.3.1 加载预训练模型
```python
# 旧API（已弃用）：model = torchvision.models.efficientnet_b0(pretrained=True).to(device)
# 新API（推荐）：
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)
```

#### 3.3.2 查看模型结构
```python
summary(model=model,
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
```

EfficientNet_B0 结构分为三部分：
1. `features`：卷积特征提取器（核心，包含大量MBConv块）
2. `avgpool`：自适应平均池化，将特征图转为特征向量
3. `classifier`：分类头，默认输出1000类（ImageNet类别）

#### 3.3.3 冻结特征提取器
```python
# 冻结features部分所有参数，训练时不更新
for param in model.features.parameters():
    param.requires_grad = False
```

#### 3.3.4 修改分类头适配任务
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 替换分类头：输入维度保持1280，输出维度改为3（三分类）
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),  # 保留原dropout防止过拟合
    nn.Linear(in_features=1280, out_features=len(class_names), bias=True)
).to(device)
```

**修改后模型参数变化**：
- 总参数：~528万 → ~401万
- 可训练参数：~528万 → **仅3843个**（仅分类头）
- 训练速度大幅提升

### 3.4 模型训练
```python
# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练5个epoch
from timeit import default_timer as timer
start_time = timer()

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

end_time = timer()
print(f"总训练时间: {end_time-start_time:.3f} 秒")
```

**典型训练结果**：
- Epoch 5: 训练准确率~78.5%，测试准确率~85.6%
- 训练时间：GPU上约10秒（远快于从头训练）
- 性能远超从头训练的TinyVGG（准确率~45%）

### 3.5 模型评估
#### 3.5.1 绘制损失曲线
```python
from helper_functions import plot_loss_curves
plot_loss_curves(results)
```

**理想损失曲线特征**：
- 训练损失和测试损失持续下降
- 两者差距较小（无明显过拟合）
- 准确率持续上升并趋于稳定

#### 3.5.2 混淆矩阵分析
```python
# 预测所有测试集样本
y_preds = []
model.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        y_logit = model(X)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
y_preds = torch.cat(y_preds)

# 绘制混淆矩阵
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=y_preds, target=torch.tensor(test_dataloader.dataset.targets))

plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                      class_names=class_names,
                      figsize=(10, 7))
```

### 3.6 单张图像预测与可视化
```python
from PIL import Image
import matplotlib.pyplot as plt

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: list[str],
                        transform=None,
                        device=device):
    img = Image.open(image_path)
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_img = transform(img).unsqueeze(0).to(device)
        pred_logit = model(transformed_img)
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = pred_prob.argmax(dim=1)
    
    plt.figure()
    plt.imshow(img)
    plt.title(f"预测: {class_names[pred_label]} | 置信度: {pred_prob.max():.3f}")
    plt.axis(False)

# 测试自定义图像
pred_and_plot_image(model, "data/04-pizza-dad.jpeg", class_names)
```

## 四、关键技巧与注意事项
1. **数据预处理一致性**：必须使用与预训练模型完全相同的归一化参数和图像尺寸
2. **学习率选择**：微调时使用较小的学习率（如1e-4 ~ 1e-3），避免破坏预训练特征
3. **冻结策略**：
   - 小数据集：仅训练分类头
   - 大数据集：解冻最后几层特征提取器一起微调
4. **模型选择**：
   - 移动端/低算力：EfficientNet_B0/B1、MobileNetV2
   - 高性能需求：EfficientNet_B4/B7、ResNet50、ViT
5. **过拟合处理**：保留预训练模型的Dropout层，必要时添加数据增强

## 五、拓展练习与进阶方向
1. 训练10个epoch，观察性能变化
2. 使用更大的数据集（如Food101的20%数据）
3. 尝试其他模型架构（EfficientNet_B2、ResNet50、ViT_B_16）
4. 实现模型微调：解冻最后2-3层特征提取器进行训练
5. 分析"最错误"的预测样本，找出模型误判原因
6. 收集自定义数据集（如猫狗分类），训练自己的迁移学习模型

需要我把这个笔记整理成**可直接运行的Jupyter Notebook代码**版本，包含所有导入和完整训练流程吗？