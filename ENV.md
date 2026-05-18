# ENV.md
## 运行环境说明
### 1. 基础环境
- Python 版本：3.9 / 3.10（推荐3.10）
- 虚拟环境工具：Anaconda / Miniconda
- 训练平台：Windows / Linux

### 2. CUDA 环境
- 显卡：RTX4060 Laptop
- 推荐CUDA：11.7 / 11.8
- 配套 cuDNN 对应版本

### 3. 核心依赖
```
torch
torchvision
numpy
matplotlib
opencv-python
scikit-learn
tensorboard
```

### 4. 环境创建命令
```bash
# 创建环境
conda create -n dl-learn python=3.10

# 激活环境
conda activate dl-learn

# 安装所有依赖
pip install -r requirements.txt
```

### 5. 环境查看命令
```bash
# 查看已建环境
conda env list

# 查看torch是否支持GPU
python -c "import torch;print(torch.cuda.is_available())"
```

### 6. 备注
- 数据集、模型权重、日志文件均不上传仓库
- 本地路径自行修改，代码尽量通用可直接运行