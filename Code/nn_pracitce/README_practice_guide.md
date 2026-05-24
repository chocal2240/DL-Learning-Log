# 神经网络实践练习路线：从线性模型到 Transformer

这套练习按“题目 → 目标 → 要求 → 验收标准 → 代码入口”设计，适合你用来补齐从鱼书/D2L 到 Transformer 之间的代码实践。

推荐顺序：

```text
线性回归 → softmax 分类 → nn.Linear 对比 → MLP → 梯度问题 → LeNet → ResNet-18 → 特征图可视化 → LSTM/GRU → Attention → Transformer
```

---

## 练习 1：从零实现线性回归

**目标**：理解最小训练闭环：前向传播、loss、反向传播、参数更新。

**要求**：不用 `nn.Linear`，只用 Tensor 和自动求导实现 `y = Xw + b`。

**验收标准**：

- 能解释 `X @ w + b` 的 shape。
- 能解释 `loss.backward()` 后 `w.grad` 是什么。
- 知道为什么每轮都要清空梯度。
- 知道学习率过大会导致 loss 震荡或 nan。

代码：`01_linear_regression_from_scratch.py`

---

## 练习 2：从零实现 softmax 分类

**目标**：掌握最基础的多分类模型。

**要求**：在 MNIST 上实现 softmax 分类，限制：不使用 `nn.Linear`，自己定义 `W/b`，自己实现 softmax 和交叉熵。

**验收标准**：

- 能解释 logits、softmax、cross entropy。
- 能解释 `argmax(dim=1)`。
- 能说清为什么分类不用 MSE 作为首选 loss。

代码：`02_softmax_from_scratch.py`

---

## 练习 3：对比从零实现和 `nn.Linear`

**目标**：理解 `nn.Linear` 和手写 `X @ W + b` 本质一致。

**要求**：用 `nn.Linear` 重写 softmax 分类模型，比较 loss、准确率和参数 shape。

**验收标准**：

- 能解释 `nn.Linear(in_features, out_features)`。
- 能解释 `model.parameters()`。
- 能对应 `nn.Linear.weight` 和手写参数 `W`。

代码：`03_compare_linear.py`

---

## 练习 4：2 层 MLP 分类 MNIST

**目标**：掌握全连接神经网络结构：`Flatten → Linear → ReLU → Linear`。

**要求**：实现 `784 → 256 → 10` 的 2 层 MLP，记录 loss、accuracy 和梯度范数。

**验收标准**：

- 能解释为什么需要非线性激活函数。
- 能解释 ReLU 的作用。
- 能观察 hidden size 改变对效果的影响。

代码：`04_mlp_mnist.py`

---

## 练习 5：调试梯度消失 / 梯度爆炸

**目标**：亲眼观察训练不稳定。

**要求**：做三组实验：正常 ReLU、超大学习率导致梯度爆炸、多层 sigmoid 导致梯度消失。

**验收标准**：

- 记录每轮 loss。
- 记录每层梯度范数。
- 能判断问题属于梯度消失还是爆炸。
- 能提出缓解方案：降低学习率、ReLU、BatchNorm、合适初始化、梯度裁剪。

代码：`05_gradient_debugging.py`

---

## 练习 6：复现 LeNet

**目标**：理解经典 CNN：`Conv → Pool → Conv → Pool → FC`。

**要求**：在 MNIST 上实现 LeNet 风格网络，并打印每层输出 shape。

**验收标准**：

- 能解释 `[batch, 1, 28, 28]`。
- 能解释 `in_channels/out_channels`。
- 能计算池化后的特征图大小。

代码：`06_lenet_mnist.py`

---

## 练习 7：复现简化版 ResNet-18

**目标**：理解残差连接 `y = F(x) + x`。

**要求**：实现 `BasicBlock`，在 CIFAR-10 上训练简化版 ResNet-18。

**验收标准**：

- 能解释残差连接为什么缓解深层网络训练困难。
- 能解释 shortcut 在通道变化时为什么要用 1×1 卷积。
- 能解释 `stride=2` 如何下采样。

代码：`07_resnet18_cifar.py`

---

## 练习 8：CNN 特征图可视化

**目标**：观察 CNN 中间层输出，而不是只看 accuracy。

**要求**：用 forward hook 提取某个卷积层 feature map，并画出若干 channel。

**验收标准**：

- 能解释 feature map shape。
- 能说明不同 channel 可能学到不同局部特征。

代码：`08_feature_map_visualization.py`

---

## 练习 9：LSTM 实现 IMDB 情感分析

**目标**：掌握序列模型：`Embedding → LSTM → Linear`。

**要求**：在 IMDB 数据集上训练 LSTM 情感分类模型。建议在 Google Colab 运行。

**验收标准**：

- 能解释 token id 到 embedding 的过程。
- 能解释 hidden state / cell state。
- 能解释为什么可以取最后一个 hidden state 做分类。

代码：`09_lstm_imdb_colab.py`

---

## 练习 10：对比 GRU 和 LSTM

**目标**：理解 GRU 和 LSTM 的区别。

**要求**：保持 embedding size、hidden size、batch size 相同，只替换 LSTM 为 GRU，比较参数量、速度、准确率。

**验收标准**：

- 能说明 GRU 比 LSTM 少了哪些结构。
- 能说明为什么 GRU 参数更少。
- 能比较二者在小数据上的表现。

代码：`10_gru_vs_lstm.py`

---

## 练习 11：从零实现 Scaled Dot-Product Attention

**目标**：掌握 Transformer 核心公式：

```text
Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
```

**要求**：自己实现 QK 点积、scale、mask、softmax 和加权求和。

**验收标准**：写清楚每一步 shape：

```text
Q/K/V: [B, T, D]
scores: [B, T, T]
weights: [B, T, T]
output: [B, T, D]
```

代码：`11_attention_from_scratch.py`

---

## 练习 12：从零实现 Multi-Head Attention

**目标**：理解多头注意力的拆分和合并。

**要求**：实现 `W_q/W_k/W_v`、split heads、attention、concat heads、output projection。

**验收标准**：

- 能解释为什么 `d_model` 要能被 `num_heads` 整除。
- 能解释每个 head 的维度。
- 能解释 concat 后为什么还要输出线性层。

代码：`12_multi_head_attention_from_scratch.py`

---

## 练习 13：跑通小型 Transformer Encoder

**目标**：用 Transformer Encoder 完成玩具序列分类任务。

**任务**：生成随机整数序列，判断序列和是否超过阈值。

流程：

```text
token id → embedding → position embedding → TransformerEncoder → pooling → classifier
```

**验收标准**：

- 能解释输入 token shape。
- 能解释 embedding 后 shape。
- 能解释 TransformerEncoder 输出 shape。
- 能解释 mean pooling 的作用。

代码：`13_tiny_transformer_classifier.py`

---

## 学习方法

每个练习至少做三遍：

1. 第一遍：照着代码跑通。
2. 第二遍：遮住代码，自己重写。
3. 第三遍：改一个条件，比如 hidden size、学习率、层数、数据集。

真正掌握的标准不是“看懂”，而是“能改、能解释、能排错”。
