# Transfomer

### 解决问题：（RNN的三大痛点）
1. RNN基于时序无法并行运算，训练极慢
2. RNN传播过程容易梯度爆炸/消失，长距离依赖能力极差
3. RNN只能单向/双向传递信息且权重分配死板，序列信息捕捉能力弱

*图1: 基于 LSTM 的seq2seq架构图*
![alt text](<ChatGPT Image 2026年5月19日 16_53_20.png>)

*图2: 基于 Transformer 的seq2seq架构图*
![alt text](<ChatGPT Image 2026年5月19日 16_53_12.png>)
(图1图2为AI生图笔记,仅供参考)

---

*图3：transformer架构图*
![alt text](transformer架构图.png)
核心结构: Multi-head Attention
主要组成部分：
- Encoder
- Decoder