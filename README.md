# Transformer (PyTorch 实现)

该仓库提供了一个基于 PyTorch 的 Transformer 模型实现，代码结构清晰、模块化，完整复现了论文 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) 中的架构。

## ✨ 特性
- 完全模块化设计：
  - PositionalEncoding（位置编码）
  - ScaledDotProductAttention（缩放点积注意力）
  - MultiHeadAttention（多头注意力）
  - PositionwiseFeedForward（位置前馈网络）
  - EncoderLayer（编码器层）
  - DecoderLayer（解码器层）
  - Encoder（编码器堆叠）
  - Decoder（解码器堆叠）
  - Transformer（完整模型）
- 支持任意序列到序列任务（例如机器翻译、文本摘要等）
- 易于扩展和自定义（例如添加 label smoothing、共享嵌入层）

## 📝 环境依赖
- Python 3.7+
- PyTorch >= 1.10

安装依赖：
```bash
pip install torch
