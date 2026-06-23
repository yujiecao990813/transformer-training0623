Markdown
# Transformer-Training-Project

这是一个基于 PyTorch 构建的 Transformer 编码器模型，旨在学习和理解序列建模的核心逻辑。本项目实现了从 Positional Encoding 到 Multi-Head Attention 的完整架构，并支持简单的序列预测训练。

## 项目结构
- `model.py`: 核心模型定义，包含 Positional Encoding、多头注意力机制及 Transformer 层。
- `train.py`: 训练脚本，演示了如何通过简单的输入序列训练模型。
- `main.py`: 项目入口及测试代码。

## 环境要求
- Python 3.10+ (推荐，建议避开 3.14+ 以避免 inspect 兼容性问题)
- PyTorch 2.0+

## 快速开始
1. 克隆本项目：
   ```bash
   git clone [https://github.com/yujiecao990813/transformer-training0623.git](https://github.com/yujiecao990813/transformer-training0623.git)
安装依赖：

Bash
pip install torch
开始训练：

Bash
python train.py
核心实现说明
本模型重点实现了以下三维张量变换：

Batch: 批量处理能力

Seq_len: 序列长度适应

Embed_dim: 词向量维度映射

问题排查
如果遇到 inspect.py 相关报错，请优先检查 Python 版本是否过高。本模型在 Python 3.10 环境下已通过验证。
