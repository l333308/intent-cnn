"""
TextCNN 模型定义
支持多语言意图识别（中文/英文/日语）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    Text CNN 模型用于意图分类

    Args:
        vocab_size: 词表大小
        embed_dim: 词嵌入维度
        num_classes: 意图类别数（包含 OOS）
        kernel_sizes: 卷积核大小列表
        num_filters: 每个卷积核的过滤器数量
        dropout: Dropout 比率
        padding_idx: padding token 的索引
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 10,
        kernel_sizes: tuple = (2, 3, 4),
        num_filters: int = 64,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        # 保存配置用于导出
        self.config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "kernel_sizes": kernel_sizes,
            "num_filters": num_filters,
            "dropout": dropout,
            "padding_idx": padding_idx,
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len] 输入的 token ID 序列

        Returns:
            logits: [batch_size, num_classes] 分类 logits
        """
        # input_ids: [B, L]
        x = self.embedding(input_ids)          # [B, L, D]
        x = x.transpose(1, 2)                  # [B, D, L]

        conv_outs = []
        for conv in self.convs:
            y = F.relu(conv(x))                        # [B, C, L']
            y = F.adaptive_max_pool1d(y, 1)            # [B, C, 1]  使用自适应池化兼容 ONNX
            conv_outs.append(y.squeeze(2))             # [B, C]

        feat = torch.cat(conv_outs, dim=1)     # [B, C * K]
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits

    def predict(self, input_ids: torch.Tensor) -> tuple:
        """
        预测接口，返回预测类别和概率

        Args:
            input_ids: [batch_size, seq_len] 输入的 token ID 序列

        Returns:
            pred_labels: [batch_size] 预测的类别
            probs: [batch_size, num_classes] 预测概率
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            probs = F.softmax(logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1)
        return pred_labels, probs
