"""
TextCNN 训练脚本
支持多语言意图识别模型的完整训练流程
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import TextCNN
from dataset import create_dataloaders
from vocab import Vocabulary, LabelEncoder


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> float:
    """
    训练一个 epoch

    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备

    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    评估模型

    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    train_path: str,
    val_path: str = None,
    output_dir: str = "./output",
    embed_dim: int = 128,
    num_filters: int = 64,
    kernel_sizes: tuple = (2, 3, 4),
    dropout: float = 0.3,
    max_len: int = 32,
    batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: str = "auto",
    early_stopping_patience: int = 5,
):
    """
    完整的训练流程

    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        output_dir: 输出目录
        embed_dim: 词嵌入维度
        num_filters: 卷积过滤器数量
        kernel_sizes: 卷积核大小
        dropout: Dropout 比率
        max_len: 最大序列长度
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        device: 设备 (auto/cpu/cuda/mps)
        early_stopping_patience: 早停耐心值
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # 创建数据加载器
    print("Loading data...")
    train_loader, val_loader, vocab, label_encoder = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        max_len=max_len,
        batch_size=batch_size,
    )
    print(f"Vocab size: {len(vocab)}")
    print(f"Num classes: {label_encoder.num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples: {len(val_loader.dataset)}")

    # 打印意图标签
    print(f"Intent labels: {list(label_encoder.label2id.keys())}")

    # 创建模型
    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_classes=label_encoder.num_classes,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters,
        dropout=dropout,
        padding_idx=vocab.pad_id,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # 训练循环
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # 评估
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            # 早停和模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型
                save_checkpoint(
                    model, vocab, label_encoder, output_dir, "best_model.pt"
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f}")
            # 没有验证集时保存最后一个模型
            if epoch == epochs:
                save_checkpoint(
                    model, vocab, label_encoder, output_dir, "best_model.pt"
                )

    print("-" * 60)
    print(f"Training completed!")
    if val_loader:
        print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.4f}")

    # 保存最终模型
    save_checkpoint(model, vocab, label_encoder, output_dir, "final_model.pt")

    # 保存词表和标签编码器
    vocab.save(output_dir / "vocab.json")
    label_encoder.save(output_dir / "labels.json")
    print(f"\nModel and artifacts saved to: {output_dir}")

    return model, vocab, label_encoder


def save_checkpoint(
    model: TextCNN,
    vocab: Vocabulary,
    label_encoder: LabelEncoder,
    output_dir: Path,
    filename: str,
):
    """保存检查点"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "vocab_size": len(vocab),
        "num_classes": label_encoder.num_classes,
    }
    torch.save(checkpoint, output_dir / filename)


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> tuple:
    """
    加载检查点

    Args:
        checkpoint_path: 检查点路径
        device: 设备

    Returns:
        model: 模型
        vocab: 词表
        label_encoder: 标签编码器
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载词表和标签编码器
    vocab = Vocabulary.load(checkpoint_dir / "vocab.json")
    label_encoder = LabelEncoder.load(checkpoint_dir / "labels.json")

    # 创建模型
    config = checkpoint["model_config"]
    model = TextCNN(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, vocab, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Train TextCNN for intent recognition")

    # 数据参数
    parser.add_argument("--train_path", type=str, required=True, help="Training data path (JSONL)")
    parser.add_argument("--val_path", type=str, default=None, help="Validation data path (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")

    # 模型参数
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_filters", type=int, default=64, help="Number of filters per kernel")
    parser.add_argument("--kernel_sizes", type=str, default="2,3,4", help="Kernel sizes (comma-separated)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--max_len", type=int, default=32, help="Maximum sequence length")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--early_stopping", type=int, default=5, help="Early stopping patience")

    args = parser.parse_args()

    # 解析 kernel_sizes
    kernel_sizes = tuple(int(k) for k in args.kernel_sizes.split(","))

    train(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        kernel_sizes=kernel_sizes,
        dropout=args.dropout,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        early_stopping_patience=args.early_stopping,
    )


if __name__ == "__main__":
    main()
