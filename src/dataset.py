"""
数据集类
支持意图识别数据的加载和处理
"""
import json
from typing import List, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import tokenize
from vocab import Vocabulary, LabelEncoder


class IntentDataset(Dataset):
    """
    意图识别数据集

    数据格式：
    - JSON Lines 格式
    - 每行包含: {"text": "...", "intent": "...", "lang": "zh/en/ja"}
    """

    def __init__(
        self,
        samples: List[Tuple[str, str, str]],  # (text, lang, intent)
        vocab: Vocabulary,
        label_encoder: LabelEncoder,
        max_len: int = 32,
    ):
        """
        初始化数据集

        Args:
            samples: 样本列表，每个样本为 (text, lang, intent)
            vocab: 词表
            label_encoder: 标签编码器
            max_len: 最大序列长度
        """
        self.samples = samples
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_len = max_len

    def encode(self, tokens: List[str]) -> List[int]:
        """
        将 token 编码为 ID 并填充/截断

        Args:
            tokens: token 列表

        Returns:
            ids: 填充后的 ID 列表
        """
        ids = self.vocab.encode(tokens)
        # 截断
        ids = ids[:self.max_len]
        # 填充
        ids += [self.vocab.pad_id] * (self.max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, lang, intent = self.samples[idx]
        tokens = tokenize(text, lang)
        input_ids = self.encode(tokens)
        label = self.label_encoder.encode(intent)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def load_data_from_jsonl(path: str) -> List[Tuple[str, str, str]]:
    """
    从 JSONL 文件加载数据

    Args:
        path: 文件路径

    Returns:
        samples: 样本列表 [(text, lang, intent), ...]
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item["text"]
            intent = item["intent"]
            lang = item.get("lang", "auto")
            samples.append((text, lang, intent))
    return samples


def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    vocab: Vocabulary = None,
    label_encoder: LabelEncoder = None,
    max_len: int = 32,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Vocabulary, LabelEncoder]:
    """
    创建训练和验证数据加载器

    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径（可选）
        vocab: 词表（可选，如果不提供则从数据构建）
        label_encoder: 标签编码器（可选）
        max_len: 最大序列长度
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（如果提供了 val_path）
        vocab: 词表
        label_encoder: 标签编码器
    """
    # 加载训练数据
    train_samples = load_data_from_jsonl(train_path)

    # 构建词表（如果未提供）
    if vocab is None:
        texts = [s[0] for s in train_samples]
        langs = [s[1] for s in train_samples]
        vocab = Vocabulary(min_freq=1)
        vocab.build_from_texts(texts, langs)

    # 构建标签编码器（如果未提供）
    if label_encoder is None:
        labels = [s[2] for s in train_samples]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

    # 创建训练数据集
    train_dataset = IntentDataset(
        samples=train_samples,
        vocab=vocab,
        label_encoder=label_encoder,
        max_len=max_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # 创建验证数据集（如果提供）
    val_loader = None
    if val_path is not None:
        val_samples = load_data_from_jsonl(val_path)
        val_dataset = IntentDataset(
            samples=val_samples,
            vocab=vocab,
            label_encoder=label_encoder,
            max_len=max_len,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader, vocab, label_encoder
