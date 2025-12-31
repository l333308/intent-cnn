"""
词表构建和管理工具
支持多语言统一词表
"""
import json
from collections import Counter
from typing import Dict, List, Optional
from pathlib import Path

from tokenizer import tokenize


# 特殊 token
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]


class Vocabulary:
    """
    词表类，管理 token 到 ID 的映射
    """

    def __init__(
        self,
        token2id: Dict[str, int] = None,
        min_freq: int = 1,
    ):
        """
        初始化词表

        Args:
            token2id: 已有的 token 到 ID 映射
            min_freq: 最小词频阈值
        """
        self.min_freq = min_freq

        if token2id is not None:
            self.token2id = token2id
            self.id2token = {v: k for k, v in token2id.items()}
        else:
            self.token2id = {}
            self.id2token = {}
            # 初始化特殊 token
            for token in SPECIAL_TOKENS:
                self._add_token(token)

    def _add_token(self, token: str) -> int:
        """添加单个 token"""
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def build_from_texts(
        self,
        texts: List[str],
        langs: List[str] = None,
    ) -> "Vocabulary":
        """
        从文本列表构建词表

        Args:
            texts: 文本列表
            langs: 语言列表

        Returns:
            self
        """
        if langs is None:
            langs = ["auto"] * len(texts)

        # 统计词频
        counter = Counter()
        for text, lang in zip(texts, langs):
            tokens = tokenize(text, lang)
            counter.update(tokens)

        # 添加高频词
        for token, freq in counter.items():
            if freq >= self.min_freq:
                self._add_token(token)

        return self

    def encode(self, tokens: List[str]) -> List[int]:
        """
        将 token 列表编码为 ID 列表

        Args:
            tokens: token 列表

        Returns:
            ids: ID 列表
        """
        unk_id = self.token2id[UNK_TOKEN]
        return [self.token2id.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """
        将 ID 列表解码为 token 列表

        Args:
            ids: ID 列表

        Returns:
            tokens: token 列表
        """
        return [self.id2token.get(i, UNK_TOKEN) for i in ids]

    @property
    def pad_id(self) -> int:
        """获取 padding token 的 ID"""
        return self.token2id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        """获取 unknown token 的 ID"""
        return self.token2id[UNK_TOKEN]

    def __len__(self) -> int:
        """词表大小"""
        return len(self.token2id)

    def __contains__(self, token: str) -> bool:
        """检查 token 是否在词表中"""
        return token in self.token2id

    def save(self, path: str):
        """
        保存词表到文件

        Args:
            path: 保存路径
        """
        data = {
            "token2id": self.token2id,
            "min_freq": self.min_freq,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """
        从文件加载词表

        Args:
            path: 文件路径

        Returns:
            vocab: 词表对象
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(
            token2id=data["token2id"],
            min_freq=data.get("min_freq", 1),
        )
        return vocab


class LabelEncoder:
    """
    标签编码器，管理意图标签到 ID 的映射
    """

    def __init__(self, label2id: Dict[str, int] = None):
        """
        初始化标签编码器

        Args:
            label2id: 已有的标签到 ID 映射
        """
        if label2id is not None:
            self.label2id = label2id
            self.id2label = {v: k for k, v in label2id.items()}
        else:
            self.label2id = {}
            self.id2label = {}

    def fit(self, labels: List[str]) -> "LabelEncoder":
        """
        从标签列表构建映射

        Args:
            labels: 标签列表

        Returns:
            self
        """
        unique_labels = sorted(set(labels))
        for idx, label in enumerate(unique_labels):
            self.label2id[label] = idx
            self.id2label[idx] = label
        return self

    def encode(self, label: str) -> int:
        """编码单个标签"""
        return self.label2id[label]

    def decode(self, idx: int) -> str:
        """解码单个 ID"""
        return self.id2label[idx]

    def encode_batch(self, labels: List[str]) -> List[int]:
        """批量编码"""
        return [self.encode(label) for label in labels]

    def decode_batch(self, ids: List[int]) -> List[str]:
        """批量解码"""
        return [self.decode(idx) for idx in ids]

    @property
    def num_classes(self) -> int:
        """类别数量"""
        return len(self.label2id)

    def save(self, path: str):
        """保存到文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        """从文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        return cls(label2id=label2id)
