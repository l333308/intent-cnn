"""
推理脚本
用于测试意图识别模型的准确度
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from model import TextCNN
from tokenizer import tokenize
from vocab import Vocabulary, LabelEncoder


class IntentClassifier:
    """意图分类器"""

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        max_len: int = 32,
    ):
        """
        初始化分类器

        Args:
            model_dir: 模型目录（包含 best_model.pt, vocab.json, labels.json）
            device: 设备
            max_len: 最大序列长度
        """
        model_dir = Path(model_dir)
        self.max_len = max_len

        # 设置设备
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 加载词表和标签
        self.vocab = Vocabulary.load(model_dir / "vocab.json")
        self.label_encoder = LabelEncoder.load(model_dir / "labels.json")

        # 加载模型
        checkpoint = torch.load(model_dir / "best_model.pt", map_location=self.device)
        config = checkpoint["model_config"]
        self.model = TextCNN(**config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str, lang: str = "auto") -> torch.Tensor:
        """将文本编码为输入张量"""
        tokens = tokenize(text, lang)
        ids = self.vocab.encode(tokens)
        ids = ids[:self.max_len]
        ids += [self.vocab.pad_id] * (self.max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def predict(
        self,
        text: str,
        lang: str = "auto",
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        预测意图

        Args:
            text: 输入文本
            lang: 语言
            top_k: 返回 top-k 个预测结果

        Returns:
            results: [(intent, probability), ...]
        """
        input_ids = self.encode(text, lang)

        with torch.no_grad():
            logits = self.model(input_ids)
            probs = F.softmax(logits, dim=-1)[0]

        # 获取 top-k
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            intent = self.label_encoder.decode(idx)
            results.append((intent, prob))

        return results

    def predict_batch(
        self,
        texts: List[str],
        langs: List[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        批量预测

        Args:
            texts: 文本列表
            langs: 语言列表

        Returns:
            results: [(intent, probability), ...]
        """
        if langs is None:
            langs = ["auto"] * len(texts)

        input_ids = torch.stack([
            self.encode(text, lang)[0]
            for text, lang in zip(texts, langs)
        ])

        with torch.no_grad():
            logits = self.model(input_ids)
            probs = F.softmax(logits, dim=-1)
            max_probs, pred_indices = torch.max(probs, dim=-1)

        results = []
        for prob, idx in zip(max_probs.tolist(), pred_indices.tolist()):
            intent = self.label_encoder.decode(idx)
            results.append((intent, prob))

        return results


def interactive_mode(classifier: IntentClassifier):
    """交互式测试模式"""
    print("\n" + "=" * 60)
    print("意图识别交互测试")
    print("输入文本进行测试，输入 'quit' 或 'q' 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            text = input("请输入: ").strip()
            if text.lower() in ("quit", "q", "exit"):
                print("再见!")
                break
            if not text:
                continue

            results = classifier.predict(text, top_k=3)

            print(f"\n预测结果:")
            for i, (intent, prob) in enumerate(results, 1):
                bar = "█" * int(prob * 20)
                print(f"  {i}. {intent:20s} {prob:.4f} {bar}")
            print()

        except KeyboardInterrupt:
            print("\n再见!")
            break


def test_from_file(classifier: IntentClassifier, test_path: str):
    """从文件测试并计算准确率"""
    correct = 0
    total = 0
    errors = []

    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            text = item["text"]
            true_intent = item["intent"]
            lang = item.get("lang", "auto")

            results = classifier.predict(text, lang, top_k=1)
            pred_intent, prob = results[0]

            total += 1
            if pred_intent == true_intent:
                correct += 1
            else:
                errors.append({
                    "text": text,
                    "true": true_intent,
                    "pred": pred_intent,
                    "prob": prob,
                })

    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"总样本数: {total}")
    print(f"正确数:   {correct}")
    print(f"准确率:   {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60)

    if errors:
        print(f"\n错误样本 ({len(errors)} 个):")
        print("-" * 60)
        for err in errors[:20]:  # 最多显示 20 个错误
            print(f"  文本: {err['text']}")
            print(f"  真实: {err['true']} | 预测: {err['pred']} (置信度: {err['prob']:.4f})")
            print()

    return accuracy, errors


def test_samples(classifier: IntentClassifier):
    """测试一些示例"""
    test_cases = [
        # 音量
        ("音量调大一点", "zh"),
        ("声音太小了", "zh"),
        ("静音", "zh"),
        ("把音量调到50", "zh"),
        ("volume up", "en"),
        # 亮度
        ("屏幕太暗了", "zh"),
        ("亮度调高", "zh"),
        ("自动亮度", "zh"),
        # 天气
        ("今天天气怎么样", "zh"),
        ("明天会下雨吗", "zh"),
        # 股票
        ("茅台股价", "zh"),
        ("看一下基金", "zh"),
        # 模式控制
        ("打开翻译模式", "zh"),
        ("开启勿扰", "zh"),
        ("关闭录音", "zh"),
        # 计时
        ("倒计时5分钟", "zh"),
        ("开始计时", "zh"),
        # 翻译
        ("翻译成英文", "zh"),
        # 打电话
        ("给妈妈打电话", "zh"),
        ("call mom", "en"),
        # 导航
        ("导航到机场", "zh"),
        ("怎么去火车站", "zh"),
        # 闲聊
        ("你好", "zh"),
        ("讲个笑话", "zh"),
        ("hello", "en"),
    ]

    print("\n" + "=" * 60)
    print("示例测试")
    print("=" * 60 + "\n")

    for text, lang in test_cases:
        results = classifier.predict(text, lang, top_k=1)
        intent, prob = results[0]
        print(f"[{lang}] {text:20s} -> {intent:20s} ({prob:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Intent Recognition Inference")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="../output",
        help="Model directory",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Test file path (JSONL format)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda/mps/auto)",
    )

    args = parser.parse_args()

    # 加载分类器
    print("Loading model...")
    classifier = IntentClassifier(
        model_dir=args.model_dir,
        device=args.device,
    )
    print(f"Model loaded. Device: {classifier.device}")

    if args.test_file:
        # 从文件测试
        test_from_file(classifier, args.test_file)
    elif args.interactive:
        # 交互式模式
        interactive_mode(classifier)
    else:
        # 默认运行示例测试
        test_samples(classifier)


if __name__ == "__main__":
    main()
