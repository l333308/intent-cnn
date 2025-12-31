"""
多语言分词器
支持中文、英文、日语的统一处理
"""
import re
from typing import List


def tokenize(text: str, lang: str = "auto") -> List[str]:
    """
    对文本进行分词

    Args:
        text: 输入文本
        lang: 语言类型 ("zh", "en", "ja", "auto")
              - zh: 中文，字符级分词
              - ja: 日语，字符级分词
              - en: 英文，空格分词 + 小写
              - auto: 自动检测语言

    Returns:
        tokens: 分词后的 token 列表
    """
    text = text.strip()

    if lang == "auto":
        lang = detect_language(text)

    if lang in ("zh", "ja"):
        # 中文/日语：字符级分词
        return list(text)
    else:
        # 英文：空格分词 + 小写化
        return re.findall(r"[a-zA-Z']+", text.lower())


def detect_language(text: str) -> str:
    """
    简单的语言检测

    Args:
        text: 输入文本

    Returns:
        lang: 检测到的语言 ("zh", "ja", "en")
    """
    # 统计字符类型
    cjk_count = 0
    hiragana_katakana_count = 0
    ascii_count = 0

    for char in text:
        code = ord(char)
        # 中文字符范围
        if 0x4E00 <= code <= 0x9FFF:
            cjk_count += 1
        # 日语平假名
        elif 0x3040 <= code <= 0x309F:
            hiragana_katakana_count += 1
        # 日语片假名
        elif 0x30A0 <= code <= 0x30FF:
            hiragana_katakana_count += 1
        # ASCII 字母
        elif 0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A:
            ascii_count += 1

    # 判断语言
    if hiragana_katakana_count > 0:
        return "ja"
    elif cjk_count > ascii_count:
        return "zh"
    else:
        return "en"


class Tokenizer:
    """
    分词器类，封装分词逻辑
    """

    def __init__(self, default_lang: str = "auto"):
        """
        初始化分词器

        Args:
            default_lang: 默认语言
        """
        self.default_lang = default_lang

    def __call__(self, text: str, lang: str = None) -> List[str]:
        """
        分词

        Args:
            text: 输入文本
            lang: 语言类型，None 则使用默认语言

        Returns:
            tokens: 分词后的 token 列表
        """
        lang = lang or self.default_lang
        return tokenize(text, lang)

    def tokenize_batch(self, texts: List[str], langs: List[str] = None) -> List[List[str]]:
        """
        批量分词

        Args:
            texts: 文本列表
            langs: 语言列表，None 则全部使用默认语言

        Returns:
            tokens_list: 分词结果列表
        """
        if langs is None:
            langs = [self.default_lang] * len(texts)

        return [tokenize(text, lang) for text, lang in zip(texts, langs)]
