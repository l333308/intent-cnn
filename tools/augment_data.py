"""
数据增强工具
基于现有数据生成更多的训练和验证样本
目标：训练集 ~800条，验证集 ~120条
"""
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# 同义词和变体模板
TEMPLATES = {
    "zh": {
        "queryVolume": [
            "音量多少", "现在音量是多少", "当前音量", "音量大小", "查一下音量",
            "音量是几", "看看音量", "音量怎么样", "音量多大", "声音多大",
            "声音是多少", "音量设置的多少", "当前声音大小", "查询音量", "音量查询"
        ],
        "volumeUp": [
            "音量调大", "声音大一点", "调高音量", "大声点", "音量加大",
            "把声音调大", "声音太小了", "音量有点低", "听不清", "声音调高",
            "音量提高", "增大音量", "音量大点", "声音开大", "放大声音"
        ],
        "volumeDown": [
            "音量调小", "声音小一点", "调低音量", "小声点", "音量减小",
            "把声音调小", "声音太大了", "音量有点高", "太吵了", "声音调低",
            "音量降低", "减小音量", "音量小点", "声音开小", "降低声音"
        ],
        "volumeMute": [
            "静音", "关闭声音", "关掉声音", "消音", "把声音关了",
            "音量静音", "静音模式", "关闭音量", "声音关掉", "不要声音",
            "关声音", "音量关闭", "静音一下", "声音静音", "关掉音量"
        ],
        "volumeSet": [
            "把音量调到50", "音量设置50", "音量调成50", "设置音量为50", "音量50",
            "调到50音量", "音量设为50", "设音量50", "音量调整到50", "把音量设成50",
            "音量改成50", "设置音量50", "音量调为50", "调整音量到50", "音量设定50"
        ],
        "queryBrightness": [
            "亮度多少", "当前亮度", "亮度是几", "查一下亮度", "看看亮度",
            "亮度怎么样", "现在亮度多少", "亮度大小", "屏幕亮度", "查询亮度",
            "亮度查询", "亮度设置的多少", "当前屏幕亮度", "亮度是多少", "屏幕多亮"
        ],
        "brightnessUp": [
            "亮度调高", "屏幕太暗了", "亮一点", "调亮屏幕", "增加亮度",
            "提高亮度", "亮度加大", "屏幕调亮", "太暗了", "看不清屏幕",
            "屏幕亮点", "亮度提升", "调高亮度", "屏幕亮度调高", "增大亮度"
        ],
        "brightnessDown": [
            "亮度调低", "屏幕太亮了", "暗一点", "调暗屏幕", "降低亮度",
            "减少亮度", "亮度减小", "屏幕调暗", "太刺眼了", "屏幕太晃眼",
            "屏幕暗点", "亮度降低", "调低亮度", "屏幕亮度调低", "减小亮度"
        ],
        "brightnessSet": [
            "亮度调到5", "设置亮度5", "亮度设为5", "把亮度调成5", "亮度5",
            "调到5亮度", "设亮度5", "亮度调整到5", "把亮度设成5", "亮度改成5",
            "设置亮度为5", "亮度调为5", "调整亮度到5", "亮度设定5", "屏幕亮度5"
        ],
        "brightnessAuto": [
            "自动亮度", "亮度自动", "开启自动亮度", "打开自动亮度", "自适应亮度",
            "亮度自动调节", "自动调节亮度", "启用自动亮度", "自动亮度模式", "亮度自适应",
            "开自动亮度", "自动屏幕亮度", "屏幕自动亮度", "亮度自动模式", "自适应屏幕"
        ],
        "weather_query": [
            "今天天气怎么样", "明天会下雨吗", "天气如何", "查天气", "看看天气",
            "今天天气", "天气预报", "明天天气", "今儿天气咋样", "天气怎么样",
            "查询天气", "今日天气", "天气情况", "今天会下雨吗", "天气好不好"
        ],
        "stock_search": [
            "茅台股价", "看一下基金", "查股票", "股票行情", "查一下股价",
            "看看股市", "股市行情", "基金怎么样", "股票查询", "查询股票",
            "看股票", "基金查询", "股价多少", "查基金", "股市情况"
        ],
        "modeOpen": [
            "打开翻译模式", "开启勿扰", "进入会议模式", "启动录音", "打开助手",
            "进入翻译模式", "启用勿扰模式", "开会议模式", "录音模式",
            "开启助手模式", "打开录音模式", "启动会议模式", "进入助手模式", "开启录音",
            # 新增：强化"开始"语义
            "开始会议记录", "启动会议记录", "开始录音会议", "打开会议记录功能", "开启会议记录"
        ],
        "modeClose": [
            "关闭录音", "退出会议模式", "关掉翻译", "结束会议记录", "关闭助手",
            "退出翻译模式", "关闭会议模式", "停止录音", "结束录音", "退出助手模式",
            "关掉会议模式", "关闭翻译模式", "停止会议", "退出录音模式", "关助手",
            # 新增：强化"结束/停止"语义
            "停止会议记录", "终止会议记录", "结束录音会议", "关闭会议记录功能", "停止会议记录功能"
        ],
        "timeCountdown": [
            "倒计时5分钟", "5分钟倒计时", "倒数5分钟", "设置倒计时5分钟", "开始倒计时5分钟",
            "倒计时五分钟", "五分钟倒计时", "计时5分钟", "定时5分钟", "5分钟计时",
            "倒数计时5分钟", "倒计时提醒5分钟", "设定倒计时5分钟", "启动倒计时5分钟", "开个5分钟倒计时"
        ],
        "timeCount": [
            "开始计时", "计时开始", "启动计时", "开启计时器", "开始计时器",
            "打开计时", "计时", "开计时", "启动计时器", "计时功能",
            "开始秒表", "秒表开始", "打开秒表", "启用计时", "开始记时"
        ],
        "translate": [
            "翻译成英文", "翻译", "帮我翻译", "翻译一下", "中译英",
            "英文翻译", "翻译成中文", "英译中", "日译中", "翻译成日语",
            "帮忙翻译", "翻译这句话", "这怎么翻译", "翻译功能", "语言翻译", "启动翻译"
        ],
        "call": [
            "给妈妈打电话", "打电话给爸爸", "拨打电话", "呼叫", "call妈妈",
            "联系妈妈", "打给妈妈", "拨号给爸爸", "电话给妈妈", "给爸爸来电话",
            "呼叫妈妈", "致电妈妈", "拨打给爸爸", "打电话", "拨号"
        ],
        "navigation": [
            "导航到机场", "怎么去火车站", "去机场怎么走", "带我去机场", "到机场的路线",
            "导航去火车站", "去火车站", "机场导航", "路线到机场", "怎么走到火车站",
            "导航", "带我去火车站", "去机场", "火车站怎么走", "到火车站怎么走"
        ],
        "chat": [
            "你好", "讲个笑话", "聊天", "说话", "在吗",
            "早上好", "晚安", "你是谁", "干嘛呢", "谢谢",
            "再见", "怎么样", "哈哈", "嗯嗯", "好的"
        ],
    },
    "en": {
        "queryVolume": [
            "what's the volume", "current volume level", "check volume", "volume level", "how loud is it",
            "what is the volume", "show volume", "volume status", "current sound level", "volume setting",
            "get volume", "volume info", "sound level", "check sound", "what's the sound level"
        ],
        "volumeUp": [
            "turn up the volume", "louder please", "increase volume", "make it louder", "volume up",
            "raise the volume", "turn it up", "boost volume", "more volume", "louder",
            "increase the sound", "turn up", "make louder", "sound up", "raise volume"
        ],
        "volumeDown": [
            "turn down the volume", "quieter please", "decrease volume", "make it quieter", "volume down",
            "lower the volume", "turn it down", "reduce volume", "less volume", "quieter",
            "decrease the sound", "turn down", "make quieter", "sound down", "lower volume"
        ],
        "volumeMute": [
            "mute", "silence", "turn off sound", "mute volume", "no sound",
            "mute it", "silence please", "mute audio", "turn off audio", "kill sound",
            "shut up", "quiet mode", "disable sound", "turn sound off", "mute mode"
        ],
        "volumeSet": [
            "set volume to 50", "volume 50", "set volume 50", "make volume 50", "adjust volume to 50",
            "change volume to 50", "volume at 50", "set to 50", "50 volume", "put volume at 50",
            "adjust to 50", "set sound to 50", "sound at 50", "volume to 50 percent"
        ],
        "queryBrightness": [
            "what's the brightness", "current brightness", "check brightness", "brightness level", "how bright",
            "what is brightness", "show brightness", "brightness status", "current screen brightness", "brightness setting",
            "get brightness", "brightness info", "screen brightness", "check screen", "display brightness",
            # 新增：强化查询语义，避免与设置混淆
            "what's the brightness level", "current brightness level", "tell me brightness level", "show me brightness", "brightness level now"
        ],
        "brightnessUp": [
            "increase brightness", "brighter", "make it brighter", "brightness up", "turn up brightness",
            "raise brightness", "more brightness", "boost brightness", "brighten screen", "screen brighter",
            "brighten up", "make brighter", "increase screen", "brighter screen", "brightness higher"
        ],
        "brightnessDown": [
            "decrease brightness", "dimmer", "make it dimmer", "brightness down", "turn down brightness",
            "lower brightness", "less brightness", "reduce brightness", "dim screen", "screen dimmer",
            "dim down", "make dimmer", "decrease screen", "dimmer screen", "brightness lower"
        ],
        "brightnessSet": [
            "set brightness to 5", "brightness 5", "set brightness 5", "make brightness 5", "adjust brightness to 5",
            "change brightness to 5", "brightness at 5", "set to 5", "5 brightness", "put brightness at 5",
            "adjust to 5", "set screen to 5", "screen at 5", "brightness to 5 percent"
        ],
        "brightnessAuto": [
            "auto brightness", "automatic brightness", "enable auto brightness", "turn on auto brightness", "adaptive brightness",
            "auto adjust brightness", "brightness auto mode", "automatic screen", "auto screen brightness", "adaptive screen",
            "enable adaptive brightness", "brightness adaptation", "auto mode", "automatic adjustment", "auto display"
        ],
        "weather_query": [
            "what's the weather", "weather today", "check weather", "weather forecast", "how's the weather",
            "today's weather", "weather report", "will it rain", "is it raining", "weather status",
            "check forecast", "weather info", "what's weather like", "forecast today", "weather update"
        ],
        "stock_search": [
            "stock price", "check stocks", "stock market", "fund status", "stock info",
            "how's the market", "market status", "check fund", "stock status", "market price",
            "fund price", "stock query", "market info", "fund info", "stock update"
        ],
        "modeOpen": [
            "enable translation mode", "turn on translation", "start recording", "enable assistant", "open meeting mode",
            "activate translation", "turn on recorder", "start meeting", "enable recorder", "activate assistant",
            "translation mode on", "recorder on", "meeting mode on", "assistant on", "start assistant",
            # 新增：强化"start/begin"语义
            "start meeting recording", "begin meeting recording", "start meeting notes", "open meeting recording", "launch meeting recorder"
        ],
        "modeClose": [
            "disable translation mode", "turn off translation", "stop recording", "disable assistant", "close meeting mode",
            "deactivate translation", "turn off recorder", "end meeting", "disable recorder", "deactivate assistant",
            "translation mode off", "recorder off", "meeting mode off", "assistant off", "stop assistant",
            # 新增：强化"stop/end"语义
            "stop meeting recording", "end meeting recording", "finish meeting notes", "close meeting recording", "terminate meeting recorder"
        ],
        "timeCountdown": [
            "countdown 5 minutes", "5 minute countdown", "start countdown 5 minutes", "timer 5 minutes", "5 min countdown",
            "countdown for 5 minutes", "set countdown 5 minutes", "5 minutes timer", "count down 5 minutes", "countdown timer 5",
            "5 minute timer", "start 5 minute countdown", "timer for 5 minutes", "countdown 5 min", "5 min timer"
        ],
        "timeCount": [
            "start timer", "begin timer", "start counting", "enable timer", "timer on",
            "start stopwatch", "begin counting", "timer start", "stopwatch start", "count time",
            "start time", "enable stopwatch", "counting on", "begin stopwatch", "activate timer"
        ],
        "translate": [
            "translate to English", "translate", "translate this", "translation", "convert to English",
            "English translation", "translate to Chinese", "how do you say this", "what's this in English", "translate please",
            "help translate", "translate it", "translation please", "convert language", "language translation"
        ],
        "call": [
            "call mom", "phone mom", "dial mom", "call dad", "phone dad",
            "ring mom", "contact mom", "call mother", "phone father", "dial dad",
            "make a call", "call someone", "dial number", "phone call", "ring dad"
        ],
        "navigation": [
            "navigate to airport", "go to station", "directions to airport", "route to airport", "how to get to station",
            "navigate to station", "take me to airport", "airport directions", "station route", "way to airport",
            "go to airport", "directions to station", "route to station", "how to reach airport", "navigate airport"
        ],
        "chat": [
            "hello", "hi", "good morning", "good night", "how are you",
            "thanks", "thank you", "bye", "goodbye", "see you",
            "what's up", "hey", "yeah", "ok", "sure"
        ],
    },
    "ja": {
        "queryVolume": [
            "音量はいくつ", "今の音量", "音量を確認", "音量は", "音量教えて",
            "音量レベル", "現在の音量", "音量状態", "音量情報", "音量を見せて",
            "音量どれくらい", "音量設定", "音量は何", "音量確認", "音量チェック"
        ],
        "volumeUp": [
            "音量を上げて", "音量アップ", "大きくして", "もっと大きく", "音量を大きく",
            "音を大きく", "ボリュームアップ", "音量を増やして", "大きい音", "音量高く",
            "聞こえない", "音小さい", "もう少し大きく", "音量上げる", "大きくなって"
        ],
        "volumeDown": [
            "音量を下げて", "音量ダウン", "小さくして", "もっと小さく", "音量を小さく",
            "音を小さく", "ボリュームダウン", "音量を減らして", "小さい音", "音量低く",
            "うるさい", "音大きい", "もう少し小さく", "音量下げる", "小さくなって"
        ],
        "volumeMute": [
            "ミュート", "音を消して", "消音", "音量オフ", "音なし",
            "静かに", "音を切って", "ミュートして", "無音", "サウンドオフ",
            "音消す", "音量ミュート", "音を止めて", "静音", "音声オフ"
        ],
        "volumeSet": [
            "音量を50に", "音量50", "音量を50に設定", "50の音量", "音量50にして",
            "音量設定50", "50に設定", "音量調整50", "ボリューム50", "音量レベル50",
            "50にする", "音量を50", "50の設定", "音量50に調整", "50に変更"
        ],
        "queryBrightness": [
            "明るさはいくつ", "今の明るさ", "明るさを確認", "明るさは", "明るさ教えて",
            "輝度レベル", "現在の明るさ", "明るさ状態", "画面の明るさ", "明るさを見せて",
            "明るさどれくらい", "明るさ設定", "明るさは何", "明るさ確認", "輝度チェック"
        ],
        "brightnessUp": [
            "明るくして", "明るさアップ", "もっと明るく", "画面明るく", "輝度を上げて",
            "明るさを上げて", "もう少し明るく", "明るく", "画面が暗い", "見えない",
            "輝度アップ", "明るさ増やして", "明るくなって", "画面を明るく", "明るさ上げる"
        ],
        "brightnessDown": [
            "暗くして", "明るさダウン", "もっと暗く", "画面暗く", "輝度を下げて",
            "明るさを下げて", "もう少し暗く", "暗く", "画面が明るい", "まぶしい",
            "輝度ダウン", "明るさ減らして", "暗くなって", "画面を暗く", "明るさ下げる"
        ],
        "brightnessSet": [
            "明るさを5に", "明るさ5", "明るさを5に設定", "5の明るさ", "明るさ5にして",
            "輝度設定5", "5に設定", "明るさ調整5", "輝度5", "明るさレベル5",
            "5にする", "明るさを5", "5の設定", "明るさ5に調整", "5に変更"
        ],
        "brightnessAuto": [
            "自動明るさ", "明るさ自動", "オート明るさ", "自動調整", "明るさオート",
            "自動輝度", "明るさ自動調整", "オート輝度", "適応明るさ", "自動で",
            "明るさ自動モード", "輝度自動", "自動調節", "アダプティブ", "自動設定"
        ],
        "weather_query": [
            "天気は", "今日の天気", "天気教えて", "天気予報", "天気どう",
            "天気確認", "天気情報", "明日の天気", "天気は何", "雨降る",
            "天気チェック", "天候", "今日天気", "天気見せて", "天気状況"
        ],
        "stock_search": [
            "株価", "株を見て", "株価確認", "株情報", "市場状況",
            "株チェック", "銘柄確認", "株価教えて", "投資信託", "ファンド",
            "株を調べて", "株価情報", "マーケット", "株式", "株価は"
        ],
        "modeOpen": [
            "翻訳モード開始", "翻訳モードオン", "録音開始", "会議モード", "アシスタントオン",
            "翻訳モード", "録音モード", "翻訳開始", "会議開始", "アシスト開始",
            "翻訳を開く", "録音を開始", "会議モード開始", "アシスタント起動", "モード開始",
            # 新增：强化"開始/スタート"语义
            "会議記録開始", "会議記録スタート", "会議録音開始", "会議メモ開始", "会議記録を開始"
        ],
        "modeClose": [
            "翻訳モード終了", "翻訳モードオフ", "録音停止", "会議終了", "アシスタントオフ",
            "翻訳終了", "録音終了", "翻訳を閉じる", "会議を終わる", "アシスト終了",
            "翻訳を止める", "録音を停止", "会議モード終了", "アシスタント停止", "モード終了",
            # 新増：强化"終了/停止"语义
            "会議記録終了", "会議記録停止", "会議録音終了", "会議メモ終了", "会議記録を終了"
        ],
        "timeCountdown": [
            "5分カウントダウン", "カウントダウン5分", "5分タイマー", "タイマー5分", "5分後",
            "5分計測", "カウント5分", "5分間タイマー", "5分のタイマー", "タイマースタート5分",
            "5分カウント", "カウントダウンタイマー5分", "5分測定", "5分セット", "5分間カウント"
        ],
        "timeCount": [
            "タイマー開始", "計測開始", "カウント開始", "タイマースタート", "時間測定",
            "ストップウォッチ", "計時開始", "タイマーオン", "測定スタート", "時間計測",
            "カウントスタート", "計測", "時間測る", "タイマー", "ストップウォッチ開始"
        ],
        "translate": [
            "翻訳して", "英語に翻訳", "翻訳", "訳して", "英訳",
            "中国語に翻訳", "日本語に", "翻訳お願い", "これ翻訳", "英語で",
            "翻訳機能", "訳す", "言語翻訳", "翻訳してください", "英語翻訳"
        ],
        "call": [
            "母に電話", "電話かけて", "ママに電話", "電話して", "発信",
            "母親に電話", "お母さんに", "電話お願い", "コール", "ダイヤル",
            "電話をかける", "連絡して", "電話する", "呼び出し", "電話かける"
        ],
        "navigation": [
            "空港までナビ", "駅への道", "空港まで", "ナビゲーション", "道案内",
            "駅へ行く", "空港への行き方", "ナビして", "駅までの道", "ルート案内",
            "空港に行く", "駅まで", "道を教えて", "経路案内", "ナビ開始"
        ],
        "chat": [
            "こんにちは", "おはよう", "こんばんは", "ありがとう", "さようなら",
            "やあ", "ハロー", "よろしく", "どうも", "バイバイ",
            "元気", "うん", "はい", "いいえ", "了解"
        ],
    }
}


def load_existing_data(file_path: str) -> List[Dict]:
    """加载现有数据"""
    data = []
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def generate_augmented_data(
    intent: str,
    lang: str,
    count: int,
    existing_texts: set
) -> List[Dict]:
    """为指定意图和语言生成增强数据"""
    if lang not in TEMPLATES or intent not in TEMPLATES[lang]:
        return []

    templates = TEMPLATES[lang][intent]
    generated = []

    for template in templates:
        if template not in existing_texts and len(generated) < count:
            generated.append({
                "text": template,
                "intent": intent,
                "lang": lang
            })
            existing_texts.add(template)

    return generated


def augment_train_data(
    input_file: str = "data/train.jsonl",
    output_file: str = "data/train_augmented.jsonl",
    target_per_intent: Dict[str, int] = None
):
    """
    增强训练数据

    目标：每个意图约40条
    - 中文: 15条
    - 英文: 15条
    - 日语: 10条
    """
    if target_per_intent is None:
        target_per_intent = {"zh": 15, "en": 15, "ja": 10}

    # 加载现有数据
    existing_data = load_existing_data(input_file)

    # 统计现有数据
    existing_by_intent_lang = defaultdict(list)
    existing_texts = set()

    for item in existing_data:
        key = (item["intent"], item["lang"])
        existing_by_intent_lang[key].append(item)
        existing_texts.add(item["text"])

    print(f"现有训练数据: {len(existing_data)} 条")

    # 生成增强数据
    augmented_data = existing_data.copy()

    all_intents = set()
    for lang in TEMPLATES:
        all_intents.update(TEMPLATES[lang].keys())

    for intent in sorted(all_intents):
        for lang in ["zh", "en", "ja"]:
            key = (intent, lang)
            existing_count = len(existing_by_intent_lang[key])
            target_count = target_per_intent.get(lang, 10)
            need_count = max(0, target_count - existing_count)

            if need_count > 0:
                new_samples = generate_augmented_data(
                    intent, lang, need_count, existing_texts
                )
                augmented_data.extend(new_samples)
                print(f"  {intent:20s} [{lang}]: {existing_count:2d} -> {existing_count + len(new_samples):2d} (+{len(new_samples)})")

    # 打乱数据
    random.shuffle(augmented_data)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n训练数据增强完成: {len(existing_data)} -> {len(augmented_data)} 条")
    print(f"保存到: {output_file}")

    return augmented_data


def augment_val_data(
    input_file: str = "data/val.jsonl",
    output_file: str = "data/val_augmented.jsonl",
    target_per_intent: Dict[str, int] = None
):
    """
    增强验证数据

    目标：每个意图约6条
    - 中文: 3条
    - 英文: 2条
    - 日语: 1条
    """
    if target_per_intent is None:
        target_per_intent = {"zh": 3, "en": 2, "ja": 1}

    # 加载现有数据
    existing_data = load_existing_data(input_file)

    # 统计现有数据
    existing_by_intent_lang = defaultdict(list)
    existing_texts = set()

    for item in existing_data:
        key = (item["intent"], item["lang"])
        existing_by_intent_lang[key].append(item)
        existing_texts.add(item["text"])

    print(f"现有验证数据: {len(existing_data)} 条")

    # 生成增强数据
    augmented_data = existing_data.copy()

    all_intents = set()
    for lang in TEMPLATES:
        all_intents.update(TEMPLATES[lang].keys())

    for intent in sorted(all_intents):
        for lang in ["zh", "en", "ja"]:
            key = (intent, lang)
            existing_count = len(existing_by_intent_lang[key])
            target_count = target_per_intent.get(lang, 1)
            need_count = max(0, target_count - existing_count)

            if need_count > 0:
                new_samples = generate_augmented_data(
                    intent, lang, need_count, existing_texts
                )
                augmented_data.extend(new_samples)
                print(f"  {intent:20s} [{lang}]: {existing_count:2d} -> {existing_count + len(new_samples):2d} (+{len(new_samples)})")

    # 打乱数据
    random.shuffle(augmented_data)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n验证数据增强完成: {len(existing_data)} -> {len(augmented_data)} 条")
    print(f"保存到: {output_file}")

    return augmented_data


def main():
    print("=" * 60)
    print("数据增强工具")
    print("=" * 60)

    # 增强训练数据
    print("\n### 增强训练数据 ###")
    train_data = augment_train_data()

    print("\n" + "-" * 60 + "\n")

    # 增强验证数据
    print("### 增强验证数据 ###")
    val_data = augment_val_data()

    print("\n" + "=" * 60)
    print("数据增强完成!")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
