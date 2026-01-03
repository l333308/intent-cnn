# intent-cnn

# 一级意图
VOLUME 音量
BRIGHTNESS 亮度
WEATHER 天气
STOCK 股票
MODE 模式开关 / 助手模式
TRANSLATE 翻译
NAVIGATION 地图 / 导航
CHAT 闲聊
OSS 其他 / 噪音

# 测试
根目录下
source .venv/bin/activate

## 测试集
python src/inference.py --model_dir output --test_file data/val.jsonl

## 交互测试
python src/inference.py --model_dir output --interactive