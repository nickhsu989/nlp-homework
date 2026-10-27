import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_DIR = Path("output/bert_multiclass_clf")
DATA_DIR = Path("data/bert_multiclass")

TEST_TEXTS = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。",
]


def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"找不到模型: {MODEL_DIR.resolve()}，請先執行 train_bert.py")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    with (DATA_DIR / "id2label.json").open(encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    results = classifier(TEST_TEXTS, truncation=True, max_length=128)

    for text, result in zip(TEST_TEXTS, results):
        label = result["label"]
        if label.startswith("LABEL_"):
            idx = int(label.split("_")[-1])
            label = id2label.get(idx, label)
        score = round(float(result["score"]), 4)
        print(f"{text} => {label} ({score})")


if __name__ == "__main__":
    main()