import json
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split


INPUT_DATASET = "Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset"
OUTPUT_DIR = Path("data/bert_multiclass")
VALID_RATIO = 0.2
SEED = 42


def main():
    ds = load_dataset(INPUT_DATASET)
    split = "train" if "train" in ds else list(ds.keys())[0]
    texts = [str(x) for x in ds[split]["text"]]
    raw_labels = [str(x) for x in ds[split]["emotion"]]

    unique = sorted(set(raw_labels))
    label2id = {label: i for i, label in enumerate(unique)}
    id2label = {i: label for label, i in label2id.items()}

    labels = [label2id[x] for x in raw_labels]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "label2id.json").open("w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "id2label.json").open("w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=VALID_RATIO, random_state=SEED, stratify=labels
    )

    def save_jsonl(texts, labels, path):
        with path.open("w", encoding="utf-8") as f:
            for t, lab in zip(texts, labels):
                f.write(json.dumps({"sentences": t, "labels": lab}, ensure_ascii=False) + "\n")

    save_jsonl(train_texts, train_labels, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid_texts, valid_labels, OUTPUT_DIR / "valid.jsonl")

    print(f"labels: {label2id}")
    print(f"train: {len(train_texts)}, valid: {len(valid_texts)}")


if __name__ == "__main__":
    main()