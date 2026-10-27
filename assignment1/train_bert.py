from pathlib import Path
import json

# 注意：Windows 上 torch 與 pyarrow (datasets 底層) 同時載入 DLL 會造成
# heap corruption (0xC0000374)。先載入 datasets 再載入 torch 可避免此問題。
from datasets import load_dataset

import numpy as np
import torch
import transformers
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "google-bert/bert-base-chinese"
DATA_DIR = Path("data/bert_multiclass")
TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"
OUTPUT_DIR = Path("output/bert_multiclass_clf")
MAX_SEQ_LENGTH = 128
SEED = 42
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
EVAL_STEPS = 50
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING_PATIENCE = 3
LOGGING_STEPS = 10
USE_FP16 = True
USE_BF16 = False


def print_gpu_info():
    print("=" * 72)
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.argmax(preds, axis=-1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = (preds == labels).mean()
    return {"f1": float(f1), "accuracy": float(acc)}


def main():
    print_gpu_info()

    train_path = DATA_DIR / TRAIN_FILE
    valid_path = DATA_DIR / VALID_FILE
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError("請先執行 python prepare_data.py")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (DATA_DIR / "label2id.json").open(encoding="utf-8") as f:
        label2id = json.load(f)
    with (DATA_DIR / "id2label.json").open(encoding="utf-8") as f:
        id2label_raw = json.load(f)
    id2label = {int(k): v for k, v in id2label_raw.items()}
    num_labels = len(label2id)

    dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(valid_path)})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(batch["sentences"], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["sentences"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    fp16_enabled = USE_FP16 and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16_enabled,
        bf16=USE_BF16,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()
    print("best checkpoint:", trainer.state.best_model_checkpoint)
    print("best eval_loss:", trainer.state.best_metric)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    trainer.save_state()
    print("Final model directory:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()