# 中文句子情緒分類（多元分類）

## 訓練資料來源
- [Datasets:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)

## 基礎模型
- [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

## 安裝套件
- torch==2.14.1+cu126
- transformers==5.19.0
- tokenizers==0.23.3
- datasets==5.1.0
- scikit-learn==1.9.1
- accelerate==1.15.0
- numpy==2.4.6
- huggingface-hub==2.2.0

## 說明
將 bert_finetune 範例從二元分類改為多元分類（8 類情緒），label 為中文文字，先轉為整數 0..7。

## 執行方法（Colab T4，已註解）
# !pip install -r requirements-colab.txt
# python prepare_data.py
# python train_bert.py
# python infer_bert.py

## 本地 GPU 執行方法
pip install -r requirementsgpu.txt
python prepare_data.py
python train_bert.py
python infer_bert.py

> 注意：Windows 上若先 `import torch` 再 `import datasets`（底層載入 pyarrow），
> 會發生 DLL 衝突導致 heap corruption (0xC0000374) 且沒有任何錯誤訊息。
> `train_bert.py` 已將 `from datasets import load_dataset` 移到 `import torch` 之前以避免此問題。

## 訓練結果（本地 GPU：NVIDIA GeForce GTX 1050 Ti）
- train_runtime: 252.7 秒
- best eval_loss: 0.4377
- eval_f1 (macro): 0.8622
- eval_accuracy: 0.8654

## 預測結果
執行 infer_bert.py 後輸出：
我每天都能跟她一起上學，我好開心！ => 開心語調 (0.9562)
最好的朋友要離開臺灣了，以後可能不容易再見面... => 悲傷語調 (0.9198)
我覺得我快不行了... => 悲傷語調 (0.9524)
剛剛收到研究所錄取的通知書！ => 開心語調 (0.9365)
今年的冬天好像比較晚來。 => 平淡語氣 (0.9475)