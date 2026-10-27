# 土味情話 / 幽默笑話專家（多輪對話 QLoRA）

## 基礎模型
- Qwen/Qwen2.5-0.5B-Instruct

## 執行方法
python download_chat_data.py
python prepare_chat_data.py
python train_qlora.py
python plot_loss.py
python infer_lora_chat.py

## 說明
system prompt：你是一個情話高手，你會以親切、溫和的口吻來回答使用者的問題。
支援至少 2 輪多輪對話。