"""
載入 train_qlora.py 儲存的最後 LoRA adapter，進行「多輪互動式對話」。

這個程式和單輪測試版最大的差別是：

    單輪版：
        system
        user
        assistant
        程式結束

    多輪版：
        system
        user
        assistant
        user
        assistant
        user
        assistant
        ...
        直到使用者輸入 exit / quit / q 才結束

模型每次回答時，都會把前面的對話歷史一起送進模型，因此可以延續上下文。

例如：

    使用者：今天下大雨，我沒帶傘。
    助理：那我陪你一起淋，反正我想跟你走同一條路。
    使用者：這樣會感冒耶。
    助理：那就抱緊一點，當你的專屬暖暖包。

第二輪的「這樣會感冒耶」本身沒有再次說「下雨」，
但模型可以從前面的 conversation history 知道使用者正在延續同一個情境。

如果要測試特定 checkpoint（需先保留該 checkpoint），也可以把 ADAPTER_DIR 改成：

    outputs/qwen25_05b_lora/checkpoint-189

注意：
本專案的 QLoRA 使用的是 bitsandbytes NF4，而不是 TorchAO quantization。

如果目前 Colab 環境有不相容的舊版 torchao，例如：

    torchao==0.10.0

而 PEFT 顯示：

    ImportError:
    Found an incompatible version of torchao

可以移除：

    python -m pip uninstall -y torchao

因為這個推論程式不需要 TorchAO。
"""

# =============================================================================
# 1. 匯入套件
# =============================================================================

# PyTorch。
#
# 這裡主要用來：
# 1. 指定模型使用 FP16。
# 2. 使用 torch.inference_mode() 關閉 gradient。
#
# 推論和訓練不同：
#
# 訓練：
#     forward
#     ↓
#     loss
#     ↓
#     backward
#     ↓
#     gradient
#     ↓
#     optimizer update
#
# 推論：
#     forward
#     ↓
#     直接產生文字
#
# 所以推論不需要 gradient。
import torch

# AutoPeftModelForCausalLM：
#
# 讀取 LoRA adapter 目錄中的 adapter_config.json，
# 找出原本的 base model，
# 然後自動把：
#
#     Base Model
#         +
#     LoRA Adapter
#
# 組合起來。
from peft import AutoPeftModelForCausalLM

# AutoTokenizer：
#
# 負責：
# - 把文字轉成 token IDs
# - 套用 Qwen chat template
# - 把模型產生的 token IDs 解碼回文字
from transformers import AutoTokenizer


# =============================================================================
# 2. 推論設定
# =============================================================================

# LoRA adapter 的位置。
#
# train_qlora.py 正常訓練完成後，會把最後模型存到：
#
#     outputs/qwen25_05b_lora
#
# 如果想比較某個特定 checkpoint：
#
#     ADAPTER_DIR = "outputs/qwen25_05b_lora/checkpoint-189"
#
# 訓練時 adapter 直接放在 outputs/ 底下；
# 在本機把下載的 adapter 解壓到 adapter/ 底下時，則是
# adapter/outputs/qwen25_05b_lora。
# 下面這段會自動挑選實際存在的那一個。
ADAPTER_CANDIDATES = [
    "outputs/qwen25_05b_lora",
    "adapter/outputs/qwen25_05b_lora",
]


def resolve_adapter_dir() -> str:
    """回傳實際存在的 LoRA adapter 目錄。"""
    from pathlib import Path as _Path

    for candidate in ADAPTER_CANDIDATES:
        if (_Path(candidate) / "adapter_config.json").is_file():
            return candidate

    raise FileNotFoundError(
        "找不到 LoRA adapter，請先執行 train_qlora.py，"
        f"或把 adapter 解壓到 {ADAPTER_CANDIDATES[1]}。\n"
        f"已搜尋：{ADAPTER_CANDIDATES}"
    )


ADAPTER_DIR = resolve_adapter_dir()


# System prompt。
#
# System prompt 可以理解為：
# 「在整段對話開始之前，先告訴模型它的角色、規則與限制。」
#
# 這段內容會保留在每一輪 conversation context 中。
SYSTEM_PROMPT = (
    "你是一個土味情話高手。\n"
    "你會以親切、溫和、俏皮但不油膩的口吻跟使用者互動。\n"
    "回答要自然、符合日常台灣/華語用法，避免過於露骨或不當內容。\n"
    "每次回應都要符合「土味情話」的風格，善用反差、關心與小心機的表達方式。"
)


# 每次 assistant 最多可以新產生多少 tokens。
#
# 這裡的 100 是「新回答」的上限，
# 不包含前面 system / user / assistant 的歷史內容。
#
# 訓練語料的 assistant 回覆很短，
# 100 tokens 已經足夠，也能避免小模型長篇離題。
MAX_NEW_TOKENS = 100


# Sampling temperature。
#
# 越低：
#     回答越穩定、越接近最高機率 token。
#
# 越高：
#     回答變化越多，但也可能更不穩定。
#
# 本版本使用 greedy decoding，因此 temperature 不參與實際生成；
# 保留 1.0 是為了避免 generate() 對未使用的 sampling 參數發出警告。
TEMPERATURE = 1.0


# Top-p sampling。
#
# 模型不會從「所有 token」中隨機抽，
# 而是只考慮累積機率達到 top_p 的候選 token。
#
# 例如：
#
#     TOP_P = 0.9
#
# 表示保留累積機率約 90% 的候選 token。
#
# 本版本使用 greedy decoding，因此 top_p 不參與實際生成。
TOP_P = 1.0


# 是否啟用 sampling。
#
# True：
#     使用 temperature / top_p，
#     每次回答可能稍有不同。
#
# False：
#     通常接近 greedy decoding，
#     回答比較固定。
#
DO_SAMPLE = False


# 最多保留多少輪 user-assistant 對話。
#
# 一輪的意思是：
#
#     user
#     assistant
#
# 如果設成 4，最多保留最近 4 輪。
#
# 為什麼需要限制？
#
# 因為多輪對話越來越長時：
# - token 數會持續增加
# - 推論速度會變慢
# - GPU 記憶體需求會增加
# - 最後可能超過模型 context window
#
# 訓練語料最多 4 輪對話，因此保留最近 4 輪即可，
# 避免推論時出現訓練分布外的超長上下文。
MAX_HISTORY_TURNS = 4


# =============================================================================
# 3. 輔助函式：裁切過長的對話歷史
# =============================================================================

def trim_history(messages):
    """
    保留 system prompt，以及最近 MAX_HISTORY_TURNS 輪對話。

    messages 的格式例如：

        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "我要冰拿鐵"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "再加濃縮"},
            {"role": "assistant", "content": "..."},
        ]

    第一筆 system message 永遠保留。

    後面的 user / assistant message，
    最多保留：

        MAX_HISTORY_TURNS × 2

    筆。

    例如 MAX_HISTORY_TURNS = 4：

        10 個 user
        +
        10 個 assistant
        =
        最多 20 筆歷史 message
    """

    # 如果只有 system prompt，
    # 或歷史還沒有超過限制，就直接回傳。
    if len(messages) <= 1 + MAX_HISTORY_TURNS * 2:
        return messages

    # 第一筆是 system prompt。
    system_message = messages[0]

    # 從最後面取最近 N 輪。
    recent_messages = messages[-MAX_HISTORY_TURNS * 2:]

    # 重新組合：
    #
    # system
    # +
    # 最近幾輪對話
    return [system_message] + recent_messages


# =============================================================================
# 4. 輔助函式：產生單次 assistant 回覆
# =============================================================================

def generate_reply(model, tokenizer, messages):
    """
    把目前完整對話歷史送進模型，
    並回傳「這一輪 assistant 新產生的文字」。

    例如目前 messages：

        system
        user: 我想點冰拿鐵
        assistant: 好的
        user: 幫我加一份濃縮

    這個函式會把全部 context 送進模型，
    讓模型知道「加濃縮」是針對前面的冰拿鐵。
    """

    # -------------------------------------------------------------------------
    # Step 1：套用 Qwen 原生 chat template
    # -------------------------------------------------------------------------

    # apply_chat_template() 會把：
    #
    # [
    #   {"role": "system", ...},
    #   {"role": "user", ...}
    # ]
    #
    # 轉成 Qwen 真正訓練時使用的特殊格式。
    #
    # 概念上可能類似：
    #
    # <|im_start|>system
    # ...
    # <|im_end|>
    # <|im_start|>user
    # ...
    # <|im_end|>
    # <|im_start|>assistant
    #
    # 不應該自己手動拼接這些特殊 token，
    # 因為 tokenizer 已經知道正確格式。
    inputs = tokenizer.apply_chat_template(
        messages,

        # 在最後補上 assistant 開始回答的 generation prompt。
        add_generation_prompt=True,

        # 直接做 tokenization。
        tokenize=True,

        # 回傳 dict，例如：
        #
        # {
        #     "input_ids": ...,
        #     "attention_mask": ...
        # }
        return_dict=True,

        # 回傳 PyTorch tensor。
        return_tensors="pt",
    )

    # -------------------------------------------------------------------------
    # Step 2：把 input tensors 放到模型所在 GPU
    # -------------------------------------------------------------------------

    # AutoPeftModel 使用 device_map="auto" 後，
    # 這個 0.5B 模型在單張 4 GB GPU 上通常整個都會放到 cuda:0。
    #
    # model.device 會告訴我們主要模型所在裝置。
    inputs = inputs.to(model.device)

    # 記錄 prompt 本身用了多少 token。
    #
    # 假設：
    #
    # input_length = 150
    #
    # 模型最後輸出：
    #
    # [原本 150 tokens] + [新產生 30 tokens]
    #
    # 我們等一下只取後面那 30 個。
    input_length = inputs["input_ids"].shape[-1]

    # -------------------------------------------------------------------------
    # Step 3：執行推論
    # -------------------------------------------------------------------------

    # torch.inference_mode()：
    #
    # 告訴 PyTorch：
    #
    # 「現在只是 inference，不需要 gradient。」
    #
    # 好處：
    # - 降低記憶體使用量
    # - 減少額外 autograd overhead
    # - 推論速度較好
    with torch.inference_mode():

        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if DO_SAMPLE:
            generation_kwargs["temperature"] = TEMPERATURE
            generation_kwargs["top_p"] = TOP_P

        output = model.generate(**inputs, **generation_kwargs)

    # -------------------------------------------------------------------------
    # Step 4：只取新生成的 assistant tokens
    # -------------------------------------------------------------------------

    # generate() 回傳的 sequence 通常包含：
    #
    # 原始 prompt
    # +
    # 新產生內容
    #
    # 所以：
    #
    # output[0][input_length:]
    #
    # 表示只留下模型新生成的部分。
    new_tokens = output[0][input_length:]

    # -------------------------------------------------------------------------
    # Step 5：token IDs → 中文文字
    # -------------------------------------------------------------------------

    reply = tokenizer.decode(
        new_tokens,

        # 不把 <|im_end|> 等特殊 token 顯示給使用者。
        skip_special_tokens=True,
    ).strip()

    return reply


# =============================================================================
# 5. 主程式
# =============================================================================

def main():
    """
    啟動多輪互動式聊天。

    可用指令：

        exit
        quit
        q

            結束程式。

        /reset

            清除目前對話歷史，
            但保留 system prompt。

        /history

            顯示目前保存的完整 conversation history。

        /help

            顯示可用指令。
    """

    # -------------------------------------------------------------------------
    # Step 1：載入 tokenizer
    # -------------------------------------------------------------------------

    print("正在載入 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR
    )

    # 某些模型沒有 pad token。
    #
    # generation 時通常可以讓 pad_token = eos_token。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    # -------------------------------------------------------------------------
    # Step 2：載入 base model + LoRA adapter
    # -------------------------------------------------------------------------

    print("正在載入 Base Model + LoRA Adapter...")

    model = AutoPeftModelForCausalLM.from_pretrained(
        # LoRA adapter 目錄。
        ADAPTER_DIR,

        # 使用 FP16 載入模型。
        dtype=torch.float16,

        # 自動把模型放到可用 GPU。
        device_map="auto",
    )

    # evaluation mode：
    #
    # 關閉 dropout 等只有 training 才需要的行為。
    model.eval()
    model.config.use_cache = True


    # -------------------------------------------------------------------------
    # Step 3：初始化 conversation history
    # -------------------------------------------------------------------------

    # messages 就是整段多輪對話的「記憶」。
    #
    # 一開始只有 system prompt。
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]


    # -------------------------------------------------------------------------
    # Step 4：顯示操作說明
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("LoRA 多輪對話模式已啟動")
    print("=" * 72)
    print("輸入訊息後按 Enter 即可與模型對話。")
    print()
    print("可用指令：")
    print("  /reset    清除目前對話歷史")
    print("  /history  顯示目前對話歷史")
    print("  /help     顯示指令說明")
    print("  exit      結束程式")
    print("  quit      結束程式")
    print("  q         結束程式")
    print("=" * 72)


    # -------------------------------------------------------------------------
    # Step 5：進入多輪對話迴圈
    # -------------------------------------------------------------------------

    while True:

        # input() 會等待使用者從 terminal 輸入文字。
        #
        # strip()：
        # 去掉字串前後空白。
        user_text = input("\n你：").strip()

        # -------------------------------------------------------------
        # 空字串不送給模型
        # -------------------------------------------------------------

        if not user_text:
            continue


        # -------------------------------------------------------------
        # 結束程式
        # -------------------------------------------------------------

        if user_text.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("\n對話結束。")
            break


        # -------------------------------------------------------------
        # /help
        # -------------------------------------------------------------

        if user_text == "/help":
            print()
            print("可用指令：")
            print("  /reset    清除對話歷史")
            print("  /history  顯示目前對話歷史")
            print("  /help     顯示這份說明")
            print("  exit      結束程式")
            print("  quit      結束程式")
            print("  q         結束程式")
            continue


        # -------------------------------------------------------------
        # /reset
        # -------------------------------------------------------------

        if user_text == "/reset":

            # 清除 user / assistant history，
            # 只留下 system prompt。
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]

            print(
                "\n系統：已清除對話歷史，"
                "開始新的點餐對話。"
            )

            continue


        # -------------------------------------------------------------
        # /history
        # -------------------------------------------------------------

        if user_text == "/history":

            print("\n目前對話歷史：")
            print("-" * 72)

            for index, message in enumerate(
                messages,
                start=1,
            ):
                print(
                    f"[{index}] "
                    f"{message['role']}: "
                    f"{message['content']}"
                )

            print("-" * 72)

            continue


        # -------------------------------------------------------------
        # Step 6：把新的 user message 加進 conversation history
        # -------------------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )


        # -------------------------------------------------------------
        # Step 7：限制 history 長度
        # -------------------------------------------------------------

        messages = trim_history(messages)


        # -------------------------------------------------------------
        # Step 8：讓模型產生回答
        # -------------------------------------------------------------

        try:
            reply = generate_reply(
                model,
                tokenizer,
                messages,
            )

        except torch.cuda.OutOfMemoryError:
            # 如果真的遇到 GPU OOM，
            # 清掉 PyTorch CUDA allocator 中沒有使用的 cached blocks。
            torch.cuda.empty_cache()

            print(
                "\n系統：GPU 記憶體不足。"
                "可以先使用 /reset 清除較長的對話歷史，"
                "或降低 MAX_HISTORY_TURNS。"
            )

            # 因為這一輪沒有成功得到 assistant 回答，
            # 把剛才新增的 user message 移除，
            # 避免 history 出現 user message 卻沒有 assistant response。
            if (
                messages
                and messages[-1]["role"] == "user"
            ):
                messages.pop()

            continue


        # -------------------------------------------------------------
        # Step 9：把 assistant 回答加入 history
        # -------------------------------------------------------------

        messages.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )


        # -------------------------------------------------------------
        # Step 10：顯示回答
        # -------------------------------------------------------------

        print(f"\n助理：{reply}")


# =============================================================================
# 6. Python 程式進入點
# =============================================================================

# 只有直接執行：
#
#     python infer_lora_chat.py
#
# 才會進入 main()。
#
# 如果其他程式只是：
#
#     import infer_lora_chat
#
# 則不會自動開始互動。
if __name__ == "__main__":
    main()
