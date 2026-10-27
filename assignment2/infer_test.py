"""Generate a deterministic multi-turn sample with the final LoRA adapter."""

from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ADAPTER = "outputs/qwen25_05b_lora"
OUTPUT_FILE = Path("infer_test_out.txt")

SYSTEM = (
    "你是一個土味情話高手。\n"
    "你會以親切、溫和、俏皮但不油膩的口吻跟使用者互動。\n"
    "回答要自然、符合日常台灣/華語用法，避免過於露骨或不當內容。\n"
    "每次回應都要符合「土味情話」的風格，善用反差、關心與小心機的表達方式。"
)

SCENARIOS = [
    [
        "我昨天夢到你耶。",
        "真的嗎？夢到什麼？",
        "不告訴你。",
    ],
    [
        "這本書我找好久，原來在你手上。",
        "你畫了什麼重點？",
        "那你現在可以問了。",
    ],
    [
        "我剛剛看到一隻好大的娃娃，好想夾。",
        "你夾娃娃很強嗎？",
        "這樣很花錢耶。",
    ],
]

MAX_NEW_TOKENS = 100


def main():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = True

    lines = []

    for scenario_index, turns in enumerate(SCENARIOS, start=1):
        messages = [{"role": "system", "content": SYSTEM}]
        lines.append(f"===== 情境 {scenario_index} =====")

        for question in turns:
            messages.append({"role": "user", "content": question})
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
            }
            input_length = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            answer = tokenizer.decode(
                output[0][input_length:],
                skip_special_tokens=True,
            ).strip()
            messages.append({"role": "assistant", "content": answer})

            lines.append(f"你：{question}")
            lines.append(f"專家：{answer}")
            lines.append("-" * 60)

        lines.append("")

    text = "\n".join(lines)
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    print(text)
    print(f"已寫入 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
