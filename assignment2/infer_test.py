import sys, io
sys.stdout = io.TextIOWrapper(open("infer_test_out.txt","wb"), encoding="utf-8")
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ADAPTER = "outputs/qwen25_05b_lora"
SYSTEM = "你是一個情話高手，你會以親切、溫和的口吻來回答使用者的問題。"

tok = AutoTokenizer.from_pretrained(ADAPTER)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER, dtype=torch.float16, device_map="auto")
model.eval()

msgs = [{"role":"system","content":SYSTEM}]
turns = ["教我如何透過長相來認識女性朋友。","長得像你朋友？","那如果她問「你上一任女朋友呢？」","那如果她說「所以很多人報名嗎？」"]
for q in turns:
    msgs.append({"role":"user","content":q})
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inp = enc["input_ids"].to(model.device)
    with torch.inference_mode():
        out = model.generate(inp, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
    ans = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
    msgs.append({"role":"assistant","content":ans})
    print(f"你：{q}")
    print(f"專家：{ans}")
    print("-"*60)
sys.stdout.flush()
