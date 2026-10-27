"""
Evaluate LoRA checkpoints on the fixed held-out multi-turn conversations.

The evaluation is intentionally generation-based: each model sees the original
user turns, its own previous replies, and the canonical system prompt. Reference
assistant replies are used only for scoring, never as model context.

Outputs are written outside the repository by default so repeated experiments
do not clutter the assignment folder.
"""

import argparse
import gc
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
HOLDOUT_FILE = HERE / "data" / "eval" / "heldout.jsonl"
OUTPUT_ROOT = Path(
    r"C:\Users\student\AppData\Local\Temp\opencode\assignment2_eval"
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = """你是一個土味情話高手。
你會以親切、溫和、俏皮但不油膩的口吻跟使用者互動。
回答要自然、符合日常台灣/華語用法，避免過於露骨或不當內容。
每次回應都要符合「土味情話」的風格，善用反差、關心與小心機的表達方式。"""

DEFAULT_CHECKPOINTS = [30, 60, 90, 120, 150, 180, 189]
MAX_NEW_TOKENS = 100


def read_heldout():
    rows = []
    for line in HOLDOUT_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line)["messages"])
    if not rows:
        raise RuntimeError(f"No held-out rows in {HOLDOUT_FILE}")
    return rows


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path):
    if model_path is None:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
        )
        tokenizer_path = BASE_MODEL
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
        )
        tokenizer_path = str(model_path)

    model.eval()
    model.config.use_cache = True
    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer


def generate_reply(model, tokenizer, history, max_new_tokens):
    inputs = tokenizer.apply_chat_template(
        history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_length:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    ended_with_eos = bool(
        new_tokens and new_tokens[-1] == tokenizer.eos_token_id
    )
    return text, len(new_tokens), ended_with_eos


def score_model(model, tokenizer, heldout, max_new_tokens):
    conversations = []
    generated_texts = []

    for conversation_index, messages in enumerate(heldout):
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        turns = []

        for message_index, message in enumerate(messages[1:], start=1):
            if message["role"] != "user":
                continue

            reference = ""
            if (
                message_index + 1 < len(messages)
                and messages[message_index + 1]["role"] == "assistant"
            ):
                reference = messages[message_index + 1]["content"]

            history.append(
                {"role": "user", "content": message["content"]}
            )
            answer, token_count, ended_with_eos = generate_reply(
                model,
                tokenizer,
                history,
                max_new_tokens,
            )
            history.append({"role": "assistant", "content": answer})

            generated_texts.append(answer)
            turns.append(
                {
                    "user": message["content"],
                    "reference": reference,
                    "generated": answer,
                    "chars": len(answer),
                    "tokens": token_count,
                    "ended_with_eos": ended_with_eos,
                    "truncated": (
                        token_count >= max_new_tokens and not ended_with_eos
                    ),
                    "reference_similarity": (
                        SequenceMatcher(None, answer, reference).ratio()
                        if reference
                        else 0.0
                    ),
                }
            )

        conversations.append(
            {
                "index": conversation_index,
                "turns": turns,
                "history": history,
            }
        )
        print(
            f"  conversation {conversation_index + 1}/{len(heldout)} "
            f"completed ({len(turns)} turns)",
            flush=True,
        )

    return summarize(conversations, generated_texts), conversations


def summarize(conversations, generated_texts):
    turns = [turn for conv in conversations for turn in conv["turns"]]
    total_chars = sum(turn["chars"] for turn in turns)
    total_tokens = sum(turn["tokens"] for turn in turns)
    repeated = len(generated_texts) - len(set(generated_texts))
    empty = sum(1 for turn in turns if not turn["generated"])
    truncated = sum(1 for turn in turns if turn["truncated"])
    eos = sum(1 for turn in turns if turn["ended_with_eos"])
    similarities = [
        turn["reference_similarity"]
        for turn in turns
        if turn["reference"]
    ]

    return {
        "conversations": len(conversations),
        "turns": len(turns),
        "empty_replies": empty,
        "truncated_replies": truncated,
        "eos_rate": round(eos / len(turns), 4) if turns else 0.0,
        "exact_duplicate_rate": (
            round(repeated / len(generated_texts), 4)
            if generated_texts
            else 0.0
        ),
        "avg_chars": round(total_chars / len(turns), 2) if turns else 0.0,
        "avg_tokens": round(total_tokens / len(turns), 2) if turns else 0.0,
        "avg_reference_similarity": (
            round(sum(similarities) / len(similarities), 4)
            if similarities
            else 0.0
        ),
    }


def write_outputs(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        name: {
            "metrics": result["metrics"],
            "conversations": result["conversations"],
        }
        for name, result in results.items()
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "transcripts.jsonl").open("w", encoding="utf-8") as file:
        for name, result in results.items():
            for conversation in result["conversations"]:
                file.write(
                    json.dumps(
                        {
                            "model": name,
                            "conversation": conversation,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    lines = ["# Checkpoint review", ""]
    for name, result in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(result["metrics"], ensure_ascii=False, indent=2)
        )
        lines.append("```")
        lines.append("")

        for conversation in result["conversations"]:
            lines.append(f"### Conversation {conversation['index'] + 1}")
            for turn in conversation["turns"]:
                lines.append(f"**User:** {turn['user']}")
                lines.append(f"**Model:** {turn['generated']}")
                lines.append(f"**Reference:** {turn['reference']}")
                lines.append("")
        lines.append("")

    (output_dir / "review.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        default=",".join(str(value) for value in DEFAULT_CHECKPOINTS),
        help="Comma-separated checkpoint steps.",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate the base model without an adapter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit held-out conversations for a smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    heldout = read_heldout()
    if args.limit is not None:
        heldout = heldout[: args.limit]

    checkpoints = [
        int(value.strip())
        for value in args.checkpoints.split(",")
        if value.strip()
    ]

    targets = []
    if args.include_base:
        targets.append(("base", None))
    for step in checkpoints:
        targets.append(
            (
                f"checkpoint-{step}",
                HERE / "outputs" / "qwen25_05b_lora" / f"checkpoint-{step}",
            )
        )

    results = {}

    for name, model_path in targets:
        if model_path is not None and not model_path.is_dir():
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")

        print("=" * 72, flush=True)
        print(f"Evaluating {name}", flush=True)
        print(f"Path: {model_path or BASE_MODEL}", flush=True)

        started = time.time()
        model, tokenizer = load_model(model_path)
        metrics, conversations = score_model(
            model,
            tokenizer,
            heldout,
            args.max_new_tokens,
        )
        metrics["seconds"] = round(time.time() - started, 2)
        results[name] = {
            "metrics": metrics,
            "conversations": conversations,
        }

        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    write_outputs(results, output_dir)

    print("=" * 72)
    print(f"Wrote results to {output_dir}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"Review:  {output_dir / 'review.md'}")


if __name__ == "__main__":
    main()
