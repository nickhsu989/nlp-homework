"""
Build the cleaned multi-turn chat corpus from data/raw/batch_*.json.

The raw files use the ShareGPT schema:

    {"conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
    ]}

This script normalizes them to the OpenAI messages format used by
prepare_chat_data.py, removes malformed conversations, and creates a fixed
held-out evaluation set.
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
CHAT_FILE = HERE / "data" / "chat.jsonl"
EVAL_DIR = HERE / "data" / "eval"
HOLDOUT_FILE = EVAL_DIR / "heldout.jsonl"
MANIFEST_FILE = HERE / "data" / "clean_manifest.json"

SYSTEM_PROMPT = """你是一個土味情話高手。
你會以親切、溫和、俏皮但不油膩的口吻跟使用者互動。
回答要自然、符合日常台灣/華語用法，避免過於露骨或不當內容。
每次回應都要符合「土味情話」的風格，善用反差、關心與小心機的表達方式。"""

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}

MIN_USER_TURNS = 2
HOLDOUT_COUNT = 15
HOLDOUT_BY_TURNS = {2: 1, 3: 12, 4: 2}
SEED = 42

ALLOWED_LATIN = {"app", "atm", "type", "you"}
LATIN_RE = re.compile(r"[A-Za-z]{3,}")
BOPOMOFO_RE = re.compile(r"[\u3100-\u312f]")
META_RE = re.compile(
    r"(?<![A-Za-z])AI(?![A-Za-z])|ChatGPT|Claude|Gemini|人工智能|語言模型",
    re.IGNORECASE,
)


def read_raw_files():
    records = []

    files = sorted(RAW_DIR.glob("batch_*.json"))
    if not files:
        raise FileNotFoundError(f"No batch_*.json files found in {RAW_DIR}")

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path.name} must contain a JSON array")

        for index, item in enumerate(data):
            conversations = item.get("conversations")
            if not isinstance(conversations, list) or not conversations:
                records.append(
                    {
                        "file": path.name,
                        "index": index,
                        "conversations": None,
                        "reason": "missing_conversations",
                    }
                )
                continue

            records.append(
                {
                    "file": path.name,
                    "index": index,
                    "conversations": conversations,
                    "reason": "",
                }
            )

    return files, records


def normalize_conversation(raw_conversations):
    messages = []

    for item in raw_conversations:
        if not isinstance(item, dict):
            return None, "invalid_message"

        role = ROLE_MAP.get(item.get("from"))
        content = str(item.get("value", "")).strip()

        if role is None:
            return None, "unknown_role"
        if not content:
            return None, "empty_message"

        messages.append({"role": role, "content": content})

    if not messages:
        return None, "empty_conversation"
    if messages[0]["role"] != "system":
        return None, "missing_system"

    messages[0]["content"] = SYSTEM_PROMPT

    trimmed = 0
    while messages and messages[-1]["role"] == "user":
        messages.pop()
        trimmed += 1

    for index, message in enumerate(messages[1:], start=1):
        expected = "user" if index % 2 == 1 else "assistant"
        if message["role"] != expected:
            return None, "invalid_alternation"

    user_turns = sum(1 for message in messages if message["role"] == "user")

    if user_turns < MIN_USER_TURNS:
        return None, "less_than_2_user_turns"
    if messages[-1]["role"] != "assistant":
        return None, "does_not_end_with_assistant"

    return messages, {"trimmed_trailing_user": trimmed}


def quality_issue(messages):
    for message in messages:
        text = message["content"]

        if "\ufffd" in text:
            return "replacement_character"
        if text.count("「") != text.count("」"):
            return "unbalanced_double_quotes"
        if text.count("『") != text.count("』"):
            return "unbalanced_single_quotes"
        if BOPOMOFO_RE.search(text):
            return "bopomofo_text"
        if META_RE.search(text):
            return "meta_ai_reference"

        for token in LATIN_RE.findall(text):
            if token.lower() not in ALLOWED_LATIN:
                return f"unexpected_latin:{token}"

    return None


def summarize(messages_list):
    role_counts = Counter()
    assistant_texts = []
    user_texts = []
    turn_counts = Counter()
    message_lengths = []

    for messages in messages_list:
        role_counts.update(message["role"] for message in messages)
        turn_counts[sum(1 for m in messages if m["role"] == "user")] += 1
        message_lengths.append(len(messages))

        assistant_texts.extend(
            m["content"] for m in messages if m["role"] == "assistant"
        )
        user_texts.extend(
            m["content"] for m in messages if m["role"] == "user"
        )

    return {
        "conversations": len(messages_list),
        "messages": sum(message_lengths),
        "message_length_min": min(message_lengths) if message_lengths else 0,
        "message_length_max": max(message_lengths) if message_lengths else 0,
        "roles": dict(sorted(role_counts.items())),
        "turn_distribution": dict(sorted(turn_counts.items())),
        "unique_user_texts": len(set(user_texts)),
        "unique_assistant_texts": len(set(assistant_texts)),
    }


def choose_holdout(conversations):
    by_turns = defaultdict(list)

    for index, messages in enumerate(conversations):
        user_turns = sum(1 for m in messages if m["role"] == "user")
        by_turns[user_turns].append(index)

    rng = random.Random(SEED)
    selected = []

    for user_turns, desired in sorted(HOLDOUT_BY_TURNS.items()):
        pool = list(by_turns.get(user_turns, []))
        rng.shuffle(pool)
        selected.extend(pool[:desired])

    selected = set(selected)

    if len(selected) < HOLDOUT_COUNT:
        remaining = [
            index
            for index in range(len(conversations))
            if index not in selected
        ]
        rng.shuffle(remaining)
        selected.update(
            remaining[: HOLDOUT_COUNT - len(selected)]
        )

    return sorted(selected)[:HOLDOUT_COUNT]


def write_jsonl(path, conversations):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for messages in conversations:
            file.write(
                json.dumps(
                    {"messages": messages},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    files, records = read_raw_files()

    dropped = Counter()
    drop_details = []
    trimmed_total = 0
    cleaned = []
    cleaned_sources = []
    seen_conversations = set()
    seen_user_sequences = set()
    raw_system_prompts = Counter()

    for record in records:
        if record["reason"]:
            dropped[record["reason"]] += 1
            drop_details.append(
                {
                    "file": record["file"],
                    "index": record["index"],
                    "reason": record["reason"],
                }
            )
            continue

        raw_conversations = record["conversations"]
        first = raw_conversations[0]
        if first.get("from") == "system":
            raw_system_prompts[str(first.get("value", ""))] += 1

        messages, meta = normalize_conversation(raw_conversations)

        if messages is None:
            dropped[meta] += 1
            drop_details.append(
                {
                    "file": record["file"],
                    "index": record["index"],
                    "reason": meta,
                }
            )
            continue

        issue = quality_issue(messages)
        if issue:
            dropped[issue] += 1
            drop_details.append(
                {
                    "file": record["file"],
                    "index": record["index"],
                    "reason": issue,
                }
            )
            continue

        signature = tuple(
            (message["role"], message["content"])
            for message in messages
        )
        if signature in seen_conversations:
            dropped["duplicate_conversation"] += 1
            drop_details.append(
                {
                    "file": record["file"],
                    "index": record["index"],
                    "reason": "duplicate_conversation",
                }
            )
            continue

        user_signature = tuple(
            message["content"]
            for message in messages
            if message["role"] == "user"
        )
        if user_signature in seen_user_sequences:
            dropped["duplicate_user_sequence"] += 1
            drop_details.append(
                {
                    "file": record["file"],
                    "index": record["index"],
                    "reason": "duplicate_user_sequence",
                }
            )
            continue

        seen_conversations.add(signature)
        seen_user_sequences.add(user_signature)
        trimmed_total += meta["trimmed_trailing_user"]
        cleaned.append(messages)
        cleaned_sources.append(
            {
                "file": record["file"],
                "index": record["index"],
            }
        )

    if len(cleaned) <= HOLDOUT_COUNT:
        raise RuntimeError(
            f"Only {len(cleaned)} clean conversations; "
            f"need more than {HOLDOUT_COUNT}."
        )

    holdout_indexes = choose_holdout(cleaned)
    holdout_set = set(holdout_indexes)

    holdout = [cleaned[index] for index in holdout_indexes]
    chat = [
        messages
        for index, messages in enumerate(cleaned)
        if index not in holdout_set
    ]

    write_jsonl(CHAT_FILE, chat)
    write_jsonl(HOLDOUT_FILE, holdout)

    manifest = {
        "source": {
            "repository": "nickhsu989/nlp_hw",
            "path": "assignment2/data/raw",
            "batch_files": len(files),
        },
        "system_prompt": SYSTEM_PROMPT,
        "seed": SEED,
        "raw_conversations": len(records),
        "raw_system_prompt_variants": len(raw_system_prompts),
        "trimmed_trailing_user_turns": trimmed_total,
        "dropped": dict(sorted(dropped.items())),
        "drop_details": drop_details,
        "cleaned": summarize(cleaned),
        "heldout": summarize(holdout),
        "chat": summarize(chat),
        "heldout_sources": [
            cleaned_sources[index] for index in holdout_indexes
        ],
    }

    manifest["heldout"]["file"] = "data/eval/heldout.jsonl"
    manifest["chat"]["file"] = "data/chat.jsonl"

    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"raw batch files          : {len(files)}")
    print(f"raw conversations        : {len(records)}")
    print(f"raw system prompt variants: {len(raw_system_prompts)}")
    print(f"trimmed trailing user turns: {trimmed_total}")
    print(f"cleaned conversations    : {len(cleaned)}")
    print(f"held-out conversations   : {len(holdout)}")
    print(f"chat conversations       : {len(chat)}")
    print(f"chat turn distribution   : {manifest['chat']['turn_distribution']}")
    print(
        "chat unique assistant texts: "
        f"{manifest['chat']['unique_assistant_texts']}"
    )
    print(f"chat JSONL               : {CHAT_FILE}")
    print(f"held-out JSONL           : {HOLDOUT_FILE}")
    print(f"manifest                 : {MANIFEST_FILE}")

    if dropped:
        print("dropped:")
        for reason, count in sorted(dropped.items()):
            print(f"  {reason}: {count}")

    print("=" * 72)


if __name__ == "__main__":
    main()
