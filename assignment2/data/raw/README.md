# Raw conversation data

## Source

Imported with permission from the private repository:

- Repository: `nickhsu989/nlp_hw`
- Path: `assignment2/data/raw/`

The corpus contains multi-turn 「土味情話」 dialogues.

## Files

- `batch_01.json` through `batch_14.json`
- `manifest.json`: file sizes and SHA256 checksums

All batches use the ShareGPT-style format:

```json
{
  "conversations": [
    {"from": "system", "value": "..."},
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

## Corpus summary

- 14 batch files
- 403 raw conversations
- 1 consistent system prompt
- 2–4 user turns per conversation before cleaning

## Cleaning

`build_chat_data.py` enforces:

- canonical system prompt
- alternating `human` / `gpt` turns
- non-empty messages
- at least 2 user turns
- final assistant turn
- no duplicate conversations
- basic quality checks for corrupted text

Cleaning drops trailing human turns and clearly malformed conversations.
The cleaned training corpus and the 15-conversation held-out evaluation set
are written under `data/` by the same script.
