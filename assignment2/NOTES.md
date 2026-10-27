# Assignment 2 — Experiment Notes

## Current status

- Training completed: 3 epochs, 189 optimizer steps, ~29 minutes.
- Final adapter: `outputs/qwen25_05b_lora` (identical to `checkpoint-189`).
- Selected checkpoint: `checkpoint-189`.
- Intermediate checkpoints were removed after selection.

## Data

- Raw: 14 batches, 403 conversations, 1 system prompt.
- Clean: 388 conversations (2–4 user turns).
- Held-out: 15 conversations, stratified by turn count, seed 42.
- Chat/train split: 373 conversations → 1,003 train / 109 validation SFT examples.
- Cleaning details: `data/clean_manifest.json`.

## Training results

- Final eval loss: `2.7012`
- Final eval token accuracy: `48.8%`
- Average training loss: `2.9993`
- Eval loss by step:

| Step | Eval loss |
|---:|---:|
| 30 | 3.0868 |
| 60 | 2.8555 |
| 90 | 2.7705 |
| 120 | 2.7242 |
| 150 | 2.7046 |
| 180 | 2.7027 |
| 189 | 2.7012 |

## Checkpoint evaluation

- Evaluated `30, 60, 90, 120, 150, 180, 189` plus the base model.
- Method: 15 held-out conversations, canonical system prompt, greedy decoding,
  `max_new_tokens=100`, no reference-answer leakage.
- `checkpoint-189` had the lowest eval loss.
- `checkpoint-120` had a slightly higher reference similarity, but manual review
  favored the later checkpoint's more complete, on-style multi-turn replies.
- Evaluation metrics are summarized in `README.md`; raw transcripts were generated
  during checkpoint selection and were not kept in the repository.

## Inference

- `infer_lora_chat.py`: interactive multi-turn chat, greedy decoding.
- `infer_lora.py`: single-turn demo.
- `infer_test.py`: deterministic 3-scenario demo; writes `infer_test_out.txt`.
- All scripts use the canonical 4-line system prompt.
- `MAX_NEW_TOKENS=100`, `MAX_HISTORY_TURNS=4`.

## Final artifacts

Keep:

- `outputs/qwen25_05b_lora/adapter_config.json`
- `outputs/qwen25_05b_lora/adapter_model.safetensors`
- tokenizer files (`tokenizer.json`, `tokenizer_config.json`) and `chat_template.jinja`
- `trainer_state.json`
- `training_args.bin`
- `loss_curve.png`

Removed:

- `outputs/qwen25_05b_lora/checkpoint-*` (final adapter is checkpoint-189)
- `outputs/qwen25_05b_lora/README.md` (auto-generated placeholder)

## Known limitations

- 0.5B capacity causes occasional incoherent or off-topic replies.
- Occasional simplified Chinese characters.
- Not suitable for production or sensitive use.
