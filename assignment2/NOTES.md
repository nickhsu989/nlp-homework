# Assignment 2 — Resume Notes

## Current status

- Final adapter currently copied from `checkpoint-105`.
- Retraining is still required for arbitrary new prompts.
- Do not treat the current adapter as final-quality.

## Main problem

The current data pipeline teaches disconnected replies:

- 296 of 300 conversations end on a user turn.
- Only the 4 seed conversations end with an assistant reply.
- Those 296 final user messages have no assistant target.
- `data/sft/train.jsonl` has 1,052 examples but only 272 unique assistant replies.
- `prepare_chat_data.py` truncates context to 6 non-system messages, dropping the opening and early context.
- `infer_lora_chat.py` samples at `temperature=0.7`, which makes the small model much less coherent.

## Required data fixes

1. Update `download_chat_data.py` so every conversation ends with an assistant message.
2. Ensure every user turn has a matching assistant target.
3. Use a consistent `open -> assistant -> follow-up -> assistant -> ...` structure.
4. Remove answers that do not answer the preceding user message.
5. Increase unique assistant replies from 272 toward at least 500.
6. Keep 300 conversations so the assignment requirement is still met.
7. Change `MAX_CONTEXT_MESSAGES` in `prepare_chat_data.py` to 8 or 10.
8. Rebuild `data/chat.jsonl` and `data/sft/{train,valid}.jsonl`.

## Evaluation

Create a fixed set of 12–15 unseen multi-turn scenarios, separate from training topics.

Evaluate with:

- `do_sample=False`
- `/reset` between scenarios
- `max_new_tokens=100`
- `MAX_HISTORY_TURNS=4` or `6`
- Optional `repetition_penalty=1.1` and `no_repeat_ngram_size=3`

Acceptance target:

- At least 80% of unseen scenarios produce coherent, on-topic replies.
- No exact repeated sentence across turns.
- No unrelated “I am an AI” responses.
- At least the first 3 turns remain contextually connected.

## Retraining plan

First attempt with `Qwen/Qwen2.5-0.5B-Instruct`:

- `EPOCHS = 3`
- `LORA_R = 16`
- `LORA_ALPHA = 32`
- Keep `LORA_DROPOUT = 0.10`
- Keep `LEARNING_RATE = 5e-5`
- Set `load_best_model_at_end=False`
- Disable early stopping or use a high patience
- Save every 20–30 steps with `CHECKPOINT_LIMIT=6` or more
- Select checkpoints manually with the unseen-prompt evaluation set

Do not select the checkpoint by `eval_loss` alone. The previous run showed that the lowest `eval_loss` checkpoint was not the best qualitative model.

Fallback only if 0.5B still fails:

- `Qwen/Qwen2.5-1.5B-Instruct`
- 4-bit NF4
- Batch size 1
- Gradient accumulation 16
- `MAX_LENGTH=384` or `512`
- Gradient checkpointing
- `paged_adamw_8bit`
- 4-bit inference to fit the 4 GB GTX 1050 Ti

## Finalization checklist

- Select the best checkpoint using the unseen evaluation set.
- Regenerate `infer_test_out.txt` with greedy decoding.
- Regenerate `loss_curve.png`.
- Update the root `README.md` with dataset composition, training config, evaluation results, and sample dialogue.
- Verify that the standalone assignment 2 folder contains only the final adapter, not checkpoints or backups.
