# Islamic SFT Format (Member 2)

This folder contains Gemma 3 / AISA function-calling training files built from
`aisa_islamic_train` and `aisa_islamic_dev`.

## Generate files

**Notebook (recommended for learning):** open and run `format_for_sft.ipynb` cell by cell.

**Script (quick CLI run):**

```bash
python format_for_sft.py
```

Outputs go to `formatted/`:

| File | Purpose |
|------|---------|
| `islamic_train_track_b.jsonl` | Training with Arabic `<think>` + tool call |
| `islamic_train_track_a.jsonl` | Training with tool call only |
| `islamic_dev_eval.jsonl` | Dev prompts + gold labels for evaluation |
| `stats.json` | Counts per tool and dialect |
| `samples_preview.json` | First 3 examples per split |

## Prompt / completion split

Each source row already has a ready-made `text` field. The script splits on:

```
<start_of_turn>model\n
```

- **prompt** = everything up to and including the model turn marker
- **completion** = model output, trimmed at `<end_function_call>`

## Track B completion

```
<think>
... Arabic reasoning ...
</think>
<start_function_call>call:TOOL_NAME{arg:<escape>value<escape>}<end_function_call>
```

## Track A completion

Same as Track B, but without the `<think>` block.

## JSONL fields

### Training (`islamic_train_track_*.jsonl`)

| Field | Description |
|-------|-------------|
| `id` | Row index |
| `track` | `A` or `B` |
| `prompt` | Input for causal LM training |
| `completion` | Target the model should learn |
| `text` | `prompt + completion` |
| `tool_called` | Gold function name |
| `arguments` | Gold arguments (nulls removed) |
| `think` | Gold reasoning text |
| `dialect` | `msa`, `gulf`, `egyptian`, `levantine`, `maghrebi` |
| `tools_sampled` | 4 candidate tools shown in the prompt |

### Evaluation (`islamic_dev_eval.jsonl`)

| Field | Description |
|-------|-------------|
| `prompt` | Input only (no gold completion) |
| `tool_called` | Gold function name |
| `arguments` | Gold arguments |
| `think` | Gold reasoning trace |

## Fine-tuning usage

**Member 3** can train with either:

1. `text` — single concatenated string (standard causal LM SFT)
2. `prompt` + `completion` — if the trainer masks loss on the prompt only

Recommended base model: `google/gemma-3-270m` or baseline
`TuwaiqAcademy/AISA-AR-FunctionCall-Think`.

## Evaluation usage

**Member 4** should:

1. Run inference on `dev_eval.prompt`
2. Parse output with the shared-task Gemma format
3. Compare predicted `tool_called` and `arguments` to gold fields

## Notes

- Current Islamic subset has **3 active tools**: `get_qibla_direction`,
  `calculate_zakat`, `search_quran`.
- `calculate_inheritance` is defined in the task schema but has **0 gold examples**
  in train/dev.
