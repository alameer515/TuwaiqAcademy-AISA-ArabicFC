#!/usr/bin/env python3
"""Convert AISA Islamic subset to Gemma 3 SFT JSONL (Track A / Track B).

For a step-by-step walkthrough, see format_for_sft.ipynb.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

MODEL_TURN = "<start_of_turn>model\n"
END_CALL = "<end_function_call>"
THINK_BLOCK_RE = re.compile(
    r"<think>\s*.*?\s*</think>\s*",
    re.DOTALL,
)

ISLAMIC_TOOLS = {
    "get_qibla_direction",
    "calculate_zakat",
    "search_quran",
    "calculate_inheritance",
}


def clean_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    if not arguments:
        return {}
    return {key: value for key, value in arguments.items() if value is not None}


def get_user_message(example: dict[str, Any]) -> str:
    for message in example["messages"]:
        if message["role"] == "user":
            return message.get("content", "")
    return ""


def get_assistant_fields(example: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    think = ""
    tool_name = example.get("tool_called", "none")
    arguments: dict[str, Any] = {}

    for message in example["messages"]:
        if message["role"] != "assistant":
            continue
        think = message.get("think") or message.get("_think_for_train") or ""
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            function = tool_calls[0]["function"]
            tool_name = function["name"]
            arguments = clean_arguments(function.get("arguments"))
        break

    return think, tool_name, arguments


def split_prompt_and_completion(text: str) -> tuple[str, str]:
    if MODEL_TURN not in text:
        raise ValueError("Missing model turn marker in text field.")
    prompt, completion = text.split(MODEL_TURN, 1)
    prompt = prompt + MODEL_TURN
    return prompt, trim_completion(completion)


def trim_completion(completion: str) -> str:
    if END_CALL in completion:
        end_index = completion.index(END_CALL) + len(END_CALL)
        return completion[:end_index]
    return completion.rstrip()


def to_track_a_completion(completion: str) -> str:
    stripped = THINK_BLOCK_RE.sub("", completion).strip()
    return stripped


def format_record(
    example: dict[str, Any],
    example_id: int,
    track: str,
) -> dict[str, Any]:
    prompt, completion_b = split_prompt_and_completion(example["text"])
    completion = completion_b if track.upper() == "B" else to_track_a_completion(completion_b)

    think, tool_name, arguments = get_assistant_fields(example)
    tools_sampled = [
        tool["function"]["name"] for tool in example.get("tools_sampled", [])
    ]

    return {
        "id": example_id,
        "track": track.upper(),
        "prompt": prompt,
        "completion": completion,
        "text": prompt + completion,
        "tool_called": tool_name,
        "arguments": arguments,
        "think": think,
        "requires_function": example.get("requires_function", tool_name != "none"),
        "dialect": example.get("dialect"),
        "user": get_user_message(example),
        "tools_sampled": tools_sampled,
        "negative_category": example.get("negative_category"),
    }


def format_eval_record(example: dict[str, Any], example_id: int) -> dict[str, Any]:
    prompt, _ = split_prompt_and_completion(example["text"])
    think, tool_name, arguments = get_assistant_fields(example)
    tools_sampled = [
        tool["function"]["name"] for tool in example.get("tools_sampled", [])
    ]

    return {
        "id": example_id,
        "prompt": prompt,
        "user": get_user_message(example),
        "tool_called": tool_name,
        "arguments": arguments,
        "think": think,
        "requires_function": example.get("requires_function", tool_name != "none"),
        "dialect": example.get("dialect"),
        "tools_sampled": tools_sampled,
        "negative_category": example.get("negative_category"),
    }


def load_split(path: Path, split_name: str) -> Dataset:
    if path.exists():
        return load_from_disk(str(path))

    dataset = load_dataset("TuwaiqAcademy/AISA-ArabicFC")
    split = dataset["train" if split_name == "train" else "dev"]
    return split.filter(lambda example: example["tool_called"] in ISLAMIC_TOOLS)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_examples": len(records),
        "tool_counts": dict(Counter(record["tool_called"] for record in records)),
        "dialect_counts": dict(Counter(record["dialect"] for record in records)),
    }


def validate_records(records: list[dict[str, Any]], track: str) -> None:
    for record in records:
        if not record["prompt"].endswith(MODEL_TURN):
            raise ValueError(f"Prompt missing model turn for id={record['id']}")
        if track.upper() == "A" and "<think>" in record["completion"]:
            raise ValueError(f"Track A completion still has think block for id={record['id']}")
        if track.upper() == "B" and record["requires_function"]:
            if "<start_function_call>" not in record["completion"]:
                raise ValueError(f"Track B completion missing function call for id={record['id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("aisa_islamic_train"),
        help="Path to filtered train dataset on disk.",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=Path("aisa_islamic_dev"),
        help="Path to filtered dev dataset on disk.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("formatted"),
        help="Directory for JSONL outputs.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of preview examples to save.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    train_dir = (root / args.train_dir).resolve()
    dev_dir = (root / args.dev_dir).resolve()
    output_dir = (root / args.output_dir).resolve()

    train_ds = load_split(train_dir, "train")
    dev_ds = load_split(dev_dir, "dev")

    train_track_b = [
        format_record(example, idx, "B") for idx, example in enumerate(train_ds)
    ]
    train_track_a = [
        format_record(example, idx, "A") for idx, example in enumerate(train_ds)
    ]
    dev_eval = [
        format_eval_record(example, idx) for idx, example in enumerate(dev_ds)
    ]

    validate_records(train_track_b, "B")
    validate_records(train_track_a, "A")

    write_jsonl(output_dir / "islamic_train_track_b.jsonl", train_track_b)
    write_jsonl(output_dir / "islamic_train_track_a.jsonl", train_track_a)
    write_jsonl(output_dir / "islamic_dev_eval.jsonl", dev_eval)

    stats = {
        "train_track_b": build_stats(train_track_b),
        "train_track_a": build_stats(train_track_a),
        "dev_eval": build_stats(dev_eval),
        "notes": [
            "Track B includes <think> before the function call.",
            "Track A keeps only the Gemma function-call block.",
            "calculate_inheritance has 0 gold examples in the source dataset.",
        ],
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    preview = {
        "track_b": train_track_b[: args.preview],
        "track_a": train_track_a[: args.preview],
        "dev_eval": dev_eval[: args.preview],
    }
    with (output_dir / "samples_preview.json").open("w", encoding="utf-8") as handle:
        json.dump(preview, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train_track_b)} train examples (Track B) -> {output_dir}")
    print(f"Wrote {len(train_track_a)} train examples (Track A) -> {output_dir}")
    print(f"Wrote {len(dev_eval)} dev eval examples -> {output_dir}")
    print("Train tool counts:", stats["train_track_b"]["tool_counts"])


if __name__ == "__main__":
    main()
