#!/usr/bin/env python3
"""Train a small Gemma 4 tool-trace LoRA from the correction set.

This is the local/MPS-friendly counterpart to the Kaggle trainer. It expects
Transformers 5.x so Gemma 4 and its native tool-call chat template are
available. The loss is masked to assistant/model turns, including tool-call
turns and the final JSON turn.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_MARKER = "<|turn>model\n"
TURN_END = "<turn|>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def find_span(ids: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    last = len(ids) - len(needle)
    for idx in range(start, last + 1):
        if ids[idx : idx + len(needle)] == needle:
            return idx
    return -1


def choose_lora_target_regex(model: Any) -> str:
    primary = (
        r".*\.language_model\.layers\.([0-9]|1[0-4])\.self_attn\.(q|k|v|o)_proj(\.linear)?$"
        r"|.*\.language_model\.layers\.(1[5-9]|2[0-9]|3[0-4])\.self_attn\.(q|o)_proj(\.linear)?$"
        r"|.*\.language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj(\.linear)?$"
    )
    hf = (
        r".*\.language_model\.model\.layers\.([0-9]|1[0-4])\.self_attn\.(q|k|v|o)_proj(\.linear)?$"
        r"|.*\.language_model\.model\.layers\.(1[5-9]|2[0-9]|3[0-4])\.self_attn\.(q|o)_proj(\.linear)?$"
        r"|.*\.language_model\.model\.layers\.\d+\.mlp\.(gate|up|down)_proj(\.linear)?$"
    )
    plain = (
        r".*\.layers\.([0-9]|1[0-4])\.self_attn\.(q|k|v|o)_proj(\.linear)?$"
        r"|.*\.layers\.(1[5-9]|2[0-9]|3[0-4])\.self_attn\.(q|o)_proj(\.linear)?$"
        r"|.*\.layers\.\d+\.mlp\.(gate|up|down)_proj(\.linear)?$"
    )
    candidates = [primary, hf, plain]
    counts = [sum(1 for name, _ in model.named_modules() if re.match(regex, name)) for regex in candidates]
    best_idx = max(range(len(candidates)), key=counts.__getitem__)
    print(f"LoRA target regex matches: primary={counts[0]} hf={counts[1]} plain={counts[2]}")
    if counts[best_idx] == 0:
        raise RuntimeError("No LoRA target modules matched Gemma 4 language layers")
    return candidates[best_idx]


def unwrap_clippable_linears(model: Any) -> int:
    """Replace Gemma4ClippableLinear wrappers with their inner nn.Linear.

    PEFT can attach LoRA to torch.nn.Linear, but not to the Gemma 4 wrapper
    class. Transformers uses these wrappers for logit clipping; for our LoRA
    target modules, replacing the wrapper with its contained linear layer
    matches the existing Kaggle training path.
    """
    replaced = 0
    for module_name, module in list(model.named_modules()):
        if type(module).__name__ != "Gemma4ClippableLinear":
            continue
        parent = model
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.linear)
        replaced += 1
    return replaced


@dataclass
class ToolTraceCollator:
    tokenizer: Any
    max_length: int

    def __post_init__(self) -> None:
        self.model_marker_ids = self.tokenizer.encode(MODEL_MARKER, add_special_tokens=False)
        self.turn_end_ids = self.tokenizer.encode(TURN_END, add_special_tokens=False)
        if not self.model_marker_ids or not self.turn_end_ids:
            raise ValueError("Could not encode Gemma 4 model-turn markers")

    def render(self, conversation: list[dict[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

    def encode(self, row: dict[str, Any]) -> dict[str, list[int]]:
        text = self.render(row["conversations"])
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        ids = list(encoded["input_ids"])
        labels = [-100] * len(ids)
        cursor = 0
        while True:
            start = find_span(ids, self.model_marker_ids, cursor)
            if start < 0:
                break
            content_start = start + len(self.model_marker_ids)
            end = find_span(ids, self.turn_end_ids, content_start)
            if end < 0:
                end = len(ids)
            for pos in range(content_start, end):
                labels[pos] = ids[pos]
            cursor = end + max(1, len(self.turn_end_ids))
        if all(label == -100 for label in labels):
            raise ValueError("Rendered row has no trainable model tokens")
        return {
            "input_ids": ids,
            "attention_mask": list(encoded["attention_mask"]),
            "labels": labels,
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = [self.encode(feature) for feature in features]
        max_len = max(len(item["input_ids"]) for item in encoded)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for item in encoded:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        default="build/llm_required/generated/gold_tool_trace_synth200_corrections/gold-tool-trace-synth200-corrections-train.jsonl",
    )
    parser.add_argument(
        "--val-path",
        default="build/llm_required/generated/gold_tool_trace_synth200_corrections/gold-tool-trace-synth200-corrections-val.jsonl",
    )
    parser.add_argument(
        "--model",
        default="/Users/thinkstudio/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it/snapshots/f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae",
    )
    parser.add_argument("--output-dir", default="build/local_tool_trace_lora")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_rows = read_jsonl(args.train_path)
    val_rows = read_jsonl(args.val_path)
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples:
        val_rows = val_rows[: args.max_val_samples]
    print(f"Rows: train={len(train_rows)} val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.model_max_length = args.max_length
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = ToolTraceCollator(tokenizer=tokenizer, max_length=args.max_length)
    sample = collator.encode(train_rows[0])
    trainable = sum(1 for label in sample["labels"] if label != -100)
    print(f"Sample tokens: total={len(sample['input_ids'])} trainable={trainable}")
    if args.dry_run:
        return 0

    dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32
    device_map: str | dict[str, str] = {"": "mps"} if torch.backends.mps.is_available() else "auto"
    print(f"Loading model: dtype={dtype} device_map={device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    replaced = unwrap_clippable_linears(model)
    print(f"Unwrapped Gemma4ClippableLinear modules: {replaced}")
    target_regex = choose_lora_target_regex(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=target_regex,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=2,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        bf16=False,
        fp16=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        data_collator=collator,
    )
    stats = trainer.train()
    print(f"Training metrics: {stats.metrics}")

    lora_dir = output_dir / "cliniq-tool-trace-local-lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"Saved LoRA: {lora_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
