#!/usr/bin/env python3
"""ClinIQ Gemma 4 E2B tool-trace LoRA training.

This kernel trains the agent/tool-use policy, not the legacy compact
single-shot JSON path. Rows are OpenAI-style conversations containing
assistant tool calls, tool results, and a final extraction JSON.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_MARKER = "<|turn>model\n"
TURN_END = "<turn|>\n"


def compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=False)


def message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text") or ""))
                elif "text" in item:
                    chunks.append(str(item.get("text") or ""))
            elif item is not None:
                chunks.append(str(item))
        return "\n".join(chunks)
    if content is None:
        return ""
    return str(content)


def render_tool_call(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    raw_args = function.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = raw_args
    else:
        args = raw_args
    payload = {
        "name": function.get("name"),
        "arguments": args,
    }
    return f"<tool_call>{compact_json(payload)}</tool_call>"


def render_conversation(conversation: list[dict[str, Any]]) -> str:
    """Render OpenAI-style tool conversations into Gemma 4 turn text.

    The base model/server already uses Gemma 4 turn delimiters. We keep those
    delimiters and serialize tool calls/results explicitly so the response
    mask can train every assistant decision: lookup calls, validation calls,
    and final JSON.
    """
    out: list[str] = []
    for message in conversation:
        role = str(message.get("role") or "user")
        if role == "assistant":
            role = "model"
        if role == "model" and message.get("tool_calls"):
            content = "\n".join(
                render_tool_call(call)
                for call in (message.get("tool_calls") or [])
                if isinstance(call, dict)
            )
        elif role == "tool":
            raw = message_content(message)
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
            content = compact_json({
                "tool_call_id": message.get("tool_call_id"),
                "name": message.get("name"),
                "content": parsed,
            })
        else:
            content = message_content(message)
        out.append(f"<|turn>{role}\n{content.strip()}{TURN_END}")
    return "".join(out)


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def find_input(patterns: list[str], explicit: str | None) -> str:
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(explicit)
        return explicit
    for pattern in patterns:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No input matched: {patterns}")


def install_training_deps() -> None:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers>=5.5",
        "peft>=0.15",
        "trl>=0.15",
        "bitsandbytes",
        "sentencepiece",
        "datasets",
        "git+https://github.com/huggingface/peft.git",
    ])


def validate_rendered(train_path: str, val_path: str) -> None:
    for label, path in (("train", train_path), ("val", val_path)):
        rows = load_rows(path)
        if not rows:
            raise ValueError(f"{path} has no rows")
        lengths = []
        assistant_turns = []
        for row in rows:
            conv = row.get("conversations")
            if not isinstance(conv, list):
                raise ValueError(f"{path} row missing conversations")
            text = render_conversation(conv)
            if "<tool_call>" not in text:
                raise ValueError(f"{path} row rendered without tool_call")
            if not text.rstrip().endswith(TURN_END.strip()):
                raise ValueError(f"{path} row missing final turn end")
            lengths.append(len(text))
            assistant_turns.append(text.count(MODEL_MARKER))
        lengths_sorted = sorted(lengths)
        p95 = lengths_sorted[int(0.95 * (len(lengths_sorted) - 1))]
        print(
            f"{label}: rows={len(rows)} chars_p50={lengths_sorted[len(lengths_sorted)//2]} "
            f"chars_p95={p95} chars_max={max(lengths)} "
            f"assistant_turns_max={max(assistant_turns)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=os.environ.get("TRAIN_PATH"))
    parser.add_argument("--val-path", default=os.environ.get("VAL_PATH"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/kaggle/working"))
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", "google/gemma-4-E2B-it"))
    parser.add_argument("--epochs", type=float, default=float(os.environ.get("EPOCHS", "4")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "5e-5")))
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "16")))
    parser.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "4")))
    parser.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "4096")))
    parser.add_argument("--dry-run-render", action="store_true")
    args = parser.parse_args()

    train_path = find_input(
        [
            "/kaggle/input/**/gold-tool-trace-synth200-corrections-train.jsonl",
            "/kaggle/input/**/gold-tool-trace-synth200-train.jsonl",
            "build/llm_required/generated/gold_tool_trace_synth200_corrections/gold-tool-trace-synth200-corrections-train.jsonl",
        ],
        args.train_path,
    )
    val_path = find_input(
        [
            "/kaggle/input/**/gold-tool-trace-synth200-corrections-val.jsonl",
            "/kaggle/input/**/gold-tool-trace-synth200-val.jsonl",
            "build/llm_required/generated/gold_tool_trace_synth200_corrections/gold-tool-trace-synth200-corrections-val.jsonl",
        ],
        args.val_path,
    )
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    validate_rendered(train_path, val_path)
    if args.dry_run_render:
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    install_training_deps()

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this training run.")
    cc = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} (sm_{cc[0]}{cc[1]})")
    if cc < (7, 0):
        raise SystemExit(f"{name} is not supported; use Kaggle T4 or better.")

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers import DataCollatorForLanguageModeling
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_seq_len
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    for module_name, module in list(model.named_modules()):
        if type(module).__name__ == "Gemma4ClippableLinear":
            parts = module_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module.linear)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    target_regex = (
        r".*\.language_model\.layers\.([0-9]|1[0-4])\.self_attn\.(q|k|v|o)_proj$"
        r"|.*\.language_model\.layers\.(1[5-9]|2[0-9]|3[0-4])\.self_attn\.(q|o)_proj$"
        r"|.*\.language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj$"
    )
    target_regex_alt = (
        r".*layers\.([0-9]|1[0-4])\.self_attn\.(q|k|v|o)_proj$"
        r"|.*layers\.(1[5-9]|2[0-9]|3[0-4])\.self_attn\.(q|o)_proj$"
        r"|.*layers\.\d+\.mlp\.(gate|up|down)_proj$"
    )
    n_primary = sum(1 for n, _ in model.named_modules() if re.match(target_regex, n))
    n_alt = sum(1 for n, _ in model.named_modules() if re.match(target_regex_alt, n))
    chosen_regex = target_regex if n_primary >= n_alt else target_regex_alt
    print(f"LoRA target matches: primary={n_primary}, alt={n_alt}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=chosen_regex,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )

    def fmt(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        return {
            "text": [
                render_conversation(conversation)
                for conversation in examples["conversations"]
            ]
        }

    dataset = dataset.map(fmt, batched=True, remove_columns=["conversations"])
    print(f"Train rows: {len(dataset['train'])} | Val rows: {len(dataset['validation'])}")

    class DataCollatorForAssistantTurns(DataCollatorForLanguageModeling):
        def __init__(self, tokenizer):
            super().__init__(tokenizer=tokenizer, mlm=False)
            self.model_ids = tokenizer.encode(MODEL_MARKER, add_special_tokens=False)
            self.end_ids = tokenizer.encode(TURN_END, add_special_tokens=False)
            self.ignore_index = -100

        @staticmethod
        def _find(ids: list[int], needle: list[int], start: int) -> int:
            if not needle:
                return -1
            last = len(ids) - len(needle)
            for idx in range(start, last + 1):
                if ids[idx:idx + len(needle)] == needle:
                    return idx
            return -1

        def torch_call(self, examples):
            batch = super().torch_call(examples)
            labels = batch["labels"]
            labels[:, :] = self.ignore_index
            input_rows = batch["input_ids"].tolist()
            for row_idx, ids in enumerate(input_rows):
                cursor = 0
                while True:
                    start = self._find(ids, self.model_ids, cursor)
                    if start < 0:
                        break
                    content_start = start + len(self.model_ids)
                    end = self._find(ids, self.end_ids, content_start)
                    if end < 0:
                        end = len(ids)
                    if end > content_start:
                        labels[row_idx, content_start:end] = batch["input_ids"][row_idx, content_start:end]
                    cursor = end + max(1, len(self.end_ids))
            return batch

    output_dir = Path(args.output_dir)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForAssistantTurns(tokenizer),
        args=SFTConfig(
            dataset_text_field="text",
            max_length=args.max_seq_len,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="no",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(output_dir / "checkpoints"),
            report_to="none",
            bf16=True,
            packing=False,
        ),
    )
    stats = trainer.train()
    print(f"Training loss: {stats.training_loss:.4f}")
    print(f"Training runtime: {stats.metrics.get('train_runtime', 0):.1f}s")

    lora_dir = output_dir / "cliniq-tool-trace-lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"Saved LoRA: {lora_dir}")

    print("Merging LoRA into bf16 HF checkpoint...")
    import bitsandbytes as bnb
    import torch.nn as nn
    from bitsandbytes.nn import Linear4bit

    merged = model.merge_and_unload()
    replaced = 0
    for _parent_name, parent in list(merged.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, Linear4bit):
                with torch.no_grad():
                    weight = bnb.functional.dequantize_4bit(
                        child.weight.data,
                        child.weight.quant_state,
                    ).to(torch.bfloat16)
                linear = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                ).to(torch.bfloat16)
                linear.weight.data = weight.contiguous()
                if child.bias is not None:
                    linear.bias.data = child.bias.data.to(torch.bfloat16)
                setattr(parent, child_name, linear)
                replaced += 1
    for attr in ("is_loaded_in_4bit", "is_loaded_in_8bit", "_is_quantized"):
        if hasattr(merged, attr):
            setattr(merged, attr, False)
    if hasattr(merged, "_quantization_config"):
        merged._quantization_config = None
    if hasattr(merged.config, "quantization_config"):
        delattr(merged.config, "quantization_config")
    merged.config.torch_dtype = "bfloat16"
    merged_dir = output_dir / "cliniq-tool-trace-merged"
    merged.save_pretrained(
        str(merged_dir),
        safe_serialization=True,
        max_shard_size="1GB",
    )
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Replaced {replaced} Linear4bit modules")
    print(f"Saved merged checkpoint: {merged_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
