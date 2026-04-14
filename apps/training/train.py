#!/usr/bin/env python3
"""
ClinIQ: Fine-tune Gemma 4 E2B with Unsloth on Jetson Orin Nano.
Trains clinical entity extraction from eICR summaries.

Usage:
  python3 train.py --data-dir /data --output-dir /output
  python3 train.py --data-dir /data --output-dir /output --epochs 3 --batch-size 1
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="ClinIQ Fine-tuning")
    parser.add_argument("--data-dir", default="/data", help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--output-dir", default="/output", help="Directory for model output")
    parser.add_argument("--model", default="unsloth/gemma-4-E2B-it", help="Base model name or path")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (use 1 for 8GB)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF export")
    parser.add_argument("--dry-run", action="store_true", help="Load model and data but don't train")
    args = parser.parse_args()

    # Verify CUDA
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU Memory: {mem:.1f} GB")
    else:
        print("WARNING: No CUDA — training will be very slow on CPU")

    # Verify training data
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found")
        print(f"Contents of {args.data_dir}: {os.listdir(args.data_dir) if os.path.isdir(args.data_dir) else 'NOT A DIR'}")
        sys.exit(1)

    # Load model with Unsloth
    print(f"\n=== Loading model: {args.model} ===")
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    print("Model loaded!")

    # Configure LoRA
    print(f"\n=== Configuring LoRA (r={args.lora_r}) ===")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    print("LoRA configured!")

    # Load training data
    print(f"\n=== Loading data from {args.data_dir} ===")
    from datasets import load_dataset

    dataset = load_dataset("json", data_files={
        "train": train_path,
        "validation": val_path,
    })
    print(f"Train: {len(dataset['train'])} samples")
    print(f"Val: {len(dataset['validation'])} samples")

    # Format for chat template
    print("\n=== Formatting dataset ===")
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"Sample length: {len(dataset['train'][0]['text'])} chars")

    if args.dry_run:
        print("\n=== DRY RUN — skipping training ===")
        print("Model, data, and formatting all validated successfully!")
        return

    # Train
    print(f"\n=== Training: {args.epochs} epochs, batch {args.batch_size}x{args.grad_accum} ===")
    from trl import SFTTrainer, SFTConfig
    from unsloth.chat_templates import train_on_responses_only

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=os.path.join(args.output_dir, "checkpoints"),
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    stats = trainer.train()
    print(f"\nTraining loss: {stats.training_loss:.4f}")
    print(f"Training time: {stats.metrics['train_runtime']:.1f}s")

    # Test inference
    print("\n=== Test inference ===")
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    test_input = (
        "Patient: Test Patient\nGender: M\nDOB: 1990-01-01\n"
        "Location: Denver, CO 80202\nEncounter: 2026-03-15\n"
        "Dx: COVID-19 (SNOMED 840539006)\n"
        "Lab: SARS-CoV-2 RNA (LOINC 94500-6) - Detected"
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Extract clinical entities from this eICR summary. Output JSON with: patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), medications (RxNorm), vitals, and a case summary. Include confidence scores. Output valid JSON only."}]},
        {"role": "user", "content": [{"type": "text", "text": test_input}]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True, temperature=1.0, top_p=0.95, top_k=64)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(response[:500])

    # Save
    print(f"\n=== Saving to {args.output_dir} ===")
    lora_dir = os.path.join(args.output_dir, "cliniq-lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"LoRA saved to {lora_dir}")

    merged_dir = os.path.join(args.output_dir, "cliniq-merged")
    model.save_pretrained_merged(merged_dir, tokenizer)
    print(f"Merged model saved to {merged_dir}")

    if not args.skip_gguf:
        print("\n=== Exporting GGUF (Q8_0) ===")
        gguf_dir = os.path.join(args.output_dir, "cliniq-gguf")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="Q8_0")
        print(f"GGUF saved to {gguf_dir}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
