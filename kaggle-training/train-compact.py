#!/usr/bin/env python3
"""
ClinIQ: Fine-tune Gemma 4 E2B on compact eICR extraction format.
Runs on Kaggle GPU (T4/P100) with Unsloth for fast LoRA training.

Outputs: LoRA adapter + merged GGUF (Q3_K_M, Q8_0) for llama.cpp deployment.
"""

import json
import os
import subprocess
import sys

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "unsloth", "datasets", "trl", "peft", "accelerate",
    "bitsandbytes", "sentencepiece", "protobuf"])

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# Training data (embedded — no external files needed)
# ============================================================

SYSTEM_PROMPT = (
    'Extract clinical entities from this eICR. Return minified JSON: '
    '{"patient":{...},"conditions":[...],"labs":[...],"meds":[...],"vitals":[...]}. '
    'All sections are arrays. Include SNOMED for conditions, LOINC for labs, '
    'RxNorm for meds. No summary. No markdown. JSON only.'
)

# Load training data from the dataset attached to this kernel
TRAIN_PATH = "/kaggle/input/cliniq-training-data/train-compact.jsonl"
VAL_PATH = "/kaggle/input/cliniq-training-data/val-compact.jsonl"

if not os.path.exists(TRAIN_PATH):
    print(f"ERROR: Training data not found at {TRAIN_PATH}")
    print("Please attach the 'cliniq-training-data' dataset to this kernel.")
    print("Or upload train-compact.jsonl and val-compact.jsonl as a dataset.")
    sys.exit(1)

# ============================================================
# Load model
# ============================================================

print("\n=== Loading Gemma 4 E2B with Unsloth ===")
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
    max_seq_length=768,
    dtype=None,
    load_in_4bit=True,
    full_finetuning=False,
)
print("Model loaded!")

# ============================================================
# Configure LoRA
# ============================================================

print("\n=== Configuring LoRA (r=16) ===")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# ============================================================
# Load and format data
# ============================================================

print("\n=== Loading training data ===")
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

dataset = load_dataset("json", data_files={
    "train": TRAIN_PATH,
    "validation": VAL_PATH,
})
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

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

# ============================================================
# Train
# ============================================================

print("\n=== Training: 3 epochs ===")
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="/kaggle/working/checkpoints",
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
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

# ============================================================
# Test inference
# ============================================================

print("\n=== Test inference ===")
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

test_input = (
    "Patient: Test Patient\nGender: M\nDOB: 1990-01-01\n"
    "Location: Denver, CO 80202\nEncounter: 2026-03-15\n"
    "Dx: COVID-19 (SNOMED 840539006)\n"
    "Lab: SARS-CoV-2 RNA (LOINC 94500-6) - Detected\n"
    "Meds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543)"
)
messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "text", "text": test_input}]},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt",
    tokenize=True, return_dict=True,
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True,
                          temperature=0.1, top_p=0.95, top_k=64)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                             skip_special_tokens=True)
print("Response:")
print(response[:500])

# Validate JSON
try:
    parsed = json.loads(response.strip())
    print(f"\nJSON valid: True")
    print(f"Keys: {list(parsed.keys())}")
except json.JSONDecodeError as e:
    print(f"\nJSON valid: False ({e})")

# ============================================================
# Save outputs
# ============================================================

OUTPUT_DIR = "/kaggle/working"

# Save LoRA adapter
print("\n=== Saving LoRA adapter ===")
lora_dir = f"{OUTPUT_DIR}/cliniq-compact-lora"
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print(f"LoRA saved to {lora_dir}")

# Save merged model
print("\n=== Saving merged model ===")
merged_dir = f"{OUTPUT_DIR}/cliniq-compact-merged"
model.save_pretrained_merged(merged_dir, tokenizer)
print(f"Merged model saved to {merged_dir}")

# Export GGUF for llama.cpp
print("\n=== Exporting GGUF (Q8_0 + Q3_K_M) ===")
gguf_dir = f"{OUTPUT_DIR}/cliniq-compact-gguf"

# Q8_0 for LoRA adapter
model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q8_0")
print(f"Q8_0 GGUF saved to {gguf_dir}")

# Q3_K_M for standalone deployment
gguf_q3_dir = f"{OUTPUT_DIR}/cliniq-compact-gguf-q3km"
model.save_pretrained_gguf(gguf_q3_dir, tokenizer, quantization_method="q3_k_m")
print(f"Q3_K_M GGUF saved to {gguf_q3_dir}")

print("\n=== DONE ===")
print(f"Files at {OUTPUT_DIR}:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1e6
        if size > 1:
            print(f"  {os.path.relpath(path, OUTPUT_DIR)}: {size:.1f} MB")
