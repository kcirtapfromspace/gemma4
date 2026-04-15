#!/usr/bin/env python3
"""
Gemma 4 E4B Fine-tuning with Unsloth — ClinIQ
==================================================
Hackathon: Gemma 4 Good Hackathon (Unsloth Track - $10K prize)
Task: Clinical entity extraction from eICR summaries — extract conditions,
      labs, medications, vitals with ontology codes and case summaries.
      Replaces cloud NLP (Comprehend Medical + IMO) with edge model.
Target: Edge deployment on Jetson Orin Nano (8GB unified memory)

Run on Kaggle with free A100 GPU.
Upload training data (train.jsonl, val.jsonl) as a Kaggle dataset first.

# %% [markdown]
# ## 1. Install Dependencies
"""

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force single GPU to avoid batch doubling
# !pip install unsloth
# !pip install --no-deps trl peft accelerate bitsandbytes

# %% [markdown]
# ## 2. Load Model with Unsloth

# %%
from unsloth import FastLanguageModel
import torch

max_seq_length = 512  # Compact output ~150 tokens + prompt ~150 = ~300 total
dtype = None  # auto-detect (bfloat16 on A100)
load_in_4bit = True  # QLoRA for memory efficiency

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# %% [markdown]
# ## 3. Configure LoRA Adapters

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=3407,
)

# %% [markdown]
# ## 4. Load Training Data

# %%
from datasets import load_dataset

import os

COMPACT = True  # Set to True for compact output (fewer tokens, faster inference)
DATASET_PATH = "/kaggle/input/eicr-fhir-training-data"

# Debug: list all available Kaggle inputs
input_dir = "/kaggle/input"
if os.path.exists(input_dir):
    print(f"Kaggle inputs: {os.listdir(input_dir)}")
    for d in os.listdir(input_dir):
        full = os.path.join(input_dir, d)
        if os.path.isdir(full):
            print(f"  {d}/: {os.listdir(full)}")

# Fallback: try alternative Kaggle mount paths
if not os.path.exists(DATASET_PATH):
    for alt in ["/kaggle/input/datasets/patrickdeutsch/eicr-fhir-training-data",
                "/kaggle/input/datasets"]:
        if os.path.exists(alt):
            DATASET_PATH = alt
            print(f"Using alternative path: {DATASET_PATH}")
            break

suffix = "-compact" if COMPACT else ""
train_file = f"{DATASET_PATH}/train{suffix}.jsonl"
val_file = f"{DATASET_PATH}/val{suffix}.jsonl"
if not os.path.exists(train_file):
    print(f"WARNING: {train_file} not found, trying verbose")
    COMPACT = False
    suffix = ""
    train_file = f"{DATASET_PATH}/train.jsonl"
    val_file = f"{DATASET_PATH}/val.jsonl"

dataset = load_dataset("json", data_files={
    "train": train_file,
    "validation": val_file,
})

print(f"Dataset: {DATASET_PATH}")
print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}")
# Verify we loaded the right data
sample_prompt = dataset['train'][0]['conversations'][0]['content']
print(f"System prompt: {sample_prompt[:80]}...")
assert "No summary" in sample_prompt or not COMPACT, "COMPACT=True but loaded verbose data!"

# %% [markdown]
# ## 5. Format Dataset for Chat Template

# %%
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-4",
)


def format_prompts(examples):
    texts = []
    for convos in examples["conversations"]:
        text = tokenizer.apply_chat_template(
            convos,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


dataset = dataset.map(format_prompts, batched=True)

# %% [markdown]
# ## 6. Train

# %%
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=5,
        learning_rate=1e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# %%
import torch
props = torch.cuda.get_device_properties(0)
total_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
print(f"GPU = {torch.cuda.get_device_name(0)}. Max memory = {total_mem / 1024**3:.3f} GB.")
print(f"{torch.cuda.memory_reserved() / 1024**3:.3f} GB of memory reserved.")

trainer_stats = trainer.train()
runtime = trainer_stats.metrics['train_runtime']
print(f"{runtime:.0f}s training time")
peak = torch.cuda.max_memory_reserved() / 1024**3
used = (torch.cuda.max_memory_reserved() - torch.cuda.memory_reserved()) / 1024**3
print(f"Peak memory = {peak:.3f} GB (training used {used:.3f} GB)")
print(f"Loss: {trainer_stats.training_loss:.4f}")
if runtime < 600:
    print("Quick proof complete — increase max_steps for full training")

# %% [markdown]
# ## 7. Test Inference

# %%
# Quick inference test (wrapped in try/except so save always runs)
try:
    FastLanguageModel.for_inference(model)
    test_input = "Patient: Maria Garcia\nGender: F\nDOB: 1985-06-14\nLocation: Denver, CO 80202\nDx: COVID-19 (SNOMED 840539006)\nLab: SARS-CoV-2 RNA (LOINC 94500-6) - Detected\nMeds: nirmatrelvir (RxNorm 2599543)"
    sys_prompt = "Extract clinical entities from this eICR. Output compact JSON with: patient, encounter, conditions (SNOMED), labs (LOINC), meds (RxNorm), vitals. No summary. Valid JSON only." if COMPACT else "Extract clinical entities from this eICR summary. Output JSON with: patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), medications (RxNorm), vitals, and a case summary. Include confidence scores. Output valid JSON only."
    inputs = tokenizer.apply_chat_template([{"role": "system", "content": sys_prompt}, {"role": "user", "content": test_input}], tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.1, top_p=0.9)
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    print("=== Model output ===")
    print(response[:500])
except Exception as e:
    print(f"Inference test failed: {e}")

# %% [markdown]
# ## 8. Save & Export

# %%
# Save LoRA adapter
import os
tag = "-compact" if COMPACT else ""
lora_dir = "cliniq_lora"
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
lora_size = sum(os.path.getsize(os.path.join(lora_dir, f)) for f in os.listdir(lora_dir) if os.path.isfile(os.path.join(lora_dir, f)))
print(f"LoRA saved to {lora_dir}/ — {lora_size / 1024 / 1024:.1f} MB ({len(os.listdir(lora_dir))} files)")

# GGUF export skipped — T4 doesn't have enough disk space for merge+export.
# Use convert_lora_to_gguf.py locally with the downloaded LoRA adapter:
#   python /tmp/llama-cpp-tools/convert_lora_to_gguf.py cliniq_lora/ --outfile model.gguf --base-model-id unsloth/gemma-4-E2B-it
# Then apply at runtime: llama-server -m base.gguf --lora model.gguf
print("Download cliniq_lora/ from Output tab for local GGUF conversion")

# %% [markdown]
# ## 9. Upload to HuggingFace (optional)

# %%
# Uncomment to push to HuggingFace Hub
# model.push_to_hub_gguf(
#     "YOUR_HF_USER/gemma4-eicr-fhir-e4b-gguf",
#     tokenizer,
#     quantization_method="q4_k_m",
#     token="YOUR_HF_TOKEN",
# )
