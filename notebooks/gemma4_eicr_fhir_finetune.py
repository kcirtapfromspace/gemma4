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
# !pip install unsloth
# !pip install --no-deps trl peft accelerate bitsandbytes

# %% [markdown]
# ## 2. Load Model with Unsloth

# %%
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024  # Extraction JSON samples are ~600 tokens
dtype = None  # auto-detect (bfloat16 on A100)
load_in_4bit = True  # QLoRA for memory efficiency

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it",
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

# If running on Kaggle, upload train.jsonl and val.jsonl as a dataset
# and update the path below
DATASET_PATH = "/kaggle/input/eicr-fhir-training-data"

dataset = load_dataset("json", data_files={
    "train": f"{DATASET_PATH}/train.jsonl",
    "validation": f"{DATASET_PATH}/val.jsonl",
})

print(f"Train: {len(dataset['train'])} samples")
print(f"Val: {len(dataset['validation'])} samples")

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
    packing=True,  # Pack short sequences for throughput
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
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
trainer_stats = trainer.train()
print(f"Training loss: {trainer_stats.training_loss:.4f}")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")

# %% [markdown]
# ## 7. Test Inference

# %%
FastLanguageModel.for_inference(model)

test_input = """Patient: Maria Garcia
Gender: F
DOB: 1985-06-14
Race: White
Ethnicity: Hispanic or Latino
Location: Denver, CO 80202
Phone: +1-303-555-0142
Facility: Denver Health Medical Center (NPI: 1234567800)
Encounter: 2026-03-15
Reason: fever (39.2C), dry cough for 5 days, shortness of breath
Dx: COVID-19 (SNOMED 840539006)
Lab: SARS-CoV-2 RNA NAA+probe Ql (Resp) (LOINC 94500-6) - Detected
Vitals: Temp 39.2C, HR 92, RR 20, SpO2 95%, BP 128
Meds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543)"""

messages = [
    {"role": "system", "content": "Extract clinical entities from this eICR summary. Output JSON with: patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), medications (RxNorm), vitals, and a case summary. Include confidence scores. Output valid JSON only."},
    {"role": "user", "content": test_input},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print("=== Raw response ===")
print(response)

# Validate JSON
import json
try:
    extraction = json.loads(response)
    print("\n=== Parsed extraction ===")
    print(json.dumps(extraction, indent=2))
    print(f"\nConditions: {[c['name'] for c in extraction.get('conditions', [])]}")
    print(f"Summary: {extraction.get('summary', 'N/A')}")
except json.JSONDecodeError as e:
    print(f"\nJSON parse error: {e}")

# %% [markdown]
# ## 8. Save & Export

# %%
# Save LoRA adapter
model.save_pretrained("gemma4-eicr-fhir-lora")
tokenizer.save_pretrained("gemma4-eicr-fhir-lora")

# Merge and save full model
model.save_pretrained_merged(
    "gemma4-eicr-fhir-merged",
    tokenizer,
    save_method="merged_16bit",
)

# Export to GGUF for Ollama/llama.cpp deployment on Jetson
# Q4_K_M is ~3GB, fits well in 8GB Jetson Orin Nano
model.save_pretrained_gguf(
    "gemma4-eicr-fhir-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)

# Also export Q8_0 for higher quality if memory permits
model.save_pretrained_gguf(
    "gemma4-eicr-fhir-gguf-q8",
    tokenizer,
    quantization_method="q8_0",
)

print("Export complete!")
print("Files ready for deployment:")
print("  - gemma4-eicr-fhir-gguf/  (Q4_K_M, ~3GB, recommended for Jetson)")
print("  - gemma4-eicr-fhir-gguf-q8/  (Q8_0, ~4.5GB, higher quality)")

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
