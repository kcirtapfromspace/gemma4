#!/usr/bin/env python3
"""
ClinIQ: Fine-tune Gemma 4 E2B LoRA on compact eICR extraction.
Uses Kaggle's pre-installed env (PyTorch 2.10, transformers 5.x).
REQUIRES T4 GPU — P100 is not compatible.
"""

import json, os, subprocess, sys, glob
# Force single GPU to avoid multi-GPU CUBLAS issues
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

# Fail fast on P100
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} (sm_{cc[0]}{cc[1]})")
    if cc < (7, 0):
        print(f"\nERROR: {name} (sm_{cc[0]}{cc[1]}) is not supported.")
        print("This kernel requires a T4 or better GPU.")
        print("In the Kaggle UI: Settings → Accelerator → GPU T4 x2")
        print("Or retry — Kaggle sometimes assigns T4 instead of P100.")
        sys.exit(1)
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print(f"PyTorch: {torch.__version__}")

# Upgrade transformers + peft for Gemma 4 support (ClippableLinear)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=5.5", "peft>=0.15", "trl>=0.15",
    "bitsandbytes", "sentencepiece", "datasets",
    "git+https://github.com/huggingface/peft.git"])

# ============================================================
# Find training data
# ============================================================

TRAIN_PATH = VAL_PATH = None
for base in glob.glob("/kaggle/input/**/train-compact.jsonl", recursive=True):
    TRAIN_PATH = base
    VAL_PATH = base.replace("train-compact", "val-compact")
    break

if not TRAIN_PATH:
    print("Downloading training data...")
    os.makedirs("/kaggle/working/data", exist_ok=True)
    subprocess.check_call(["kaggle", "datasets", "download",
        "patrickdeutsch/cliniq-training-data",
        "-p", "/kaggle/working/data", "--unzip"])
    TRAIN_PATH = "/kaggle/working/data/train-compact.jsonl"
    VAL_PATH = "/kaggle/working/data/val-compact.jsonl"

assert os.path.exists(TRAIN_PATH), f"Not found: {TRAIN_PATH}"
print(f"Train: {TRAIN_PATH}\nVal: {VAL_PATH}")

# ============================================================
# Load model
# ============================================================

print("\n=== Loading Gemma 4 E2B in 4-bit ===")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model_name = "google/gemma-4-E2B-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded! Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ============================================================
# LoRA
# ============================================================

print("\n=== Configuring LoRA (r=16) ===")
from peft import LoraConfig, get_peft_model

# Gemma 4 uses ClippableLinear wrappers that PEFT doesn't recognize.
# Unwrap them to expose the inner Linear4bit modules.
from torch import nn
for name, module in list(model.named_modules()):
    if type(module).__name__ == "Gemma4ClippableLinear":
        # Replace wrapper with its inner linear
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], module.linear)
print("Unwrapped Gemma4ClippableLinear → Linear4bit")

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
))
model.print_trainable_parameters()

# ============================================================
# Data
# ============================================================

print("\n=== Loading data ===")
from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

def fmt(examples):
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                     for c in examples["conversations"]]}

dataset = dataset.map(fmt, batched=True, remove_columns=["conversations"])

# ============================================================
# Train
# ============================================================

print("\n=== Training: 3 epochs ===")
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=dataset["train"], eval_dataset=dataset["validation"],
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        warmup_steps=10, num_train_epochs=3, learning_rate=1e-4,
        logging_steps=20, eval_strategy="no",
        save_strategy="no", optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=3407,
        output_dir="/kaggle/working/checkpoints", report_to="none",
        bf16=True, packing=False,
    ),
)

stats = trainer.train()
print(f"\nLoss: {stats.training_loss:.4f} | Time: {stats.metrics['train_runtime']:.0f}s")

# ============================================================
# Test
# ============================================================

print("\n=== Test inference ===")
SYSTEM_PROMPT = (
    'Extract clinical entities from this eICR. Return minified JSON: '
    '{"patient":{...},"conditions":[...],"labs":[...],"meds":[...],"vitals":[...]}. '
    'All sections are arrays. Include SNOMED for conditions, LOINC for labs, '
    'RxNorm for meds. No summary. No markdown. JSON only.'
)
test_msgs = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Patient: Test Patient\nGender: M\nDOB: 1990-01-01\n"
     "Dx: COVID-19 (SNOMED 840539006)\nLab: SARS-CoV-2 RNA (LOINC 94500-6) - Detected\n"
     "Meds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543)"},
]
inputs = tokenizer.apply_chat_template(test_msgs, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=True).to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
resp = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(resp[:500])
try:
    print(f"JSON valid: {bool(json.loads(resp.strip()))}")
except:
    print("JSON valid: False")

# ============================================================
# Save
# ============================================================

print("\n=== Saving LoRA ===")
model.save_pretrained("/kaggle/working/cliniq-compact-lora")
tokenizer.save_pretrained("/kaggle/working/cliniq-compact-lora")

print("\n=== Merging ===")
merged = model.merge_and_unload()
merged.save_pretrained("/kaggle/working/cliniq-compact-merged")
tokenizer.save_pretrained("/kaggle/working/cliniq-compact-merged")

print("\n=== DONE ===")
for root, _, files in os.walk("/kaggle/working"):
    for f in files:
        p = os.path.join(root, f)
        s = os.path.getsize(p) / 1e6
        if s > 1: print(f"  {os.path.relpath(p, '/kaggle/working')}: {s:.0f} MB")
