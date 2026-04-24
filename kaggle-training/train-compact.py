#!/usr/bin/env python3
"""
ClinIQ: Fine-tune Gemma 4 E2B LoRA on compact eICR extraction.
Uses Kaggle's pre-installed env (PyTorch 2.10, transformers 5.x).
REQUIRES T4 GPU — P100 is not compatible.

Team C9 retrain v2 (2026-04-23)
--------------------------------
Targets two failure modes from Team C8 validation (13/18, target >=0.9):

1. Code elision (COVID case): model drops numeric SNOMED/RxNorm codes even
   when present in input. Fix: +20 code-preservation training examples
   (see dataset/train_v2_additions.jsonl) + bump LoRA rank 16 -> 32 for
   more capacity + train_on_responses_only so gradient only flows on the
   JSON output tokens (not the input where codes already appear).

2. Sequence degeneration (negative-lab case): output collapses to repeating
   digits on "Not detected" inputs. Fix: +20 negative-lab examples across
   diverse conditions + 10 length-stress cases (300-500 tok inputs) to push
   past the ~20-30 token shoulder where degeneration starts.

3. LoRA target_modules fix (from Team C7 DIAGNOSIS.md): Gemma 4 E2B has
   num_kv_shared_layers=20 (layers 15..34 reuse K/V from the first 15).
   Training k_proj/v_proj on those shared layers produces bit-exact-zero
   deltas (dead weight). Exclude them via per-layer regex below.
"""

import json, os, subprocess, sys, glob, re
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
        print("In the Kaggle UI: Settings -> Accelerator -> GPU T4 x2")
        print("Or retry - Kaggle sometimes assigns T4 instead of P100.")
        sys.exit(1)
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print(f"PyTorch: {torch.__version__}")

# Upgrade transformers + peft for Gemma 4 support (ClippableLinear).
# NOTE: previously imported unsloth for get_chat_template + train_on_responses_only,
# but unsloth's 2026 releases hit ImportError in Kaggle's TRL (ConstantLengthDataset
# moved). Bypass: inline the gemma4 template string vendored from unsloth source
# at kaggle-training/templates/gemma4_unsloth.jinja, use TRL's stock
# DataCollatorForCompletionOnlyLM for response-only masking (same behavior).
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=5.5", "peft>=0.15", "trl>=0.15",
    "bitsandbytes", "sentencepiece", "datasets",
    "git+https://github.com/huggingface/peft.git"])

# ============================================================
# Find training data — prefer v2 (C9) over legacy compact
# ============================================================

TRAIN_PATH = VAL_PATH = None
# Prefer v2 dataset first, fall back to legacy
for pattern in ("/kaggle/input/**/train_v2.jsonl", "/kaggle/input/**/train-compact.jsonl"):
    for base in glob.glob(pattern, recursive=True):
        TRAIN_PATH = base
        # val stays on val-compact; v2 only adds train examples
        val_guess = base.replace("train_v2", "val-compact").replace("train-compact", "val-compact")
        VAL_PATH = val_guess
        break
    if TRAIN_PATH:
        break

if not TRAIN_PATH:
    print("Downloading training data...")
    os.makedirs("/kaggle/working/data", exist_ok=True)
    subprocess.check_call(["kaggle", "datasets", "download",
        "patrickdeutsch/cliniq-training-data",
        "-p", "/kaggle/working/data", "--unzip"])
    TRAIN_PATH = "/kaggle/working/data/train_v2.jsonl"
    if not os.path.exists(TRAIN_PATH):
        TRAIN_PATH = "/kaggle/working/data/train-compact.jsonl"
    VAL_PATH = "/kaggle/working/data/val-compact.jsonl"

assert os.path.exists(TRAIN_PATH), f"Not found: {TRAIN_PATH}"
assert os.path.exists(VAL_PATH), f"Not found: {VAL_PATH}"
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
# Force the unsloth gemma-4 chat template so the tokenized training text
# uses <|turn> / <turn|> delimiters (matching apps/mobile/convert/validate_litertlm.py
# and the original cliniq-compact-lora.gguf which was trained via the notebook
# using the same template). HF's default Gemma tokenizer template emits
# <start_of_turn> / <end_of_turn> — that would mis-align training vs inference.
# Template vendored verbatim from
# github.com/unslothai/unsloth/unsloth/chat_templates.py `gemma4_template`
# (2026-04-23). Embedded inline because Kaggle kernel push only uploads
# the single code_file.
_GEMMA4_UNSLOTH_TEMPLATE = '{%- macro strip_thinking(text) -%}\n    {%- set ns = namespace(result=\'\') -%}\n    {%- for part in text.split(\'<channel|>\') -%}\n        {%- if \'<|channel>\' in part -%}\n            {%- set ns.result = ns.result + part.split(\'<|channel>\')[0] -%}\n        {%- else -%}\n            {%- set ns.result = ns.result + part -%}\n        {%- endif -%}\n    {%- endfor -%}\n    {{- ns.result | trim -}}\n{%- endmacro -%}\n{%- set thinking = enable_thinking is defined and enable_thinking -%}\n{%- set loop_messages = messages -%}\n{%- if messages[0][\'role\'] in [\'system\', \'developer\'] or thinking -%}\n    {{ \'<|turn>system\\n\' }}\n    {%- if thinking -%}\n        {{ \'<|think>\' }}\n    {%- endif -%}\n    {%- if messages[0][\'role\'] in [\'system\', \'developer\'] -%}\n        {{ messages[0][\'content\'] | trim }}\n        {%- set loop_messages = messages[1:] -%}\n    {%- endif -%}\n    {{ \'<turn|>\\n\' }}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- set role = message[\'role\'] -%}\n    {%- if role == \'assistant\' -%}\n        {%- set role = \'model\' -%}\n    {%- endif -%}\n    {{ \'<|turn>\' + role + \'\\n\' }}\n    {%- if message[\'content\'] is string -%}\n        {%- if role == \'model\' -%}\n            {{ strip_thinking(message[\'content\']) }}\n        {%- else -%}\n            {{ message[\'content\'] | trim }}\n        {%- endif -%}\n    {%- else -%}\n        {%- for content in message[\'content\'] -%}\n            {%- if content[\'type\'] == \'text\' -%}\n                {%- if role == \'model\' -%}\n                    {{ strip_thinking(content[\'text\']) }}\n                {%- else -%}\n                    {{ content[\'text\'] | trim }}\n                {%- endif -%}\n            {%- elif content[\'type\'] == \'image\' -%}\n                {{ \'<start_of_image>\' }}\n            {%- elif content[\'type\'] == \'audio\' -%}\n                {{ \'<start_of_audio>\' }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- endif -%}\n    {{ \'<turn|>\\n\' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{\'<|turn>model\\n\'}}\n{%- endif -%}'
tokenizer.chat_template = _GEMMA4_UNSLOTH_TEMPLATE
print(f"Loaded unsloth gemma-4 template inline ({len(tokenizer.chat_template)} chars)")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded! Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# ============================================================
# LoRA — r=32, exclude KV-shared layers 15..34 k/v_proj
# ============================================================

print("\n=== Configuring LoRA (r=32, KV-shared k/v excluded) ===")
from peft import LoraConfig, get_peft_model

# Gemma 4 uses ClippableLinear wrappers that PEFT doesn't recognize.
# Unwrap them to expose the inner Linear4bit modules.
from torch import nn
for name, module in list(model.named_modules()):
    if type(module).__name__ == "Gemma4ClippableLinear":
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], module.linear)
print("Unwrapped Gemma4ClippableLinear -> Linear4bit")

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

# Per-layer target selection (Team C7 DIAGNOSIS.md):
#   Gemma 4 E2B has 35 hidden layers; layers 15..34 are KV-shared.
#   Training k_proj/v_proj on shared layers is dead weight (bit-exact zero).
#   -> attach LoRA to k_proj/v_proj only on layers 0..14,
#      and to q_proj/o_proj + gate/up/down_proj on ALL 35 layers.
#
# PEFT LoraConfig.target_modules accepts a regex. Build an OR of:
#   - layers 0..14 any of [q|k|v|o]_proj
#   - layers 15..34 any of [q|o]_proj  (skip k|v on shared)
#   - all layers any of [gate|up|down]_proj
target_regex = (
    r".*\.language_model\.layers\.("
    r"[0-9]|1[0-4]"                                # layers 0..14
    r")\.self_attn\.(q|k|v|o)_proj$"
    r"|"
    r".*\.language_model\.layers\.("
    r"1[5-9]|2[0-9]|3[0-4]"                         # layers 15..34
    r")\.self_attn\.(q|o)_proj$"
    r"|"
    r".*\.language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj$"
)

# Fallback for model layouts missing the language_model prefix (defensive)
target_regex_alt = (
    r".*layers\.("
    r"[0-9]|1[0-4]"
    r")\.self_attn\.(q|k|v|o)_proj$"
    r"|"
    r".*layers\.("
    r"1[5-9]|2[0-9]|3[0-4]"
    r")\.self_attn\.(q|o)_proj$"
    r"|"
    r".*layers\.\d+\.mlp\.(gate|up|down)_proj$"
)

# Pick whichever regex actually matches modules in this model
def _n_matches(pat):
    return sum(1 for n, _ in model.named_modules() if re.match(pat, n))

n1 = _n_matches(target_regex)
n2 = _n_matches(target_regex_alt)
chosen_regex = target_regex if n1 >= n2 else target_regex_alt
print(f"Regex match counts: language_model-prefix={n1}, alt={n2} -> using {'primary' if n1>=n2 else 'alt'}")
print(f"Total LoRA target modules matched: {max(n1,n2)}")

model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=32,
    target_modules=chosen_regex,
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
# Train — 3 epochs, train_on_responses_only masking
# ============================================================

print("\n=== Training: 3 epochs, response-only masking ===")
from trl import SFTTrainer, SFTConfig
# DataCollatorForCompletionOnlyLM moves around across TRL versions:
#   <=0.10: from trl import DataCollatorForCompletionOnlyLM
#   0.11-0.14: from trl.trainer import DataCollatorForCompletionOnlyLM
#   0.15+: from trl.trainer.utils import DataCollatorForCompletionOnlyLM
# Try all three so the script runs on whatever Kaggle has preinstalled.
DataCollatorForCompletionOnlyLM = None
for _module in ("trl.trainer.utils", "trl.trainer", "trl"):
    try:
        _m = __import__(_module, fromlist=["DataCollatorForCompletionOnlyLM"])
        DataCollatorForCompletionOnlyLM = _m.DataCollatorForCompletionOnlyLM
        print(f"Loaded DataCollatorForCompletionOnlyLM from {_module}")
        break
    except (ImportError, AttributeError):
        continue
assert DataCollatorForCompletionOnlyLM is not None, \
    "DataCollatorForCompletionOnlyLM not found in any TRL submodule"

# Response-only loss masking: compute loss only on the assistant JSON
# tokens, not the user/system prompt where the codes already appear
# verbatim. Uses TRL's stock collator; response_template is tokenized
# and every occurrence in each sequence flips the preceding tokens'
# labels to -100. Unsloth's train_on_responses_only does the same thing
# with more plumbing; bypassed here because unsloth's latest releases
# break on Kaggle's TRL version (ConstantLengthDataset import moved).
#
# Unsloth gemma-4 template emits <|turn>model\n as the assistant-turn
# opener (verified: templates/gemma4_unsloth.jinja line "<|turn>model\n").
_response_template = "<|turn>model\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=_response_template,
    tokenizer=tokenizer,
)
print(f"DataCollatorForCompletionOnlyLM with response_template={_response_template!r}")

trainer = SFTTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=dataset["train"], eval_dataset=dataset["validation"],
    data_collator=collator,
    args=SFTConfig(
        dataset_text_field="text",
        # Kaggle's TRL rejects max_seq_length in SFTConfig; tokenizer's
        # model_max_length governs truncation instead (Gemma 4 default
        # 131072 so our ~605-token p99 examples don't get truncated).
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        warmup_steps=10, num_train_epochs=3, learning_rate=1e-4,
        logging_steps=20, eval_strategy="no",
        save_strategy="no", optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=3407,
        output_dir="/kaggle/working/checkpoints", report_to="none",
        bf16=True, packing=False,
    ),
)

# Legacy dead code below (kept commented so git blame preserves the
# decision trail about why we dropped unsloth). DataCollatorForCompletionOnlyLM
# now handles response-only masking above.
if False:
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
        )
        print("train_on_responses_only applied (unsloth, <|turn> delimiters)")
    except ImportError:
        pass
# End legacy dead code.

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
# Two probes: (1) COVID w/ codes (code-elision fail-mode), (2) Neg-lab Hep C
probes = [
    ("COVID code preservation",
     "Patient: Maria Garcia\nGender: F\nDOB: 1985-06-14\nLocation: Denver, CO 80202\n"
     "Dx: COVID-19 (SNOMED 840539006)\n"
     "Lab: SARS-CoV-2 RNA (LOINC 94500-6) - Detected\n"
     "Meds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543)"),
    ("Negative Hep C lab",
     "Patient: Jennifer Brown\nGender: F\nDOB: 1985-10-05\nLocation: Portland, OR 97201\n"
     "Dx: Hepatitis C (SNOMED 50711007)\n"
     "Lab: Hepatitis C virus Ab [Presence] in Serum (LOINC 11259-9) - Not detected [Serum, final]\n"
     "Meds: sofosbuvir 400 MG / velpatasvir 100 MG (RxNorm 1940261)"),
]
for label, user_msg in probes:
    print(f"\n--- Probe: {label} ---")
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(resp[:500])
    try:
        print(f"JSON valid: {bool(json.loads(resp.strip()))}")
    except Exception:
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
