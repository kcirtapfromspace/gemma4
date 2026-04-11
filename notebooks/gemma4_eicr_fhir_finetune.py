#!/usr/bin/env python3
"""
Gemma 4 E4B Fine-tuning with Unsloth for eICR → FHIR Conversion
================================================================
Hackathon: Gemma 4 Good Hackathon (Unsloth Track - $10K prize)
Task: Convert eICR CDA/XML documents to FHIR R4 Bundles
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

max_seq_length = 4096
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
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
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

test_eicr = """<?xml version="1.0" encoding="UTF-8"?>
<ClinicalDocument xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <realmCode code="US"/>
  <typeId root="2.16.840.1.113883.1.3" extension="POCD_HD000040"/>
  <templateId root="2.16.840.1.113883.10.20.15.2" extension="2021-01-01"/>
  <id root="test-doc-001"/>
  <code code="55751-2" codeSystem="2.16.840.1.113883.6.1" displayName="Public Health Case Report"/>
  <title>Initial Public Health Case Report - eICR</title>
  <effectiveTime value="20260401120000-0600"/>
  <recordTarget>
    <patientRole>
      <id extension="PT-TEST" root="2.16.840.1.113883.19.5"/>
      <patient>
        <name use="L"><given>Test</given><family>Patient</family></name>
        <administrativeGenderCode code="M" codeSystem="2.16.840.1.113883.5.1"/>
        <birthTime value="19900101"/>
      </patient>
    </patientRole>
  </recordTarget>
  <component>
    <structuredBody>
      <component>
        <section>
          <code code="11450-4" codeSystem="2.16.840.1.113883.6.1" displayName="Problem list"/>
          <entry>
            <act classCode="ACT" moodCode="EVN">
              <entryRelationship typeCode="SUBJ">
                <observation classCode="OBS" moodCode="EVN">
                  <value xsi:type="CD" code="840539006" codeSystem="2.16.840.1.113883.6.96" displayName="COVID-19"/>
                </observation>
              </entryRelationship>
            </act>
          </entry>
        </section>
      </component>
    </structuredBody>
  </component>
</ClinicalDocument>"""

messages = [
    {"role": "system", "content": "You are a clinical informatics assistant. Convert the provided eICR CDA/XML document into a valid HL7 FHIR R4 Bundle JSON. Output valid JSON only."},
    {"role": "user", "content": f"Convert this eICR to a FHIR R4 Bundle:\n\n{test_eicr}"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=2048,
    temperature=0.1,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)

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
