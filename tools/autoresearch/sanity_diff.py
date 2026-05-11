"""Sanity: confirm FT generates different output than base on the same prompt."""
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

BASE = "/Volumes/models/hf/hub/models--google--gemma-4-E2B-it/snapshots/6b7e72c67d3c4556f42b56d5a68b4b8e864c63b4"
FT = "/Users/thinkstudio/mnt/models/cliniq/v2-fp16-merged/cliniq-compact-merged-fp16"
DEVICE = "mps"
DTYPE = torch.float16

prompt_text = (
    "Patient with COVID-19. Lab: SARS-CoV-2 RNA NAA+probe Ql Resp (LOINC 94500-6) - Detected. "
    "Meds: nirmatrelvir 150 MG / ritonavir 100 MG (RxNorm 2599543). "
    "Extract SNOMED, LOINC, RxNorm codes."
)

for label, path in [("base", BASE), ("ft", FT)]:
    tok = AutoTokenizer.from_pretrained(path)
    msgs = [{"role": "user", "content": prompt_text}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    print(f"\n=== {label} ===")
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
    m.training = False  # equivalent to .eval() but doesn't trigger hook
    for module in m.modules():
        module.training = False
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=64, do_sample=False)
    new = out[0, inputs["input_ids"].shape[1]:]
    print(tok.decode(new, skip_special_tokens=True))
    del m
    torch.mps.empty_cache()
