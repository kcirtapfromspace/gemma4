# Draft section to add to `hackathon-submission-2026-04-27.md`

Insert before the existing "Open uncertainties" section.

---

## Unsloth-track contribution — v62 fine-tune

The pipeline's three-tier extraction (regex → RAG → 6-turn agent) is the
project's quality story. The v62 Unsloth fine-tune is its *speed* story.

**What it is.** A LoRA adapter on `unsloth/gemma-4-E2B-it` (`r=16`, all 7
attention + MLP target modules, 124 MB), trained for 5 epochs on 800
synthetic eICRs with `eicr-fhir-training-data`. Replaces the 6-turn agent
loop with a single forward pass for the fastest demo path.

**Trained on Kaggle's free T4 ×2 in 1h 4m.** Final loss 0.2446 on the
held-out 200-case val set (healthy generalization, not memorization).

**Result on `val-compact` (200 cases held out from training):**

| Metric | Base `gemma-4-E2B-it` Q3_K_M | v62 LoRA Q3_K_M | Delta |
|---|---:|---:|---:|
| Micro-F1 (code-level) | 0.337 | **0.823** | **+0.486** |
| Micro-precision | 0.469 | **0.979** | +0.510 |
| Micro-recall | 0.263 | 0.710 | +0.447 |
| JSON validity | 100% | 86% | -14pp* |
| Latency p50 (Mac M-series) | 6.6s | **4.1s** | 1.6× faster |
| Cases above F1 = 0.70 | 0 / 200 | **162 / 200** | — |

\* JSON-invalid cases score F1=0 — v62 truncates mid-output on 28 of 200
cases at the 2048-token cap. We tested a GBNF-constrained decode path
(`apps/mobile/convert/cliniq_v62_compact.gbnf`); it regressed F1 to 0.780
because grammar paths hit the length limit before closing the JSON. The
shipped submission uses **unconstrained decoding at F1 = 0.823**. A v63
retrain with longer `max_seq_length` would close both gaps; tracked but
not in this submission.

The fine-tune isn't redundant with the agent path — it solves a *different*
problem. The agent hits F1=1.000 in 5–35s using a verified RAG database +
multi-turn lookup. The fine-tune does single-shot extraction in ~4.1s
without RAG, at shipped F1=0.823 and 0.979 precision; the JSON-valid subset
is F1=0.895, while the measured grammar path regressed to F1=0.780. For
demo, v62 is still the fast single-shot path on known-format inputs. For
LMIC clinic deployment, the fine-tune's
smaller footprint (no embedded ~60-entry RAG db) matters.

**What Unsloth specifically did:**
- `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` — base fits
  on T4 16 GB at 4-bit
- `use_gradient_checkpointing="unsloth"` — 30% VRAM reduction; let us
  train at `max_seq_length=512` with packing=True
- `unsloth.chat_templates.get_chat_template(tokenizer, "gemma-4")` —
  emits the correct `<|turn>system\n` markers Gemma 4 was pretrained on;
  HF's default tokenizer would have used `<start_of_turn>` and broken
  inference

**Submission artifacts:**
- Public Kaggle notebook: <PUBLIC_KAGGLE_URL>
- HF Hub model: <HF_HUB_URL>
- LoRA GGUF (~124 MB int8 quant of the adapter): `models/cliniq-gemma4-e2b-v62-lora.gguf`
- Bench harness: `apps/mobile/convert/bench_v62_singleshot.py`
- Bench output: `apps/mobile/convert/build/v62_val_compact_bench.json`

**Honest framing.** The v62 LoRA is not a replacement for the verified
agent-path F1=1.000 claim — that remains the headline. v62 is the
fastest-path demo option for inputs in the eICR pseudo-format the
training set covers. Free-text clinician dictation and full HL7 CDA
XML stay on the agent path.

---

## v63 follow-up — same recipe, longer context

After v62 shipped, we ran the future-path experiment the v62 doc
above named: same recipe, `max_seq_length` 512 → 1024, no extra training
data. Run record and full log live at
`tools/autoresearch/v63-experiment/EXPERIMENT.md` and
`tools/autoresearch/v63-experiment/v63_kernel.log`; LoRA at
`models/cliniq-gemma4-e2b-v63-lora/`.

**Result on the same `val-compact` 200-case held-out split:**

| Metric | v62 (max_seq=512) | v63 (max_seq=1024) | Delta |
|---|---:|---:|---:|
| Micro-F1 | 0.823 | **0.9989** | +0.176 |
| Micro-precision | 0.979 | **0.9989** | +0.020 |
| Micro-recall | 0.710 | **0.9989** | +0.289 |
| JSON validity | 0.86 | **1.00** | +14 pp |
| Schema-complete | 0.86 | **1.00** | +14 pp |
| Cases ≥ F1 0.70 | 162 / 200 | **200 / 200** | +38 |
| Train wall-clock (T4) | 1h 04m | 3h 04m | 2.85× |

The v62 future-path cell predicted F1 ~0.90 / JSON-valid ~95%; v63 cleared
both. The 14-pp truncation gap was entirely max-context-driven —
`packing=True` at `max_seq_length=512` was clipping the long-expansion
training cases mid-example, so the model never learned to close their
JSON. Doubling the context let those cases train fully and the gap
collapsed.

**The Unsloth-specific lever that made this affordable on T4:**
`use_gradient_checkpointing="unsloth"` (~30% peak-VRAM cut vs vanilla
`gradient_checkpointing=True`). Without it, `max_seq_length=1024` +
`packing=True` would have OOM'd on a 16 GB T4 at batch=1; v63's 3h 04m
T4 train is the direct payoff of that recipe.

**Latency claim status.** The 38.1 s p50 in the kernel log is unquantized
PyTorch on T4 with `model.generate(max_new_tokens=1024)` — not comparable
to v62's 4.1 s Mac-Metal-Q3_K_M number. A like-for-like Mac re-bench of
the v63 GGUF is pending and not part of this submission's reported
latency line.

---

## 2026-05-11 update — v63 Mac re-bench discovers a quantization regression

The Mac Q3_K_M re-bench landed at F1 = 0.548 / precision = 0.393 vs v62's
F1 = 0.837 / precision = 0.837 on the same val-compact 200-case split with
the same `bench_v62_singleshot.py` harness. The 0.9989 Kaggle number was
unquantized PyTorch and did not survive Q3_K_M quantization. Diagnosis:
v63's LoRA safetensors has 410 tensors vs v62's 490 — missing k_proj +
v_proj LoRA weights on the 20 global-attention layers (15-34) of Gemma 4's
hybrid attention. Probable cause: a release of unsloth/peft ≥ 0.18 silently
treats global-attention k/v as non-trainable for Gemma 4. v62 (trained
2026-04-30) predates this regression. Full diagnosis at
`tools/autoresearch/v63-experiment/EXPERIMENT.md` → "Mac Q3_K_M re-bench".

**v63's latency win is real** (p50 2.79 s vs v62 3.08 s, 9% faster) — that
part survives quantization. Only quality regresses.

**For this submission, v62 is the shipped Unsloth-track LoRA.** v63b
(retrain with explicit `layers_to_transform=list(range(35))` and a
fail-fast tensor-count assertion) is queued at
`tools/autoresearch/v63b-experiment/`. If v63b recovers F1 ≥ v62 at v63
latency before the 2026-05-18 deadline, it becomes the shipped LoRA.
Either way v62 remains on HF Hub for transparency and as the
deadline-safe option.
