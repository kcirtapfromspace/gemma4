# Public Kaggle notebook — v62 Unsloth submission

## Path

Duplicate `patrickdeutsch/cliniq-gemma4-finetune` (private, 62 versions) →
public version at the same slug *or* a clean
`patrickdeutsch/cliniq-gemma4-unsloth-submission`.

The public version should be a single notebook that:
1. Documents the project goal and the Unsloth-track-specific value
2. Trains end-to-end on Kaggle T4 ×2 (~1h 4m)
3. Saves the LoRA adapter (124 MB) to the kernel output
4. Runs the held-out val bench inline so judges see the numbers
5. Exports the LoRA in a format that's downloadable from the Output tab

## Cell plan

| Cell | Type | Content |
|---|---|---|
| 1 | md | Title + 200-word abstract: "Why this notebook exists, what it does, what's novel" |
| 2 | md | Track eligibility: explicit list of Unsloth APIs used + why each one matters |
| 3 | code | `pip install unsloth trl peft accelerate bitsandbytes` |
| 4 | md | Background: eICR / EZeCR / public health context (~150 words) |
| 5 | code | `FastLanguageModel.from_pretrained("unsloth/gemma-4-E2B-it", load_in_4bit=True)` |
| 6 | code | `FastLanguageModel.get_peft_model(...)` with the 7-target-module config |
| 7 | code | `get_chat_template(tokenizer, "gemma-4")` — explain why the template matters |
| 8 | code | Load `eicr-fhir-training-data` from Kaggle dataset attachment |
| 9 | code | `format_prompts` — apply the chat template to the conversations |
| 10 | code | `SFTTrainer` config: 5 epochs, lr=1e-4, packing=True, adamw_8bit |
| 11 | code | `trainer.train()` — runs ~1h 4m on T4 ×2 |
| 12 | md | Training curve: paste loss-by-step (final 0.2446) + brief read |
| 13 | code | Inline bench against val-compact (200 cases): code-level F1 + JSON validity |
| 14 | md | Results table: v62 vs base; per-axis breakdown |
| 15 | code | `model.save_pretrained("cliniq_lora")` — adapter to Output tab |
| 16 | md | "What to do with the adapter" — link to convert_lora_to_gguf.py + repo demo |
| 17 | md | Limitations + future work + citations |

## Public notebook diffs from private v62

| Change | Why |
|---|---|
| Add abstract + track-eligibility cells | Judges read top-of-notebook |
| Inline val-compact bench cell | Judges should see numbers without leaving Kaggle |
| Add per-axis F1 breakdown | Strengthens the story |
| Add "what's novel" callouts | Frames the contribution |
| Remove TODO / WIP markdown | Polish |
| Add citation block | Reproducibility |

## Steps to ship (post-bench-verdict)

1. Make `eicr-fhir-training-data` public (currently private)
2. Push notebook as a *new* public kernel: `cliniq-gemma4-unsloth-submission`
3. Run it from clean — verify reproducibility
4. Get the public URL
5. Add URL to:
   - `tools/autoresearch/hackathon-submission-2026-04-27.md`
   - Repo root `README.md`
   - HF model card "training notebook" link
   - Hackathon submission form

## Estimated time

- Notebook authoring: 2 hours
- Re-run from clean on public kernel: 1h 4m wall-clock (mostly waiting)
- Cross-link + verify: 30 min
- **Total: ~4h to a fully-shipped public Unsloth submission**

This is gated on the bench verdict landing as SHIP per the decision rule
in task #34.
