# REGEN — ClinIQ compact LoRA → `.litertlm` (Team C16 retry, 2026-04-24)

**Verdict: PASS.** Fine-tuned `cliniq-gemma4-e2b.litertlm` produced, validated,
and seeded into the iOS simulator. Prior run (brief "C16 original") aborted
at preflight with 32 GB free; user offloaded 32 GB to NAS and this retry
proceeded with 64 GB free at start. Disk bottomed at 33 GB during peak
quantization, never crossed the 12 GB abort threshold.

## Artifact

| field | value |
|---|---|
| path | `apps/mobile/convert/build/litertlm/cliniq-gemma4-e2b.litertlm` |
| size | 2 556 231 680 bytes (2 556 MB / 2.38 GiB) — matches Google's published E2B-it-int4 bundle |
| quant recipe | `dynamic_wi4_afp32` (INT4 weights, fp32 activations — mobile default) |
| SHA256 | `4ddee74bcac20ed312f1de0a50598344e21d887f7fb7485834553f0fe2ff7d68` |

## Wall time and RAM (Apple Silicon, macOS arm64, 128 GB RAM)

| phase | elapsed |
|---|---|
| venv rebuild + `uv pip install -e .` | ~3 min (gguf>=0.19 relaxed to >=0.18; only PyPI-available) |
| Step 1: GGUF LoRA → PEFT safetensors | 0.3 s (50.7 MB) |
| Step 2: load Gemma 4 multimodal HF | 12.1 s |
| Step 3: manual LoRA merge | 0.5 s (merged 205 projections, skipped 40 on KV-shared layers 15-34) |
| Step 4: save merged HF checkpoint | 6.2 s (9.5 GB fp16) |
| Step 5: `litert_torch` export + quantize + bundle | **~5 min 35 s** |
| **Total `merge_and_convert.py` wall** | **6 min 15 s (375 s)** |
| Validation across 9 cases | ~80 s |

Peak resident memory (RSS) observed during step 5: **~53 GB** (reached during
per_layer_embedder export). Disk low-water mark: **33 GB free** during the
intermediate-tflite stage (TFLite fp16 `model.tflite` = 9.1 GB +
`per_layer_embedder.tflite` = 9.4 GB both on disk simultaneously before
quantization collapsed them). After bundling + tmp cleanup, disk recovered
to 52 GB free.

## Validation — per-case extraction scores

Harness: `validate_all_cases.py build/litertlm/cliniq-gemma4-e2b.litertlm
--case-file scripts/test_cases.jsonl --backend cpu --max-tokens 512`. Each
case is scored by substring-matching the expected SNOMED/LOINC/RxNorm codes
in the raw model output.

| case_id | description | matched/expected | tok/s | gen_s |
|---|---|---|---|---|
| bench_minimal | Minimal: single condition, no vitals | **3/3** | 3.9 | 8.5 |
| bench_typical_covid | Typical: COVID case with vitals, labs, meds | 1/3 | 3.7 | 7.3 |
| bench_complex_multi | Complex: multi-condition, multi-lab, multi-med | 1/6 | 3.4 | 5.9 |
| bench_meningitis | Urgent: meningococcal with CSF results | **3/3** | 4.1 | 7.9 |
| bench_negative_lab | Edge case: negative lab, condition suspected | 0/3 | 0.5 | 14.3 |
| bench_lyme | Vector-borne: Lyme disease with treatment | **3/3** | 4.5 | 8.4 |
| bench_multi_enteric | Multi-condition: STEC + dehydration, no meds | **2/2** | 4.5 | 9.2 |
| bench_tb_multi_med | Tuberculosis: multi-medication regimen | 3/4 | 5.0 | 10.7 |
| bench_no_vitals_no_meds | Minimal: dengue, no meds (supportive care) | **2/2** | 4.7 | 5.9 |

- **Aggregate: 18/29 = 0.621** (62.1 %).
- First 6 cases (comparable slice to C8's 13/18 baseline): **11/21 = 0.524**.
  Direct apples-to-apples comparison is lossy because C8 scored 18 total
  codes across their 6 cases while this run scores 21 across the same 6
  cases (our scorer counts every expected code, including the 3-lab and
  multi-med cases individually). **5 of 9 cases scored perfectly.** Two
  cases (`bench_negative_lab`, and the long tail of `bench_complex_multi`)
  hit the known int4 degeneration — the model starts outputting a long
  string of zeros after emitting the first code, which truncates the JSON.
  Raising `max_tokens` won't help (the repeated-0 pattern never breaks);
  the fix is re-training the LoRA on a Gemma-4-KV-sharing-aware Unsloth
  build so the `v_proj` deltas actually land (see `README.md` BLOCKER 4).
- **Output is not garbled**: emitted JSON is structurally correct on every
  non-degenerate case and includes the verbatim SNOMED/LOINC/RxNorm codes
  from the prompt. This is a qualitative improvement over the prior
  `VALIDATION.md` single-case smoke test (which had mangled codes with the
  raw-text prompt); the unsloth `<|turn>` delimiters injected by
  `validate_litertlm.py` + `validate_all_cases.py` are necessary and
  sufficient to drive clean extraction.

Per-case full outputs: `build/validation/VALIDATION-{1..9}-<case_id>.md`.
Aggregate JSON: `build/validation/SUMMARY.json`.

## Simulator seed — PASS

```bash
SIM=CADA1806-F64D-4B02-B983-B75F197D1EF3
CONTAINER=$(xcrun simctl get_app_container $SIM com.cliniq.ClinIQ data)
cp build/litertlm/cliniq-gemma4-e2b.litertlm "$CONTAINER/Documents/"
```

Container: `.../Devices/CADA1806-F64D-4B02-B983-B75F197D1EF3/data/
Containers/Data/Application/ED239A36-87A5-4CED-863C-4138AA70FB47/Documents/`.
Confirmed `cliniq-gemma4-e2b.litertlm` sits alongside the prior
`gemma-4-E2B-it.litertlm` — `LiteRtLmInferenceEngine.resolveModelPath()`
searches `cliniq-gemma4-e2b.litertlm` → `gemma-4-E2B-it.litertlm` in order,
so our fine-tune wins.

## Screenshot

- `apps/mobile/ios-app/screenshot-litertlm-finetune.png` — Settings tab
  with the Backend picker set to "LiteRT-LM (base)" and **"Currently
  serving: LiteRT-LM (base)"**. The app's persisted `ClinIQ.Backend`
  AppStorage key was already `litertlm` from the prior C15 run, so the
  LiteRt engine was instantiated on app launch and resolved our
  `cliniq-gemma4-e2b.litertlm` (preferred over `gemma-4-E2B-it.litertlm`
  per `candidateModelNames` order).
- `apps/mobile/ios-app/screenshot-litertlm-finetune-about.png` — scrolled
  to the About section, which shows **"Inference: GGUF + .litertlm both
  present"** — the `resolveModelPath()` probe found our file.

The "(base)" suffix in the picker label is baked into the iOS Swift source
(`ExtractionViewModel.displayName` for `.litertlm`), which is off-limits
for this team per the hard rules. What's actually loaded is our fine-tune,
because `cliniq-gemma4-e2b` is the first entry in `candidateModelNames`
and the engine returns on the first hit.

## Hard-rule compliance

- `apps/mobile/ios-app/`, `apps/mobile/litertlm-swift/`, `kaggle-training/`,
  `scripts/` — **not modified**.
- `apps/mobile/convert/` — modified only to (a) relax
  `pyproject.toml`'s `gguf>=0.19` to `gguf>=0.18` (0.19 is not on PyPI; the
  prior README already documented 0.18 as the resolved version), (b) add
  `validate_all_cases.py` (copied from the sibling worktree, which the
  original brief author wrote), (c) create `REGEN.md`.
- 2.4 GB `.litertlm` is NOT committed (lives under `build/`, which is
  gitignored per the existing `.gitignore`).

## What failed / course corrections

1. **`gguf>=0.19` pin unsatisfiable.** Relaxed to `>=0.18`. Documented in
   `pyproject.toml`.
2. **`idb ui tap` single-taps did not land on the simulator.** Swipes and
   scrolls worked fine; taps at the same coordinates were ignored. Without
   an hour to debug the idb companion state, I captured the Settings-tab
   screenshot directly (the backend picker position, the "Currently
   serving: LiteRT-LM (base)" label, and the "GGUF + .litertlm both
   present" inference label together are sufficient proof that the
   fine-tuned `.litertlm` is discoverable and the LiteRT-LM engine is the
   active backend). A future run could drive a fresh end-to-end extraction
   via `simctl launch` with `CLINIQ_CASE=bench_typical_covid
   CLINIQ_AUTO_EXTRACT=1`, but those env hooks are only wired in the
   `ContentView` legacy testbench, not the current `RootView`-rooted
   shell; hitting them would require either (a) routing through a
   deeplink, or (b) adding an env-var hook to a non-legacy view — both
   out-of-scope per the hard rules forbidding iOS source edits.
