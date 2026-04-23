# Phase 2 Sketch — Minimal LiteRT-LM integration for iOS

Goal: prove the integration surface without actually building. Shows the three critical steps: (1) bundle/download the merged `.litertlm`, (2) instantiate the LiteRT-LM engine, (3) run one inference.

**This code is not compiled.** It mirrors the published LiteRT-LM Kotlin API (stable as of v0.10.2) and the iOS Swift path documented as "In Dev, specialized Metal support coming soon" in the LiteRT-LM overview; the Swift surface in `google-ai-edge/gallery` iOS app is the canonical reference. If the iOS Swift SDK is still in-dev at integration time (worst case), fall back to embedding the LiteRT-LM C++ core via a Swift bridging header — the AI Edge Gallery iOS source does exactly this.

---

## Android (Kotlin) — fully stable

```kotlin
// app/src/main/java/com/cliniq/demo/EicrExtractor.kt
package com.cliniq.demo

import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Backend

class EicrExtractor(modelPath: String) {
    private val engine: Engine

    init {
        val config = EngineConfig(
            modelPath = modelPath, // e.g. context.filesDir.resolve("cliniq-gemma4-e2b.litertlm").path
            backend = Backend.GPU(), // GPU on flagships; fall back to Backend.CPU() on low-end
        )
        engine = Engine(config).also { it.initialize() }
    }

    suspend fun extract(eicrXml: String): String {
        val convo = engine.createConversation()
        val prompt = buildString {
            append(SYSTEM_PROMPT_CLINICAL_EXTRACTION) // ~200 tokens, shipped in res/raw
            append("\n\n<eICR>\n")
            append(eicrXml)
            append("\n</eICR>\n\nReturn JSON only.")
        }
        return convo.sendMessage(prompt) // streams internally; returns full string
    }

    fun close() = engine.close()
}
```

Dependencies (Gradle):

```kotlin
// app/build.gradle.kts
dependencies {
    implementation("com.google.ai.edge.litertlm:litertlm-core:0.10.2")
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.10.2")
}
```

Model asset: `app/src/main/assets/cliniq-gemma4-e2b.litertlm` (2.58 GB — over Play Store's 200 MB base APK limit; use Play Asset Delivery or first-run download from a signed URL).

---

## iOS (Swift) — LiteRT-LM path, expected GA imminent

```swift
// CliniQ/EicrExtractor.swift
import Foundation
import LiteRtLM // from AI Edge Gallery iOS; bridges the LiteRT-LM C++ core through Metal

final class EicrExtractor {
    private let engine: LiteRtLmEngine

    init() throws {
        let modelURL = Bundle.main.url(forResource: "cliniq-gemma4-e2b", withExtension: "litertlm")!
        let config = LiteRtLmEngineConfig(
            modelPath: modelURL.path,
            backend: .gpu, // Metal on A17/A18/M-series; .cpu fallback otherwise
            maxTokens: 1024,
            temperature: 0.2,          // tighter for structured JSON extraction
            topK: 1                    // greedy for determinism in clinical outputs
        )
        engine = try LiteRtLmEngine(config: config)
        try engine.initialize()
    }

    func extract(eicrXml: String) async throws -> String {
        let convo = engine.createConversation()
        let prompt = """
        \(Self.systemPromptClinicalExtraction)

        <eICR>
        \(eicrXml)
        </eICR>

        Return JSON only.
        """
        return try await convo.sendMessage(prompt)
    }

    private static let systemPromptClinicalExtraction: String = {
        guard let url = Bundle.main.url(forResource: "cliniq_system_prompt", withExtension: "txt"),
              let s = try? String(contentsOf: url) else { return "" }
        return s
    }()
}
```

Swift Package Manager dependency (from the `google-ai-edge/gallery` iOS source — package name may shift on GA release):

```swift
// Package.swift or Xcode SPM UI
.package(url: "https://github.com/google-ai-edge/LiteRT-LM.git", from: "0.10.2"),
```

If the Swift package isn't published yet at build time, the current approach (documented in AI Edge Gallery iOS source) is to vendor the `libLiteRtLm.a` static library and expose a Swift wrapper via a bridging header. That's what the gallery app does today.

Info.plist addition for model download if not bundled:

```xml
<key>NSAppTransportSecurity</key>
<dict>
  <key>NSAllowsArbitraryLoads</key>
  <false/>
</dict>
```

Model distribution: bundle in `.app` for TestFlight (under 4 GB IPA limit, fine at 2.6 GB), or download from our signed HF endpoint on first launch for production (lets us retrain LoRA and push new models without an App Store review).

---

## LoRA merge → `.litertlm` conversion pipeline (host-side, run once per LoRA retrain)

```bash
# 1. Merge compact LoRA into base (HF transformers + PEFT)
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("google/gemma-4-e2b-it", torch_dtype="auto")
peft = PeftModel.from_pretrained(base, "models/cliniq-compact-lora")
merged = peft.merge_and_unload()
merged.save_pretrained("build/cliniq-gemma4-e2b-merged")
AutoTokenizer.from_pretrained("google/gemma-4-e2b-it").save_pretrained("build/cliniq-gemma4-e2b-merged")
PY

# 2. Convert merged HF model to int4 .litertlm via ai-edge-torch
pip install ai-edge-torch ai-edge-litert
python -m ai_edge_torch.generative.examples.gemma4.convert_to_litertlm \
    --checkpoint-path build/cliniq-gemma4-e2b-merged \
    --output-path build/cliniq-gemma4-e2b.litertlm \
    --quantize int4

# 3. Smoke-test on desktop via LiteRT-LM CLI (v0.10.1+)
litert-lm-cli --model build/cliniq-gemma4-e2b.litertlm \
              --prompt "Extract JSON from this eICR: ..."
```

Final artifact: `build/cliniq-gemma4-e2b.litertlm` ≈ **2.58 GB** at int4. Fits comfortably in a TestFlight IPA or a one-time app download on any phone with ≥4 GB free storage.

---

## Proof points this sketch is correct

- Kotlin snippet mirrors the exact example in [LiteRT-LM overview docs](https://ai.google.dev/edge/litert-lm/overview).
- Swift shape mirrors the deprecated-but-similar MediaPipe `LlmInference` Swift API, which the AI Edge Gallery iOS app currently wraps around the LiteRT-LM C++ core; see [google-ai-edge/gallery](https://github.com/google-ai-edge/gallery).
- `.litertlm` int4 file size 2.58 GB confirmed on [HF litert-community/gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm).
- `ai-edge-torch` Gemma 4 conversion script exists in the Google AI Edge repo per the Gemma 4 edge launch blog.
