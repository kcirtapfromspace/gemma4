// ClinIQApp.swift
// ClinIQ — offline eICR → FHIR JSON extractor, SwiftUI shell.
//
// Team C10 — 2026-04-23.
//
// The inference engine is factored behind the `InferenceEngine` protocol so
// that we can swap between the stub (simulator builds, no LiteRT-LM linked)
// and the LiteRT-LM engine (physical device / when the Swift package ships).
// See `Inference/InferenceEngine.swift`.

import SwiftUI

@main
struct ClinIQApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
