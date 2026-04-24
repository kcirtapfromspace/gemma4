# Legacy: C10/C12 developer testbench

Team C13 (2026-04-23) replaced the developer-facing JSON-dumper UI with a
clinician field case-reporting PoC. The previous `ContentView` is retained
in the source tree as a developer testbench and is **deprecated** for
user-facing purposes.

## What stays

- `ClinIQ/ClinIQ/Views/ContentView.swift` — legacy "paste eICR, see JSON"
  screen. Still compiled but not wired into `ClinIQApp`. Useful for
  inspecting raw model output while iterating on prompts.
- `ClinIQ/ClinIQ/Views/ExtractionViewModel.swift` — the view model that
  drives `ContentView`. Preserves the headless env-var auto-extract path
  (`CLINIQ_AUTO_EXTRACT=1`, `CLINIQ_CASE=<id>`) used by
  `BUILD.md` and `VALIDATION.md` for per-case scoring.
- `ClinIQ/ClinIQ/Models/TestCase.swift` — the five C12 test cases the
  testbench references.
- `validate.swift` — headless validator (stub regex path).
- `screenshot.png`, `screenshot-llamacpp.png`,
  `screenshot-llamacpp-neglab.png` — C10/C12 evidence of the developer
  path working.

## What was removed

- `ContentView` is no longer the app root. That role is now
  `RootView.swift`.
- No public navigation into the testbench exists in the C13 shell; re-
  enabling it is a one-line swap in `ClinIQApp.body` if an engineer needs
  to compare raw JSON output.

## Why not delete outright?

1. `VALIDATION.md` references the auto-extract path for reproducibility.
2. The five bundled `TestCase` fixtures are still useful for future
   benchmarking.
3. The stub engine is still required for CI / SwiftUI Previews, and the
   current C13 `ReviewFlowView` uses it via `ExtractionService`. Removing
   the testbench view without reshaping CI would be disruptive.

## Turning the testbench back on (for future debugging)

In `ClinIQApp.swift`, replace `RootView()` with `ContentView()` inside the
`WindowGroup` builder. That's the entire flip.
