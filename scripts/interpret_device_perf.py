#!/usr/bin/env python3
"""Summarize a ClinIQ physical-device performance JSON export."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def pct(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = round((len(values) - 1) * fraction)
    return values[max(0, min(len(values) - 1, idx))]


def fmt(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"


def codes(row: dict[str, Any], bucket: str) -> list[str]:
    return [item.get("code", "") for item in row.get("extracted", {}).get(bucket, []) if item.get("code")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_json", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.export_json.read_text())
    cases = artifact.get("cases", [])
    measured = [c for c in cases if c.get("metrics", {}).get("elapsedSeconds", 0) > 0]
    elapsed = [c["metrics"]["elapsedSeconds"] for c in measured]
    tps = [c["metrics"]["tokensPerSecond"] for c in measured if c["metrics"].get("tokensPerSecond", 0) > 0]
    token_cases = [c for c in measured if c["metrics"].get("tokensGenerated", 0) > 0]

    app = artifact.get("app", {})
    device = artifact.get("device", {})
    config = artifact.get("configuration", {})

    print("# ClinIQ Device Performance Summary")
    print()
    print(f"- Export: `{args.export_json}`")
    print(f"- Generated: {artifact.get('generatedAt', 'unknown')}")
    print(f"- App: {app.get('displayName', 'ClinIQ')} {app.get('version', '?')} ({app.get('build', '?')})")
    print(f"- Device: {device.get('hardwareIdentifier', 'unknown')} / {device.get('systemName', 'iOS')} {device.get('systemVersion', '?')}")
    print(f"- Backend: {config.get('activeBackendLabel', 'unknown')}")
    print(f"- Cases: {len(cases)} total, {len(measured)} measured, {len(token_cases)} generated model tokens")
    print()
    print("## Latency")
    print(f"- Mean elapsed: {fmt(statistics.fmean(elapsed) if elapsed else None, ' s')}")
    print(f"- Median elapsed: {fmt(statistics.median(elapsed) if elapsed else None, ' s')}")
    print(f"- P95 elapsed: {fmt(pct(elapsed, 0.95), ' s')}")
    print(f"- Median token rate: {fmt(statistics.median(tps) if tps else None, ' tok/s')}")
    print()
    print("## Cases")
    for row in cases:
        metrics = row.get("metrics", {})
        conds = ",".join(codes(row, "conditions")) or "-"
        labs = ",".join(codes(row, "labs")) or "-"
        meds = ",".join(codes(row, "medications")) or "-"
        print(
            f"- {row.get('caseId', '?')[:8]} status={row.get('status', '?')} "
            f"elapsed={metrics.get('elapsedSeconds', 0):.3f}s "
            f"tok={metrics.get('tokensGenerated', 0)} "
            f"tps={metrics.get('tokensPerSecond', 0):.2f} "
            f"SNOMED=[{conds}] LOINC=[{labs}] RxNorm=[{meds}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
