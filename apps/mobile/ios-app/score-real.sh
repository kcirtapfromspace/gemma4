#!/usr/bin/env bash
# score-real.sh — parse `extractions.log` from the running simulator and
# score real-inference outputs against the expected codes in TestCase.bundled.
# Team C12 — 2026-04-23.
#
# Usage:
#   ./score-real.sh                 # uses default simulator UDID
#   ./score-real.sh <UDID>
#   ./score-real.sh --file <path>   # score a saved copy of the log
#
# Reads the simulator's `Documents/extractions.log` for the ClinIQ app and
# emits a per-case CSV + total score.

set -euo pipefail

if [ "${1:-}" = "--file" ]; then
  LOG="$2"
else
  UDID="${1:-CADA1806-F64D-4B02-B983-B75F197D1EF3}"
  CONTAINER=$(xcrun simctl get_app_container "$UDID" com.cliniq.ClinIQ data)
  LOG="$CONTAINER/Documents/extractions.log"
fi

if [ ! -f "$LOG" ]; then
  echo "extractions.log not found at $LOG" >&2
  exit 1
fi

python3 - "$LOG" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path).read()

expected = {
    "bench_minimal":       {"cond": ["76272004"],                  "loinc": ["20507-0"],           "rx": ["105220"]},
    "bench_typical_covid": {"cond": ["840539006"],                 "loinc": ["94500-6"],           "rx": ["2599543"]},
    "bench_complex_multi": {"cond": ["86406008"],                  "loinc": ["75622-1","57021-8","24467-3"], "rx": ["1999563","197696"]},
    "bench_meningitis":    {"cond": ["23511006"],                  "loinc": ["49672-8"],           "rx": ["1665021"]},
    "bench_negative_lab":  {"cond": ["50711007"],                  "loinc": ["11259-9"],           "rx": ["1940261"]},
}

# Parse log into (case, output) pairs. Last entry per case wins.
entries = re.findall(
    r'### case=(\S+) @ (\S+)\ntokens=(\d+) elapsed=(\S+) tps=(\S+)\nOUTPUT:\n(.+?)\n---',
    text, re.DOTALL)
# The very first run (before case= prefix was added) maps to the default
# loaded case `bench_typical_covid`.
first_entries = re.findall(
    r'### extraction @ (\S+)\ntokens=(\d+) elapsed=(\S+) tps=(\S+)\nOUTPUT:\n(.+?)\n---',
    text, re.DOTALL)
# Map: caseID -> (ts, tokens, elapsed, tps, output). Each later match overrides.
latest = {}
for ts, toks, ela, tps, out in first_entries:
    latest["bench_typical_covid"] = (ts, toks, ela, tps, out)
for case_id, ts, toks, ela, tps, out in entries:
    latest[case_id] = (ts, toks, ela, tps, out)

print("case_id,tokens,elapsed_s,tps,cond_hit,cond_max,loinc_hit,loinc_max,rx_hit,rx_max,score,max")
total_hit = 0
total_max = 0
for case_id, exp in expected.items():
    if case_id not in latest:
        cmax = len(exp["cond"]) + len(exp["loinc"]) + len(exp["rx"])
        total_max += cmax
        print(f'{case_id},-,-,-,0,{len(exp["cond"])},0,{len(exp["loinc"])},0,{len(exp["rx"])},0,{cmax}')
        continue
    ts, toks, ela, tps, out = latest[case_id]
    cond_hit = sum(1 for c in exp["cond"] if c in out)
    loinc_hit = sum(1 for c in exp["loinc"] if c in out)
    rx_hit = sum(1 for c in exp["rx"] if c in out)
    hit = cond_hit + loinc_hit + rx_hit
    cmax = len(exp["cond"]) + len(exp["loinc"]) + len(exp["rx"])
    total_hit += hit
    total_max += cmax
    print(f'{case_id},{toks},{ela},{tps},{cond_hit},{len(exp["cond"])},{loinc_hit},{len(exp["loinc"])},{rx_hit},{len(exp["rx"])},{hit},{cmax}')
print(f"# total: {total_hit}/{total_max}")
PY
