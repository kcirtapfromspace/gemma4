#!/usr/bin/env bash
# Tail llama-server logs looking for spec-decode acceptance stats.
# After each /v1/chat/completions completes, llama-server prints lines like:
#   slot print_timing: id 0 | task N
#   draft accept stats: n_drafted = X, n_accept = Y
# Capture the ratio.
kubectl -n gemma4 logs --tail=400 llama-server-67d59fb5fb-sxmg5 2>&1 | \
  grep -iE "draft|n_accept|speculative|predicted_per_second|eval time" | tail -30
