# Convenience wrappers around the Bazel + SwiftPM build flow. All of these
# assume you've cloned LiteRT-LM to /tmp/c11-litertlm/LiteRT-LM (the build
# script can override via the LITERTLM_UPSTREAM env var).

SHELL := /bin/bash

LITERTLM_UPSTREAM ?= /tmp/c11-litertlm/LiteRT-LM
MODEL_PATH        ?= /tmp/gemma-model/gemma-4-E2B-it.litertlm
# First booted iOS simulator on this machine; override with SIMULATOR_ID.
SIMULATOR_ID      ?= $(shell xcrun simctl list devices booted | awk '/\(Booted\)/ { print $$NF }' | tr -d '()' | head -1)

.PHONY: help
help:
	@echo "Targets:"
	@echo "  clone            — shallow clone of LiteRT-LM into $$LITERTLM_UPSTREAM"
	@echo "  xcframework      — build LiteRtLmCore.xcframework via Bazel"
	@echo "  model            — download gemma-4-E2B-it.litertlm into $$MODEL_PATH"
	@echo "  test             — swift test (requires xcframework + model)"
	@echo "  clean            — wipe build/ and SwiftPM derived data"

.PHONY: clone
clone:
	@test -d "$(LITERTLM_UPSTREAM)" || \
	  git clone --depth 1 https://github.com/google-ai-edge/LiteRT-LM.git "$(LITERTLM_UPSTREAM)"

.PHONY: xcframework
xcframework: clone
	LITERTLM_UPSTREAM="$(LITERTLM_UPSTREAM)" ./scripts/build_xcframework.sh

.PHONY: model
model:
	./scripts/fetch_model.sh "$(MODEL_PATH)"

.PHONY: test
test:
	@if [ -z "$(SIMULATOR_ID)" ]; then \
	  echo "error: no booted iOS simulator. Boot one in Simulator.app, then re-run." >&2; \
	  exit 1; \
	fi
	@echo "==> copying model into simulator sandbox"
	@SIM_ROOT="$$HOME/Library/Developer/CoreSimulator/Devices/$(SIMULATOR_ID)/data"; \
	 mkdir -p "$$SIM_ROOT/private/tmp/gemma-model"; \
	 cp -n "$(MODEL_PATH)" "$$SIM_ROOT/private/tmp/gemma-model/" || true
	TEST_RUNNER_LITERTLM_MODEL_PATH=/private/tmp/gemma-model/$$(basename "$(MODEL_PATH)") \
	  xcodebuild -scheme LiteRtLm \
	    -destination 'platform=iOS Simulator,id=$(SIMULATOR_ID)' \
	    test

# Same as `test` but also opts in to the decode smoke test. Decode on
# the iOS Simulator CPU backend is SLOW (minutes per 32-token run), so
# we gate it behind LITERTLM_RUN_DECODE=1 to keep routine test runs fast.
.PHONY: test-decode
test-decode:
	@if [ -z "$(SIMULATOR_ID)" ]; then \
	  echo "error: no booted iOS simulator. Boot one in Simulator.app, then re-run." >&2; \
	  exit 1; \
	fi
	@echo "==> copying model into simulator sandbox"
	@SIM_ROOT="$$HOME/Library/Developer/CoreSimulator/Devices/$(SIMULATOR_ID)/data"; \
	 mkdir -p "$$SIM_ROOT/private/tmp/gemma-model"; \
	 cp -n "$(MODEL_PATH)" "$$SIM_ROOT/private/tmp/gemma-model/" || true
	TEST_RUNNER_LITERTLM_MODEL_PATH=/private/tmp/gemma-model/$$(basename "$(MODEL_PATH)") \
	TEST_RUNNER_LITERTLM_RUN_DECODE=1 \
	  xcodebuild -scheme LiteRtLm \
	    -destination 'platform=iOS Simulator,id=$(SIMULATOR_ID)' \
	    -only-testing:LiteRtLmTests/DecodeTest \
	    test 2>&1 | tee build/decode-test.log | \
	      grep -E 'LITERTLM_DECODE_RESULT|LITERTLM_DECODE_SAMPLE|Test Suite.*(passed|failed)|error:' || true
	@echo ''
	@echo '==> tok/s (parsed from test output):'
	@grep -E 'LITERTLM_DECODE_RESULT' build/decode-test.log | tail -1 || echo '  (no result line found; see build/decode-test.log)'

# Build + invoke the CLI on the booted simulator. Pass a prompt via PROMPT=,
# e.g. `make cli PROMPT='The capital of France is'`.
PROMPT ?= The capital of France is
.PHONY: cli
cli:
	@if [ -z "$(SIMULATOR_ID)" ]; then \
	  echo "error: no booted iOS simulator." >&2; exit 1; fi
	@SIM_ROOT="$$HOME/Library/Developer/CoreSimulator/Devices/$(SIMULATOR_ID)/data"; \
	 mkdir -p "$$SIM_ROOT/private/tmp/gemma-model"; \
	 cp -n "$(MODEL_PATH)" "$$SIM_ROOT/private/tmp/gemma-model/" || true
	xcodebuild -scheme LiteRtLmCli \
	  -destination 'platform=iOS Simulator,id=$(SIMULATOR_ID)' \
	  build -quiet
	@CLI=$$(find "$$HOME/Library/Developer/Xcode/DerivedData" \
	  -name LiteRtLmCli -path '*Debug-iphonesimulator*' -type f | head -1); \
	 test -n "$$CLI" || (echo "error: LiteRtLmCli binary not found after build" >&2; exit 1); \
	 SIMCTL_CHILD_LITERTLM_MODEL_PATH=/private/tmp/gemma-model/$$(basename "$(MODEL_PATH)") \
	 SIMCTL_CHILD_LITERTLM_MAX_TOKENS=64 \
	   xcrun simctl spawn $(SIMULATOR_ID) "$$CLI" $(PROMPT)

.PHONY: clean
clean:
	rm -rf build/ .build/ .swiftpm/ Package.resolved
