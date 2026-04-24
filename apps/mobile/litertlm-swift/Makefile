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

.PHONY: clean
clean:
	rm -rf build/ .build/ .swiftpm/ Package.resolved
