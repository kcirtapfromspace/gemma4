// C ABI shim around litert::lm::Engine and litert::lm::Session. Translates
// the blocking RunPrefill/RunDecode methods into the five calls declared in
// litertlm_c_shim.h.
//
// This file is NOT compiled by SPM — SPM consumes the headers and the
// prebuilt xcframework. The .cc is compiled by the Bazel rule declared in
// apps/mobile/litertlm-swift/scripts/bazel_overlay/BUILD.bazel, which is
// staged into the upstream LiteRT-LM tree by build_xcframework.sh.
//
// iOS builds disable C++ exceptions (`-fno-exceptions`), so every entry
// point must return status codes and avoid try/catch. Upstream LiteRT-LM
// is StatusOr-first, so exceptions are almost never raised; on the rare
// case where an internal library throws, the process will terminate — an
// acceptable trade for the much cleaner symbol layout.

#include "include/litertlm_c_shim.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"

namespace {

using ::litert::lm::Backend;
using ::litert::lm::DecodeConfig;
using ::litert::lm::Engine;
using ::litert::lm::EngineFactory;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputData;
using ::litert::lm::InputText;
using ::litert::lm::ModelAssets;
using ::litert::lm::Responses;
using ::litert::lm::SessionConfig;

Backend BackendFromInt(int backend) {
  switch (backend) {
    case LITERTLM_BACKEND_GPU:
      return Backend::GPU;
    case LITERTLM_BACKEND_NPU:
      return Backend::NPU;
    case LITERTLM_BACKEND_CPU:
    default:
      return Backend::CPU;
  }
}

}  // namespace

struct LiteRtLmEngineOpaque {
  std::unique_ptr<Engine> engine;
};

struct LiteRtLmSessionOpaque {
  std::unique_ptr<Engine::Session> session;
  // Buffer holding the most recently produced response text. decode_step
  // pops bytes off the front until drained, then advances the model one
  // more step. Naive — enough to prove the binding path, not efficient.
  std::string pending_output;
  // Number of tokens produced by the most recent RunDecode. Populated
  // from Responses::GetTokenLengths() when available; falls back to a
  // whitespace-ish heuristic otherwise so tok/s can still be estimated.
  int last_token_count = 0;
  // User-supplied cap applied on the next RunDecode. <=0 means use the
  // model's default. Consumed at the first decode_step call for the turn.
  int max_output_tokens = 0;
  bool finished = false;
};

extern "C" {

const char* litertlm_shim_version(void) {
  // 0.2.0+team-c14: adds decode_set_max_tokens + last_token_count, exposes
  // token counts to Swift so tok/s can be measured from the test harness.
  return "0.2.0+team-c14";
}

LiteRtLmEngine* litertlm_engine_create(const char* model_path, int backend) {
  if (model_path == nullptr) return nullptr;
  auto assets = ModelAssets::Create(std::string(model_path));
  if (!assets.ok()) return nullptr;
  auto settings =
      EngineSettings::CreateDefault(*assets, BackendFromInt(backend));
  if (!settings.ok()) return nullptr;
  auto engine = EngineFactory::CreateDefault(*std::move(settings));
  if (!engine.ok()) return nullptr;
  auto* out = new LiteRtLmEngineOpaque();
  out->engine = std::move(*engine);
  return out;
}

void litertlm_engine_destroy(LiteRtLmEngine* engine) {
  delete engine;  // nullptr-safe
}

LiteRtLmSession* litertlm_session_create(LiteRtLmEngine* engine) {
  if (engine == nullptr || engine->engine == nullptr) return nullptr;
  auto cfg = SessionConfig::CreateDefault();
  auto session = engine->engine->CreateSession(cfg);
  if (!session.ok()) return nullptr;
  auto* out = new LiteRtLmSessionOpaque();
  out->session = std::move(*session);
  return out;
}

void litertlm_session_destroy(LiteRtLmSession* session) {
  delete session;  // nullptr-safe
}

int litertlm_session_prefill(LiteRtLmSession* session, const char* text) {
  if (session == nullptr || session->session == nullptr || text == nullptr) {
    return -1;
  }
  std::vector<InputData> inputs;
  // InputText's ctor accepts std::variant<std::string, TensorBuffer>;
  // std::string converts to the variant's first alternative.
  inputs.emplace_back(InputText(std::string(text)));
  auto status = session->session->RunPrefill(inputs);
  return status.ok() ? 0 : -2;
}

int litertlm_session_decode_step(LiteRtLmSession* session,
                                 char* out_buf,
                                 int out_cap) {
  if (session == nullptr || session->session == nullptr || out_buf == nullptr ||
      out_cap <= 0) {
    return -1;
  }
  // Naive: on first call, run a full blocking decode and buffer the first
  // candidate's text. Subsequent calls drain the buffer.
  if (session->pending_output.empty() && !session->finished) {
    absl::StatusOr<Responses> responses = [&]() {
      if (session->max_output_tokens > 0) {
        DecodeConfig cfg = DecodeConfig::CreateDefault();
        cfg.SetMaxOutputTokens(session->max_output_tokens);
        return session->session->RunDecode(cfg);
      }
      return session->session->RunDecode();
    }();
    if (!responses.ok()) return -2;
    const auto& texts = responses->GetTexts();
    if (!texts.empty()) {
      session->pending_output = texts.front();
    }
    // Capture token count for tok/s reporting. Prefer the model's own
    // tally when it reports one; otherwise fall back to a whitespace
    // approximation so the Swift test can still compute a meaningful
    // throughput number.
    const auto& maybe_token_lengths = responses->GetTokenLengths();
    int tokens = 0;
    if (maybe_token_lengths.has_value() && !maybe_token_lengths->empty()) {
      tokens = (*maybe_token_lengths)[0];
    } else if (!session->pending_output.empty()) {
      // Space-tokenised estimate; deliberately rough but non-zero.
      tokens = 1;
      for (char c : session->pending_output) {
        if (c == ' ' || c == '\n' || c == '\t') tokens++;
      }
    }
    session->last_token_count = tokens;
    session->finished = true;
  }
  if (session->pending_output.empty()) return 0;  // EOS
  int n = static_cast<int>(session->pending_output.size());
  if (n > out_cap) n = out_cap;
  std::memcpy(out_buf, session->pending_output.data(),
              static_cast<size_t>(n));
  session->pending_output.erase(0, static_cast<size_t>(n));
  return n;
}

int litertlm_session_decode_set_max_tokens(LiteRtLmSession* session,
                                           int max_output_tokens) {
  if (session == nullptr || session->session == nullptr) return -1;
  // Ignored once a decode is in flight — the cap is sticky per turn and
  // is applied lazily when decode_step runs RunDecode. Calling this after
  // decode_step has returned data is a no-op rather than an error so the
  // Swift wrapper can set it defensively before every prefill.
  session->max_output_tokens = max_output_tokens;
  return 0;
}

int litertlm_session_last_token_count(LiteRtLmSession* session) {
  if (session == nullptr || session->session == nullptr) return -1;
  return session->last_token_count;
}

}  // extern "C"
