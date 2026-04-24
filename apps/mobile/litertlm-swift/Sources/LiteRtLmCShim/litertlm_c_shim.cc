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
  bool finished = false;
};

extern "C" {

const char* litertlm_shim_version(void) {
  return "0.1.0+team-c11";
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
    auto responses = session->session->RunDecode();
    if (!responses.ok()) return -2;
    const auto& texts = responses->GetTexts();
    if (!texts.empty()) {
      session->pending_output = texts.front();
    }
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

}  // extern "C"
