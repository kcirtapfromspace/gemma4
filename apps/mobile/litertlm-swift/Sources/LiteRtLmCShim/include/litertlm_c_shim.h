// Minimal C ABI wrapping a small subset of LiteRT-LM's C++ Engine/Session.
// Designed to be callable from Swift via a module map. Keep the surface
// absolutely minimal — five calls are enough to prove the binding path.
//
// This header intentionally does NOT include any LiteRT-LM headers so that
// Swift consumers only depend on this file plus a statically-linked
// xcframework that embeds all of LiteRT-LM's transitive deps.
//
// Error convention: functions returning int use 0 == OK, negative == error.
// Functions returning pointers return NULL on failure.

#ifndef LITERTLM_C_SHIM_H_
#define LITERTLM_C_SHIM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct LiteRtLmEngineOpaque LiteRtLmEngine;
typedef struct LiteRtLmSessionOpaque LiteRtLmSession;

// Backend selector. Mirrors LiteRT-LM's `Backend` enum, but held as an int
// here so the ABI does not depend on the upstream header.
//   0 = CPU
//   1 = GPU (Metal on iOS)
//   2 = NPU  (unsupported on iOS today, reserved)
typedef enum {
  LITERTLM_BACKEND_CPU = 0,
  LITERTLM_BACKEND_GPU = 1,
  LITERTLM_BACKEND_NPU = 2,
} LiteRtLmBackend;

// Create an engine from a `.litertlm` model file on disk.
// Returns NULL on failure.
LiteRtLmEngine* litertlm_engine_create(const char* model_path, int backend);

// Tear down an engine. Safe to pass NULL.
void litertlm_engine_destroy(LiteRtLmEngine* engine);

// Create a session against an existing engine.
// Returns NULL on failure.
LiteRtLmSession* litertlm_session_create(LiteRtLmEngine* engine);

// Destroy a session. Safe to pass NULL.
void litertlm_session_destroy(LiteRtLmSession* session);

// Run prefill on a UTF-8 prompt. Returns 0 on success.
int litertlm_session_prefill(LiteRtLmSession* session, const char* text);

// Run one decode step. Writes the next token's UTF-8 bytes (not
// necessarily null-terminated) into `out_buf`, up to `out_cap` bytes.
// Returns the number of bytes written on success, 0 for end-of-sequence,
// and a negative value on error.
//
// First call triggers a full blocking `RunDecode` (bounded by any prior
// `litertlm_session_decode_begin` call). Subsequent calls drain the
// buffered bytes. This is enough to prove the binding path and is
// compatible with Swift's `AsyncThrowingStream<String>` consumer.
int litertlm_session_decode_step(LiteRtLmSession* session,
                                 char* out_buf,
                                 int out_cap);

// Configure the next decode with an explicit max-output-token cap. Must
// be called BEFORE the first `litertlm_session_decode_step` for the
// current turn — once decoding has started, this is a no-op.
//
// Passing `max_output_tokens <= 0` clears any previous cap (model uses
// its default, e.g. the trained max sequence length).
//
// Returns 0 on success, negative on error. Used by the CLI and tests to
// bound decode latency on the simulator.
int litertlm_session_decode_set_max_tokens(LiteRtLmSession* session,
                                           int max_output_tokens);

// Returns the number of tokens produced by the most recent decode for
// this session, or 0 if none has run yet. Drives tok/s measurement in
// the Swift test. Negative on error.
int litertlm_session_last_token_count(LiteRtLmSession* session);

// Report the version of the shim itself (not LiteRT-LM's version). This is
// cheap to call from Swift tests to confirm the dylib/xcframework is loaded.
const char* litertlm_shim_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LITERTLM_C_SHIM_H_
