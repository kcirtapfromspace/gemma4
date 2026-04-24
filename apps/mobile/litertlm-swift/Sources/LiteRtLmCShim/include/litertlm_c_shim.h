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
int litertlm_session_decode_step(LiteRtLmSession* session,
                                 char* out_buf,
                                 int out_cap);

// Report the version of the shim itself (not LiteRT-LM's version). This is
// cheap to call from Swift tests to confirm the dylib/xcframework is loaded.
const char* litertlm_shim_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LITERTLM_C_SHIM_H_
