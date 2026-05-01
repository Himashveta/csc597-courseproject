/*
 * mc_probe.h -- branch-variable probe macros.
 *
 * The PolyFuzz design harvests, at every branch, the runtime values
 * of the variables in the predicate. We do this without an LLVM
 * pass: the C source manually wraps each branch predicate in a
 * MC_PROBE(branch_id, var, ...) macro that emits
 *
 *     branch_id<TAB>name=value;name=value;...<NEWLINE>
 *
 * to the file descriptor named by the MC_TRACE_FD environment
 * variable (decimal integer). When the env var is unset, the
 * macros are no-ops; ordinary fuzzing runs are unaffected.
 *
 * This approach is the pragmatic fallback for not having an LLVM
 * pass. Its limitation is that it requires source modification --
 * which for our mock target is fine (we wrote the source) but for
 * PyTorch is exactly the obstacle we document in the report.
 *
 * Probe IDs are assigned manually as small integers; we keep them
 * stable across runs so the harness can index a per-branch state.
 */
#ifndef MC_PROBE_H
#define MC_PROBE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The probe stream is opened lazily on first use. */
extern FILE *_mc_probe_stream(void);

/* Emit a single probe record. branch_id is a small integer; payload
 * is a printf-style format string and varargs that should render
 * "name=value" pairs separated by ';'. */
static inline void _mc_probe_emit(int branch_id, const char *payload) {
    FILE *f = _mc_probe_stream();
    if (f != NULL) {
        fprintf(f, "%d\t%s\n", branch_id, payload);
    }
}

/* Convenience macros for common shapes. Each one renders the
 * relevant branch variables into a small buffer and emits.
 *
 * MC_PROBE_I1(id, var)            -- one int variable
 * MC_PROBE_I2(id, a, b)           -- two int vars
 * MC_PROBE_I3(id, a, b, c)        -- three int vars
 *
 * A more general macro takes a pre-formatted payload string.
 */
#define MC_PROBE_I1(id, var) do { \
    char _mc_buf[64]; \
    snprintf(_mc_buf, sizeof(_mc_buf), #var "=%d", (int)(var)); \
    _mc_probe_emit((id), _mc_buf); \
} while (0)

#define MC_PROBE_I2(id, a, b) do { \
    char _mc_buf[128]; \
    snprintf(_mc_buf, sizeof(_mc_buf), #a "=%d;" #b "=%d", \
             (int)(a), (int)(b)); \
    _mc_probe_emit((id), _mc_buf); \
} while (0)

#define MC_PROBE_I3(id, a, b, c) do { \
    char _mc_buf[192]; \
    snprintf(_mc_buf, sizeof(_mc_buf), #a "=%d;" #b "=%d;" #c "=%d", \
             (int)(a), (int)(b), (int)(c)); \
    _mc_probe_emit((id), _mc_buf); \
} while (0)

#define MC_PROBE_RAW(id, payload) _mc_probe_emit((id), (payload))

#ifdef __cplusplus
}
#endif

#endif  /* MC_PROBE_H */
