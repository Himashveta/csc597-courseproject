/*
 * mc_probe.c -- lazy initialisation of the branch-probe stream.
 *
 * Reads MC_TRACE_FD on first call. If unset, returns NULL and the
 * MC_PROBE macros become no-ops. If set to a numeric file descriptor,
 * fdopen()s it for write. This avoids one fopen-per-probe.
 *
 * The harness opens a per-seed probe file before exec, dups the fd
 * to a known number, and sets MC_TRACE_FD to that number.
 */
#include "mc_probe.h"

#include <stdio.h>
#include <stdlib.h>

static FILE *g_probe_stream = NULL;
static int   g_probe_initialised = 0;

FILE *_mc_probe_stream(void) {
    if (g_probe_initialised) {
        return g_probe_stream;
    }
    g_probe_initialised = 1;

    const char *env = getenv("MC_TRACE_FD");
    if (env == NULL || env[0] == '\0') {
        g_probe_stream = NULL;
        return NULL;
    }

    char *endp = NULL;
    long fd = strtol(env, &endp, 10);
    if (endp == env || *endp != '\0' || fd < 0) {
        g_probe_stream = NULL;
        return NULL;
    }

    g_probe_stream = fdopen((int)fd, "w");
    if (g_probe_stream != NULL) {
        /* Line-buffered so the parent process can read records as
         * the seed runs without the child needing to fflush. */
        setvbuf(g_probe_stream, NULL, _IOLBF, 0);
    }
    return g_probe_stream;
}
