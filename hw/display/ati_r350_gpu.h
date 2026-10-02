/*
 * ATI R300/R350 -- the host-GPU backends behind ati_r350_gl.h.
 *
 * The device talks to ONE interface, ati_r350_gl.h, and never learns
 * which host API is underneath it. There are two:
 *
 *   r350_ogl_*   ati_r350_gl.c    OpenGL 3.3 core (CGL, WGL)
 *   r350_mtl_*   ati_r350_mtl.m   Metal, Apple-silicon GPUs
 *
 * Each implementation file defines the interface's own names to its
 * prefix before including ati_r350_gl.h, so its source reads exactly
 * like the interface and its symbols come out under the prefix. The
 * public ati_r350_gl_* entry points are then ati_r350_gpu.c, which
 * holds whichever one was opened and forwards to it. The prototypes
 * below are what that forwarding compiles against.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_R350_GPU_H
#define ATI_R350_GPU_H

#include "ati_r350_gl.h"

#define R350_GPU_DECLARE(T, P)                                              \
    typedef struct T T;                                                     \
    T *P##_open(const char **err);                                          \
    void P##_close(T *g);                                                   \
    bool P##_target(T *g, int w, int h, bool *lost);                        \
    bool P##_seed(T *g, int x0, int y0, int w, int h,                       \
                  const uint8_t *base, unsigned pitch, unsigned xr);        \
    bool P##_fetch(T *g, int x0, int y0, int w, int h,                      \
                   uint8_t *base, unsigned pitch, unsigned xr);             \
    bool P##_draw(T *g, const R350GlReq *req);                              \
    const char *P##_describe(T *g);                                         \
    void P##_prog_stats(T *g, uint64_t *hits, uint64_t *links,              \
                        uint64_t *failed);                                  \
    uint64_t P##_barriers(T *g);                                            \
    void P##_queue_stats(T *g, uint64_t *units, uint64_t *flushes,          \
                         uint64_t *waves);

/*
 * An implementation file defines its R350_GPU_IMPL_* first: its own
 * prototypes then come from ati_r350_gl.h under its names, and are not
 * declared a second time here.
 */
#ifndef R350_GPU_IMPL_OGL
R350_GPU_DECLARE(R350OglCtx, r350_ogl)
#endif

#if defined(CONFIG_DARWIN) && !defined(R350_GPU_IMPL_MTL)
R350_GPU_DECLARE(R350MtlCtx, r350_mtl)
/* the depth buffer, which only the Metal backend has */
bool r350_mtl_zseed(R350MtlCtx *g, int x0, int y0, int w, int h,
                    const uint32_t *z);
bool r350_mtl_zfetch(R350MtlCtx *g, int x0, int y0, int w, int h,
                     uint32_t *z);
/* zero-copy: emulated VRAM as the backend's own buffer */
bool r350_mtl_vram(R350MtlCtx *g, void *ptr, uint64_t size);
bool r350_mtl_sync(R350MtlCtx *g);
uint64_t r350_mtl_commit(R350MtlCtx *g);
uint64_t r350_mtl_done(R350MtlCtx *g);
uint64_t r350_mtl_next(R350MtlCtx *g);
bool r350_mtl_idle(R350MtlCtx *g);
void r350_mtl_notify(R350MtlCtx *g, void (*fn)(void *), void *opaque);
#endif

#endif /* ATI_R350_GPU_H */
