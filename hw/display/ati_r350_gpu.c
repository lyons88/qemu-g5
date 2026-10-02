/*
 * ATI R300/R350 -- choosing the host-GPU backend.
 *
 * The device calls the ati_r350_gl_* interface and nothing else. This
 * file is that interface: it opens the backend the `gl-api` property
 * names and forwards every call to it. See ati_r350_gpu.h for how the
 * two implementations share the interface's names.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "ati_r350_gpu.h"

typedef enum R350GpuApi {
    R350_GPU_OGL,
    R350_GPU_MTL,
} R350GpuApi;

struct R350GlCtx {
    R350GpuApi api;
    R350OglCtx *ogl;
#ifdef CONFIG_DARWIN
    R350MtlCtx *mtl;
#endif
};

#ifdef CONFIG_DARWIN
#define R350_GPU_CALL(g, fn, ...)                                           \
    ((g)->api == R350_GPU_MTL ? r350_mtl_##fn((g)->mtl, __VA_ARGS__)        \
                              : r350_ogl_##fn((g)->ogl, __VA_ARGS__))
#define R350_GPU_CALL0(g, fn)                                               \
    ((g)->api == R350_GPU_MTL ? r350_mtl_##fn((g)->mtl)                     \
                              : r350_ogl_##fn((g)->ogl))
#else
#define R350_GPU_CALL(g, fn, ...)  r350_ogl_##fn((g)->ogl, __VA_ARGS__)
#define R350_GPU_CALL0(g, fn)      r350_ogl_##fn((g)->ogl)
#endif

R350GlCtx *ati_r350_gl_open_api(const char *api, const char **err)
{
    R350GlCtx *g;

    *err = NULL;
    if (!api || !api[0] || !strcmp(api, "opengl")) {
        R350OglCtx *o = r350_ogl_open(err);

        if (!o) {
            return NULL;
        }
        g = g_new0(R350GlCtx, 1);
        g->api = R350_GPU_OGL;
        g->ogl = o;
        return g;
    }
    if (!strcmp(api, "metal")) {
#ifdef CONFIG_DARWIN
        R350MtlCtx *m = r350_mtl_open(err);

        if (!m) {
            return NULL;
        }
        g = g_new0(R350GlCtx, 1);
        g->api = R350_GPU_MTL;
        g->mtl = m;
        return g;
#else
        *err = "gl-api=metal is only available on macOS";
        return NULL;
#endif
    }
    *err = "gl-api must be one of opengl, metal";
    return NULL;
}

R350GlCtx *ati_r350_gl_open(const char **err)
{
    return ati_r350_gl_open_api(NULL, err);
}

void ati_r350_gl_close(R350GlCtx *g)
{
    if (!g) {
        return;
    }
#ifdef CONFIG_DARWIN
    if (g->api == R350_GPU_MTL) {
        r350_mtl_close(g->mtl);
    } else
#endif
    {
        r350_ogl_close(g->ogl);
    }
    g_free(g);
}

bool ati_r350_gl_ordered(R350GlCtx *g)
{
    return g && g->api == R350_GPU_MTL;
}

bool ati_r350_gl_depth(R350GlCtx *g)
{
    return g && g->api == R350_GPU_MTL;
}

bool ati_r350_gl_direct(R350GlCtx *g)
{
    return g && g->api == R350_GPU_MTL;
}

bool ati_r350_gl_vram(R350GlCtx *g, void *ptr, uint64_t size)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        return r350_mtl_vram(g->mtl, ptr, size);
    }
#endif
    return false;
}

bool ati_r350_gl_wait(R350GlCtx *g)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        return r350_mtl_sync(g->mtl);
    }
#endif
    return true;
}

uint64_t ati_r350_gl_commit(R350GlCtx *g)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        return r350_mtl_commit(g->mtl);
    }
#endif
    return 0;
}

uint64_t ati_r350_gl_done(R350GlCtx *g)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        return r350_mtl_done(g->mtl);
    }
#endif
    return 0;
}

uint64_t ati_r350_gl_next(R350GlCtx *g)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        return r350_mtl_next(g->mtl);
    }
#endif
    return 0;
}

bool ati_r350_gl_idle(R350GlCtx *g)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        return r350_mtl_idle(g->mtl);
    }
#endif
    return true;
}

void ati_r350_gl_notify(R350GlCtx *g, void (*fn)(void *), void *opaque)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        r350_mtl_notify(g->mtl, fn, opaque);
    }
#endif
}

bool ati_r350_gl_zseed(R350GlCtx *g, int x0, int y0, int w, int h,
                       const uint32_t *z)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        return r350_mtl_zseed(g->mtl, x0, y0, w, h, z);
    }
#endif
    return false;
}

bool ati_r350_gl_zfetch(R350GlCtx *g, int x0, int y0, int w, int h,
                        uint32_t *z)
{
#ifdef CONFIG_DARWIN
    if (g && g->api == R350_GPU_MTL) {
        return r350_mtl_zfetch(g->mtl, x0, y0, w, h, z);
    }
#endif
    return false;
}

bool ati_r350_gl_target(R350GlCtx *g, int w, int h, bool *lost)
{
    if (!g) {
        *lost = false;
        return false;
    }
    return R350_GPU_CALL(g, target, w, h, lost);
}

bool ati_r350_gl_seed(R350GlCtx *g, int x0, int y0, int w, int h,
                      const uint8_t *base, unsigned pitch, unsigned xr)
{
    return g && R350_GPU_CALL(g, seed, x0, y0, w, h, base, pitch, xr);
}

bool ati_r350_gl_fetch(R350GlCtx *g, int x0, int y0, int w, int h,
                       uint8_t *base, unsigned pitch, unsigned xr)
{
    return g && R350_GPU_CALL(g, fetch, x0, y0, w, h, base, pitch, xr);
}

bool ati_r350_gl_draw(R350GlCtx *g, const R350GlReq *req)
{
    return g && R350_GPU_CALL(g, draw, req);
}

const char *ati_r350_gl_describe(R350GlCtx *g)
{
    return g ? R350_GPU_CALL0(g, describe) : "none";
}

void ati_r350_gl_prog_stats(R350GlCtx *g, uint64_t *hits, uint64_t *links,
                            uint64_t *failed)
{
    if (!g) {
        *hits = *links = *failed = 0;
        return;
    }
    R350_GPU_CALL(g, prog_stats, hits, links, failed);
}

uint64_t ati_r350_gl_barriers(R350GlCtx *g)
{
    return g ? R350_GPU_CALL0(g, barriers) : 0;
}

void ati_r350_gl_queue_stats(R350GlCtx *g, uint64_t *units, uint64_t *flushes,
                             uint64_t *waves)
{
    if (!g) {
        *units = *flushes = *waves = 0;
        return;
    }
    R350_GPU_CALL(g, queue_stats, units, flushes, waves);
}
