/*
 * ATI R300/R350 -- the host-GPU rendering backend interface.
 *
 * This header is the whole contract between the device and whatever
 * draws its triangles on the host. It deliberately names no QEMU type:
 * a request is plain data, the backend keeps no device state, and the
 * three entry points below are the only ones the device calls. That is
 * what lets phase 2's milestone M3 move the backend onto its own thread
 * -- a request is already everything a worker needs -- and what lets a
 * Metal or SDL_GPU implementation replace ati_r350_gl.c without the
 * draw path noticing.
 *
 * Coordinates in a request are the device's own: y increases downward
 * and the origin is the render target's top-left corner. The backend
 * never touches a QEMU display backend or a window.
 *
 * The render target is RESIDENT (milestone M3). ati_r350_gl_target()
 * sizes it, ati_r350_gl_seed() copies emulated VRAM into it and
 * ati_r350_gl_fetch() copies it back out, and between those the caller
 * may draw into it as often as it likes without a byte crossing the
 * bus. M2 uploaded the destination rectangle twice and read it back for
 * every single draw; on this host that was 5.2 ms of the 6.5 ms a
 * full-screen draw cost (doc/radeon9800/glbench).
 *
 * What that buys has to be paid for in coherency, and the rules are
 * stated where they are enforced -- see "GL-OWNED RENDER TARGET" in
 * ati_r350_3d.c. The backend's own part of the contract is only this:
 * it holds one target, it holds it until told otherwise, and seed and
 * fetch are the only ways bytes move.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ATI_R350_GL_H
#define ATI_R350_GL_H

/*
 * Interpolated texture coordinate SETS a request carries, and texture
 * UNITS it can bind.
 *
 * ONE coordinate set, and that is a consequence rather than a limit.
 * The caller only offloads a draw whose fragment program the translator
 * could express, and that shape is a single fetch from unit 0 addressed
 * by coordinate SET 0 -- anything else needs a fetch inside the program,
 * which this contract's `us_main()` cannot perform. So a request can
 * never carry a second set, and the vertex layout below does not spend
 * a GL attribute on one; the sixteen GL 3.3 core guarantees are exactly
 * filled as it stands. The device's own R300_TEXCOORDS is eight,
 * because its software rasterizer really does interpolate that many for
 * Mac OS X 10.5.
 *
 * The unit count is the device's, and asserted equal to it: a request
 * names a texture per unit even though today's translated programs read
 * only unit 0, so that a backend growing multi-unit sampling does not
 * need the contract changed underneath it.
 */
#define R350_GL_TEXCOORDS 4   /* four: Metal samples up to four units */
#define R350_GL_TEXUNITS  8

/*
 * Floats per vertex in a request's vertex array. Each vertex carries
 * its own position, colour and texture coordinates AND its whole
 * triangle's, flat: the fragment stage rebuilds the software
 * rasterizer's own barycentric weights from them, which is what makes
 * the two paths agree to the last bit rather than merely to the eye.
 * The layout, in order, with C = R350_GL_TEXCOORDS:
 *
 *   0..1              x, y            this vertex
 *   2..5              r, g, b, a
 *   6..(5+2C)         s, t per coordinate set, set 0 first
 *   (6+2C)..(11+2C)   triangle vertex 0/1/2 positions
 *   (12+2C)..(23+2C)  triangle vertex 0/1/2 colours
 *   (24+2C)..(23+8C)  triangle vertex 0/1/2 coordinates, all sets each
 *   (24+8C)           1.0f / signed area, computed on the HOST
 *   (25+8C)..(36+8C)  triangle vertex 0/1/2 SECOND colours
 *   (37+8C)           1.0f / signed area again
 *   (38+8C)..(40+8C)  triangle vertex 0/1/2 1/w
 *   (41+8C)..(43+8C)  triangle vertex 0/1/2 Z (screen-linear)
 *   (44+8C)           1.0f if the triangle is back-facing, else 0.0f
 *
 * The last four are the depth test's and only a backend with a depth
 * buffer reads them; the GL backend's attribute table stops before them
 * and only the stride grew.
 *
 * The second colour is the one a fragment program can add to the
 * modulated texel -- Chess.app's specular term -- and it is carried at
 * the corners like the first so the fragment stage interpolates it with
 * the same weights.
 */
/*
 * ...and after those, each vertex's own coordinate sets RAW, four floats
 * per set for all R350_GL_RAWSETS sets: (45+8C)..(44+8C+4R), what a
 * fragment program reads from a set no fetch addresses directly (Metal
 * reads the triangle's three from the buffer by primitive id; GL ignores
 * them). Eight: the ATI demos' bump-mapped shaders route a tangent
 * basis through sets 4-7 and only ever do arithmetic on it.
 */
#define R350_GL_RAWSETS 8
#define R350_GL_VSTRIDE (45 + 8 * R350_GL_TEXCOORDS + 4 * R350_GL_RAWSETS)

/*
 * How many uploaded textures the backend keeps, plus one: slot
 * R350_GL_TEXSLOTS is a scratch the caller uses for a texture it is not
 * tracking, and it is uploaded every time. See R350GlReq.tex_slot.
 *
 * MEASURED, not guessed: at 8 a live Flurry session reported 620 of its
 * 2546 decodes as entries the LRU had evicted -- the second largest
 * miss class after the ones a writer killed. The device-side cache
 * bounds each entry at R300_GL_TEXCACHE_MAX texels, so this many slots
 * is a bounded worst case rather than an open-ended allocation.
 *
 * And at 32, JK2's timedemo evicted 199254 entries in one run -- a
 * lightmapped Quake 3-engine frame touches more textures than that, so
 * every frame decoded nearly all of them again.
 */
#define R350_GL_TEXSLOTS 128

/*
 * Metal's reasons for handing a draw back, counted (darwin only):
 * 0 request it cannot serve, 1 no pipeline, 2 texture upload failed,
 * 3 slot bookkeeping mismatch, 4 vertex buffer full; 5 is not a
 * decline -- a stale slot re-uploaded instead of refused.
 */
#define R350_MTL_DECLINE_N 6
extern uint64_t r350_mtl_decline[R350_MTL_DECLINE_N];

/* US_ALU_CONST vectors a translated fragment program may name */
#define R350_GL_USK 32

/* rectangles the texture-barrier bookkeeping keeps apart */
#define R350_GL_WRITTEN 32

typedef struct R350GlReq {
    /*
     * Every coordinate below is a coordinate IN THE RESIDENT TARGET,
     * with (0,0) its top-left pixel and y increasing downward. Row k of
     * the target is row k of the GL texture -- there is no flip
     * anywhere, in either direction, which is the only arrangement in
     * which a seed, a draw and a fetch can be composed in any order and
     * still agree.
     */
    /* destination rectangle: the bounding box this draw may write */
    int x0, y0, w, h;
    /* scissor, same coordinates, bottom-right exclusive */
    int sx0, sy0, sx1, sy1;
    /* the whole target's extent, which is what the draw renders into */
    int surf_w, surf_h;

    const float *verts;         /* R350_GL_VSTRIDE floats per vertex */
    unsigned nvert;             /* 3 * triangle count */

    /*
     * The vertices are ordered into PASSES. A blended draw whose own
     * primitives overlap cannot be rendered in one go -- the shader
     * blends against a snapshot of the destination, while the device
     * paints primitives in order and each blends against what the last
     * one left. So the caller partitions the triangles so that no two in
     * a pass overlap and any overlapping pair lands in the device's own
     * order, and the backend refreshes the blend's source between
     * passes. `pass[k]` is the first vertex of pass k and there are
     * npass + 1 entries, the last being nvert. A single pass (the
     * ordinary case) may leave both NULL and 0.
     */
    const unsigned *pass;
    unsigned npass;

    /*
     * The bound textures, RGBA8, one entry per texture UNIT, and WHICH
     * of the backend's texture objects each belongs in. The caller
     * already decides when a decoded texture is still current -- it owns
     * the VRAM ranges the answer depends on -- so it names a slot and
     * says whether the bytes are new. A slot whose bytes are unchanged
     * is bound and not re-uploaded, which on this host is 0.93 ms of
     * caller time per full-screen draw. `tex` may be NULL when
     * tex_fresh is false. A slot of exactly R350_GL_TEXSLOTS is the
     * scratch, which is always uploaded -- and no two units in one
     * request may name it, because there is only one.
     *
     * `textured` is a MASK of the units this draw samples; a unit
     * outside it reads white, which is what leaves a modulate program
     * computing its colour operand alone.
     */
    const uint8_t *tex[R350_GL_TEXUNITS];
    unsigned tex_slot[R350_GL_TEXUNITS];    /* <= R350_GL_TEXSLOTS */
    int tex_fresh[R350_GL_TEXUNITS];        /* upload `tex` into that slot */
    int tex_w[R350_GL_TEXUNITS], tex_h[R350_GL_TEXUNITS];
    /* TX_FILTER0 clamp modes; <= 1 is repeat */
    int clamp_s[R350_GL_TEXUNITS], clamp_t[R350_GL_TEXUNITS];
    /*
     * Filtering, for a unit whose `filt[u][0]` is set: `tex` then holds
     * `levels` images, level l being max(w >> l, 1) x max(h >> l, 1),
     * one after another, and the shader samples them as the device's
     * r300_tex_filter() does. filt = { on, need_lod, mag, min, mip,
     * log2 max aniso, first level, last level, LOD bias (1/256), floor
     * log2 w, floor log2 h }; `border` is the border colour as RGBA.
     */
    int levels[R350_GL_TEXUNITS];
    /* 1/size of the unit each coordinate set 0-3 is carried in */
    float set_inv[4][2];
    /* sets the program reads raw (s, t, r, q): r300_fs_setup()'s tc_raw */
    unsigned tc_raw;
    int filt[R350_GL_TEXUNITS][12];     /* [11]: a cube map, faces stacked */
    uint8_t border[R350_GL_TEXUNITS][4];
    uint32_t textured;

    uint32_t wmask;             /* RB3D_COLOR_CHANNEL_MASK as an ARGB mask */

    int alpha_test, af_func;
    float af_ref;
    int discard;                /* DISCARD_SRC_PIXELS selector */

    int blend, blend_read;
    int src_factor, dst_factor, comb_fcn;
    int a_src_factor, a_dst_factor, a_comb_fcn;
    float k_r, k_g, k_b, k_a;

    /*
     * Hand this draw's blend to GL's OWN blender instead of computing it
     * in the fragment shader, and render every primitive in one pass
     * however much they overlap each other.
     *
     * The caller sets it only when the blend is
     *     dst' = dst + f(src)
     * -- destination factor ONE, combine ADD, and a source factor that
     * does not read the destination, for colour and alpha alike. Under
     * that shape the destination term is the destination unchanged, so
     * a per-primitive quantisation can be reproduced exactly without
     * ever reading it: the shader emits floor(255*f(src)) and GL's
     * blender adds it to a byte that is already an integer. `pass` and
     * `npass` are then not used and no snapshot of the destination is
     * needed or taken. See the r300_gl_addblend() comment in
     * ati_r350_3d.c for the predicate and why it is exactly this shape.
     *
     * It is not exact -- the per-primitive rounding decomposition
     * differs from the device's and accumulates over a pixel's overlap
     * depth -- so the caller only sets it under `gl=fast`. The numbers
     * are in that same comment.
     */
    int add_blend;

    /*
     * Where to leave a packed RGBA8 copy of the drawn rectangle, or
     * NULL. Only gl=verify wants one: the target is resident, so the
     * ordinary path leaves the pixels on the GPU and fetches them when
     * something outside the 3D engine needs to look.
     */
    uint8_t *out;

    /*
     * The fragment program, translated to GLSL by ati_r350_us_glsl.c:
     * the text of a `void us_main(vec4 tex0, vec4 col0, vec4 col1, out
     * vec4 outc)` the backend splices into its own fragment shader,
     * `us_key` a signature the backend caches the linked program under,
     * and `us_konst` the thirty-two constant vectors the program may
     * name, uploaded per draw.
     *
     * The Metal backend compiles this same text as Metal Shading
     * Language, through a handful of aliases (see ati_r350_mtl.m), so
     * the translator stays the single source for both.
     *
     * Every draw carries one. The backend does not have a default: the
     * shading the caller wants is whatever the guest's program says, and
     * there is no arithmetic here that is not in that text.
     */
    const char *us_glsl;
    uint64_t us_key;
    const float *us_konst;      /* 32 * 4 floats */

    /*
     * The depth and stencil test, for a backend with a resident depth
     * buffer (ati_r350_gl_depth()); zero `zmode` for any other, and the
     * caller never sends a depth-tested draw to one. r300_zb_pixel() is
     * the definition: `zmode` 1 is 24-bit Z above 8 stencil bits, 2 is
     * 16-bit Z with no stencil; the rest is the ZB_* decode in s->zb.
     * The per-vertex Z and the triangle's facing are in the vertices.
     */
    int zmode;
    int z_test, z_wr, s_en, s_fb;
    uint32_t zsc;               /* ZB_ZSTENCILCNTL */
    int s_ref, s_mask, s_wmask;

    /*
     * ZERO-COPY (ati_r350_gl_direct()): the backend renders straight into
     * emulated VRAM, so it is told where the buffers ARE rather than
     * being handed copies of them. The colour buffer is 32bpp at byte
     * `cb_off`, `cb_pitch` bytes a row, through the swapper lane xor
     * `cb_xr`; the depth buffer is r300_zaddr()'s layout at `z_off`
     * under `z_xr`. Coordinates are still the device's own.
     */
    uint32_t cb_off, cb_pitch;
    unsigned cb_xr;
    /*
     * 0: the colour buffer is ARGB8888. Otherwise it is 16bpp and this is
     * its R300 COLORFORMAT (ARGB1555, RGB565 or ARGB4444), two bytes a
     * pixel, packed and unpacked as r300_cb_pack16()/_unpack16() do.
     */
    unsigned cb_fmt;
    uint32_t z_off, z_pitch;
    int z_macro, z_micro, z_aa;
    unsigned z_xr;
} R350GlReq;

typedef struct R350GlCtx R350GlCtx;

/*
 * Create a backend. Returns NULL and points *err at a static reason
 * string on failure -- a host without a usable GL context is a
 * configuration fact to report, not an abort.
 */
R350GlCtx *ati_r350_gl_open(const char **err);
void ati_r350_gl_close(R350GlCtx *g);

/*
 * Create a backend on a named host API: "opengl" is ati_r350_gl.c,
 * "metal" is ati_r350_mtl.m (macOS, Apple-silicon GPUs only). NULL or ""
 * means "opengl", so ati_r350_gl_open() is this with no name. The
 * dispatch between the two is ati_r350_gpu.c; each implementation is
 * compiled with its own names for the entry points below, and nothing
 * outside those three files can tell which one it is talking to.
 */
R350GlCtx *ati_r350_gl_open_api(const char *api, const char **err);

/*
 * True when the backend blends every primitive against what the one
 * before it left at that pixel, in submission order, within ONE draw.
 * Metal's raster order groups on an Apple GPU are exactly that. The caller
 * then partitions no self-overlapping blended draw into passes and never
 * needs gl=fast's add-blend approximation: the draw is rendered in one
 * go and still blends the way the device does.
 */
bool ati_r350_gl_ordered(R350GlCtx *g);

/*
 * A resident depth/stencil buffer beside the colour target, the same
 * size, holding one Z word per pixel exactly as r300_zb_pixel() reads
 * it from VRAM: (z24 << 8) | stencil, or the 16-bit Z. The caller does
 * the tiling and the swapper; the backend sees packed rows of words.
 * Only the Metal backend has one; for any other depth() is false and
 * zseed()/zfetch() refuse.
 */
bool ati_r350_gl_depth(R350GlCtx *g);

/*
 * ZERO-COPY. A direct backend renders into emulated VRAM itself: give it
 * the RAM with ati_r350_gl_vram() once, after open, and there is no
 * resident target to seed or fetch -- ati_r350_gl_wait() waits until
 * every draw submitted so far has landed in VRAM, which is all a reader
 * of VRAM then needs. Only the Metal backend is direct (Apple GPUs share
 * memory with the CPU).
 */
bool ati_r350_gl_direct(R350GlCtx *g);
bool ati_r350_gl_vram(R350GlCtx *g, void *ptr, uint64_t size);
bool ati_r350_gl_wait(R350GlCtx *g);
/*
 * ASYNCHRONOUS COMPLETION, direct backends only. commit() sends what has
 * been drawn so far without waiting and returns the serial that covers
 * it; done() is the newest serial known complete (every earlier one is
 * too); idle() says nothing is open or running. notify() names a
 * function the backend calls, from a thread of its own, each time a
 * committed batch completes.
 */
uint64_t ati_r350_gl_commit(R350GlCtx *g);
uint64_t ati_r350_gl_done(R350GlCtx *g);
/* the serial the next draw will be part of */
uint64_t ati_r350_gl_next(R350GlCtx *g);
bool ati_r350_gl_idle(R350GlCtx *g);
void ati_r350_gl_notify(R350GlCtx *g, void (*fn)(void *), void *opaque);
bool ati_r350_gl_zseed(R350GlCtx *g, int x0, int y0, int w, int h,
                       const uint32_t *z);
bool ati_r350_gl_zfetch(R350GlCtx *g, int x0, int y0, int w, int h,
                        uint32_t *z);

/*
 * Size the resident render target to at least w x h. Returns false if
 * it could not be created, and sets *lost when the previous contents
 * did not survive -- the caller then has to seed again whatever it
 * needs, because nothing else can tell it.
 */
bool ati_r350_gl_target(R350GlCtx *g, int w, int h, bool *lost);

/*
 * Move one rectangle between emulated VRAM and the resident target.
 * `base` addresses the target's own pixel (0,0) in VRAM, `pitch` is its
 * bytes per row, and `xr` is the aperture swapper's byte-lane xor over
 * it -- byte (2^xr) of a pixel is red, (1^xr) green, (0^xr) blue and
 * (3^xr) alpha. GL can be asked for two of the four orders directly,
 * and measurably should not be: see the comment above ati_r350_gl_seed()
 * for the numbers.
 */
bool ati_r350_gl_seed(R350GlCtx *g, int x0, int y0, int w, int h,
                      const uint8_t *base, unsigned pitch, unsigned xr);
bool ati_r350_gl_fetch(R350GlCtx *g, int x0, int y0, int w, int h,
                       uint8_t *base, unsigned pitch, unsigned xr);

/*
 * Render one request into the resident target. Returns false if the
 * backend could not run it, in which case the target is unchanged and
 * the caller must fall back.
 */
bool ati_r350_gl_draw(R350GlCtx *g, const R350GlReq *req);

/* a one-line description of the backend actually in use, for `qom-get gl` */
const char *ati_r350_gl_describe(R350GlCtx *g);

/* the fragment-shader cache: hits, links, and programs that would not build */
void ati_r350_gl_prog_stats(R350GlCtx *g, uint64_t *hits, uint64_t *links,
                            uint64_t *failed);
uint64_t ati_r350_gl_barriers(R350GlCtx *g);
void ati_r350_gl_queue_stats(R350GlCtx *g, uint64_t *units, uint64_t *flushes,
                             uint64_t *waves);

#endif /* ATI_R350_GL_H */
