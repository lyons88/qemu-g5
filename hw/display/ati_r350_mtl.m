/*
 * ATI R300/R350 -- the Metal rendering backend.
 *
 * The same contract as ati_r350_gl.c (see ati_r350_gl.h), the same
 * shading, and one thing GL on this host could never do: the fragment
 * shader reads the pixel it is about to write. On an Apple GPU that is
 * framebuffer fetch -- the colour attachment is an input to the shader,
 * `[[color(0)]]` -- and the hardware guarantees that fragments landing
 * on one pixel run in primitive order, each seeing what the one before
 * it left. That is exactly what the device does, so:
 *
 *   - the blend is still computed in the shader, with the device's
 *     truncating pack, but against the LIVE destination rather than a
 *     snapshot. There is no dst copy, no texture barrier, no draw
 *     queue sorted into waves, and no pass partition: a self-overlapping
 *     blended draw is one draw call and blends the way the chip does.
 *     ati_r350_gl_ordered() tells the caller so, and it stops
 *     partitioning (and stops falling back when the partition refuses).
 *   - gl=fast's add-blend is not needed and is not used: the exact path
 *     is already one pass. A request with add_blend set is rendered
 *     exactly.
 *   - the colour write mask is applied in the shader too (the unmasked
 *     channels keep the fetched value), so one pipeline serves every
 *     mask and the pipeline cache is keyed on the program alone.
 *
 * THE ARITHMETIC is Cat_7's, statement for statement: the fragment
 * shader below is fs_src from ati_r350_gl.c rewritten in MSL, including
 * the explicit fma() calls, the host-computed 1/area, and the k/255
 * table (bound as a buffer, so its bits are the host's). GLSL's
 * `precise` has no MSL spelling; the equivalent here is compiling with
 * fast math off (MTLMathModeSafe) and floating-point contraction off,
 * so a multiply and an add are fused only where the source says fma().
 *
 * THE FRAGMENT PROGRAM is the GLSL text ati_r350_us_glsl.c already
 * emits, compiled as MSL. Its vocabulary is small -- vec3/vec4,
 * `precise`, clamp() with scalar bounds, inversesqrt(), and one `out`
 * parameter -- so a few typedefs and two scoped macros make it valid
 * MSL, and the one `out vec4 outc` becomes a reference. The translator
 * stays the single source for both backends.
 *
 * COORDINATES are the device's with no flip, as in GL: target row k is
 * device row k. Metal's framebuffer origin is top-left, so the vertex
 * stage negates the NDC y GL computes -- the viewport transform then
 * yields the same window y bit for bit -- and [[position]] is the
 * device pixel plus a half, which is what gl_FragCoord was.
 *
 * RESIDENCY. The target is a private RGBA8Uint texture that lives on
 * the GPU. Seeds and fetches are blits in the same command stream as
 * the draws, so they are ordered with them without the CPU waiting;
 * only a fetch waits, because its bytes are needed now. Draws stay in
 * one render pass until something outside it (a seed, a fetch, a new
 * target) has to happen, and command buffers are committed without
 * waiting in between.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/error-report.h"

/* the "metal" backend's names for the interface; see ati_r350_gpu.h */
#define R350_GPU_IMPL_MTL 1
#define R350GlCtx               R350MtlCtx
#define ati_r350_gl_open        r350_mtl_open
#define ati_r350_gl_close       r350_mtl_close
#define ati_r350_gl_target      r350_mtl_target
#define ati_r350_gl_seed        r350_mtl_seed
#define ati_r350_gl_fetch       r350_mtl_fetch
#define ati_r350_gl_draw        r350_mtl_draw
#define ati_r350_gl_describe    r350_mtl_describe
#define ati_r350_gl_prog_stats  r350_mtl_prog_stats
#define ati_r350_gl_barriers    r350_mtl_barriers
#define ati_r350_gl_queue_stats r350_mtl_queue_stats
#include "ati_r350_gpu.h"

#import <Metal/Metal.h>

/* linked pipelines kept, one per distinct guest fragment program */
#define MTL_PROGSLOTS 32
/* bytes in one vertex/staging arena */
#define MTL_ARENA (8 * 1024 * 1024)
/* command buffers committed and not yet known to be complete */
#define MTL_INFLIGHT 32
/* draws after which the open command buffer is committed anyway */
#define MTL_CB_DRAWS 4096
/* compile failures reported in full before going quiet */
#define MTL_REPORTS 8

typedef struct MtlProg {
    uint64_t key;                       /* 0 = empty */
    id<MTLRenderPipelineState> pso;     /* nil with a key: would not build */
} MtlProg;

typedef struct MtlFlight {
    id<MTLCommandBuffer> cb;
    uint64_t serial;
    NSMutableArray *hold;               /* kept alive until it completes */
} MtlFlight;

/*
 * The fragment stage's per-draw values. Laid out to match `FsU` in the
 * shader exactly: two float4, a float padded to 16, then int arrays.
 */
enum {
    IV_TEXW, IV_TEXH, IV_CLAMPS, IV_CLAMPT, IV_TEXTURED, IV_ATEST,
    IV_AFUNC, IV_DISCARD, IV_BLEND, IV_BREAD, IV_CSRC, IV_CDST, IV_CCOMB,
    IV_ASRC, IV_ADST, IV_ACOMB, IV_WMASK, IV_N = 20
};

typedef struct MtlFsU {
    float konst[4];
    float afref;
    float pad[3];
    int32_t tf[16];
    int32_t iv[IV_N];
} MtlFsU;

QEMU_BUILD_BUG_ON(sizeof(MtlFsU) != 32 + 64 + 4 * IV_N);

struct R350MtlCtx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLBuffer> n255;                 /* k / 255.0f, as the host rounds it */
    id<MTLTexture> white;               /* what an untextured draw binds */

    /* the resident target */
    id<MTLTexture> cbuf;
    int fb_w, fb_h;

    /* uploaded textures by caller slot, plus the scratch at the end */
    id<MTLTexture> tex[R350_GL_TEXSLOTS + 1];
    int tex_w[R350_GL_TEXSLOTS + 1], tex_h[R350_GL_TEXSLOTS + 1];
    int tex_nl[R350_GL_TEXSLOTS + 1];
    uint64_t tex_use[R350_GL_TEXSLOTS + 1];     /* serial last drawn in */

    MtlProg prog[MTL_PROGSLOTS];
    unsigned prog_next;
    uint64_t prog_hits, prog_links, prog_failed;
    unsigned reports;

    /*
     * The open command buffer and, inside it, the open render pass.
     * `serial` is the open buffer's number (or the last committed one's
     * when none is open); `done` is the newest one known complete, and
     * every buffer before it is complete too, since one queue runs them
     * in order.
     */
    id<MTLCommandBuffer> cb;
    id<MTLRenderCommandEncoder> enc;
    NSMutableArray *hold;               /* the open buffer's keep-alives */
    uint64_t serial, done;
    unsigned cb_draws;
    bool failed;                        /* a committed buffer errored */

    /* what the open render pass already has set */
    id<MTLRenderPipelineState> enc_pso;
    id<MTLBuffer> enc_vb;
    id<MTLTexture> enc_tex;
    int enc_vw, enc_vh;

    /* vertex and staging arenas: the one being filled, and spares */
    id<MTLBuffer> arena;                /* owned by `hold` */
    size_t arena_used;
    NSMutableArray *arena_free;

    MtlFlight fl[MTL_INFLIGHT];
    unsigned fl_head, fl_n;

    id<MTLBuffer> rb;                   /* readback */
    size_t rb_sz;

    uint64_t draws, cbs, passes;
    char desc[160];
};

/*
 * THE SHADERS. Three pieces are joined per program: the head, the
 * guest's translated us_main() between two scoped macros, and the body.
 */
static const char *mtl_head =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"#if __METAL_VERSION__ >= 320\n"
"#pragma METAL fp math_mode(safe)\n"
"#pragma METAL fp contract(off)\n"
"#endif\n"
/* the GLSL names the translated program uses */
"typedef float2 vec2;\n"
"typedef float3 vec3;\n"
"typedef float4 vec4;\n"
"static inline float inversesqrt(float x) { return rsqrt(x); }\n"
/*
 * clamp() with scalar bounds, as us_clamp01() in ati_r350_us.c computes
 * it: two comparisons, so a NaN passes through unchanged there and here.
 */
"static inline float r3_clamp(float v, float lo, float hi)\n"
"{ return v < lo ? lo : (v > hi ? hi : v); }\n"
"static inline float3 r3_clamp(float3 v, float lo, float hi)\n"
"{ return select(select(v, float3(hi), v > hi), float3(lo), v < lo); }\n"
"static inline float4 r3_clamp(float4 v, float lo, float hi)\n"
"{ return select(select(v, float4(hi), v > hi), float4(lo), v < lo); }\n";

static const char *mtl_body =
"struct FsU {\n"
"    float4 konst;\n"
"    float afref;\n"
"    float pad0;\n"
"    float pad1;\n"
"    float pad2;\n"
"    int tf[16];\n"
"    int iv[20];\n"
"};\n"
"#define IV_TEXW 0\n"
"#define IV_TEXH 1\n"
"#define IV_CLAMPS 2\n"
"#define IV_CLAMPT 3\n"
"#define IV_TEXTURED 4\n"
"#define IV_ATEST 5\n"
"#define IV_AFUNC 6\n"
"#define IV_DISCARD 7\n"
"#define IV_BLEND 8\n"
"#define IV_BREAD 9\n"
"#define IV_CSRC 10\n"
"#define IV_CDST 11\n"
"#define IV_CCOMB 12\n"
"#define IV_ASRC 13\n"
"#define IV_ADST 14\n"
"#define IV_ACOMB 15\n"
"#define IV_WMASK 16\n"
"\n"
"struct VOut {\n"
"    float4 pos [[position]];\n"
"    float2 p0 [[flat]];\n"
"    float2 p1 [[flat]];\n"
"    float2 p2 [[flat]];\n"
"    float4 c0 [[flat]];\n"
"    float4 c1 [[flat]];\n"
"    float4 c2 [[flat]];\n"
"    float2 t0 [[flat]];\n"
"    float2 t1 [[flat]];\n"
"    float2 t2 [[flat]];\n"
"    float4 inv [[flat]];\n"
"    float4 s0 [[flat]];\n"
"    float4 s1 [[flat]];\n"
"    float4 s2 [[flat]];\n"
"};\n"
"\n"
"static float2 ld2(const device float *v, int o)\n"
"{ return float2(v[o], v[o + 1]); }\n"
"static float4 ld4(const device float *v, int o)\n"
"{ return float4(v[o], v[o + 1], v[o + 2], v[o + 3]); }\n"
"\n"
"vertex VOut r350_vs(uint vid [[vertex_id]],\n"
"                    const device float *vb [[buffer(0)]],\n"
"                    constant float4 &rect [[buffer(1)]])\n"
"{\n"
"    const device float *v = vb + vid * R350_VSTRIDE;\n"
"    VOut o;\n"
"    float nx = (v[0] - rect.x) / rect.z * 2.0f - 1.0f;\n"
"    float ny = (v[1] - rect.y) / rect.w * 2.0f - 1.0f;\n"
/*
 * GL's NDC, negated in y: Metal's viewport puts NDC +1 at row 0, GL's
 * put -1 there, so this lands every vertex on the same window y.
 */
"    o.pos = float4(nx, -ny, 0.0f, 1.0f);\n"
"    o.p0 = ld2(v, OFF_P0); o.p1 = ld2(v, OFF_P1); o.p2 = ld2(v, OFF_P2);\n"
"    o.c0 = ld4(v, OFF_C0); o.c1 = ld4(v, OFF_C1); o.c2 = ld4(v, OFF_C2);\n"
"    o.t0 = ld2(v, OFF_T0); o.t1 = ld2(v, OFF_T1); o.t2 = ld2(v, OFF_T2);\n"
"    o.inv = ld4(v, OFF_INV);\n"
"    o.s0 = ld4(v, OFF_S0); o.s1 = ld4(v, OFF_S1); o.s2 = ld4(v, OFF_S2);\n"
"    return o;\n"
"}\n"
"\n"
"static float bf(int code, float sc, float sa, float dc, float da,\n"
"               float kc, float ka)\n"
"{\n"
"    if (code == 1 || code == 32) return 0.0f;\n"
"    if (code == 2 || code == 33) return 1.0f;\n"
"    if (code == 3 || code == 34) return sc;\n"
"    if (code == 4 || code == 35) return 1.0f - sc;\n"
"    if (code == 9 || code == 36) return dc;\n"
"    if (code == 10 || code == 37) return 1.0f - dc;\n"
"    if (code == 5 || code == 38) return sa;\n"
"    if (code == 6 || code == 39) return 1.0f - sa;\n"
"    if (code == 7 || code == 40) return da;\n"
"    if (code == 8 || code == 41) return 1.0f - da;\n"
"    if (code == 11 || code == 42) return min(sa, 1.0f - da);\n"
"    if (code == 43) return kc;\n"
"    if (code == 44) return 1.0f - kc;\n"
"    if (code == 45) return ka;\n"
"    if (code == 46) return 1.0f - ka;\n"
"    return 1.0f;\n"
"}\n"
"\n"
"static float comb(int f, float s, float d)\n"
"{\n"
"    if (f == 2 || f == 3) return s - d;\n"
"    if (f == 4) return min(s, d);\n"
"    if (f == 5) return max(s, d);\n"
"    if (f == 6 || f == 7) return d - s;\n"
"    return s + d;\n"
"}\n"
"\n"
/* r300_tex_filter() and its helpers; see fs_src in ati_r350_gl.c */
"static float tc_pre(float c, int n, int m)\n"
"{\n"
"    float r = c;\n"
"    if (m == 3 || m == 5 || m == 7) r = abs(r);\n"
"    if (m == 4 || m == 5) r = min(max(r, 0.0f), float(n));\n"
"    return min(max(r, -16777216.0f), 16777216.0f);\n"
"}\n"
"\n"
"static int tc_mod(int i, int n)\n"
"{\n"
"    return i >= 0 ? i % n : n - 1 - (-1 - i) % n;\n"
"}\n"
"\n"
"static int tc_idx(int i, int n, int m, bool pt)\n"
"{\n"
"    if (m == 0) return tc_mod(i, n);\n"
"    if (m == 1) {\n"
"        int r = tc_mod(i, 2 * n);\n"
"        return r >= n ? 2 * n - 1 - r : r;\n"
"    }\n"
"    if (m == 2 || m == 3 || ((m == 4 || m == 5) && pt))\n"
"        return clamp(i, 0, n - 1);\n"
"    return i < 0 || i >= n ? -1 : i;\n"
"}\n"
"\n"
"static uint4 tfetch(texture2d<uint, access::read> t, constant FsU &u,\n"
"                    int l, int i, int j)\n"
"{\n"
"    if (i < 0 || j < 0)\n"
"        return uint4(uint(u.tf[11]), uint(u.tf[12]), uint(u.tf[13]),\n"
"                     uint(u.tf[14]));\n"
/* the guard GL's texelFetch did not need: never read outside the chain */
"    uint ll = min(uint(max(l, 0)), t.get_num_mip_levels() - 1u);\n"
"    uint2 p = min(uint2(i, j), uint2(t.get_width(ll) - 1u,\n"
"                                      t.get_height(ll) - 1u));\n"
"    return t.read(p, ll);\n"
"}\n"
"\n"
"static uint4 tlerp(uint4 a, uint4 b, int f)\n"
"{\n"
"    return uint4((int4(a) * (256 - f) + int4(b) * f + 128) >> 8);\n"
"}\n"
"\n"
"static uint4 tlevel(texture2d<uint, access::read> t, constant FsU &u,\n"
"                    int l, float fs, float ft, bool lin)\n"
"{\n"
"    int w = max(u.iv[IV_TEXW] >> l, 1), h = max(u.iv[IV_TEXH] >> l, 1);\n"
"    float ss = tc_pre(ldexp(fs, -min(l, u.tf[9])), w, u.iv[IV_CLAMPS]);\n"
"    float tt = tc_pre(ldexp(ft, -min(l, u.tf[10])), h, u.iv[IV_CLAMPT]);\n"
"    if (!lin)\n"
"        return tfetch(t, u, l,\n"
"                      tc_idx(int(floor(ss)), w, u.iv[IV_CLAMPS], true),\n"
"                      tc_idx(int(floor(tt)), h, u.iv[IV_CLAMPT], true));\n"
"    float fx = ss - 0.5f;\n"
"    float fy = tt - 0.5f;\n"
"    float x0 = floor(fx);\n"
"    float y0 = floor(fy);\n"
"    float qx = fx - x0;\n"
"    float qy = fy - y0;\n"
"    int wx = int(qx * 256.0f + 0.5f);\n"
"    int wy = int(qy * 256.0f + 0.5f);\n"
"    int i0 = int(x0), j0 = int(y0);\n"
"    if (wx == 256) { i0++; wx = 0; }\n"
"    if (wy == 256) { j0++; wy = 0; }\n"
"    int i1 = tc_idx(i0 + 1, w, u.iv[IV_CLAMPS], false);\n"
"    int j1 = tc_idx(j0 + 1, h, u.iv[IV_CLAMPT], false);\n"
"    i0 = tc_idx(i0, w, u.iv[IV_CLAMPS], false);\n"
"    j0 = tc_idx(j0, h, u.iv[IV_CLAMPT], false);\n"
"    uint4 t00 = tfetch(t, u, l, i0, j0);\n"
"    uint4 t10 = wx != 0 ? tfetch(t, u, l, i1, j0) : t00;\n"
"    if (wy == 0) return wx != 0 ? tlerp(t00, t10, wx) : t00;\n"
"    uint4 t01 = tfetch(t, u, l, i0, j1);\n"
"    uint4 t11 = wx != 0 ? tfetch(t, u, l, i1, j1) : t01;\n"
"    int4 top = int4(t00) * (256 - wx) + int4(t10) * wx;\n"
"    int4 bot = int4(t01) * (256 - wx) + int4(t11) * wx;\n"
"    return uint4((top * (256 - wy) + bot * wy + 32768) >> 16);\n"
"}\n"
"\n"
"static uint4 tmip(texture2d<uint, access::read> t, constant FsU &u,\n"
"                  int lod, float fs, float ft)\n"
"{\n"
"    bool lin = u.tf[3] != 1;\n"
"    int lo = min(max(lod, u.tf[6] * 256), u.tf[7] * 256);\n"
"    if (u.tf[4] == 1)\n"
"        return tlevel(t, u, min((lo + 128) >> 8, u.tf[7]), fs, ft, lin);\n"
"    int l = lo >> 8;\n"
"    if (u.tf[4] != 2 || l >= u.tf[7] || (lo & 255) == 0)\n"
"        return tlevel(t, u, l, fs, ft, lin);\n"
"    return tlerp(tlevel(t, u, l, fs, ft, lin),\n"
"                 tlevel(t, u, l + 1, fs, ft, lin), lo & 255);\n"
"}\n"
"\n"
"static int tlog2(float v)\n"
"{\n"
"    if (!(v > 0.0f)) return -65536;\n"
"    uint b = as_type<uint>(v);\n"
"    if (b >= 0x7f800000u) return 65536;\n"
"    return (int(b >> 23) - 127) * 256 + int((b >> 15) & 0xffu);\n"
"}\n"
"\n"
"static uint4 tfilter(texture2d<uint, access::read> t, constant FsU &u,\n"
"                     float fs, float ft, float4 der)\n"
"{\n"
"    int lod = 0, nl = 0;\n"
"    float ax = 0.0f, ay = 0.0f;\n"
"    if (u.tf[1] != 0) {\n"
"        float m = der.x * der.x;\n"
"        float n = der.y * der.y;\n"
"        float px = m + n;\n"
"        m = der.z * der.z;\n"
"        n = der.w * der.w;\n"
"        float py = m + n;\n"
"        if (u.tf[3] == 3) {\n"
"            int lmaj = tlog2(px >= py ? px : py);\n"
"            int lmin = tlog2(px >= py ? py : px);\n"
"            nl = min(max(((lmaj - lmin) / 2 + 255) >> 8, 0), u.tf[5]);\n"
"            lod = (lmaj >> 1) - nl * 256;\n"
"            ax = px >= py ? der.x : der.z;\n"
"            ay = px >= py ? der.y : der.w;\n"
"        } else {\n"
"            lod = tlog2(px >= py ? px : py) >> 1;\n"
"        }\n"
"        lod += u.tf[8];\n"
"    }\n"
"    if (lod <= 0) return tlevel(t, u, u.tf[6], fs, ft, u.tf[2] != 1);\n"
"    if (nl == 0) return tmip(t, u, lod, fs, ft);\n"
"    int N = 1 << nl;\n"
"    uint4 sum = uint4(0u);\n"
"    for (int k = 0; k < N; k++) {\n"
"        float ok = float(2 * k + 1 - N) / float(2 * N);\n"
"        float ds = ax * ok;\n"
"        float dt = ay * ok;\n"
"        float s1 = fs + ds;\n"
"        float t1 = ft + dt;\n"
"        sum += tmip(t, u, lod, s1, t1);\n"
"    }\n"
"    return (sum + uint(N >> 1)) >> uint(nl);\n"
"}\n"
"\n"
/* r300_tc_der() */
"static void tc_der(float3 ga, float3 gb, float t0, float t1, float t2,\n"
"                   float v, float iq, thread float &dx, thread float &dy)\n"
"{\n"
"    float e0 = t0 - v;\n"
"    float e1 = t1 - v;\n"
"    float e2 = t2 - v;\n"
"    float m0 = ga.x * e0;\n"
"    float m1 = ga.y * e1;\n"
"    float m2 = ga.z * e2;\n"
"    float r = m0 + m1;\n"
"    r = r + m2;\n"
"    float rx = r * iq;\n"
"    m0 = gb.x * e0;\n"
"    m1 = gb.y * e1;\n"
"    m2 = gb.z * e2;\n"
"    r = m0 + m1;\n"
"    r = r + m2;\n"
"    float ry = r * iq;\n"
"    dx = rx;\n"
"    dy = ry;\n"
"}\n"
"\n"
"struct FOut {\n"
"    uint4 c [[color(0)]];\n"
"};\n"
"\n"
"fragment FOut r350_fs(VOut in [[stage_in]],\n"
"                      uint4 dst [[color(0)]],\n"
"                      constant FsU &u [[buffer(0)]],\n"
"                      constant float4 *USK [[buffer(1)]],\n"
"                      constant float *N255 [[buffer(2)]],\n"
"                      texture2d<uint, access::read> tex [[texture(0)]])\n"
"{\n"
"    float4 c;\n"
"    float ts, tt;\n"
/* r300_raster_tri()'s own weights, expression for expression */
"    float inv = in.inv.x;\n"
"    float px = in.pos.x;\n"
"    float py = in.pos.y;\n"
"    float q0 = (in.p2.y - in.p1.y) * (px - in.p1.x);\n"
"    float q1 = (in.p0.y - in.p2.y) * (px - in.p2.x);\n"
"    float d0 = fma(in.p2.x - in.p1.x, py - in.p1.y, -q0);\n"
"    float d1 = fma(in.p0.x - in.p2.x, py - in.p2.y, -q1);\n"
"    float w0 = d0 * inv;\n"
"    float w1 = d1 * inv;\n"
"    float w2 = 1.0f - w0 - w1;\n"
"    float iq = 1.0f;\n"
"    bool persp = in.inv.y != in.inv.z || in.inv.z != in.inv.w;\n"
"    if (persp) {\n"
"        float pq0 = w0 * in.inv.y;\n"
"        float pq1 = w1 * in.inv.z;\n"
"        float pq2 = w2 * in.inv.w;\n"
"        iq = 1.0f / (pq0 + pq1 + pq2);\n"
"        w0 = pq0 * iq; w1 = pq1 * iq; w2 = pq2 * iq;\n"
"    }\n"
"    c = fma(float4(w2), in.c2, fma(float4(w1), in.c1, w0 * in.c0));\n"
"    float2 st = fma(float2(w2), in.t2, fma(float2(w1), in.t1, w0 * in.t0));\n"
"    ts = st.x; tt = st.y;\n"
"    float4 c1 = fma(float4(w2), in.s2, fma(float4(w1), in.s1, w0 * in.s0));\n"
"    float4 texel = float4(1.0f);\n"
"    if (u.iv[IV_TEXTURED] != 0 && u.tf[0] != 0) {\n"
"        float4 der = float4(0.0f);\n"
"        if (u.tf[1] != 0) {\n"
"            float a0 = -(in.p2.y - in.p1.y) * in.inv.x;\n"
"            float b0 = (in.p2.x - in.p1.x) * in.inv.x;\n"
"            float a1 = -(in.p0.y - in.p2.y) * in.inv.x;\n"
"            float b1 = (in.p0.x - in.p2.x) * in.inv.x;\n"
"            float3 ga = float3(a0, a1, -(a0 + a1));\n"
"            float3 gb = float3(b0, b1, -(b0 + b1));\n"
"            if (persp) {\n"
"                ga = ga * in.inv.yzw;\n"
"                gb = gb * in.inv.yzw;\n"
"            }\n"
"            float dx, dy;\n"
"            tc_der(ga, gb, in.t0.x, in.t1.x, in.t2.x, ts, iq, dx, dy);\n"
"            der.x = dx; der.z = dy;\n"
"            tc_der(ga, gb, in.t0.y, in.t1.y, in.t2.y, tt, iq, dx, dy);\n"
"            der.y = dx; der.w = dy;\n"
"        }\n"
"        uint4 tu = tfilter(tex, u, ts, tt, der);\n"
"        texel = float4(N255[tu.r], N255[tu.g], N255[tu.b], N255[tu.a]);\n"
"    } else if (u.iv[IV_TEXTURED] != 0) {\n"
"        int tw = u.iv[IV_TEXW], th = u.iv[IV_TEXH];\n"
"        int tx = int(ts);\n"
"        int ty = int(tt);\n"
"        if (u.iv[IV_CLAMPS] <= 1 && tw > 0) {\n"
"            tx = tx % tw;\n"
"            if (tx < 0) tx += tw;\n"
"        } else {\n"
"            tx = clamp(tx, 0, tw - 1);\n"
"        }\n"
"        if (u.iv[IV_CLAMPT] <= 1 && th > 0) {\n"
"            ty = ty % th;\n"
"            if (ty < 0) ty += th;\n"
"        } else {\n"
"            ty = clamp(ty, 0, th - 1);\n"
"        }\n"
"        uint4 tu = tfetch(tex, u, 0, tx, ty);\n"
"        texel = float4(N255[tu.r], N255[tu.g], N255[tu.b], N255[tu.a]);\n"
"    }\n"
"    {\n"
"        float4 shaded;\n"
"        us_main(texel, c, c1, shaded, USK);\n"
"        c = shaded;\n"
"    }\n"
"    if (u.iv[IV_ATEST] != 0) {\n"
"        int f = u.iv[IV_AFUNC];\n"
"        bool pass;\n"
"        if (f == 0)      pass = false;\n"
"        else if (f == 1) pass = c.a <  u.afref;\n"
"        else if (f == 2) pass = c.a == u.afref;\n"
"        else if (f == 3) pass = c.a <= u.afref;\n"
"        else if (f == 4) pass = c.a >  u.afref;\n"
"        else if (f == 5) pass = c.a != u.afref;\n"
"        else if (f == 6) pass = c.a >= u.afref;\n"
"        else             pass = true;\n"
"        if (!pass) discard_fragment();\n"
"    }\n"
"    if (u.iv[IV_DISCARD] != 0) {\n"
"        int k = u.iv[IV_DISCARD];\n"
"        bool a0 = c.a == 0.0f, a1 = c.a == 1.0f;\n"
"        bool z0 = c.r == 0.0f && c.g == 0.0f && c.b == 0.0f;\n"
"        bool z1 = c.r == 1.0f && c.g == 1.0f && c.b == 1.0f;\n"
"        bool kill = false;\n"
"        if (k == 1) kill = a0;\n"
"        else if (k == 2) kill = z0;\n"
"        else if (k == 3) kill = a0 && z0;\n"
"        else if (k == 4) kill = a1;\n"
"        else if (k == 5) kill = z1;\n"
"        else if (k == 6) kill = a1 && z1;\n"
"        if (kill) discard_fragment();\n"
"    }\n"
/*
 * The blend, against the pixel as the previous primitive left it --
 * framebuffer fetch, in primitive order. Cat_7's expressions verbatim.
 */
"    if (u.iv[IV_BLEND] != 0) {\n"
"        float4 d = u.iv[IV_BREAD] != 0\n"
"                   ? float4(N255[dst.r], N255[dst.g], N255[dst.b],\n"
"                            N255[dst.a])\n"
"                   : float4(0.0f);\n"
"        int cs = u.iv[IV_CSRC], cd = u.iv[IV_CDST], cf = u.iv[IV_CCOMB];\n"
"        int as = u.iv[IV_ASRC], ad = u.iv[IV_ADST], af = u.iv[IV_ACOMB];\n"
"        float4 k = u.konst;\n"
"        float nr = comb(cf,\n"
"            c.r * bf(cs, c.r, c.a, d.r, d.a, k.r, k.a),\n"
"            d.r * bf(cd, c.r, c.a, d.r, d.a, k.r, k.a));\n"
"        float ng = comb(cf,\n"
"            c.g * bf(cs, c.g, c.a, d.g, d.a, k.g, k.a),\n"
"            d.g * bf(cd, c.g, c.a, d.g, d.a, k.g, k.a));\n"
"        float nb = comb(cf,\n"
"            c.b * bf(cs, c.b, c.a, d.b, d.a, k.b, k.a),\n"
"            d.b * bf(cd, c.b, c.a, d.b, d.a, k.b, k.a));\n"
"        c.a = comb(af,\n"
"            c.a * bf(as, c.a, c.a, d.a, d.a, k.a, k.a),\n"
"            d.a * bf(ad, c.a, c.a, d.a, d.a, k.a, k.a));\n"
"        c.r = nr; c.g = ng; c.b = nb;\n"
"    }\n"
/* the device's truncating pack, then RB3D_COLOR_CHANNEL_MASK */
"    uint4 o = uint4(floor(r3_clamp(c, 0.0f, 1.0f) * 255.0f));\n"
"    uint wm = uint(u.iv[IV_WMASK]);\n"
"    bool4 wr = bool4((wm & 0x00ff0000u) != 0u, (wm & 0x0000ff00u) != 0u,\n"
"                     (wm & 0x000000ffu) != 0u, (wm & 0xff000000u) != 0u);\n"
"    FOut f;\n"
"    f.c = select(dst, o, wr);\n"
"    return f;\n"
"}\n";

/*
 * What every program's us_main() must be adapted from, and to. The
 * translator's signature is `(vec4 tex0, vec4 col0, vec4 col1, out vec4
 * outc)`; MSL has no `out`, and program-scope uniforms do not exist, so
 * the output becomes a reference and USK a parameter.
 */
static const char *mtl_us_from = "out vec4 outc)";
static const char *mtl_us_to = "thread vec4 &outc, constant vec4 *USK)";

/* compiled at open, so a head or body that will not build says so then */
static const char *mtl_us_probe =
"void us_main(vec4 tex0, vec4 col0, vec4 col1,\n"
"             out vec4 outc)\n"
"{\n"
"    precise vec4 R0 = col0;\n"
"    R0.rgb = clamp(R0.rgb * tex0.rgb + USK[0].rgb, 0.0, 1.0);\n"
"    outc = vec4(R0.rgb, inversesqrt(abs(col1.a) + 1.0));\n"
"}\n";

/* ------------------------------------------------------------------ */

/* fast math off, contraction off: `precise`, in MTLCompileOptions */
static void mtl_exact_math(MTLCompileOptions *o)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#if defined(MAC_OS_VERSION_15_0) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0
    if (@available(macOS 15.0, *)) {
        o.mathMode = MTLMathModeSafe;
        o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
        o.fastMathEnabled = NO;
    }
#else
    o.fastMathEnabled = NO;
#endif
#pragma clang diagnostic pop
}

static void mtl_report(R350MtlCtx *g, uint64_t key, const char *src,
                       const char *what, NSError *e)
{
    char *path;

    if (g->reports >= MTL_REPORTS) {
        return;
    }
    g->reports++;
    path = g_strdup_printf("/tmp/r350-mtl-%016" PRIx64 ".metal", key);
    if (!g_file_set_contents(path, src, -1, NULL)) {
        g_free(path);
        path = g_strdup("(could not be written)");
    }
    error_report("ati-radeon9800: Metal %s failed for program %016" PRIx64
                 "; the draws fall back to software. Source: %s\n%s",
                 what, key, path,
                 e ? e.localizedDescription.UTF8String : "");
    g_free(path);
}

/*
 * One pipeline: head, the guest's program adapted, body. Returns a
 * retained pipeline or nil; `why` gets a short reason on nil.
 */
static id<MTLRenderPipelineState> mtl_build(R350MtlCtx *g, const char *us,
                                            uint64_t key, const char **why)
{
    const char *hook = strstr(us, mtl_us_from);
    const int C = R350_GL_TEXCOORDS;
    MTLRenderPipelineDescriptor *pd;
    MTLCompileOptions *opt;
    id<MTLRenderPipelineState> pso;
    id<MTLLibrary> lib;
    id<MTLFunction> vf, ff;
    NSError *e = nil;
    GString *s;

    if (!hook) {
        *why = "translated program has no `out vec4 outc)` to adapt";
        return nil;
    }
    s = g_string_new(mtl_head);
    /*
     * The vertex layout, from the same C expressions as the GL
     * backend's attribute table, so neither can drift from
     * R350_GL_VSTRIDE.
     */
    g_string_append_printf(s,
        "#define R350_VSTRIDE %d\n"
        "#define OFF_P0 %d\n#define OFF_P1 %d\n#define OFF_P2 %d\n"
        "#define OFF_C0 %d\n#define OFF_C1 %d\n#define OFF_C2 %d\n"
        "#define OFF_T0 %d\n#define OFF_T1 %d\n#define OFF_T2 %d\n"
        "#define OFF_INV %d\n"
        "#define OFF_S0 %d\n#define OFF_S1 %d\n#define OFF_S2 %d\n",
        R350_GL_VSTRIDE,
        6 + 2 * C, 8 + 2 * C, 10 + 2 * C,
        12 + 2 * C, 16 + 2 * C, 20 + 2 * C,
        24 + 2 * C, 24 + 4 * C, 24 + 6 * C,
        37 + 8 * C,
        25 + 8 * C, 29 + 8 * C, 33 + 8 * C);
    g_string_append(s, "#define precise\n#define clamp r3_clamp\n");
    g_string_append_len(s, us, hook - us);
    g_string_append(s, mtl_us_to);
    g_string_append(s, hook + strlen(mtl_us_from));
    g_string_append(s, "\n#undef clamp\n#undef precise\n");
    g_string_append(s, mtl_body);

    opt = [[MTLCompileOptions alloc] init];
    mtl_exact_math(opt);
    lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:s->str]
                               options:opt
                                 error:&e];
    [opt release];
    if (!lib) {
        mtl_report(g, key, s->str, "shader compile", e);
        g_string_free(s, true);
        *why = "Metal shader would not compile";
        return nil;
    }
    vf = [lib newFunctionWithName:@"r350_vs"];
    ff = [lib newFunctionWithName:@"r350_fs"];
    pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vf;
    pd.fragmentFunction = ff;
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Uint;
    pd.colorAttachments[0].blendingEnabled = NO;
    e = nil;
    pso = (vf && ff) ? [g->dev newRenderPipelineStateWithDescriptor:pd
                                                              error:&e]
                     : nil;
    if (!pso) {
        mtl_report(g, key, s->str, "pipeline creation", e);
        *why = "Metal pipeline would not build";
    }
    [pd release];
    [vf release];
    [ff release];
    [lib release];
    g_string_free(s, true);
    return pso;
}

/* ------------------------------------------------------------------ */

/*
 * Retire command buffers that have finished: their keep-alives go, and
 * their vertex arenas come back for reuse. With `all`, wait for every
 * one first.
 */
static void mtl_reap(R350MtlCtx *g, bool all)
{
    while (g->fl_n) {
        MtlFlight *f = &g->fl[g->fl_head];
        MTLCommandBufferStatus st;

        if (all) {
            [f->cb waitUntilCompleted];
        }
        st = f->cb.status;
        if (st != MTLCommandBufferStatusCompleted &&
            st != MTLCommandBufferStatusError) {
            break;
        }
        if (st == MTLCommandBufferStatusError && !g->failed) {
            g->failed = true;
            error_report("ati-radeon9800: Metal command buffer failed: %s",
                         f->cb.error.localizedDescription.UTF8String);
        }
        g->done = f->serial;
        for (id o in f->hold) {
            if ([o conformsToProtocol:@protocol(MTLBuffer)] &&
                [(id<MTLBuffer>)o length] == MTL_ARENA) {
                [g->arena_free addObject:o];
            }
        }
        [f->hold release];
        [f->cb release];
        f->hold = nil;
        f->cb = nil;
        g->fl_head = (g->fl_head + 1) % MTL_INFLIGHT;
        g->fl_n--;
    }
}

static void mtl_end_enc(R350MtlCtx *g)
{
    if (g->enc) {
        [g->enc endEncoding];
        [g->enc release];
        g->enc = nil;
    }
}

static id<MTLCommandBuffer> mtl_cb(R350MtlCtx *g)
{
    if (!g->cb) {
        g->cb = [[g->q commandBuffer] retain];
        g->hold = [[NSMutableArray alloc] init];
        g->serial++;
        g->cb_draws = 0;
    }
    return g->cb;
}

/* keep `o` alive until the open command buffer has run */
static void mtl_hold(R350MtlCtx *g, id o)
{
    if (o) {
        mtl_cb(g);
        [g->hold addObject:o];
    }
}

/*
 * Commit the open command buffer. With `wait`, return only once every
 * committed buffer has run, and say whether they all ran cleanly.
 */
static bool mtl_submit(R350MtlCtx *g, bool wait)
{
    mtl_end_enc(g);
    if (g->cb) {
        MtlFlight *f;

        if (g->fl_n == MTL_INFLIGHT) {
            [g->fl[g->fl_head].cb waitUntilCompleted];
            mtl_reap(g, false);
        }
        [g->cb commit];
        f = &g->fl[(g->fl_head + g->fl_n) % MTL_INFLIGHT];
        f->cb = g->cb;
        f->serial = g->serial;
        f->hold = g->hold;
        g->fl_n++;
        g->cb = nil;
        g->hold = nil;
        g->arena = nil;
        g->arena_used = 0;
        g->cbs++;
    }
    mtl_reap(g, wait);
    return !g->failed;
}

/* the open render pass, begun on the resident target if there is none */
static id<MTLRenderCommandEncoder> mtl_enc(R350MtlCtx *g)
{
    if (!g->enc) {
        MTLRenderPassDescriptor *rp =
            [MTLRenderPassDescriptor renderPassDescriptor];

        rp.colorAttachments[0].texture = g->cbuf;
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        g->enc = [[mtl_cb(g) renderCommandEncoderWithDescriptor:rp] retain];
        [g->enc setFragmentBuffer:g->n255 offset:0 atIndex:2];
        g->enc_pso = nil;
        g->enc_vb = nil;
        g->enc_tex = nil;
        g->enc_vw = g->enc_vh = -1;
        g->passes++;
    }
    return g->enc;
}

/* `len` bytes of CPU-written, GPU-read memory for the open buffer */
static id<MTLBuffer> mtl_alloc(R350MtlCtx *g, size_t len, size_t *off)
{
    id<MTLBuffer> b;

    len = ROUND_UP(len, 256);
    mtl_cb(g);
    if (len > MTL_ARENA) {
        b = [g->dev newBufferWithLength:len
                                options:MTLResourceStorageModeShared |
                                        MTLResourceCPUCacheModeWriteCombined];
        mtl_hold(g, b);
        [b release];
        *off = 0;
        return b;
    }
    if (!g->arena || g->arena_used + len > MTL_ARENA) {
        b = [g->arena_free lastObject];
        if (b) {
            [b retain];
            [g->arena_free removeLastObject];
        } else {
            b = [g->dev newBufferWithLength:MTL_ARENA
                                    options:MTLResourceStorageModeShared |
                                            MTLResourceCPUCacheModeWriteCombined];
            if (!b) {
                return nil;
            }
        }
        mtl_hold(g, b);
        [b release];
        g->arena = b;
        g->arena_used = 0;
    }
    *off = g->arena_used;
    g->arena_used += len;
    return g->arena;
}

/*
 * Copy [x0,x0+w) x [y0,y0+h) of the target out, packed RGBA rows, and
 * wait for it. Returns the bytes, valid until the next call, or NULL.
 */
static const uint8_t *mtl_read(R350MtlCtx *g, int x0, int y0, int w, int h)
{
    size_t need = (size_t)w * h * 4;
    id<MTLBlitCommandEncoder> b;

    if (need > g->rb_sz) {
        [g->rb release];
        g->rb = [g->dev newBufferWithLength:need
                                    options:MTLResourceStorageModeShared];
        g->rb_sz = g->rb ? need : 0;
        if (!g->rb) {
            return NULL;
        }
    }
    mtl_end_enc(g);
    b = [mtl_cb(g) blitCommandEncoder];
    [b copyFromTexture:g->cbuf
           sourceSlice:0
           sourceLevel:0
          sourceOrigin:MTLOriginMake(x0, y0, 0)
            sourceSize:MTLSizeMake(w, h, 1)
              toBuffer:g->rb
     destinationOffset:0
destinationBytesPerRow:(NSUInteger)w * 4
destinationBytesPerImage:need];
    [b endEncoding];
    if (!mtl_submit(g, true)) {
        return NULL;
    }
    return g->rb.contents;
}

/* ------------------------------------------------------------------ */

R350MtlCtx *ati_r350_gl_open(const char **err)
{
    static char why[512];
    R350MtlCtx *g;

    *err = NULL;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        const char *e2 = NULL;
        id<MTLRenderPipelineState> probe;
        float n255[256];
        unsigned k;

        if (!dev) {
            *err = "no Metal device";
            return NULL;
        }
        /*
         * Framebuffer fetch is an Apple-GPU feature. An Intel or AMD Mac
         * would need the snapshot-and-passes scheme the GL backend has;
         * it is not reproduced here, so such a host is refused and
         * gl-api=opengl remains the way to run it.
         */
        if (![dev supportsFamily:MTLGPUFamilyApple1]) {
            [dev release];
            *err = "gl-api=metal needs an Apple-silicon GPU "
                   "(framebuffer fetch); use gl-api=opengl";
            return NULL;
        }
        g = g_new0(R350MtlCtx, 1);
        g->dev = dev;
        g->q = [dev newCommandQueue];
        for (k = 0; k < 256; k++) {
            n255[k] = k / 255.0f;
        }
        g->n255 = [dev newBufferWithBytes:n255
                                   length:sizeof(n255)
                                  options:MTLResourceStorageModeShared];
        g->arena_free = [[NSMutableArray alloc] init];
        {
            static const uint8_t white[4] = { 255, 255, 255, 255 };
            MTLTextureDescriptor *d = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                             width:1
                                            height:1
                                         mipmapped:NO];

            d.storageMode = MTLStorageModeShared;
            d.usage = MTLTextureUsageShaderRead;
            g->white = [dev newTextureWithDescriptor:d];
            [g->white replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                        mipmapLevel:0
                          withBytes:white
                        bytesPerRow:4];
        }
        if (!g->q || !g->n255 || !g->white) {
            ati_r350_gl_close(g);
            *err = "could not create the Metal queue and resources";
            return NULL;
        }
        probe = mtl_build(g, mtl_us_probe, 0, &e2);
        if (!probe) {
            snprintf(why, sizeof(why), "%s (the fixed part of the shader; "
                     "see /tmp/r350-mtl-0000000000000000.metal)", e2);
            ati_r350_gl_close(g);
            *err = why;
            return NULL;
        }
        [probe release];
        snprintf(g->desc, sizeof(g->desc),
                 "Metal, %s, framebuffer fetch", dev.name.UTF8String);
    }
    return g;
}

void ati_r350_gl_close(R350MtlCtx *g)
{
    unsigned k;

    if (!g) {
        return;
    }
    @autoreleasepool {
        mtl_submit(g, true);
        for (k = 0; k < MTL_PROGSLOTS; k++) {
            [g->prog[k].pso release];
        }
        for (k = 0; k <= R350_GL_TEXSLOTS; k++) {
            [g->tex[k] release];
        }
        [g->rb release];
        [g->arena_free release];
        [g->cbuf release];
        [g->white release];
        [g->n255 release];
        [g->q release];
        [g->dev release];
    }
    g_free(g);
}

const char *ati_r350_gl_describe(R350MtlCtx *g)
{
    return g ? g->desc : "none";
}

void ati_r350_gl_prog_stats(R350MtlCtx *g, uint64_t *hits, uint64_t *links,
                            uint64_t *failed)
{
    *hits = g ? g->prog_hits : 0;
    *links = g ? g->prog_links : 0;
    *failed = g ? g->prog_failed : 0;
}

/* nothing to barrier: framebuffer fetch is ordered by the hardware */
uint64_t ati_r350_gl_barriers(R350MtlCtx *g)
{
    return 0;
}

/* draws, command buffers committed, render passes begun */
void ati_r350_gl_queue_stats(R350MtlCtx *g, uint64_t *units,
                             uint64_t *flushes, uint64_t *waves)
{
    *units = g ? g->draws : 0;
    *flushes = g ? g->cbs : 0;
    *waves = g ? g->passes : 0;
}

bool ati_r350_gl_target(R350MtlCtx *g, int w, int h, bool *lost)
{
    MTLTextureDescriptor *d;
    id<MTLTexture> t;

    *lost = false;
    if (!g || w <= 0 || h <= 0) {
        return false;
    }
    if (w <= g->fb_w && h <= g->fb_h) {
        return true;
    }
    /* grow only, as the GL backend does, and say the contents went */
    w = MAX(w, g->fb_w);
    h = MAX(h, g->fb_h);
    if (w > 16384 || h > 16384) {
        return false;
    }
    @autoreleasepool {
        mtl_end_enc(g);
        d = [MTLTextureDescriptor
             texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                          width:w
                                         height:h
                                      mipmapped:NO];
        d.storageMode = MTLStorageModePrivate;
        d.usage = MTLTextureUsageRenderTarget;
        t = [g->dev newTextureWithDescriptor:d];
        if (!t) {
            return false;
        }
        mtl_hold(g, g->cbuf);
        [g->cbuf release];
        g->cbuf = t;
        g->fb_w = w;
        g->fb_h = h;
    }
    *lost = true;
    return true;
}

/*
 * VRAM into the target: permuted into RGBA on the CPU (see the GL
 * backend's measurement of why), then a blit in the command stream, so
 * it lands after every draw already encoded and before every later one.
 */
bool ati_r350_gl_seed(R350MtlCtx *g, int x0, int y0, int w, int h,
                      const uint8_t *base, unsigned pitch, unsigned xr)
{
    size_t need = (size_t)w * h * 4, off = 0;
    id<MTLBlitCommandEncoder> b;
    id<MTLBuffer> sb;
    uint8_t *st;
    int x, y;

    if (!g || w <= 0 || h <= 0 ||
        x0 < 0 || y0 < 0 || x0 + w > g->fb_w || y0 + h > g->fb_h) {
        return false;
    }
    @autoreleasepool {
        sb = mtl_alloc(g, need, &off);
        if (!sb) {
            return false;
        }
        st = (uint8_t *)sb.contents + off;
        for (y = 0; y < h; y++) {
            const uint8_t *p = base + (size_t)(y0 + y) * pitch +
                               (size_t)x0 * 4;
            uint8_t *o = st + (size_t)y * w * 4;

            for (x = 0; x < w; x++, p += 4, o += 4) {
                o[0] = p[2 ^ xr];       /* R */
                o[1] = p[1 ^ xr];       /* G */
                o[2] = p[0 ^ xr];       /* B */
                o[3] = p[3 ^ xr];       /* A */
            }
        }
        mtl_end_enc(g);
        b = [mtl_cb(g) blitCommandEncoder];
        [b copyFromBuffer:sb
             sourceOffset:off
        sourceBytesPerRow:(NSUInteger)w * 4
      sourceBytesPerImage:need
               sourceSize:MTLSizeMake(w, h, 1)
                toTexture:g->cbuf
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(x0, y0, 0)];
        [b endEncoding];
    }
    return !g->failed;
}

bool ati_r350_gl_fetch(R350MtlCtx *g, int x0, int y0, int w, int h,
                       uint8_t *base, unsigned pitch, unsigned xr)
{
    const uint8_t *st;
    int x, y;

    if (!g || w <= 0 || h <= 0 ||
        x0 < 0 || y0 < 0 || x0 + w > g->fb_w || y0 + h > g->fb_h) {
        return false;
    }
    @autoreleasepool {
        st = mtl_read(g, x0, y0, w, h);
    }
    if (!st) {
        return false;
    }
    for (y = 0; y < h; y++) {
        uint8_t *p = base + (size_t)(y0 + y) * pitch + (size_t)x0 * 4;
        const uint8_t *i = st + (size_t)y * w * 4;

        for (x = 0; x < w; x++, p += 4, i += 4) {
            p[2 ^ xr] = i[0];
            p[1 ^ xr] = i[1];
            p[0 ^ xr] = i[2];
            p[3 ^ xr] = i[3];
        }
    }
    return true;
}

/* the pipeline for this request's fragment program, building on a miss */
static id<MTLRenderPipelineState> mtl_prog_for(R350MtlCtx *g,
                                               const R350GlReq *r)
{
    const char *why = NULL;
    MtlProg *sl;
    unsigned k;

    for (k = 0; k < MTL_PROGSLOTS; k++) {
        if (g->prog[k].key == r->us_key) {
            g->prog_hits++;
            return g->prog[k].pso;
        }
    }
    sl = &g->prog[g->prog_next];
    g->prog_next = (g->prog_next + 1) % MTL_PROGSLOTS;
    /* the open pass may still be drawing with the one evicted */
    mtl_hold(g, sl->pso);
    [sl->pso release];
    sl->key = r->us_key;
    sl->pso = mtl_build(g, r->us_glsl, r->us_key, &why);
    if (!sl->pso) {
        g->prog_failed++;
        return nil;
    }
    g->prog_links++;
    return sl->pso;
}

static int mtl_levels(const R350GlReq *r)
{
    return r->filt[0][0] ? MAX(r->levels[0], 1) : 1;
}

/*
 * Unit 0's texture into a slot: every level the request carries, each
 * max(w >> l, 1) x max(h >> l, 1). A slot the GPU may still be reading
 * gets a NEW texture rather than being overwritten under it; the old one
 * lives on in the buffers that use it.
 */
static bool mtl_upload(R350MtlCtx *g, unsigned slot, const R350GlReq *r)
{
    int nl = mtl_levels(r), w0 = r->tex_w[0], h0 = r->tex_h[0], l, full;
    const uint8_t *p = r->tex[0];

    if (w0 <= 0 || h0 <= 0 || w0 > 16384 || h0 > 16384) {
        return false;
    }
    /* Metal will not make a chain past 1x1; the shader clamps to it */
    for (full = 1; (MAX(w0, h0) >> full) > 0; full++) {
    }
    if (!g->tex[slot] || g->tex_w[slot] != w0 || g->tex_h[slot] != h0 ||
        g->tex_nl[slot] != nl || g->tex_use[slot] > g->done) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                         width:w0
                                        height:h0
                                     mipmapped:NO];
        id<MTLTexture> t;

        d.mipmapLevelCount = MIN(nl, full);
        d.storageMode = MTLStorageModeShared;
        d.usage = MTLTextureUsageShaderRead;
        t = [g->dev newTextureWithDescriptor:d];
        if (!t) {
            return false;
        }
        mtl_hold(g, g->tex[slot]);
        [g->tex[slot] release];
        g->tex[slot] = t;
        g->tex_w[slot] = w0;
        g->tex_h[slot] = h0;
        g->tex_nl[slot] = nl;
        g->tex_use[slot] = 0;
    }
    for (l = 0; l < nl; l++) {
        int w = MAX(w0 >> l, 1), h = MAX(h0 >> l, 1);

        if (l < full) {
            [g->tex[slot] replaceRegion:MTLRegionMake2D(0, 0, w, h)
                            mipmapLevel:l
                              withBytes:p
                            bytesPerRow:(NSUInteger)w * 4];
        }
        p += (size_t)w * h * 4;
    }
    return true;
}

bool ati_r350_gl_draw(R350MtlCtx *g, const R350GlReq *r)
{
    id<MTLRenderPipelineState> pso;
    id<MTLRenderCommandEncoder> enc;
    id<MTLTexture> tex = nil;
    id<MTLBuffer> vb;
    size_t vlen, voff = 0;
    unsigned slot = R350_GL_TEXSLOTS + 1, k;
    int sx0, sy0, sx1, sy1;
    float rect[4];
    MtlFsU fu;

    if (!g || r->w <= 0 || r->h <= 0 || !r->nvert || !r->us_glsl ||
        r->surf_w > g->fb_w || r->surf_h > g->fb_h || g->failed) {
        return false;
    }
    @autoreleasepool {
        pso = mtl_prog_for(g, r);
        if (!pso) {
            return false;               /* the caller renders it instead */
        }
        if ((r->textured & 1) && r->tex_slot[0] <= R350_GL_TEXSLOTS) {
            slot = r->tex_slot[0];
            if (r->tex_fresh[0] && r->tex[0]) {
                if (!mtl_upload(g, slot, r)) {
                    return false;
                }
            } else if (!g->tex[slot] || g->tex_w[slot] != r->tex_w[0] ||
                       g->tex_h[slot] != r->tex_h[0] ||
                       g->tex_nl[slot] != mtl_levels(r)) {
                /* the caller's bookkeeping and ours disagree: refuse */
                return false;
            }
            tex = g->tex[slot];
        } else {
            tex = g->white;
        }

        /* scissor, the draw's rectangle, and the attachment's bounds */
        sx0 = MAX(MAX(r->sx0, r->x0), 0);
        sy0 = MAX(MAX(r->sy0, r->y0), 0);
        sx1 = MIN(MIN(r->sx1, r->x0 + r->w), g->fb_w);
        sy1 = MIN(MIN(r->sy1, r->y0 + r->h), g->fb_h);

        if (sx1 > sx0 && sy1 > sy0) {
            vlen = sizeof(float) * R350_GL_VSTRIDE * r->nvert;
            vb = mtl_alloc(g, vlen, &voff);
            if (!vb) {
                return false;
            }
            memcpy((uint8_t *)vb.contents + voff, r->verts, vlen);

            memset(&fu, 0, sizeof(fu));
            fu.konst[0] = r->k_r;
            fu.konst[1] = r->k_g;
            fu.konst[2] = r->k_b;
            fu.konst[3] = r->k_a;
            fu.afref = r->af_ref;
            for (k = 0; k < 11; k++) {
                fu.tf[k] = r->filt[0][k];
            }
            for (k = 0; k < 4; k++) {
                fu.tf[11 + k] = r->border[0][k];
            }
            fu.iv[IV_TEXW] = r->tex_w[0];
            fu.iv[IV_TEXH] = r->tex_h[0];
            fu.iv[IV_CLAMPS] = r->clamp_s[0];
            fu.iv[IV_CLAMPT] = r->clamp_t[0];
            fu.iv[IV_TEXTURED] = r->textured & 1;
            fu.iv[IV_ATEST] = r->alpha_test;
            fu.iv[IV_AFUNC] = r->af_func;
            fu.iv[IV_DISCARD] = r->discard;
            fu.iv[IV_BLEND] = r->blend;
            fu.iv[IV_BREAD] = r->blend_read;
            fu.iv[IV_CSRC] = r->src_factor;
            fu.iv[IV_CDST] = r->dst_factor;
            fu.iv[IV_CCOMB] = r->comb_fcn;
            fu.iv[IV_ASRC] = r->a_src_factor;
            fu.iv[IV_ADST] = r->a_dst_factor;
            fu.iv[IV_ACOMB] = r->a_comb_fcn;
            fu.iv[IV_WMASK] = (int32_t)r->wmask;
            rect[0] = 0.0f;
            rect[1] = 0.0f;
            rect[2] = (float)r->surf_w;
            rect[3] = (float)r->surf_h;

            enc = mtl_enc(g);
            if (g->enc_pso != pso) {
                [enc setRenderPipelineState:pso];
                g->enc_pso = pso;
            }
            if (g->enc_vw != r->surf_w || g->enc_vh != r->surf_h) {
                MTLViewport vp = {
                    0.0, 0.0, (double)r->surf_w, (double)r->surf_h, 0.0, 1.0
                };

                [enc setViewport:vp];
                g->enc_vw = r->surf_w;
                g->enc_vh = r->surf_h;
            }
            if (g->enc_vb != vb) {
                [enc setVertexBuffer:vb offset:voff atIndex:0];
                g->enc_vb = vb;
            } else {
                [enc setVertexBufferOffset:voff atIndex:0];
            }
            if (g->enc_tex != tex) {
                [enc setFragmentTexture:tex atIndex:0];
                g->enc_tex = tex;
            }
            [enc setVertexBytes:rect length:sizeof(rect) atIndex:1];
            [enc setFragmentBytes:&fu length:sizeof(fu) atIndex:0];
            if (r->us_konst) {
                [enc setFragmentBytes:r->us_konst
                               length:sizeof(float) * 4 * R350_GL_USK
                              atIndex:1];
            } else {
                static const float zero[4 * R350_GL_USK];

                [enc setFragmentBytes:zero length:sizeof(zero) atIndex:1];
            }
            [enc setScissorRect:(MTLScissorRect){
                (NSUInteger)sx0, (NSUInteger)sy0,
                (NSUInteger)(sx1 - sx0), (NSUInteger)(sy1 - sy0) }];
            /*
             * Every triangle, in the order given, in one call. A pass
             * partition, if the caller made one, is already an order in
             * which overlapping triangles keep the device's order, so
             * drawing the passes back to back is the same picture.
             */
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:r->nvert];
            if (slot <= R350_GL_TEXSLOTS) {
                g->tex_use[slot] = g->serial;
            }
            g->draws++;
            if (++g->cb_draws >= MTL_CB_DRAWS) {
                mtl_submit(g, false);
            }
        }

        if (r->out) {
            /* gl=verify only; the resident target keeps the pixels */
            const uint8_t *px = mtl_read(g, r->x0, r->y0, r->w, r->h);

            if (!px) {
                return false;
            }
            memcpy(r->out, px, (size_t)r->w * r->h * 4);
        }
    }
    return !g->failed;
}
