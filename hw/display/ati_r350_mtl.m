/*
 * ATI R300/R350 -- the Metal rendering backend, ZERO-COPY.
 *
 * An Apple GPU shares memory with the CPU, so emulated VRAM is handed to
 * Metal as a buffer over the very same pages (ati_r350_gl_vram()), and
 * the fragment stage reads and writes the guest's colour and depth
 * buffers in it directly: the device's own byte lanes (the aperture
 * swapper, `lanes()`), the device's own tiled depth layout (`zaddr()`,
 * r300_zaddr() transcribed). There is no render target, nothing is
 * seeded and nothing is fetched. Making VRAM coherent for a reader is
 * waiting for the GPU (ati_r350_gl_wait()), which the caller does at
 * exactly the points the copy-based design used to copy.
 *
 * ORDER. The render pass has no attachments; the fragment stage writes
 * VRAM through pointers in raster order group 0, and an Apple GPU runs
 * the fragments that land on one pixel in primitive order, each seeing
 * what the one before it left. So the depth and stencil test, the blend
 * against the destination and the colour write mask are all done in the
 * shader in r300_raster_tri()'s own order -- alpha test, depth and
 * stencil (a failing fragment still writes its stencil result), discard,
 * blend, truncating pack -- and a self-overlapping blended draw is one
 * draw call that blends the way the device does.
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
 * COORDINATES are the device's, unflipped: the vertex stage negates the
 * NDC y GL computes, so Metal's top-left raster puts device row k on
 * raster row k, and [[position]] is the device pixel plus a half.
 *
 * TEXTURES are still decoded by the caller and uploaded; vertices go
 * through per-command-buffer arenas. Both are copies of data the GPU
 * only reads.
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
#define ati_r350_gl_zseed       r350_mtl_zseed
#define ati_r350_gl_zfetch      r350_mtl_zfetch
#define ati_r350_gl_vram        r350_mtl_vram
#define ati_r350_gl_wait        r350_mtl_sync
#define ati_r350_gl_commit      r350_mtl_commit
#define ati_r350_gl_done        r350_mtl_done
#define ati_r350_gl_next        r350_mtl_next
#define ati_r350_gl_idle        r350_mtl_idle
#define ati_r350_gl_notify      r350_mtl_notify
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
    IV_ASRC, IV_ADST, IV_ACOMB, IV_WMASK,
    /* the depth and stencil test, as R350GlReq carries it */
    IV_ZMODE, IV_ZTEST, IV_ZWR, IV_SEN, IV_SFB, IV_ZSC, IV_SREF, IV_SMASK,
    IV_SWMASK,
    /* zero-copy: where the buffers are in VRAM */
    IV_CBOFF, IV_CBPITCH, IV_CBX, IV_ZOFF, IV_ZPITCH, IV_ZMACRO, IV_ZMICRO,
    IV_ZAA, IV_ZX, IV_VSZ,
    IV_N = 40
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

    /*
     * Emulated VRAM itself, wrapped without a copy: the fragment stage
     * reads and writes the guest's colour and depth buffers in it.
     */
    id<MTLBuffer> vbuf;
    uint64_t vsz;

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

    uint64_t draws, cbs, passes;
    char desc[160];

    /* called from Metal's own thread as each committed buffer completes */
    void (*notify)(void *);
    void *notify_op;
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
"    int iv[40];\n"
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
"#define IV_ZMODE 17\n"
"#define IV_ZTEST 18\n"
"#define IV_ZWR 19\n"
"#define IV_SEN 20\n"
"#define IV_SFB 21\n"
"#define IV_ZSC 22\n"
"#define IV_SREF 23\n"
"#define IV_SMASK 24\n"
"#define IV_SWMASK 25\n"
"#define IV_CBOFF 26\n"
"#define IV_CBPITCH 27\n"
"#define IV_CBX 28\n"
"#define IV_ZOFF 29\n"
"#define IV_ZPITCH 30\n"
"#define IV_ZMACRO 31\n"
"#define IV_ZMICRO 32\n"
"#define IV_ZAA 33\n"
"#define IV_ZX 34\n"
"#define IV_VSZ 35\n"
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
"    float4 zb [[flat]];\n"
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
"    o.zb = ld4(v, OFF_Z);\n"
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
/* r300_zs_cmp(): `a` is the incoming value, `b` the stored one */
"static bool zs_cmp(uint fn, uint a, uint b)\n"
"{\n"
"    switch (fn) {\n"
"    case 0: return false;\n"
"    case 1: return a < b;\n"
"    case 2: return a <= b;\n"
"    case 3: return a == b;\n"
"    case 4: return a >= b;\n"
"    case 5: return a > b;\n"
"    case 6: return a != b;\n"
"    default: return true;\n"
"    }\n"
"}\n"
"\n"
/* r300_stencil_op() */
"static uint zs_sop(uint op, uint v, uint ref)\n"
"{\n"
"    switch (op) {\n"
"    case 1: return 0u;\n"
"    case 2: return ref;\n"
"    case 3: return min(v + 1u, 0xffu);\n"
"    case 4: return v - (v != 0u ? 1u : 0u);\n"
"    case 5: return ~v & 0xffu;\n"
"    case 6: return (v + 1u) & 0xffu;\n"
"    case 7: return (v - 1u) & 0xffu;\n"
"    default: return v;\n"
"    }\n"
"}\n"
"\n"
/*
 * VRAM. A dword through the aperture swapper: byte i of the value is
 * memory byte (i ^ x) of its aligned dword -- r300_ld32() -- and the
 * same permutation stores it back, being its own inverse.
 */
"static uint lanes(uint w, uint x)\n"
"{\n"
"    if (x == 0u) return w;\n"
"    uint b0 = w & 0xffu, b1 = (w >> 8) & 0xffu;\n"
"    uint b2 = (w >> 16) & 0xffu, b3 = w >> 24;\n"
"    uint b[4] = { b0, b1, b2, b3 };\n"
"    return b[0u ^ x] | (b[1u ^ x] << 8) | (b[2u ^ x] << 16) |\n"
"           (b[3u ^ x] << 24);\n"
"}\n"
"\n"
/* r300_zaddr(), the depth buffer's tiled layout */
"static uint zaddr(constant FsU &u, uint x, uint y, uint smp)\n"
"{\n"
"    uint off = uint(u.iv[IV_ZOFF]), pitch = uint(u.iv[IV_ZPITCH]);\n"
"    bool macro = u.iv[IV_ZMACRO] != 0, micro = u.iv[IV_ZMICRO] != 0;\n"
"    uint a;\n"
"    if (!macro && !micro) return off + (y * pitch + x) * 4u;\n"
"    if (u.iv[IV_ZAA] != 0) {\n"
"        a = ((x & 1u) << 2) | ((y & 1u) << 3) | (smp << 4);\n"
"        if (macro) {\n"
"            a |= (((x >> 1) & 1u) << 5) | (((y >> 1) & 3u) << 6) |\n"
"                 (((x >> 2) & 7u) << 8);\n"
"            a += ((y >> 3) * (pitch / 32u) + (x >> 5)) * 2048u;\n"
"        } else {\n"
"            a += ((y >> 1) * (pitch / 2u) + (x >> 1)) * 32u;\n"
"        }\n"
"    } else {\n"
"        a = ((x & 1u) << 2) | (((x >> 1) & 1u) << 3) | ((y & 1u) << 4);\n"
"        if (macro) {\n"
"            a |= (((x >> 2) & 1u) << 5) | (((y >> 1) & 3u) << 6) |\n"
"                 (((x >> 3) & 3u) << 8) | (((y >> 3) & 1u) << 10);\n"
"            a += ((y >> 4) * (pitch / 32u) + (x >> 5)) * 2048u;\n"
"        } else {\n"
"            a += ((y >> 1) * (pitch / 4u) + (x >> 2)) * 32u;\n"
"        }\n"
"    }\n"
"    return off + a;\n"
"}\n"
"\n"
/*
 * The fragment stage writes emulated VRAM itself. raster_order_group(0)
 * is what makes that exact: fragments landing on one pixel run their
 * reads and writes of it in primitive order, each seeing what the one
 * before it left -- the device's own order, with no render target.
 */
"fragment void r350_fs(VOut in [[stage_in]],\n"
"                      constant FsU &u [[buffer(0)]],\n"
"                      constant float4 *USK [[buffer(1)]],\n"
"                      constant float *N255 [[buffer(2)]],\n"
"                      device uint *vram [[buffer(3), raster_order_group(0)]],\n"
"                      device uchar *vram8 [[buffer(4), raster_order_group(0)]],\n"
"                      texture2d<uint, access::read> tex [[texture(0)]])\n"
"{\n"
"    uint pxi = uint(in.pos.x), pyi = uint(in.pos.y);\n"
"    uint vsz = uint(u.iv[IV_VSZ]);\n"
"    uint wm = uint(u.iv[IV_WMASK]);\n"
"    uint caddr = uint(u.iv[IV_CBOFF]) + pyi * uint(u.iv[IV_CBPITCH]) +\n"
"                 pxi * 4u;\n"
"    uint cbx = uint(u.iv[IV_CBX]);\n"
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
/* Z is screen-linear: these weights, before any perspective correction */
"    float sw0 = w0, sw1 = w1, sw2 = w2;\n"
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
/*
 * r300_zb_pixel(), after the alpha test and before DISCARD_SRC_PIXELS,
 * as the software path orders them. A fragment the depth or stencil
 * test fails is NOT discarded here: the stencil op on failure still
 * writes, so it goes through with its colour left as it was.
 */
"    bool live = true;\n"
"    if (u.iv[IV_ZMODE] != 0) {\n"
"        uint zx = uint(u.iv[IV_ZX]);\n"
"        float zf = fma(sw2, in.zb.z, fma(sw1, in.zb.y, sw0 * in.zb.x));\n"
"        float zt = zf > 0.0f ? zf : 0.0f;\n"
"        zt = zt < 1.0f ? zt : 1.0f;\n"
"        uint zfn = uint(u.iv[IV_ZSC]) & 7u;\n"
"        bool ztest = u.iv[IV_ZTEST] != 0, zwr = u.iv[IV_ZWR] != 0;\n"
"        if (u.iv[IV_ZMODE] == 2) {\n"
/* 16-bit Z: linear, byte by byte through the swapper */
"            uint za = uint(u.iv[IV_ZOFF]) +\n"
"                      (pyi * uint(u.iv[IV_ZPITCH]) + pxi) * 2u;\n"
"            if (za + 2u <= vsz) {\n"
"                uint zold = uint(vram8[za ^ zx]) |\n"
"                            (uint(vram8[(za + 1u) ^ zx]) << 8);\n"
"                uint znew = uint(zt * 65535.0f);\n"
"                bool zpass = !ztest || zs_cmp(zfn, znew, zold);\n"
"                if (zpass && zwr && znew != zold) {\n"
"                    vram8[za ^ zx] = uchar(znew & 0xffu);\n"
"                    vram8[(za + 1u) ^ zx] = uchar(znew >> 8);\n"
"                }\n"
"                live = zpass;\n"
"            }\n"
"        } else {\n"
"            uint za = zaddr(u, pxi, pyi, 0u);\n"
"            if (za + 4u <= vsz) {\n"
"            uint zin = lanes(vram[za >> 2], zx);\n"
"            uint zold = zin >> 8, sold = zin & 0xffu, snew = sold;\n"
"            uint znew = uint(zt * 16777215.0f);\n"
"            bool zpass = !ztest || zs_cmp(zfn, znew, zold);\n"
"            bool spass = true;\n"
"            if (u.iv[IV_SEN] != 0) {\n"
"                uint zsc = uint(u.iv[IV_ZSC]);\n"
"                uint sref = uint(u.iv[IV_SREF]), smask = uint(u.iv[IV_SMASK]);\n"
"                uint swm = uint(u.iv[IV_SWMASK]);\n"
"                uint f = in.zb.w != 0.0f && u.iv[IV_SFB] != 0\n"
"                         ? (zsc >> 15) & 0xfffu : (zsc >> 3) & 0xfffu;\n"
"                spass = zs_cmp(f & 7u, sref & smask, sold & smask);\n"
"                uint op = !spass ? (f >> 3) & 7u\n"
"                        : !zpass ? (f >> 9) & 7u : (f >> 6) & 7u;\n"
"                snew = zs_sop(op, sold, sref);\n"
"                snew = (sold & ~swm) | (snew & swm);\n"
"            }\n"
"            if (!(spass && zpass && zwr)) znew = zold;\n"
"            if (znew != zold || snew != sold) {\n"
"                uint val = lanes((znew << 8) | snew, zx);\n"
"                vram[za >> 2] = val;\n"
"                if (u.iv[IV_ZAA] != 0) {\n"
"                    uint zb = zaddr(u, pxi, pyi, 1u);\n"
"                    if (zb + 4u <= vsz) vram[zb >> 2] = val;\n"
"                }\n"
"            }\n"
"            live = spass && zpass;\n"
"            }\n"
"        }\n"
"    }\n"
"    if (live && u.iv[IV_DISCARD] != 0) {\n"

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
"        if (kill) live = false;\n"
"    }\n"
/*
 * The blend, against the pixel as the previous primitive left it --
 * read from VRAM in raster order. Cat_7's expressions verbatim.
 */
"    if (!live || wm == 0u || caddr + 4u > vsz) return;\n"
"    uint dv = lanes(vram[caddr >> 2], cbx);\n"
"    if (u.iv[IV_BLEND] != 0) {\n"
"        float4 d = u.iv[IV_BREAD] != 0\n"
"                   ? float4(N255[(dv >> 16) & 0xffu], N255[(dv >> 8) & 0xffu],\n"
"                            N255[dv & 0xffu], N255[dv >> 24])\n"
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
"    uint argb = (o.a << 24) | (o.r << 16) | (o.g << 8) | o.b;\n"
/* r300_write_dst(): masked channels keep what the destination holds */
"    argb = (argb & wm) | (dv & ~wm);\n"
"    vram[caddr >> 2] = lanes(argb, cbx);\n"
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
        "#define OFF_S0 %d\n#define OFF_S1 %d\n#define OFF_S2 %d\n"
        "#define OFF_Z %d\n",
        R350_GL_VSTRIDE,
        6 + 2 * C, 8 + 2 * C, 10 + 2 * C,
        12 + 2 * C, 16 + 2 * C, 20 + 2 * C,
        24 + 2 * C, 24 + 4 * C, 24 + 6 * C,
        37 + 8 * C,
        25 + 8 * C, 29 + 8 * C, 33 + 8 * C,
        41 + 8 * C);
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
    /* no attachments: the fragment stage writes VRAM, see r350_fs */
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
        if (g->notify) {
            void (*fn)(void *) = g->notify;
            void *op = g->notify_op;

            [g->cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
                fn(op);
            }];
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
/*
 * The open render pass, begun if there is none: attachment-less, `w` x
 * `h` pixels of raster, with VRAM bound for the fragment stage. A
 * different size ends the pass and starts another.
 */
static id<MTLRenderCommandEncoder> mtl_enc(R350MtlCtx *g, int w, int h)
{
    if (g->enc && (g->enc_vw != w || g->enc_vh != h)) {
        mtl_end_enc(g);
    }
    if (!g->enc) {
        MTLRenderPassDescriptor *rp =
            [MTLRenderPassDescriptor renderPassDescriptor];
        MTLViewport vp = { 0.0, 0.0, (double)w, (double)h, 0.0, 1.0 };

        rp.renderTargetWidth = w;
        rp.renderTargetHeight = h;
        rp.defaultRasterSampleCount = 1;
        g->enc = [[mtl_cb(g) renderCommandEncoderWithDescriptor:rp] retain];
        [g->enc setFragmentBuffer:g->n255 offset:0 atIndex:2];
        [g->enc setFragmentBuffer:g->vbuf offset:0 atIndex:3];
        [g->enc setFragmentBuffer:g->vbuf offset:0 atIndex:4];
        [g->enc setViewport:vp];
        g->enc_pso = nil;
        g->enc_vb = nil;
        g->enc_tex = nil;
        g->enc_vw = w;
        g->enc_vh = h;
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
        if (![dev supportsFamily:MTLGPUFamilyApple1] ||
            !dev.rasterOrderGroupsSupported || !dev.hasUnifiedMemory) {
            [dev release];
            *err = "gl-api=metal needs an Apple-silicon GPU (unified "
                   "memory, raster order groups); use gl-api=opengl";
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
                 "Metal, %s, zero-copy VRAM", dev.name.UTF8String);
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
        [g->arena_free release];
        [g->vbuf release];
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

/* nothing to barrier: raster order groups keep each pixel in order */
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

/*
 * ZERO-COPY. There is no resident target: the draws write emulated VRAM
 * where it is, so the copy-based half of the interface has nothing to
 * do. The caller never calls it for a direct backend; these answer
 * honestly if it ever does.
 */
bool ati_r350_gl_target(R350MtlCtx *g, int w, int h, bool *lost)
{
    *lost = false;
    return g && w > 0 && h > 0 && w <= 16384 && h <= 16384;
}

bool ati_r350_gl_seed(R350MtlCtx *g, int x0, int y0, int w, int h,
                      const uint8_t *base, unsigned pitch, unsigned xr)
{
    return false;
}

bool ati_r350_gl_fetch(R350MtlCtx *g, int x0, int y0, int w, int h,
                       uint8_t *base, unsigned pitch, unsigned xr)
{
    return false;
}

bool ati_r350_gl_zseed(R350MtlCtx *g, int x0, int y0, int w, int h,
                       const uint32_t *z)
{
    return false;
}

bool ati_r350_gl_zfetch(R350MtlCtx *g, int x0, int y0, int w, int h,
                        uint32_t *z)
{
    return false;
}

/*
 * Emulated VRAM, as a Metal buffer over the same pages: QEMU's RAM block
 * is page-aligned and a whole number of pages, which is all
 * newBufferWithBytesNoCopy asks. The GPU's stores ARE the guest's VRAM.
 */
bool ati_r350_gl_vram(R350MtlCtx *g, void *ptr, uint64_t size)
{
    if (!g || g->vbuf || ((uintptr_t)ptr & (qemu_real_host_page_size() - 1)) ||
        (size & (qemu_real_host_page_size() - 1)) || size > UINT32_MAX) {
        return false;
    }
    @autoreleasepool {
        g->vbuf = [g->dev newBufferWithBytesNoCopy:ptr
                                            length:size
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
    }
    g->vsz = size;
    return g->vbuf != nil;
}

/* every draw submitted so far, landed in VRAM */
bool ati_r350_gl_wait(R350MtlCtx *g)
{
    bool ok;

    if (!g) {
        return true;
    }
    @autoreleasepool {
        ok = mtl_submit(g, true);
    }
    return ok;
}

/* send what is drawn so far, without waiting; the serial that covers it */
uint64_t ati_r350_gl_commit(R350MtlCtx *g)
{
    if (!g) {
        return 0;
    }
    @autoreleasepool {
        mtl_submit(g, false);
    }
    return g->serial;
}

/* the newest serial known complete */
uint64_t ati_r350_gl_done(R350MtlCtx *g)
{
    if (!g) {
        return 0;
    }
    @autoreleasepool {
        mtl_reap(g, false);
    }
    return g->done;
}

/* the serial the next draw lands in: the open buffer's, or a new one's */
uint64_t ati_r350_gl_next(R350MtlCtx *g)
{
    if (!g) {
        return 0;
    }
    return g->cb ? g->serial : g->serial + 1;
}

/* nothing open, nothing running */
bool ati_r350_gl_idle(R350MtlCtx *g)
{
    if (!g) {
        return true;
    }
    @autoreleasepool {
        mtl_reap(g, false);
    }
    return !g->cb && !g->fl_n;
}

void ati_r350_gl_notify(R350MtlCtx *g, void (*fn)(void *), void *opaque)
{
    if (g) {
        g->notify = fn;
        g->notify_op = opaque;
    }
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

    /* r->out is gl=verify, which a zero-copy backend cannot serve */
    if (!g || !g->vbuf || r->out || r->w <= 0 || r->h <= 0 || !r->nvert ||
        !r->us_glsl || r->surf_w <= 0 || r->surf_h <= 0 ||
        r->surf_w > 16384 || r->surf_h > 16384 || g->failed) {
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

        /* scissor, the draw's rectangle, and the raster's bounds */
        sx0 = MAX(MAX(r->sx0, r->x0), 0);
        sy0 = MAX(MAX(r->sy0, r->y0), 0);
        sx1 = MIN(MIN(r->sx1, r->x0 + r->w), r->surf_w);
        sy1 = MIN(MIN(r->sy1, r->y0 + r->h), r->surf_h);

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
            fu.iv[IV_ZMODE] = r->zmode;
            fu.iv[IV_ZTEST] = r->z_test;
            fu.iv[IV_ZWR] = r->z_wr;
            fu.iv[IV_SEN] = r->s_en;
            fu.iv[IV_SFB] = r->s_fb;
            fu.iv[IV_ZSC] = (int32_t)r->zsc;
            fu.iv[IV_SREF] = r->s_ref;
            fu.iv[IV_SMASK] = r->s_mask;
            fu.iv[IV_SWMASK] = r->s_wmask;
            fu.iv[IV_CBOFF] = (int32_t)r->cb_off;
            fu.iv[IV_CBPITCH] = (int32_t)r->cb_pitch;
            fu.iv[IV_CBX] = (int32_t)r->cb_xr;
            fu.iv[IV_ZOFF] = (int32_t)r->z_off;
            fu.iv[IV_ZPITCH] = (int32_t)r->z_pitch;
            fu.iv[IV_ZMACRO] = r->z_macro;
            fu.iv[IV_ZMICRO] = r->z_micro;
            fu.iv[IV_ZAA] = r->z_aa;
            fu.iv[IV_ZX] = (int32_t)r->z_xr;
            fu.iv[IV_VSZ] = (int32_t)g->vsz;
            rect[0] = 0.0f;
            rect[1] = 0.0f;
            rect[2] = (float)r->surf_w;
            rect[3] = (float)r->surf_h;

            enc = mtl_enc(g, r->surf_w, r->surf_h);
            if (g->enc_pso != pso) {
                [enc setRenderPipelineState:pso];
                g->enc_pso = pso;
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
    }
    return !g->failed;
}
