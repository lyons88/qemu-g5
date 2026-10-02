/*
 * ATI R300/R350 programmable vertex shader (PVS) interpreter.
 *
 * Mac OS X's accelerator paints the desktop with a vertex program that is
 * nothing but a 4x4 matrix multiply, which is why approximating every
 * program by that matrix rendered the whole compositor correctly. An
 * application's own program is a different animal: Chess.app uploads seven,
 * the longest thirty instructions, and its board's vertices carry a
 * position and a normal and no colour at all -- the colour a board square
 * is painted with is a lighting term the program computes.
 *
 * Opcode semantics here are transcribed from the R5xx Acceleration guide's
 * vertex-shader chapter, which documents the R300 instruction set. Two
 * details of it are easy to get wrong and were: the math engine reads only
 * the *w* channel of its sources, and its third source operand disappears
 * when PVS_DST_DUAL_MATH_OP is set, becoming a second, math-engine
 * instruction encoded in that word. A disassembler that does not know
 * about the second one cannot see the reciprocal square roots that every
 * lighting program in the corpus normalises its vectors with.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include <math.h>
#include <float.h>
#include "qemu/host-utils.h"
#include "ati_r350_pvs.h"

static inline float r300_pvs_f32(uint32_t v)
{
    union { uint32_t u; float f; } c = { .u = v };
    return c.f;
}

void r300_pvs_out_layout(uint32_t fmt0, unsigned *first_color,
                         unsigned *ncolor, unsigned *first_texcoord)
{
    unsigned n = 0, i;

    for (i = 0; i < 4; i++) {
        if (fmt0 & (2u << i)) {
            n++;
        }
    }
    *first_color = (fmt0 & 1) ? 1 : 0;
    *ncolor = n;
    *first_texcoord = *first_color + n;
}

void r300_pvs_const(const R300PvsProgram *p, unsigned off, float v[4])
{
    unsigned idx = (p->cbase + off) * 4, c;

    /*
     * PVS_MAX_CONST_ADDR is the highest constant the current shader may
     * name; the hardware returns (0,0,0,0) above it. Honouring it is what
     * keeps a program from reading constants a previous one left behind --
     * the constant file is RAM and nothing else clears it.
     */
    if ((p->bounded && off > p->cmax) || idx + 4 > p->const_slots * 4) {
        v[0] = v[1] = v[2] = v[3] = 0.0f;
        return;
    }
    for (c = 0; c < 4; c++) {
        v[c] = r300_pvs_f32(p->cnst[idx + c]);
    }
}

/* an ordinary source operand: register file, swizzle, negate, abs */
static void r300_pvs_src(const R300PvsProgram *p, const R300PvsRegs *r,
                         uint32_t dw, float out[4])
{
    unsigned type = dw & R300_PVS_SRC_REG_TYPE_MASK;
    unsigned off = (dw >> R300_PVS_SRC_OFFSET_SHIFT) &
                   R300_PVS_SRC_OFFSET_MASK;
    float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    unsigned c;

    switch (type) {
    case R300_PVS_SRC_REG_INPUT:
        memcpy(v, r->in[off % R300_PVS_IN_REGS], sizeof(v));
        break;
    case R300_PVS_SRC_REG_CONSTANT:
        r300_pvs_const(p, off, v);
        break;
    case R300_PVS_SRC_REG_ALT_TEMP:
        memcpy(v, r->atmp[off % R300_PVS_ATMP_REGS], sizeof(v));
        break;
    default:
        memcpy(v, r->tmp[off % R300_PVS_TMP_REGS], sizeof(v));
        break;
    }

    for (c = 0; c < 4; c++) {
        unsigned sel = (dw >> (R300_PVS_SRC_SWIZZLE_SHIFT + 3 * c)) &
                       R300_PVS_SRC_SWIZZLE_MASK;
        float f;

        if (sel < 4) {
            f = v[sel];
        } else {
            f = sel == R300_PVS_SRC_SELECT_FORCE_1 ? 1.0f : 0.0f;
        }
        if (dw & R300_PVS_SRC_ABS_XYZW) {
            f = fabsf(f);
        }
        if ((dw >> (R300_PVS_SRC_MODIFIER_SHIFT + c)) & 1) {
            f = -f;
        }
        out[c] = f;
    }
}

/*
 * The math engine, given the three source vectors of the instruction. It
 * only ever looks at their w channels; the compiler is expected to have
 * replicated the last meaningful operand into the ones an opcode does not
 * use, so a single-source op reading in_a.w and a three-source one reading
 * in_c.w can be written the way the guide writes them.
 */
static bool r300_pvs_math(unsigned opcode, const float a[4], const float b[4],
                          const float c[4], float res[4])
{
    float x = a[3], y;

    switch (opcode) {
    case R300_ME_LIGHT_COEFF_DX:
        /*
         * The lighting coefficients, and the only math opcode whose four
         * channels differ: the diffuse term is the clamped n.l in b.w, the
         * specular term the n.h in a.w raised to the exponent in c.w, and
         * it is suppressed entirely on a surface facing away from the
         * light. Chess.app runs this once per vertex of every lit piece.
         */
        res[0] = 1.0f;
        res[1] = MAX(b[3], 0.0f);
        if (b[3] > 0.0f) {
            res[2] = powf(MAX(a[3], 0.0f), MIN(MAX(c[3], -128.0f), 128.0f));
        } else {
            res[2] = 0.0f;
        }
        res[3] = 1.0f;
        return true;
    case R300_ME_EXP_BASE2_DX:
        res[0] = exp2f(floorf(x));
        res[1] = x > 128.0f ? 0.0f : x - floorf(x);
        res[2] = exp2f(x);
        res[3] = 1.0f;
        return true;
    case R300_ME_LOG_BASE2_DX:
        if (x == 0.0f) {
            res[0] = res[2] = -FLT_MAX;
            res[1] = res[3] = 1.0f;
        } else {
            int e;

            res[1] = fabsf(frexpf(x, &e)) * 2.0f;   /* mantissa, 1.0-2.0 */
            res[0] = (float)(e - 1);
            res[2] = log2f(fabsf(x));
            res[3] = 1.0f;
        }
        return true;
    case R300_ME_RECIP_DX:
        y = x != 0.0f ? 1.0f / x : FLT_MAX;
        break;
    case R300_ME_RECIP_FF:
        y = x != 0.0f ? 1.0f / x : 0.0f;
        break;
    case R300_ME_RECIP_SQRT_DX:
        y = x != 0.0f ? 1.0f / sqrtf(fabsf(x)) : FLT_MAX;
        break;
    case R300_ME_RECIP_SQRT_FF:
        y = x != 0.0f ? 1.0f / sqrtf(fabsf(x)) : 0.0f;
        break;
    case R300_ME_MULTIPLY:
        y = x * b[3];
        break;
    case R300_ME_POWER_FUNC_FF:
        /* base in a.w, exponent in b.w; a negative base keeps its sign */
        y = powf(fabsf(x), b[3]);
        if (x < 0.0f) {
            y = -y;
        }
        break;
    case R300_ME_EXP_BASE2_FULL_DX:
        y = exp2f(x);
        break;
    case R300_ME_LOG_BASE2_FULL_DX:
        y = x != 0.0f ? log2f(fabsf(x)) : -FLT_MAX;
        break;
    default:
        return false;
    }
    res[0] = res[1] = res[2] = res[3] = y;
    return true;
}

static bool r300_pvs_vector(unsigned opcode, const float a[4],
                            const float b[4], const float c[4], float res[4])
{
    unsigned i;

    switch (opcode) {
    case R300_VE_DOT_PRODUCT:
        res[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        res[1] = res[2] = res[3] = res[0];
        return true;
    case R300_VE_MULTIPLY:
        for (i = 0; i < 4; i++) {
            res[i] = a[i] * b[i];
        }
        return true;
    case R300_VE_ADD:
        for (i = 0; i < 4; i++) {
            res[i] = a[i] + b[i];
        }
        return true;
    case R300_VE_MULTIPLY_ADD:
        for (i = 0; i < 4; i++) {
            res[i] = a[i] * b[i] + c[i];
        }
        return true;
    case R300_VE_MULTIPLYX2_ADD:
        for (i = 0; i < 4; i++) {
            res[i] = 2.0f * (a[i] * b[i]) + c[i];
        }
        return true;
    case R300_VE_DISTANCE_VECTOR:
        res[0] = 1.0f;
        res[1] = a[1] * b[1];
        res[2] = a[2];
        res[3] = b[3];
        return true;
    case R300_VE_FRACTION:
        for (i = 0; i < 4; i++) {
            res[i] = a[i] - floorf(a[i]);
        }
        return true;
    case R300_VE_MAXIMUM:
        for (i = 0; i < 4; i++) {
            res[i] = MAX(a[i], b[i]);
        }
        return true;
    case R300_VE_MINIMUM:
        for (i = 0; i < 4; i++) {
            res[i] = MIN(a[i], b[i]);
        }
        return true;
    case R300_VE_SET_GREATER_THAN_EQUAL:
        for (i = 0; i < 4; i++) {
            res[i] = a[i] >= b[i] ? 1.0f : 0.0f;
        }
        return true;
    case R300_VE_SET_LESS_THAN:
        for (i = 0; i < 4; i++) {
            res[i] = a[i] < b[i] ? 1.0f : 0.0f;
        }
        return true;
    case R300_VE_MULTIPLY_CLAMP:
        /* point-size clamp: one scalar, replicated */
        if (c[3] < a[3] * b[3]) {
            res[0] = c[3];
        } else if (c[0] >= a[0] * b[0]) {
            res[0] = c[0];
        } else {
            res[0] = a[0] * b[0];
        }
        res[1] = res[2] = res[3] = res[0];
        return true;
    default:
        return false;
    }
}

/*
 * The math-engine half of a dual-issue instruction. Its operand vector is
 * a single register with only two swizzled channels, which stand in for
 * the w channels the math engine would otherwise read.
 */
static void r300_pvs_dual_math(const R300PvsProgram *p, R300PvsRegs *r,
                               uint32_t dw, R300PvsGaps *gaps)
{
    unsigned opcode = ((dw >> R300_PVS_DUAL_OPCODE_SHIFT) &
                       R300_PVS_DUAL_OPCODE_MASK) |
                      ((dw & R300_PVS_DUAL_OPCODE_MSB) ? 16 : 0);
    unsigned doff = (dw >> R300_PVS_DUAL_DST_OFF_SHIFT) &
                    R300_PVS_DUAL_DST_OFF_MASK;
    unsigned we = (dw >> R300_PVS_DUAL_WE_SEL_SHIFT) &
                  R300_PVS_DUAL_WE_SEL_MASK;
    float src[4], a[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float b[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float res[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    unsigned c;

    if (opcode == R300_ME_NO_OP) {
        return;
    }
    /*
     * The operand word keeps the ordinary register, swizzle-x/y, abs and
     * negate-x/y fields, but bits 19-24 -- where a full operand would hold
     * the z and w swizzles -- carry this instruction's destination and
     * opcode instead. Clear them so the shared decoder reads the two
     * channels that do exist and nothing else, then present each as a w
     * channel, which is all the math engine ever reads.
     */
    r300_pvs_src(p, r, dw & ~(0x3fu << 19), src);
    a[3] = src[0];
    b[3] = src[1];
    if (!r300_pvs_math(opcode, a, b, b, res)) {
        if (gaps && !gaps->has_math_op) {
            gaps->has_math_op = true;
            gaps->math_op = opcode;
        }
        return;
    }
    c = we;
    r->atmp[doff][c] = res[c];
}

/*
 * One instruction, its sources already read. The dual-issue math half
 * runs here, after `a` and `b` were read and before the vector half
 * writes -- the order the plain interpreter has always used.
 */
static inline void r300_pvs_ins(const R300PvsProgram *p, R300PvsRegs *r,
                                R300PvsGaps *gaps, const uint32_t *w,
                                const float a[4], const float b[4],
                                const float c[4])
{
    uint32_t op = w[0];
    unsigned opcode = op & R300_PVS_DST_OPCODE_MASK;
    bool math = op & R300_PVS_DST_MATH_INST;
    unsigned dtype = (op >> R300_PVS_DST_REG_TYPE_SHIFT) &
                     R300_PVS_DST_REG_TYPE_MASK;
    unsigned doff = (op >> R300_PVS_DST_OFFSET_SHIFT) &
                    R300_PVS_DST_OFFSET_MASK;
    unsigned we = (op >> R300_PVS_DST_WE_SHIFT) & R300_PVS_DST_WE_MASK;
    float res[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float *dst;
    unsigned k;

    if (op & R300_PVS_DST_DUAL_MATH_OP) {
        r300_pvs_dual_math(p, r, w[3], gaps);
    }
    do {
        if (op & R300_PVS_DST_MACRO_INST) {
            /*
             * The macro bit only ever marks a multiply-add whose three
             * temporaries the hardware has to read in two passes; the
             * arithmetic is the plain one, with opcode 1 selecting the
             * doubling form.
             */
            for (k = 0; k < 4; k++) {
                res[k] = a[k] * b[k] * (opcode ? 2.0f : 1.0f) + c[k];
            }
        } else if (math) {
            if (opcode == R300_ME_NO_OP) {
                return;
            }
            if (!r300_pvs_math(opcode, a, b, c, res)) {
                if (gaps && !gaps->has_math_op) {
                    gaps->has_math_op = true;
                    gaps->math_op = opcode;
                }
                return;
            }
        } else {
            if (opcode == R300_VE_NO_OP) {
                return;
            }
            if (!r300_pvs_vector(opcode, a, b, c, res)) {
                if (gaps && !gaps->has_vec_op) {
                    gaps->has_vec_op = true;
                    gaps->vec_op = opcode;
                }
                return;
            }
        }

        if (op & (math ? R300_PVS_DST_ME_SAT : R300_PVS_DST_VE_SAT)) {
            for (k = 0; k < 4; k++) {
                res[k] = MIN(MAX(res[k], 0.0f), 1.0f);
            }
        }

        switch (dtype) {
        case R300_PVS_DST_REG_OUT:
        case R300_PVS_DST_REG_OUT_REPL_X:
            dst = r->out[doff % R300_PVS_OUT_REGS];
            r->out_written |= 1u << (doff % R300_PVS_OUT_REGS);
            break;
        case R300_PVS_DST_REG_TEMPORARY:
            dst = r->tmp[doff % R300_PVS_TMP_REGS];
            break;
        case R300_PVS_DST_REG_ALT_TEMP:
            dst = r->atmp[doff % R300_PVS_ATMP_REGS];
            break;
        default:
            /*
             * The address register drives the relative addressing this
             * interpreter does not model, and writing the input file back
             * is a shader-model-3 trick; either would give a wrong answer
             * silently rather than an approximate one.
             */
            if (gaps && !gaps->has_dst_file) {
                gaps->has_dst_file = true;
                gaps->dst_file = dtype;
            }
            return;
        }
        for (k = 0; k < 4; k++) {
            if (we & (1u << k)) {
                dst[k] = dtype == R300_PVS_DST_REG_OUT_REPL_X ?
                         res[0] : res[k];
            }
        }
    } while (0);
}

void r300_pvs_run(const R300PvsProgram *p, R300PvsRegs *r, R300PvsGaps *gaps)
{
    unsigned i;

    if (!p->valid) {
        return;
    }
    for (i = p->first; i <= p->last; i++) {
        const uint32_t *w = &p->code[i * 4];
        float a[4], b[4], c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        r300_pvs_src(p, r, w[1], a);
        r300_pvs_src(p, r, w[2], b);
        if (!(w[0] & R300_PVS_DST_DUAL_MATH_OP)) {
            r300_pvs_src(p, r, w[3], c);
        }
        r300_pvs_ins(p, r, gaps, w, a, b, c);
    }
}

/*
 * THE COMPILED FORM. r300_pvs_run() decodes every operand's register
 * file, offset, four swizzles, abs and negate -- and converts and bounds
 * checks every constant -- for every instruction of every vertex. None
 * of that changes within a draw, so it is done once per draw here and
 * the per-vertex loop reads decoded operands; a constant operand is
 * stored finished, swizzle and modifiers applied. The arithmetic is
 * r300_pvs_ins(), shared with the interpreter, so the two cannot drift.
 */
static void r300_pvs_csrc_make(const R300PvsProgram *p, uint32_t dw,
                               R300PvsCSrc *o)
{
    unsigned type = dw & R300_PVS_SRC_REG_TYPE_MASK;
    unsigned off = (dw >> R300_PVS_SRC_OFFSET_SHIFT) &
                   R300_PVS_SRC_OFFSET_MASK;
    unsigned c;

    o->abs = (dw & R300_PVS_SRC_ABS_XYZW) != 0;
    o->neg = 0;
    o->plain = !o->abs;
    for (c = 0; c < 4; c++) {
        o->sel[c] = (dw >> (R300_PVS_SRC_SWIZZLE_SHIFT + 3 * c)) &
                    R300_PVS_SRC_SWIZZLE_MASK;
        if ((dw >> (R300_PVS_SRC_MODIFIER_SHIFT + c)) & 1) {
            o->neg |= 1u << c;
        }
        if (o->sel[c] != c) {
            o->plain = false;
        }
    }
    if (o->neg) {
        o->plain = false;
    }
    switch (type) {
    case R300_PVS_SRC_REG_INPUT:
        o->file = 0;
        o->idx = off % R300_PVS_IN_REGS;
        break;
    case R300_PVS_SRC_REG_CONSTANT:
        o->file = 1;
        o->idx = 0;
        /* the whole operand, exactly as r300_pvs_src() would finish it */
        {
            static const R300PvsRegs none;

            r300_pvs_src(p, &none, dw, o->kv);
        }
        break;
    case R300_PVS_SRC_REG_ALT_TEMP:
        o->file = 2;
        o->idx = off % R300_PVS_ATMP_REGS;
        break;
    default:
        o->file = 3;
        o->idx = off % R300_PVS_TMP_REGS;
        break;
    }
}

static inline void r300_pvs_csrc(const R300PvsCSrc *o, const R300PvsRegs *r,
                                 float out[4])
{
    const float *v;
    unsigned c;

    switch (o->file) {
    case 1:
        memcpy(out, o->kv, sizeof(o->kv));
        return;
    case 0:
        v = r->in[o->idx];
        break;
    case 2:
        v = r->atmp[o->idx];
        break;
    default:
        v = r->tmp[o->idx];
        break;
    }
    if (o->plain) {
        memcpy(out, v, 4 * sizeof(float));
        return;
    }
    for (c = 0; c < 4; c++) {
        unsigned sel = o->sel[c];
        float f;

        if (sel < 4) {
            f = v[sel];
        } else {
            f = sel == R300_PVS_SRC_SELECT_FORCE_1 ? 1.0f : 0.0f;
        }
        if (o->abs) {
            f = fabsf(f);
        }
        if ((o->neg >> c) & 1) {
            f = -f;
        }
        out[c] = f;
    }
}

/* note the register an operand reads, for the batched form */
static void r300_pvs_cmark(R300PvsCompiled *cp, const R300PvsCSrc *o)
{
    switch (o->file) {
    case 0:
        cp->in_used |= 1u << o->idx;
        break;
    case 2:
        cp->atmp_used |= 1u << o->idx;
        break;
    case 3:
        cp->tmp_used |= 1u << o->idx;
        break;
    }
}

void r300_pvs_compile(const R300PvsProgram *p, R300PvsCompiled *cp)
{
    unsigned i;

    cp->p = p;
    cp->n = 0;
    cp->in_used = cp->tmp_used = cp->atmp_used = cp->out_used = 0;
    if (!p->valid) {
        return;
    }
    for (i = p->first; i <= p->last && cp->n < R300_PVS_CODE_SLOTS; i++) {
        const uint32_t *w = &p->code[i * 4];
        R300PvsCIns *in = &cp->ins[cp->n++];
        uint32_t op = w[0];
        bool math = op & R300_PVS_DST_MATH_INST;

        memset(in, 0, sizeof(*in));
        in->w = w;
        r300_pvs_csrc_make(p, w[1], &in->a);
        r300_pvs_csrc_make(p, w[2], &in->b);
        in->dual = (op & R300_PVS_DST_DUAL_MATH_OP) != 0;
        if (!in->dual) {
            r300_pvs_csrc_make(p, w[3], &in->c);
            r300_pvs_cmark(cp, &in->c);
        } else {
            uint32_t dw = w[3];

            in->dop = ((dw >> R300_PVS_DUAL_OPCODE_SHIFT) &
                       R300_PVS_DUAL_OPCODE_MASK) |
                      ((dw & R300_PVS_DUAL_OPCODE_MSB) ? 16 : 0);
            in->ddoff = (dw >> R300_PVS_DUAL_DST_OFF_SHIFT) &
                        R300_PVS_DUAL_DST_OFF_MASK;
            in->dwe = (dw >> R300_PVS_DUAL_WE_SEL_SHIFT) &
                      R300_PVS_DUAL_WE_SEL_MASK;
            /* the operand exactly as r300_pvs_dual_math() reads it */
            r300_pvs_csrc_make(p, dw & ~(0x3fu << 19), &in->d);
            if (in->dop != R300_ME_NO_OP) {
                r300_pvs_cmark(cp, &in->d);
                cp->atmp_used |= 1u << in->ddoff;
            }
        }
        r300_pvs_cmark(cp, &in->a);
        r300_pvs_cmark(cp, &in->b);
        in->opcode = op & R300_PVS_DST_OPCODE_MASK;
        in->kind = (op & R300_PVS_DST_MACRO_INST) ? 2 : math ? 1 : 0;
        in->sat = (op & (math ? R300_PVS_DST_ME_SAT :
                         R300_PVS_DST_VE_SAT)) != 0;
        in->dtype = (op >> R300_PVS_DST_REG_TYPE_SHIFT) &
                    R300_PVS_DST_REG_TYPE_MASK;
        in->doff = (op >> R300_PVS_DST_OFFSET_SHIFT) &
                   R300_PVS_DST_OFFSET_MASK;
        in->we = (op >> R300_PVS_DST_WE_SHIFT) & R300_PVS_DST_WE_MASK;
        switch (in->dtype) {
        case R300_PVS_DST_REG_OUT:
        case R300_PVS_DST_REG_OUT_REPL_X:
            cp->out_used |= 1u << (in->doff % R300_PVS_OUT_REGS);
            break;
        case R300_PVS_DST_REG_TEMPORARY:
            cp->tmp_used |= 1u << (in->doff % R300_PVS_TMP_REGS);
            break;
        case R300_PVS_DST_REG_ALT_TEMP:
            cp->atmp_used |= 1u << (in->doff % R300_PVS_ATMP_REGS);
            break;
        }
    }
}

void r300_pvs_exec(const R300PvsCompiled *cp, R300PvsRegs *r,
                   R300PvsGaps *gaps)
{
    unsigned i;

    for (i = 0; i < cp->n; i++) {
        const R300PvsCIns *in = &cp->ins[i];
        float a[4], b[4], c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        r300_pvs_csrc(&in->a, r, a);
        r300_pvs_csrc(&in->b, r, b);
        if (!in->dual) {
            r300_pvs_csrc(&in->c, r, c);
        }
        r300_pvs_ins(cp->p, r, gaps, in->w, a, b, c);
    }
}

/*
 * THE BATCHED FORM. Everything below mirrors r300_pvs_exec() and
 * r300_pvs_ins() step for step -- the same checks in the same order, the
 * same expressions written the same way so the compiler contracts them
 * the same way -- with a loop over lanes where those have a single value.
 */
typedef float R300PvsLane[R300_PVS_LANES];

void r300_pvs_soa_reset(const R300PvsCompiled *cp, R300PvsSoa *r)
{
    uint32_t m;

    for (m = cp->tmp_used; m; m &= m - 1) {
        memset(r->tmp[ctz32(m)], 0, sizeof(r->tmp[0]));
    }
    for (m = cp->atmp_used; m; m &= m - 1) {
        memset(r->atmp[ctz32(m)], 0, sizeof(r->atmp[0]));
    }
    for (m = cp->out_used; m; m &= m - 1) {
        memset(r->out[ctz32(m)], 0, sizeof(r->out[0]));
    }
    r->out_written = 0;
}

/*
 * One operand over n lanes. A plain register is handed back in place
 * when `alias` allows it -- i.e. when nothing will write the register
 * before the instruction has consumed it -- and copied otherwise.
 */
static const R300PvsLane *r300_pvs_ssrc(const R300PvsCSrc *o,
                                        const R300PvsSoa *r, unsigned n,
                                        R300PvsLane *buf, bool alias)
{
    const R300PvsLane *v;
    unsigned c, l;

    switch (o->file) {
    case 1:
        for (c = 0; c < 4; c++) {
            float k = o->kv[c];

            for (l = 0; l < n; l++) {
                buf[c][l] = k;
            }
        }
        return buf;
    case 0:
        v = r->in[o->idx];
        break;
    case 2:
        v = r->atmp[o->idx];
        break;
    default:
        v = r->tmp[o->idx];
        break;
    }
    if (o->plain) {
        if (alias) {
            return v;
        }
        for (c = 0; c < 4; c++) {
            memcpy(buf[c], v[c], n * sizeof(float));
        }
        return buf;
    }
    for (c = 0; c < 4; c++) {
        unsigned sel = o->sel[c];
        bool neg = (o->neg >> c) & 1;

        if (sel >= 4) {
            float f = sel == R300_PVS_SRC_SELECT_FORCE_1 ? 1.0f : 0.0f;

            if (o->abs) {
                f = fabsf(f);
            }
            if (neg) {
                f = -f;
            }
            for (l = 0; l < n; l++) {
                buf[c][l] = f;
            }
            continue;
        }
        {
            const float *src = v[sel];

            if (o->abs && neg) {
                for (l = 0; l < n; l++) {
                    buf[c][l] = -fabsf(src[l]);
                }
            } else if (o->abs) {
                for (l = 0; l < n; l++) {
                    buf[c][l] = fabsf(src[l]);
                }
            } else if (neg) {
                for (l = 0; l < n; l++) {
                    buf[c][l] = -src[l];
                }
            } else {
                memcpy(buf[c], src, n * sizeof(float));
            }
        }
    }
    return buf;
}

static void r300_pvs_note_math(R300PvsGaps *gaps, unsigned opcode)
{
    if (gaps && !gaps->has_math_op) {
        gaps->has_math_op = true;
        gaps->math_op = opcode;
    }
}

/* the vector engine over n lanes; false for an opcode it does not know */
static bool r300_pvs_svector(unsigned opcode, const R300PvsLane *a,
                             const R300PvsLane *b, const R300PvsLane *c,
                             R300PvsLane *res, unsigned n)
{
    unsigned i, l;

    switch (opcode) {
    case R300_VE_DOT_PRODUCT:
        for (l = 0; l < n; l++) {
            float t = a[0][l] * b[0][l] + a[1][l] * b[1][l] +
                      a[2][l] * b[2][l] + a[3][l] * b[3][l];

            res[0][l] = t;
            res[1][l] = t;
            res[2][l] = t;
            res[3][l] = t;
        }
        return true;
    case R300_VE_MULTIPLY:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = a[i][l] * b[i][l];
            }
        }
        return true;
    case R300_VE_ADD:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = a[i][l] + b[i][l];
            }
        }
        return true;
    case R300_VE_MULTIPLY_ADD:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = a[i][l] * b[i][l] + c[i][l];
            }
        }
        return true;
    case R300_VE_MULTIPLYX2_ADD:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = 2.0f * (a[i][l] * b[i][l]) + c[i][l];
            }
        }
        return true;
    case R300_VE_DISTANCE_VECTOR:
        for (l = 0; l < n; l++) {
            res[0][l] = 1.0f;
            res[1][l] = a[1][l] * b[1][l];
            res[2][l] = a[2][l];
            res[3][l] = b[3][l];
        }
        return true;
    case R300_VE_FRACTION:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = a[i][l] - floorf(a[i][l]);
            }
        }
        return true;
    case R300_VE_MAXIMUM:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = MAX(a[i][l], b[i][l]);
            }
        }
        return true;
    case R300_VE_MINIMUM:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = MIN(a[i][l], b[i][l]);
            }
        }
        return true;
    case R300_VE_SET_GREATER_THAN_EQUAL:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = a[i][l] >= b[i][l] ? 1.0f : 0.0f;
            }
        }
        return true;
    case R300_VE_SET_LESS_THAN:
        for (i = 0; i < 4; i++) {
            for (l = 0; l < n; l++) {
                res[i][l] = a[i][l] < b[i][l] ? 1.0f : 0.0f;
            }
        }
        return true;
    case R300_VE_MULTIPLY_CLAMP:
        for (l = 0; l < n; l++) {
            float t;

            if (c[3][l] < a[3][l] * b[3][l]) {
                t = c[3][l];
            } else if (c[0][l] >= a[0][l] * b[0][l]) {
                t = c[0][l];
            } else {
                t = a[0][l] * b[0][l];
            }
            res[0][l] = t;
            res[1][l] = t;
            res[2][l] = t;
            res[3][l] = t;
        }
        return true;
    default:
        return false;
    }
}

void r300_pvs_exec_soa(const R300PvsCompiled *cp, R300PvsSoa *r, unsigned n,
                       R300PvsGaps *gaps)
{
    static const R300PvsLane zero[4];
    R300PvsLane A[4], B[4], C[4], D[4], R[4];
    unsigned i, k, l;

    if (n > R300_PVS_LANES) {
        n = R300_PVS_LANES;
    }
    for (i = 0; i < cp->n; i++) {
        const R300PvsCIns *in = &cp->ins[i];
        const R300PvsLane *a, *b, *c;
        R300PvsLane *dst;

        /*
         * The operands are read before the dual-issue half writes the
         * alternate temporaries, as the scalar form copies them first:
         * one in that file is copied here too.
         */
        a = r300_pvs_ssrc(&in->a, r, n, A, !(in->dual && in->a.file == 2));
        b = r300_pvs_ssrc(&in->b, r, n, B, !(in->dual && in->b.file == 2));
        c = in->dual ? zero : r300_pvs_ssrc(&in->c, r, n, C, true);

        if (in->dual && in->dop != R300_ME_NO_OP) {
            float fa[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float fb[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float res[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            const R300PvsLane *d = r300_pvs_ssrc(&in->d, r, n, D, false);

            if (!r300_pvs_math(in->dop, fa, fb, fb, res)) {
                r300_pvs_note_math(gaps, in->dop);
            } else {
                for (l = 0; l < n; l++) {
                    fa[3] = d[0][l];
                    fb[3] = d[1][l];
                    r300_pvs_math(in->dop, fa, fb, fb, res);
                    r->atmp[in->ddoff][in->dwe][l] = res[in->dwe];
                }
            }
        }

        if (in->kind == 2) {
            float m = in->opcode ? 2.0f : 1.0f;

            for (k = 0; k < 4; k++) {
                for (l = 0; l < n; l++) {
                    R[k][l] = a[k][l] * b[k][l] * m + c[k][l];
                }
            }
        } else if (in->kind == 1) {
            float fa[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float fb[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float fc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            float res[4];

            if (in->opcode == R300_ME_NO_OP) {
                continue;
            }
            if (!r300_pvs_math(in->opcode, fa, fb, fc, res)) {
                r300_pvs_note_math(gaps, in->opcode);
                continue;
            }
            for (l = 0; l < n; l++) {
                fa[3] = a[3][l];
                fb[3] = b[3][l];
                fc[3] = c[3][l];
                r300_pvs_math(in->opcode, fa, fb, fc, res);
                R[0][l] = res[0];
                R[1][l] = res[1];
                R[2][l] = res[2];
                R[3][l] = res[3];
            }
        } else {
            if (in->opcode == R300_VE_NO_OP) {
                continue;
            }
            if (!r300_pvs_svector(in->opcode, a, b, c, R, n)) {
                if (gaps && !gaps->has_vec_op) {
                    gaps->has_vec_op = true;
                    gaps->vec_op = in->opcode;
                }
                continue;
            }
        }

        if (in->sat) {
            for (k = 0; k < 4; k++) {
                for (l = 0; l < n; l++) {
                    R[k][l] = MIN(MAX(R[k][l], 0.0f), 1.0f);
                }
            }
        }

        switch (in->dtype) {
        case R300_PVS_DST_REG_OUT:
        case R300_PVS_DST_REG_OUT_REPL_X:
            dst = r->out[in->doff % R300_PVS_OUT_REGS];
            r->out_written |= 1u << (in->doff % R300_PVS_OUT_REGS);
            break;
        case R300_PVS_DST_REG_TEMPORARY:
            dst = r->tmp[in->doff % R300_PVS_TMP_REGS];
            break;
        case R300_PVS_DST_REG_ALT_TEMP:
            dst = r->atmp[in->doff % R300_PVS_ATMP_REGS];
            break;
        default:
            if (gaps && !gaps->has_dst_file) {
                gaps->has_dst_file = true;
                gaps->dst_file = in->dtype;
            }
            continue;
        }
        for (k = 0; k < 4; k++) {
            if (in->we & (1u << k)) {
                memcpy(dst[k],
                       R[in->dtype == R300_PVS_DST_REG_OUT_REPL_X ? 0 : k],
                       n * sizeof(float));
            }
        }
    }
}

/* is this operand register file `type` index `off`, read straight through? */
static bool r300_pvs_src_is(uint32_t dw, unsigned type, unsigned off)
{
    unsigned c;

    if ((dw & R300_PVS_SRC_REG_TYPE_MASK) != type ||
        ((dw >> R300_PVS_SRC_OFFSET_SHIFT) & R300_PVS_SRC_OFFSET_MASK) != off ||
        (dw & (R300_PVS_SRC_ABS_XYZW | R300_PVS_SRC_ADDR_MODE_0 |
               R300_PVS_SRC_ADDR_MODE_1))) {
        return false;
    }
    for (c = 0; c < 4; c++) {
        if (((dw >> (R300_PVS_SRC_SWIZZLE_SHIFT + 3 * c)) &
             R300_PVS_SRC_SWIZZLE_MASK) != c ||
            ((dw >> (R300_PVS_SRC_MODIFIER_SHIFT + c)) & 1)) {
            return false;
        }
    }
    return true;
}

/* is this operand the constant vector (1,1,1,1) the swizzler can force? */
static bool r300_pvs_src_is_one(uint32_t dw)
{
    unsigned c;

    for (c = 0; c < 4; c++) {
        if (((dw >> (R300_PVS_SRC_SWIZZLE_SHIFT + 3 * c)) &
             R300_PVS_SRC_SWIZZLE_MASK) != R300_PVS_SRC_SELECT_FORCE_1 ||
            ((dw >> (R300_PVS_SRC_MODIFIER_SHIFT + c)) & 1)) {
            return false;
        }
    }
    return true;
}

/*
 * One row of a texture-coordinate matrix:
 *
 *     out[o].<x|y|z|w> = const[c + row] . in[k]
 *
 * read straight through on both operands. The row is the write enable,
 * which is how the four instructions of a 4x4 are told apart, and it is
 * the same shape the position matrix is recognised by a few lines below
 * -- with the constant's index free rather than pinned to the row, since
 * a texture matrix sits wherever the compiler put it in the file.
 */
static bool r300_pvs_texmat_row(const uint32_t *w, unsigned *out,
                                unsigned *row, unsigned *in, unsigned *cbase)
{
    uint32_t op = w[0];
    unsigned dtype = (op >> R300_PVS_DST_REG_TYPE_SHIFT) &
                     R300_PVS_DST_REG_TYPE_MASK;
    unsigned we = (op >> R300_PVS_DST_WE_SHIFT) & R300_PVS_DST_WE_MASK;
    unsigned k = (w[2] >> R300_PVS_SRC_OFFSET_SHIFT) &
                 R300_PVS_SRC_OFFSET_MASK;
    unsigned c = (w[1] >> R300_PVS_SRC_OFFSET_SHIFT) &
                 R300_PVS_SRC_OFFSET_MASK;

    if ((op & R300_PVS_DST_OPCODE_MASK) != R300_VE_DOT_PRODUCT ||
        (op & (R300_PVS_DST_MATH_INST | R300_PVS_DST_MACRO_INST |
               R300_PVS_DST_DUAL_MATH_OP | R300_PVS_DST_PRED_ENABLE |
               R300_PVS_DST_VE_SAT | R300_PVS_DST_ME_SAT |
               R300_PVS_DST_ADDR_MODE_0 | R300_PVS_DST_ADDR_MODE_1)) ||
        (dtype != R300_PVS_DST_REG_OUT &&
         dtype != R300_PVS_DST_REG_OUT_REPL_X) ||
        (we != 1 && we != 2 && we != 4 && we != 8) ||
        k >= R300_PVS_IN_REGS ||
        !r300_pvs_src_is(w[2], R300_PVS_SRC_REG_INPUT, k) ||
        !r300_pvs_src_is(w[1], R300_PVS_SRC_REG_CONSTANT, c)) {
        return false;
    }
    *out = (op >> R300_PVS_DST_OFFSET_SHIFT) & R300_PVS_DST_OFFSET_MASK;
    *row = we == 1 ? 0 : we == 2 ? 1 : we == 4 ? 2 : 3;
    *in = k;
    *cbase = c - *row;
    return true;
}

bool r300_pvs_texmat(const R300PvsProgram *p, unsigned out,
                     R300PvsTexMat *tm)
{
    unsigned rows = 0, in = 0, cbase = 0, i, r;

    if (!r300_pvs_computes(p, out)) {
        return false;
    }
    for (i = p->first; i <= p->last; i++) {
        unsigned o, row, k, c;

        if (!r300_pvs_texmat_row(&p->code[i * 4], &o, &row, &k, &c) ||
            o != out) {
            continue;
        }
        if (rows && (k != in || c != cbase)) {
            return false;       /* two different matrices on one output */
        }
        in = k;
        cbase = c;
        rows |= 1u << row;
    }
    if (rows != 0xf) {
        return false;
    }
    tm->in = in;
    for (r = 0; r < 4; r++) {
        r300_pvs_const(p, cbase + r, tm->m[r]);
    }
    return true;
}

void r300_pvs_analyse(R300PvsProgram *p, const uint32_t *code,
                      const uint32_t *slot_valid, unsigned code_slots,
                      const uint32_t *cnst, unsigned const_slots,
                      uint32_t code_cntl, uint32_t const_cntl,
                      unsigned first_texcoord)
{
    unsigned matrix_rows = 0, i;
    bool plain;

    memset(p, 0, sizeof(*p));
    for (i = 0; i < R300_PVS_OUT_REGS; i++) {
        p->out_src[i] = -1;
    }
    p->code = code;
    p->cnst = cnst;
    p->code_slots = code_slots;
    p->const_slots = const_slots;
    p->first = code_cntl & 0x3ff;
    p->last = (code_cntl >> 20) & 0x3ff;
    p->cbase = const_cntl & 0xff;
    p->cmax = (const_cntl >> 16) & 0xff;
    p->bounded = const_cntl != 0;

    if (p->last < p->first || p->last >= code_slots) {
        return;
    }
    /*
     * Program RAM is written a slot at a time and never cleared, so the
     * bounds alone do not say the instructions are the guest's: a range
     * reaching past what has been uploaded would execute whatever the last
     * program left there. That is what made an earlier attempt at this
     * depend on the upload history -- the same draw rendering differently
     * according to what had run before it.
     */
    for (i = p->first; i <= p->last; i++) {
        if (!((slot_valid[i / 32] >> (i % 32)) & 1)) {
            return;
        }
    }
    p->valid = true;

    plain = true;
    for (i = p->first; i <= p->last; i++) {
        const uint32_t *w = &p->code[i * 4];
        uint32_t op = w[0];
        unsigned opcode = op & R300_PVS_DST_OPCODE_MASK;
        unsigned dtype = (op >> R300_PVS_DST_REG_TYPE_SHIFT) &
                         R300_PVS_DST_REG_TYPE_MASK;
        unsigned doff = (op >> R300_PVS_DST_OFFSET_SHIFT) &
                        R300_PVS_DST_OFFSET_MASK;
        unsigned we = (op >> R300_PVS_DST_WE_SHIFT) & R300_PVS_DST_WE_MASK;
        bool is_out = dtype == R300_PVS_DST_REG_OUT ||
                      dtype == R300_PVS_DST_REG_OUT_REPL_X;

        if (is_out && doff < R300_PVS_OUT_REGS) {
            p->out_mask |= 1u << doff;
            /* out[n] = in[k] * (1,1,1,1): an attribute forwarded intact */
            if (opcode == R300_VE_MULTIPLY && we == 0xf &&
                !(op & (R300_PVS_DST_MATH_INST | R300_PVS_DST_MACRO_INST |
                        R300_PVS_DST_DUAL_MATH_OP | R300_PVS_DST_PRED_ENABLE |
                        R300_PVS_DST_VE_SAT)) &&
                r300_pvs_src_is_one(w[2]) &&
                r300_pvs_src_is(w[1], R300_PVS_SRC_REG_INPUT,
                                (w[1] >> R300_PVS_SRC_OFFSET_SHIFT) &
                                R300_PVS_SRC_OFFSET_MASK) &&
                (((w[1] >> R300_PVS_SRC_OFFSET_SHIFT) &
                  R300_PVS_SRC_OFFSET_MASK) < R300_PVS_IN_REGS)) {
                p->out_src[doff] = (w[1] >> R300_PVS_SRC_OFFSET_SHIFT) &
                                   R300_PVS_SRC_OFFSET_MASK;
            }
        }
        if (!plain) {
            continue;
        }
        if (op & (R300_PVS_DST_MATH_INST | R300_PVS_DST_MACRO_INST |
                  R300_PVS_DST_DUAL_MATH_OP | R300_PVS_DST_PRED_ENABLE |
                  R300_PVS_DST_VE_SAT | R300_PVS_DST_ME_SAT |
                  R300_PVS_DST_ADDR_MODE_0 | R300_PVS_DST_ADDR_MODE_1) ||
            !is_out) {
            plain = false;
            continue;
        }
        if (opcode == R300_VE_DOT_PRODUCT && doff == 0 &&
            (we == 1 || we == 2 || we == 4 || we == 8) &&
            r300_pvs_src_is(w[2], R300_PVS_SRC_REG_INPUT, 0)) {
            unsigned row = we == 1 ? 0 : we == 2 ? 1 : we == 4 ? 2 : 3;

            if (r300_pvs_src_is(w[1], R300_PVS_SRC_REG_CONSTANT, row)) {
                matrix_rows |= 1u << row;
                continue;
            }
        }
        if (doff && p->out_src[doff] >= 0) {
            continue;               /* a forwarded attribute, recorded above */
        }
        /*
         * An instruction landing on a texture-coordinate output does not
         * make the POSITION matrix wrong, so it does not decide `plain`
         * here -- but it is not "beside the point" either, which is what
         * the comment that used to sit here said. It was checked against
         * Mac OS X 10.4, whose blit program multiplies its coordinate by
         * exactly diag(1/w, 1/h, 1, 1) -- the inverse of this model's own
         * attribute scaling, so ignoring the instruction was free.
         * Mac OS X 10.5 sends the same shape over a vertex that puts the
         * coordinate somewhere the fixed positional read does not look,
         * and there ignoring it costs the whole coordinate. The loop
         * below decides `plain` for these: a computed coordinate keeps
         * the fast path only while r300_pvs_texmat() can hand the caller
         * the matrix to apply.
         */
        if (doff >= first_texcoord) {
            continue;
        }
        plain = false;
    }
    for (i = first_texcoord; i < R300_PVS_OUT_REGS; i++) {
        R300PvsTexMat tm;

        if (r300_pvs_computes(p, i) && !r300_pvs_texmat(p, i, &tm)) {
            plain = false;
        }
    }
    p->plain_matrix = plain && matrix_rows == 0xf;
}
