# PowerPC FPU speed-up — work log

Goal: a large (~50%) real-world speed-up of guest floating point on the
G5 (qemu-system-ppc64, Tiger 10.4.11 guest, M2 MacBook Pro host).

## 2026-10-04

### 1. Measured where the time goes (JK2 timedemo, CPU sample)
- JK2's game runs on one guest CPU: CPU 1 75% busy, CPU 0 33%, CPUs 2-3 under 15%.
- 55% of that busy thread is software floating point (QEMU softfloat:
  parts64_muladd, parts64_canonicalize, parts64_uncanon_normal, helper_todouble).
- Why it matters: if FP got ~5x cheaper, that thread would run ~1.6-1.8x faster.

### 2. Tried QEMU's existing "hardfloat" (native host FPU) path
- Finding: QEMU only uses the host FPU when the "inexact" flag is already set
  before an operation; the PPC code clears all flags before every FP
  instruction, so hardfloat never runs on PPC.
- Change: target/ppc/fpu_helper.c helper_reset_fpstatus() — with the
  environment variable QEMU_PPC_HARDFLOAT=1, start each operation with
  "inexact" raised so hardfloat can run. Off by default; no rebuild to switch.
- Trade-off: FPSCR[FI] reads 1 and FPSCR[XX] stays set; a guest that enables
  the inexact exception (FPSCR[XE]) would trap on every FP instruction.
  Mac OS X leaves it off.
- Result (sample with the switch on): no gain. Games use single-precision
  instructions (fadds/fmuls/fmadds/fdivs), which QEMU implements with the
  float64r32_* functions — and those have no hardfloat path at all.

### 3. Plan (agreed)
- Stage 1: add a hardfloat path to float64r32_add/sub/mul/div/muladd (native
  M2 instruction, round to single, fall back to softfloat for NaN/Inf/denormal
  inputs and overflow/underflow results). Behind QEMU_PPC_HARDFLOAT=1.
- Stage 2: lazy FPSCR — stop resetting flags, computing FPRF and checking
  exceptions on every FP instruction while FP exceptions are disabled;
  compute the status bits only when the guest reads FPSCR (mffs etc.).
- Not now: native FP code straight from the JIT (needs new TCG ops, weeks),
  AltiVec float on NEON (JK2 barely uses AltiVec).
- NEON note: on Apple Silicon scalar FP and NEON are the same hardware;
  hardfloat already uses it.

### 4. Baseline benchmarks (current build, QEMU_PPC_HARDFLOAT unset)
Earlier numbers: Cinebench 2003 CPU 37 (1 CPU) / 111 (x CPU); JK2 41 fps.
Each app also gets a 15 s CPU sample to see which FP instructions it hits:
s-jk2, s-q3, s-cb, s-fractal-fpu, s-fractal-altivec (deleted after reading).

| Benchmark | Baseline | Stage 1 | Stage 2 |
|---|---|---|---|
| JK2 timedemo (fps) | | | |
| Quake 3 timedemo (fps) | | | |
| Cinebench 2003 CPU, 1 / x CPU | 37 / 111 | | |
| Fractal Carbon, FPU mode | | | |
| Fractal Carbon, AltiVec mode | | | |

### 5. Per-app FP profile (15 s samples, current build, hardfloat switch off)
Share of busy guest-CPU time. Samples deleted after reading. Scores taken
while sampling are not used as baselines (sampling slows the guest).

| App | All FP | softfloat core | FPSCR bookkeeping | lfs/stfs/frsp | Main FP instructions |
|---|---|---|---|---|---|
| JK2 | 47% | 27% | 7% | 5% | fmadds, fmuls, fadds, fctiw |
| Quake 3 | 31% | 15% | 6% | 4% | fmadds, fmuls |
| Cinebench 2003 | 33% | 16% | 6% | 3% | fmadd (double), fmuls, fadds |
| Fractal Carbon, FPU | 83% | 55% | 15% | — | fmuls, fmadds, fsubs, fcmpo (all single) |
| Fractal Carbon, AltiVec | 70% | 18% | — | — | vmaddfp (float32_muladd 50%) |

The rest is mostly JIT-generated code (9–30%), translation-block lookup
(5–11%) and memory access (3–8%).

Expected ceiling if FP cost drops ~80%: JK2 ~1.6x, Q3 and Cinebench ~1.3x,
Fractal FPU ~3x, Fractal AltiVec ~2x.

Order of work:
- A. AltiVec: vmaddfp etc. still run the softfloat path (parts_is_snan_frac,
  softfloat core in the profile). AltiVec has no inexact status bit, so
  letting its float ops use the host FPU costs no guest-visible accuracy.
- B. Stage 1: native single-precision scalar math (float64r32_*).
- C. Stage 2: lazy FPSCR (removes the 6–15% bookkeeping).

### 6. All three changes in, each switchable on the QEMU command line
User decision: no baselines (known peak scores per app); build everything,
make each change a separate switch, test with all on.

Switches (CPU properties, default off; turn on with
`-global powerpc64-cpu.<name>=on`):

| Switch | Files | What it does | Why | Guest-visible cost |
|---|---|---|---|---|
| fp-hardfloat | fpu/softfloat.c, target/ppc/fpu_helper.c | Pre-raises "inexact" each FP op so QEMU's hardfloat runs; adds a host-FPU path to float64r32_add/sub/mul/div/muladd (fadds, fsubs, fmuls, fdivs, fmadds/fmsubs/fnmadds/fnmsubs) | Games and Fractal FPU are single precision, which had no native path | FPSCR FI=1 and XX always set; last-bit rounding can differ in rare halfway cases (0 mismatches in a 10M-op fmadds test) |
| vmx-hardfloat | target/ppc/int_helper.c | vaddfp, vsubfp, vmaddfp, vnmsubfp: all four lanes on the host FPU in one go | vmaddfp alone was 50% of Fractal AltiVec time, one softfloat call per lane | None: AltiVec keeps no FP status; NaN/Inf/denormal/overflow/underflow vectors fall back to softfloat |
| fpscr-lazy | target/ppc/translate.c, translate/fp-impl.c.inc | Skips the FPRF and FPSCR status helpers after FP arithmetic | 6–15% of busy time was this bookkeeping | FPRF, FR/FI and sticky OX/UX/XX go stale; enabled overflow/underflow/inexact exceptions are not raised (invalid-op and divide-by-zero still are) |

The old QEMU_PPC_HARDFLOAT environment variable is replaced by fp-hardfloat.
Every fast path falls back to softfloat for NaN, infinity, denormal inputs
and for overflow/underflow results, so those keep exact IEEE behaviour.

Checks done: full qemu-system-ppc64 build clean on Linux; the three
properties read back "true" through the monitor when set with -global;
fmadds rounding (double fma then round to single vs single fma): 0
differences in 10 million random cases.

### 7. Results — all three switches on (user's runs, vs his known peaks)

| Benchmark | Before | All on | Gain |
|---|---|---|---|
| Jedi Knight II timedemo | 26.1 fps | 51 fps | +95% |
| Quake 3 timedemo | 136 fps | 168 fps | +24% |
| Cinebench 2003 C4D Shading | 74 | 123 | +66% |
| AltiVec Fractal Carbon, AltiVec mode | 671 | 989 | +47% |
| AltiVec Fractal Carbon, FPU mode | 650 | 890 | +37% |

No rendering or calculation problems reported with all three on.

### 8. Quake 3 audio "racing / lasers" -- FIXED
- Symptom: with sound on, Q3 audio races and sounds like lasers; with
  `s_initsound 0` Q3 runs ~200 fps instead of 168.
- The G5 sound (hw/misc/macio/k2_sound.c) retires each audio DMA descriptor
  on a QEMU timer at the pace the audio would play. On macOS, QEMU's poll
  rounded every timer deadline up to the next millisecond, so those timers
  fire late and then catch up in bursts — the DMA position Q3 mixes against
  jumps instead of advancing smoothly.
- Change: took only Cat's util/qemu-timer.c fix (917e1bd, "wake
  qemu_poll_ns on time on macOS": a nanosecond kqueue timer polled with the
  descriptors). Nothing else from his merge.
- Test run carries a self-starting audio trace (k2_i2s_dma) so the DMA
  timing is on record if the fix isn't enough.
- Result: timer fix did not change Q3's racing audio; kept (harmless, Cat's
  fix). Q3 audio left as a known bug for now.
- Q3 audio trace (deleted after reading): the OS X driver feeds the K2 I2S
  output in 256-byte DMA descriptors at 48 kHz, 64 frames = 1.33 ms each,
  ~750 descriptor retirements a second per direction, each on a QEMU timer.

### 9. iTunes skips/pops with screen activity (since mac99 audio was added)
- Not OpenBIOS: it only describes the sound hardware in the device tree;
  Mac OS X's own driver programs the I2S cell and DBDMA itself.
- Not Quartz Extreme (turning QE off changes nothing).
- Suspect: the output DMA is paced by QEMU timers firing 750 times a second;
  when the main loop is busy (window moves = display/2D work), they fire late,
  and anything later than 10 ms (K2_I2S_MAX_DEBT_NS) is thrown away, so the
  guest is fed less audio than it plays and the host FIFO runs dry.
- Instrument first: macio-newworld property audio-log=FILE writes one line a
  second — descriptors retired, timer lateness avg/max, debt thrown away,
  host callbacks short/empty/refill, silence written, bytes dropped, FIFO
  level min..max in ms.
- Audio log result (iTunes MP3; still / dragging windows / still; deleted
  after reading):
  - Still: timer lateness ~40 us average, ~2 ms worst; nothing thrown away;
    host FIFO steady at 34–66 ms.
  - Dragging: timers late by up to 50–106 ms; 6–198 ms of audio time thrown
    away per second by the 10 ms catch-up limit; host FIFO driven down to
    9 ms, and every host callback found less audio than it asked for.
  - After dragging stopped: the FIFO never recovered (stayed at 9–12 ms with
    every callback short) and finally ran empty — the pops continue after
    the drag because the lost time is never made up.
- Cause: the main loop stalls for up to ~100 ms while windows move; the K2
  model then discards anything over 10 ms of lateness instead of catching up.
- Change: new macio-newworld property audio-catchup-ms (default 10 = old
  behaviour). With a larger value (test: 250) the DMA catches up after a
  stall, so the guest refills the host FIFO.
- Still to find: what holds the main loop for ~100 ms during window moves.
- audio-catchup-ms=250 froze Tiger the moment a song started: the audio log
  shows not one output descriptor retired. A stream that (re)starts has
  deadline 0, which the clamp turned into "250 ms behind", so the model
  retired a quarter-second of the guest's ring instantly at start. Do not
  use large audio-catchup-ms values.
- Changes:
  - A (re)started stream now starts its timeline at "now" with no debt
    (also with the default; before, it started 10 ms behind).
  - New macio-newworld property audio-low-ms (default 0 = off): if a stall
    has pushed the host FIFO below this level, go back to prebuffering —
    one short gap of silence, then the full 60 ms cushion again — instead of
    popping on every host callback for the rest of the song.

## 10. Audio skips while dragging windows: display refresh starved on gl_tex_lock

- With `audio-low-ms=20` the FIFO now recovers to 40-74 ms after a drag; the permanent popping after a drag is gone. Drags still caused 40-235 ms timer stalls (one of 1.2 s).
- A sample taken during a drag showed the QEMU main loop (holding the BQL) blocked ~19% of the time in `ati_r350_update_display -> ati_r350_take_dirty -> gl_tex_lock`. The command-processor thread takes that lock for every draw in `r300_run_prims` and grabs it straight back, so the refresh can wait a long time. While it waits, the BQL is held and the K2 DBDMA timers (QEMU_CLOCK_VIRTUAL, main loop) cannot run, so audio underruns.
- Fix: `gl_tex_waiters` counter in ATIR350State. The display side increments it around taking the lock (take_dirty and the `s->mode` update). After each draw unlocks, `r300_run_prims` calls `sched_yield()` (up to 2000 times) while a waiter is pending, so the refresh gets the lock between draws. take_dirty now unlocks explicitly instead of QEMU_LOCK_GUARD.
- Result (iTunes MP3 with window drags): 93 s of playback, 0 refills, 1 short callback, worst DMA lateness 52 ms (before: 40-235 ms stalls, one of 1.2 s, several refills a second). James: "you fixed it" -- no audible skips.
- `audio-low-ms` default changed 0 -> 20 (set `audio-low-ms=0` to turn it off).
- `fp-hardfloat`, `vmx-hardfloat`, `fpscr-lazy` now default on; turn each off with `-global powerpc64-cpu.<name>=off`.

## Status (pushed as branch FPU-Audio-Fixes on lyons88/qemu-g5)
- Quake 3 "racing / lasers" audio: FIXED (confirmed by James).
- iTunes skips/pops during window drags: ~90% fixed, still tracking.
- FPU switches and audio-low-ms=20 are on by default; each can be turned off on the command line.
