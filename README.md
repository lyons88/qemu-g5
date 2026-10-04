# QEMU Power Mac G5 — Radeon 9800 Metal back end

This is a fork of [Cat_7's QEMU G5 work](https://github.com/cat7/qemu/tree/powermac73) (`powermac73`, as of late September 2026) that speeds up the emulated **ATI Radeon 9800 (R350, `ati-radeon9800`)** on Apple Silicon Macs. It draws the guest's 3D on the host GPU through **Metal** instead of QEMU's software rasterizer.

Guest tested: **Mac OS X 10.4.11 Tiger** on an emulated quad-CPU G5, running on a 13" M2 MacBook Pro (16 GB).

## FPU and audio fixes (branch `FPU-Audio-Fixes`)

This branch builds on `g5-metal` with faster floating point and smoother sound. Every change can be turned off on the command line.

### FPU: host floating point (on by default)

QEMU normally does every PowerPC floating-point instruction in software, because the PowerPC exception flags (FPSCR) have to be exact. These three switches let the M2's own floating point do the work instead:

| Switch | What it does |
|---|---|
| `fp-hardfloat` | Lets QEMU's softfloat use the host FPU for double precision, and adds a host path for single precision (`fadds`, `fsubs`, `fmuls`, `fdivs`, `fmadds` and the rest), which previously had none. It falls back to software for overflow, underflow and divide-by-zero. |
| `vmx-hardfloat` | Runs AltiVec `vaddfp`, `vsubfp`, `vmaddfp` and `vnmsubfp` on the host, four lanes at a time. It falls back per vector when a lane needs special handling. |
| `fpscr-lazy` | Stops recomputing the FPSCR result-class bits (FPRF) after every FP instruction. Mac OS X applications almost never read them. |

Turn any of them off individually:

```
-global powerpc64-cpu.fp-hardfloat=off
-global powerpc64-cpu.vmx-hardfloat=off
-global powerpc64-cpu.fpscr-lazy=off
```

| Benchmark (M2, Tiger) | Before | All three on |
|---|---|---|
| Jedi Knight II timedemo | 26.1 fps | 51 fps |
| Quake 3 timedemo | 136 fps | 168 fps |
| Cinebench 2003 C4D shading | 74 | 123 |
| Fractal Demo Carbon, AltiVec | 671 | 989 |
| Fractal Demo Carbon, FPU | 650 | 890 |

### Audio (K2 I2S sound)

- **iTunes skipping while windows move: about 90% fixed; still being tracked.** The cause was the Radeon's draw thread taking the texture lock back after every draw. That starved the display refresh, which holds QEMU's big lock while it waits, and the sound DMA timers stalled behind it for 40 ms to over 1 s. The draw thread now steps aside between draws whenever the refresh is waiting. In testing, 93 s of playback with window dragging had no buffer underruns, and the worst timer delay was 52 ms.
- **Low-watermark rebuffering** (`audio-low-ms`, default 20). If the host buffer drops below this, it refills before playing on, instead of popping on every callback from then on. Turn it off with `-global macio-newworld.audio-low-ms=0`.
- **Catch-up limit** (`audio-catchup-ms`, default 10). Caps how far behind the sound DMA is allowed to fall before it resyncs. Keep it small; large values burst-retire buffers and can freeze the guest.
- **Opt-in audio log** (`-global macio-newworld.audio-log=FILE`). Writes one line a second with DMA lateness, FIFO depth, underruns and refills. It does nothing unless set.
- **Quake 3 "racing / laser" sound: fixed.**
- Includes Cat_7's macOS nanosecond kqueue poll timeout (`util/qemu-timer.c`).

### Cat_7's newer changes were rolled back

We merged Cat_7's latest `powermac73` work (USB 2.0, CoreAudio, GL general programs and others) and then rolled most of it back. It conflicted with the Metal back end and broke rendering and the build on macOS. This branch is still based on `powermac73` from late September 2026, plus only his kqueue timer fix.

## What `g5-metal` adds

- **Texturing:** whole-block DXT1/3/5 decoding; packed YUV 4:2:2; AGP/PCI textures copied through the GART in bulk and cached by content; cube maps.
- **Fragment programs on Metal:** up to four texture units per program (any four of the eight); programs that fetch inside themselves (dependent reads, several indirection levels, TEXKILL); raw coordinate sets 4–7.
- **Geometry:** near-plane clipping for strips, fans, quads and polygons; guard-band clipping, so triangles that pass beside the camera stay on the GPU.
- **Occlusion queries** (ZB_ZPASS_DATA/ADDR). Without them Halo hung.
- **OpenPIC fix:** no crash when a guest touches the timer registers on the U3 MPIC.
- **Diagnostics:** detailed `gl-stats`, plus opt-in debugging that does nothing unless you name it on `-device`: `regtrace=FILE`, `us-dump=FILE` and `stats-log=FILE`.

## Graphics results (M2, Tiger 10.4.11)

| Test | Before | Now |
|---|---|---|
| Jedi Knight II timedemo | 16.1 fps | 41 fps (51 with the FPU fixes) |
| Cinebench 2003 OpenGL HW-L | 119 (software) | 389 CB-GFX |
| Halo | hung in the menus | timedemo runs through; playable |
| ATI SmartShader demos | Tubes black | Bubbles, Brains, Tubes render; Ocean partly |

## Build

```
git clone -b FPU-Audio-Fixes https://github.com/lyons88/qemu-g5.git
cd qemu-g5
mkdir build && cd build
../configure --target-list=ppc64-softmmu --enable-cocoa
ninja
```

## Run

You need a Tiger disk image, the OpenBIOS from Cat's G5 releases, and a Radeon 9800 Mac ROM. Run from the folder holding the disk image:

```
./build/qemu-system-ppc64 -M mac99,via=pmu -smp 4 -m 8G \
  -bios /path/to/openbios-ppc \
  -display cocoa -vga none \
  -device ati-radeon9800,bus=pci.0,addr=0x10,romfile="/path/to/Radeon 9800 ME rev 130.rom",gl=on,gl-api=metal,gl-async=off,id=gpu \
  -global adb-mouse.extended-protocol=on \
  -drive file=Tiger.img,format=raw,if=ide,index=0,media=disk -boot c
```

- `gl-api=metal` turns on the Metal back end.
- `gl-async` is **off** by default and should stay off (see the known issues).
- `raster-threads` defaults to half your cores. On an M2, that's the 4 performance cores.

## Playable settings: Halo

Tested at **640×480 with every option at its lowest** and **Hardware Shaders: None**, Detail Objects off, FSAA off. It runs, and the timedemo completes with only occasional audio pops.

## Known issues

- Halo's sand and dirt show a regular checkerboard of dark rectangles. The cause isn't found yet; DXT decoding has been ruled out.
- Halo with hardware shaders on is slower than with them off. Most of the remaining time is spent waiting on the GPU at fences.
- `gl-async=on` can leave Tiger stuck at the desktop waiting on a held-back fence.
- The ATI Ocean demo's water is too dark.
- Audio can still pop occasionally under heavy screen activity (about 90% fixed; being tracked).
- Based on powermac73 from late September 2026; Cat's newer changes were rolled back (see above).

## Credits

Cat_7 (Howard) for the G5 machine and the R350 emulation this builds on; linuxkid473's R300-to-Metal work as a reference.
