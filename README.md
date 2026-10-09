# QEMU Power Mac G5 — Radeon 9800 Metal back end

This is a fork of [Cat_7's QEMU G5 work](https://github.com/cat7/qemu/tree/G5-openbios) (his branch `G5-openbios`, formerly `powermac73`; forked as of late September 2026) that speeds up the emulated **ATI Radeon 9800 (R350, `ati-radeon9800`)** on Apple Silicon Macs. It draws the guest's 3D on the host GPU through **Metal** instead of QEMU's software rasterizer.

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
- **Low-watermark rebuffering** (`audio-low-ms`, default 0 = off). If set, the host buffer refills before playing on whenever it drops below this many milliseconds. The 200 ms FIFO cap now covers what this used to do.
- **Catch-up limit** (`audio-catchup-ms`, default 250). Caps how far behind the sound DMA is allowed to fall before it resyncs. Large values used to freeze the guest; that was a recursion bug, now fixed (see below).
- **Opt-in audio log** (`-global macio-newworld.audio-log=FILE`). Writes one line a second with DMA lateness, FIFO depth, underruns and refills. It does nothing unless set.
- **Quake 3 "racing / laser" sound: fixed.**
- Includes Cat_7's macOS nanosecond kqueue poll timeout (`util/qemu-timer.c`).

### QuickTime video and sound

- **Colours fixed.** QuickTime's YUV-to-RGB fragment program now renders correctly. Volume (3D) textures are supported, which ColorSync's 32x32x32 lookup table needs. The alpha `DP` op now returns the full DP4 result when the RGB side runs DP4. The packed 4:2:2 formats (`0x14`/`0x15`) are decoded too. `yuv-cat-order=on` on the device switches to the byte order of Cat's `g5-yuvtex`.
- **On Metal.** QuickTime's program fetches through texture coordinate sets 4 and up, which the Metal general path used to refuse, so every video frame ran in the software interpreter at about 34 ms a frame. Sets 4-7 now reach the shader divided by q, as the interpreter does. Draws that would need a LOD from one of those sets still fall back. The translated-shader buffer grew from 16 KB to 128 KB, because this program overflowed it.
- **Faster frame upload.** A 32-bit texture row in VRAM under one swap is now read with the swap resolved once per row, not once per texel. For a 720x480 frame that is about 345,000 fewer swap lookups per frame. This matters most when the host CPU is throttled, for example in Low Power Mode.
- **Sound in step.** The K2 sound DMA timers run in their own thread (`k2-sound`), so a slow display refresh no longer holds them up. While waiting for the draw thread, the display refresh now releases QEMU's big lock, so the guest's CPUs keep running. The host FIFO is capped at 200 ms, so a host stall costs one skip instead of seconds of lag. Catching up after a stall no longer recurses once per descriptor; that recursion overflowed the stack and locked QEMU up.
- New on `macio-newworld`: `audio-count-lag-ms` (default 20), how far the I2S frame counter trails the DMA. The opt-in audio log also reports `fcread` (frame-counter reads per second) and `zero` (how much of the guest's audio was silent when the DMA read it).
- Known issue: the first time QuickTime opens after a fresh build, Metal compiles its shader. The display freezes for a few seconds and the mouse can leave the window. Later runs use the cached shader.
- **Audio defaults.** The values QuickTime was tested with are now the defaults, so this is all a G5 needs:

  ```
  -audiodev coreaudio,id=snd -global macio-newworld.audiodev=snd
  ```

  To adjust one, add it to the command line with a different value:

  | Setting | Default | What it does |
  |---|---|---|
  | `-audiodev coreaudio,...,out.buffer-count=N` | 8 (QEMU's own default is 4) | CoreAudio output buffers; more rides out longer host stalls, at the cost of latency |
  | `-global macio-newworld.audio-catchup-ms=N` | 250 | How far behind the sound DMA may fall before audio is dropped to resync |
  | `-global macio-newworld.audio-low-ms=N` | 0 (off) | Refill the host buffer before playing on when it falls below N ms |
  | `-global macio-newworld.audio-count-lag-ms=N` | 20 | How far the I2S frame counter trails the DMA (debugging) |
  | `-global macio-newworld.audio-log=FILE` | off | One line a second of sound timing statistics (debugging) |

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

QEMU needs a few libraries from [Homebrew](https://brew.sh). Install them, then build with only the features a Power Mac needs. Everything else (GTK, SDL, VNC, TLS, USB pass-through and so on) is left out, so the build pulls in five Homebrew libraries instead of several dozen.

```
brew install meson ninja pkgconf glib pixman libslirp
git clone -b FPU-Audio-Fixes https://github.com/lyons88/qemu-g5.git
cd qemu-g5
mkdir build && cd build
../configure --target-list=ppc64-softmmu --enable-cocoa --enable-coreaudio \
  --enable-slirp --enable-pixman \
  --disable-gtk --disable-sdl --disable-sdl-image --disable-opengl \
  --disable-vnc --disable-vnc-jpeg --disable-png --disable-curses \
  --disable-gnutls --disable-nettle --disable-gcrypt --disable-libssh \
  --disable-curl --disable-libusb --disable-usb-redir --disable-smartcard \
  --disable-zstd --disable-lzfse --disable-snappy --disable-lzo \
  --disable-capstone --disable-gio --disable-dbus-display --disable-spice \
  --disable-virglrenderer --disable-xkbcommon --disable-tools \
  --disable-guest-agent --disable-docs
ninja
```

The Metal and CGL renderers for the Radeon 9800 use macOS's own frameworks, not QEMU's `opengl` option, so `--disable-opengl` does not affect them. To check what the binary links, run `otool -L qemu-system-ppc64`. Everything outside `/System` and `/usr/lib` should come from the libraries in the table below.

### Homebrew libraries QEMU uses

These are not part of this repository and are not modified. They are installed from Homebrew on your own Mac. Each Homebrew formula page lists the exact version, its license and the source tarball it was built from.

| Library | Used for | License | Homebrew | Source code |
|---|---|---|---|---|
| GLib (`libglib-2.0`, `libgmodule-2.0`) | QEMU's core utility library | LGPL-2.1-or-later | [glib](https://formulae.brew.sh/formula/glib) | [gitlab.gnome.org/GNOME/glib](https://gitlab.gnome.org/GNOME/glib) |
| gettext (`libintl`, needed by GLib) | Message translation | libintl: LGPL-2.1-or-later (gettext's tools: GPL-3.0-or-later) | [gettext](https://formulae.brew.sh/formula/gettext) | [gnu.org/software/gettext](https://www.gnu.org/software/gettext/) |
| PCRE2 (`libpcre2-8`, needed by GLib) | Regular expressions | BSD-3-Clause | [pcre2](https://formulae.brew.sh/formula/pcre2) | [github.com/PCRE2Project/pcre2](https://github.com/PCRE2Project/pcre2) |
| pixman | Display pixel operations | MIT | [pixman](https://formulae.brew.sh/formula/pixman) | [gitlab.freedesktop.org/pixman/pixman](https://gitlab.freedesktop.org/pixman/pixman) |
| libslirp | User-mode networking (`-nic user`) | BSD-3-Clause | [libslirp](https://formulae.brew.sh/formula/libslirp) | [gitlab.freedesktop.org/slirp/libslirp](https://gitlab.freedesktop.org/slirp/libslirp) |

zlib, bzip2, libiconv and the Cocoa, CoreAudio, Metal and OpenGL frameworks come with macOS.

The G5 OpenBIOS is GPLv2. Its source is [Cat_7's `G5-openbios` branch](https://github.com/cat7/qemu/tree/G5-openbios).

## Run

You need a Tiger disk image, the G5 OpenBIOS (`openbios-ppc`) built from [Cat_7's `G5-openbios` branch](https://github.com/cat7/qemu/tree/G5-openbios), and a Radeon 9800 Mac ROM. Run from the folder holding the disk image:

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
- Sound stutters and pops when the host M2 is in Low Power Mode. Turn Low Power Mode off for smooth audio.
- Based on powermac73 from late September 2026; Cat's newer changes were rolled back (see above).

## Credits

Cat_7 (Howard) for the G5 machine and the R350 emulation this builds on; linuxkid473's R300-to-Metal work as a reference.
