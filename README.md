# QEMU Power Mac G5 — Radeon 9800 Metal back end

This is a branch of [Cat_7's QEMU G5 work](https://github.com/cat7/qemu/tree/powermac73) (`powermac73`, as of late September 2026) that speeds up the emulated **ATI Radeon 9800 (R350, `ati-radeon9800`)** on Apple Silicon Macs. It draws the guest's 3D on the host GPU through **Metal** instead of QEMU's software rasterizer.

Guest tested: **Mac OS X 10.4.11 Tiger** on an emulated quad-CPU G5, running on a 13" M2 MacBook Pro (16 GB).

## What it adds

- **Texturing:** whole-block DXT1/3/5 decoding; packed YUV 4:2:2; AGP/PCI textures copied through the GART in bulk and cached by content; cube maps.
- **Fragment programs on Metal:** up to four texture units per program (any four of the eight); programs that fetch inside themselves (dependent reads, several indirection levels, TEXKILL); raw coordinate sets 4–7.
- **Geometry:** near-plane clipping for strips, fans, quads and polygons; guard-band clipping, so triangles that pass beside the camera stay on the GPU.
- **Occlusion queries** (ZB_ZPASS_DATA/ADDR). Without them Halo hung.
- **OpenPIC fix:** no crash when a guest touches the timer registers on the U3 MPIC.
- **Diagnostics:** detailed `gl-stats`, plus opt-in debugging that does nothing unless you name it on `-device`: `regtrace=FILE`, `us-dump=FILE` and `stats-log=FILE`.

## Results (M2, Tiger 10.4.11)

| Test | Before | Now |
|---|---|---|
| Jedi Knight II timedemo | 16.1 fps | 41 fps |
| Cinebench 2003 OpenGL HW-L | 119 (software) | 389 CB-GFX |
| Halo | hung in the menus | timedemo runs through; playable |
| ATI SmartShader demos | Tubes black | Bubbles, Brains, Tubes render; Ocean partly |

## Build

```
git clone -b g5-metal https://github.com/lyons88/qemu-g5.git
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
- Based on powermac73 from late September 2026. Cat's newer changes (USB 2.0, CoreAudio, GL general programs and others) are not merged yet.

## Credits

Cat_7 (Howard) for the G5 machine and the R350 emulation this builds on; linuxkid473's R300-to-Metal work as a reference.
