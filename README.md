# ML Video Tools

Turn a clip into compositing passes on your own Windows PC, with one command:

- **depth** - 16-bit depth maps (Video Depth Anything, or Depth Anything 3)
- **flow** - 16-bit PNG and 32-bit EXR motion vectors (MEMFOF), or EXR only (WAFT)
- **mattes** - black-and-white mattes from a word like "person" or "car" (SAM 3.1)

It extracts the frames once, runs each tool in turn, restarts anything that crashes or
freezes, and picks up where it left off if you stop it. Long clips are fine.

Made by Lab—Meta. Code is MIT licensed. **The AI models are not included** - each is
downloaded from its makers and has its own licence. Read `THIRD_PARTY_LICENCES.md`
before using results in paid work.

---

## What you need
- Windows 10 or 11 with an NVIDIA graphics card (RTX 20-series or newer, 8GB+ memory recommended)
- Python 3.12, git and ffmpeg
- The three tools installed - see **`INSTALL.md`**

## Setting it up
1. Put this folder somewhere, e.g. `C:\ML\ml-video-tools`
2. Install the tools you want (see `INSTALL.md`)
3. Copy `settings.example.json` to `settings.json` (or just run step 4 once - it makes it for you),
   then open `settings.json` and check the three install folders are right
4. Check everything is ready:
   ```
   cd C:\ML\ml-video-tools
   python process_clip.py --check
   ```
   You want `OK` next to each tool.

## Using it
```
cd C:\ML\ml-video-tools
python process_clip.py "D:\clips\test.mp4" --prompt person
```
No need to switch on any environment - it uses each tool's own one for you.
Leave out `--prompt` and it does depth and flow only.

### What you get
Everything goes in one folder next to the clip:
```
test_shot\
   Color\      test_0001.png ...                          the extracted frames (kept)
   Depth\      test_depth_0001.png ...                    16-bit, white = far, black = near
   Optical\    png\test_optical-flow_0001.png ...         16-bit motion vectors, 0.5 = still (see below)
               exr\test_optical-flow_0001.exr ...         32-bit float motion vectors in pixels (R = x, G = y)
               _flow_range.txt                          the top speed, to turn PNG values back into pixels
               preview\test_optical-flow_preview_0001.png colour check pictures
   Segm\       person\test_segmentation_person_0001.png   one folder per --prompt
   log.txt     what happened, with times
```
Every file is named after the clip, then the process, with the frame number last, so
After Effects and Nuke pick them up as sequences and you can always tell passes apart.
The folders are the same whichever engine made them (Depth for VDA or DA3, Optical for
MEMFOF or WAFT). Shots made before 0.7.0 have their old folders (frames, depth, flow, mattes)
renamed automatically the next time you run anything on them.

### Choosing what to run
| Add this | To |
|---|---|
| `--only depth` | run just one tool (`depth`, `flow` or `mattes`) |
| `--skip flow` | leave a tool out |
| `--prompt person --prompt "red car"` | get a matte folder for each word |

Adding a tool later reuses the frames already extracted:
`python process_clip.py "D:\clips\test.mp4" --only mattes --prompt car`

### Options for each tool
| Tool | Option | What it does |
|---|---|---|
| depth | `--depth-engine da3` | use Depth Anything 3 instead of Video Depth Anything (default set in settings.json) |
| depth | `--vda-encoder vits` | Video Depth Anything model: `vits` Small (commercial OK, default), `vitl` Large (non-commercial) |
| depth | `--depth-chunk 160` | Video Depth Anything: frames per chunk, lower = less memory (default 320) |
| depth | `--depth-model depth-anything/DA3MONO-LARGE` | Depth Anything 3 model (default DA3-BASE) |
| depth | `--near-white` | near = white instead of far = white (or `"depth_white": "near"` in settings.json) |
| flow | `--flow-engine waft` | use WAFT instead of MEMFOF (default set in settings.json) |
| flow | `--memfof-model spring` | MEMFOF tuned for fine detail (default `tskh`, real-world footage) |
| flow | `--flow-iters 12` | more refinement per frame: slower, slightly better (MEMFOF default 8, WAFT 5) |
| flow | `--backward` | motion to the previous frame instead of the next |
| flow | `--flow-format png16` | MEMFOF: 16-bit PNGs only, no EXRs (`exr` for EXRs only; default both) |
| flow | `--flow-range 40` | MEMFOF PNGs: fixed top speed in pixels per frame, so several clips match |
| flow | `--half-res` | WAFT only: faster and lighter on the graphics card |
| mattes | `--threshold 0.4` | lower finds more, higher is stricter (default 0.5) |
| mattes | `--max-objects 30` | for crowds (default 16) |
| mattes | `--sam-chunk 100` | less memory (default 200) |

### Stopping, carrying on, redoing
| Do this | And |
|---|---|
| run the same command again | it carries on from where it stopped |
| `--redo depth` | throws away just that tool's results and does them again (use after changing its options) |
| `--fresh` | throws away everything for this clip, frames included |
| `--delete-frames` | deletes the extracted frames at the end, if everything finished |

## Good to know
- **Mattes:** simple words work best (person, car, dog, hand, bottle). Every match for a word
  goes in the same matte. Edges are hard - soften them in your comp.
- **MEMFOF vs WAFT:** MEMFOF looks at three frames at once (before, now, after), which gives
  steadier motion and cleaner edges around small moving things. Its previews use one colour
  scale for the whole clip (WAFT's are scaled per frame), so they don't flicker or hide detail -
  the scale is saved in `Optical\_preview_scale.txt`. Changing flow engine or settings on a shot
  needs `--redo flow`.
- **Flow PNGs (MEMFOF default):** 16-bit, R = x movement, G = y movement, B = 0. Mid-grey (0.5)
  means still; 0 and 1 mean the clip's top speed in each direction. The top speed is in
  `Optical\_flow_range.txt` - to get pixels per frame back: `(value * 2 - 1) * range`.
  One range covers the whole clip, so a value means the same speed on every frame. It's picked
  automatically, or set it yourself with `--flow-range` so several clips use the same scale.
  The 32-bit EXRs are kept alongside, in pixels. To save disk space, `--flow-format png16`
  deletes them once the PNGs are done.
- **Flow direction:** in both PNG and EXR, G is positive moving *down*. If things move the wrong
  way vertically in your comp or in Houdini, flip G.
- **Depth** is far = white, near = black. To switch a finished shot either way, just run depth
  again with `--near-white` or `--far-white`: the PNGs are flipped in place in seconds, with no
  need to work the depth out again.
- **Depth** is normalised across the whole clip, so values don't jump between frames.
  The raw range used is saved in `Depth\_depth_range.txt`.
- **Video Depth Anything vs Depth Anything 3:** Video Depth Anything looks at runs of frames
  together, so its depth is steadier over time (less flicker). Long clips go through it in
  overlapping chunks that are matched and cross-faded, so there's no jump where they meet.
  Changing engine or model on a shot that already has depth needs `--redo depth`.
- **Shots made with an older version** (files like `depth_0001.png`) are spotted automatically:
  their old results are moved into `_old_naming` (nothing is deleted) and redone with the new names.

## If something goes wrong
| You see | Do this |
|---|---|
| `The depth tool isn't ready` (or flow / mattes) | run `python process_clip.py --check` and fix what it lists |
| `Restarting won't fix this one` | read the message just above it - usually a missing file, or a setting changed part-way (then use `--redo`) |
| a tool says FAILED in the summary | the others still ran. Scroll up or open `log.txt`, then run the same command again |
| keeps restarting | try `--depth-chunk 160` (depth) or `--sam-chunk 100` (mattes), and close other apps using the graphics card |
| `settings.json has a typo` | use forward slashes in paths: `C:/ML/waft` |

## Versions and updating
The version is in the `VERSION` file, shown by `python process_clip.py --version`, and written
at the top of every shot's `log.txt`, so you can always tell which version made a shot.
What changed in each version is in `CHANGELOG.md`.

To update, unzip the new version over the old folder. Your `settings.json` is never in the
zip, so your settings are kept.

## For people looking after this toolkit
- `make_release.py` makes a versioned zip (`ml-video-tools_v0.2.0.zip`) to share. Bump `VERSION`
  and add a line to `CHANGELOG.md` first.
- `lock_versions.py` writes the exact package versions from a working PC into `requirements\`.
  Re-run it after updating a tool that still works.
- `lib\babysitter.py` is the restart-on-crash logic, `lib\common.py` the shared bits.
- `tools\depth_vda.py`, `tools\depth_da3.py`, `tools\flow_memfof.py`, `tools\flow_waft.py`, `tools\mattes.py` are the tools. They're run by
  `process_clip.py` with each tool's own Python.
