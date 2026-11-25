# Changelog

The version number lives in the `VERSION` file. To release a new one:
1. change the number in `VERSION`
2. add a section at the top of this file saying what changed
3. run `python make_release.py`, which makes `ml-video-tools_v<version>.zip`

Numbering: the last number for fixes (0.2.0 -> 0.2.1), the middle number for new
features or anything that changes file names or options (0.2.1 -> 0.3.0).

## 0.7.0 - 2026-10-14
- **New folder names** inside each `<clip>_shot`: `Color` (the extracted frames, still kept),
  `Depth`, `Optical` and `Segm`. File names inside are unchanged.
- Older shots are renamed automatically (`frames` -> `Color`, `depth` -> `Depth`, `flow` -> `Optical`,
  `mattes` -> `Segm`) the next time you run anything on them. Nothing is worked out again.
  Other folders you've made yourself (e.g. `depth_da3`) are left alone.
- The tool names in commands stay the same: `--only depth`, `--redo flow`, `--skip mattes`.

## 0.6.1 - 2026-10-14
- Video Depth Anything now defaults to Small (`vits`): lighter on the graphics card and fine for
  client work. `--vda-encoder vitl` for Large.
- Note: if your `settings.json` says `"vda_encoder": "vitl"`, that still wins - change it to `"vits"`.

## 0.6.0 - 2026-10-14
- **Depth is now far = white, near = black** by default, for both engines. `--near-white` (or
  `"depth_white": "near"` in `settings.json`) gives the old way round.
- Running depth again on a finished shot with the other setting flips its PNGs in place, in
  seconds - no need to work the depth out again. Shots made before 0.6.0 were near = white, and
  are flipped the first time depth is run on them with 0.6.0.
- Fix: changing near/far part-way through a run could mix the two in one shot. The setting is now
  recorded in `depth\_white.txt`, and frames already written are flipped to match.

## 0.5.1 - 2026-10-14
- MEMFOF now keeps the EXRs by default, alongside the 16-bit PNGs (`flow_format` default is `both`).
  `--flow-format png16` gives PNGs only, as 0.5.0 did.
- Shots run with 0.5.0 have already lost their EXRs: add `--redo flow` to get them back.

## 0.5.0 - 2026-10-14
- **MEMFOF flow now comes out as 16-bit PNGs** (`flow\png\`) instead of EXRs. R = x, G = y,
  B = 0, with 0.5 = still and 0 / 1 = the clip's top speed. The top speed is saved in
  `flow\_flow_range.txt`, with the sum to turn values back into pixels.
- One range for the whole clip, picked automatically, or fixed with `--flow-range` so several
  clips share the same scale.
- `--flow-format exr` brings back the 32-bit float EXRs; `--flow-format both` writes the two.
  Also `"flow_format"` in `settings.json`.
- EXRs are still used as working files while MEMFOF runs, so crashes resume as before; they're
  deleted once the PNGs are finished. Previews can be rebuilt from the PNGs if needed.
- WAFT is unchanged: it always writes EXRs.
- Shots already made with 0.4.0 don't need `--redo flow`: running flow again turns their EXRs into PNGs.

## 0.4.0 - 2026-10-09
- **MEMFOF is now the default flow engine.** It looks at three frames at once, giving steadier
  motion and cleaner edges around small moving things (tested on a fish shoal against WAFT and
  RAFT). WAFT is still there: add `--flow-engine waft`, or set `"flow_engine": "waft"` in `settings.json`.
- MEMFOF goes frame by frame, so memory stays flat on long clips. Resumable, like the other tools.
- MEMFOF previews use one colour scale for the whole clip, so they don't flicker or hide detail.
- New options: `--flow-engine`, `--memfof-model` (tskh / spring / sintel / kitti), `--flow-iters`
  (works with both engines).
- `settings.json` now has `flow_memfof` and `flow_waft` install folders. Old settings files that
  say `flow` are read as `flow_waft` automatically.
- `--check` shows the default and optional engine for both depth and flow.
- A shot whose flow was made with a different engine or settings stops and asks for `--redo flow`.
- `requirements/flow.txt` is now `requirements/flow_waft.txt`; new `requirements/flow_memfof.txt`.

## 0.3.0 - 2026-10-09
- **Video Depth Anything is now the default depth engine.** Depth Anything 3 is still there:
  add `--depth-engine da3`, or set `"depth_engine": "da3"` in `settings.json`.
- Long clips go through Video Depth Anything in overlapping chunks (`--depth-chunk`, default 320)
  that are matched to each other and cross-faded, so memory stays flat and there's no jump
  where chunks meet. Resumable, like the other tools.
- New options: `--depth-engine`, `--vda-encoder` (vits / vitb / vitl), `--depth-chunk`.
- `settings.json` now has `depth_vda` and `depth_da3` install folders. Old settings files that
  say `depth` are read as `depth_da3` automatically.
- `--check` shows the default depth engine, and the other one as optional.
- A shot whose depth was made by a different engine or model stops and asks for `--redo depth`,
  rather than mixing the two.
- New `requirements/depth_vda.txt` (with matplotlib below 3.9 - Video Depth Anything needs it).
  `requirements/depth.txt` is now `requirements/depth_da3.txt`.

## 0.2.0 - 2026-10-08
- Every output file is named after the clip, then the process, with the frame number last:
  `clip_depth_0001.png`, `clip_optical-flow_0001.exr`, `clip_optical-flow_preview_0001.png`,
  `clip_segmentation_<word>_0001.png`. Extracted frames are `clip_0001.png`.
- Shots made with 0.1.0 are spotted automatically: their old results are moved into
  `_old_naming` inside the shot (nothing deleted) and redone with the new names.
- Version number shown by `--version`, `--check` and at the top of each shot's `log.txt`.
- Ships `settings.example.json` instead of `settings.json`, so unzipping an update never
  overwrites your settings. `settings.json` is made from the example on first run.
- Added `make_release.py` and this changelog.

## 0.1.0 - 2026-10-08
- First tidy version: one `process_clip.py` runs depth (Depth Anything 3), flow (WAFT)
  and mattes (SAM 3.1), extracting frames once.
- Install folders in `settings.json`, checked with `python process_clip.py --check`.
- One shared babysitter (`lib/babysitter.py`): restarts crashed or frozen tools, resumes
  where they left off, doesn't retry deliberate stops.
- Starter `requirements/` files with the known fixes, plus `lock_versions.py` to record exact
  versions from a working PC.
- MIT licence for the toolkit code; `THIRD_PARTY_LICENCES.md` for the models.
