# Third-party licences

This toolkit's own code (the scripts in this folder) is MIT licensed - see `LICENSE`.

It does **not** include any of the AI models or their code. Each tool is downloaded
separately, onto your own machine, from its makers, and each has its **own licence**.
Those licences are what decide whether you can use the results in paid work.

The short version:

| Tool | What it's used for | Paid / client work? |
|---|---|---|
| Video Depth Anything - Small (`vits`) | depth | Yes (Apache 2.0) |
| Video Depth Anything - Base (`vitb`), Large (`vitl`) | depth | **No - non-commercial only** (CC BY-NC 4.0) |
| Depth Anything 3 - Small, Base, Mono-Large, Metric-Large | depth | Yes (Apache 2.0) |
| Depth Anything 3 - Large, Giant, Nested | depth | **No - non-commercial only** (CC BY-NC 4.0) |
| MEMFOF | optical flow | Yes (BSD 3-Clause) |
| WAFT | optical flow | Yes (BSD 3-Clause) |
| Depth Anything V2 Small (used inside WAFT) | optical flow | Yes (Apache 2.0) |
| SAM 3 / SAM 3.1 | mattes | Yes, under Meta's SAM Licence (conditions below) |

This is a plain-English summary to point you in the right direction, not legal advice.
Always check the licence that comes with each tool, as they can change.

---

## Video Depth Anything (ByteDance)
- Code: https://github.com/DepthAnything/Video-Depth-Anything - Apache 2.0
- Models (from Hugging Face, `depth-anything/...`):
  - **Apache 2.0 - fine for commercial use:** `Video-Depth-Anything-Small` (`vits`)
  - **CC BY-NC 4.0 - non-commercial only:** `Video-Depth-Anything-Base` (`vitb`),
    `Video-Depth-Anything-Large` (`vitl`)
- **The toolkit defaults to Small (`vits`)**, which is fine for client work. Large (`vitl`) has
  more detail - use it for tests and personal projects with `--vda-encoder vitl`.
  The toolkit prints a reminder whenever Base or Large is used.

## Depth Anything 3 (ByteDance Seed)
- Code: https://github.com/ByteDance-Seed/Depth-Anything-3 - Apache 2.0
- Models (from Hugging Face, `depth-anything/...`):
  - **Apache 2.0 - fine for commercial use:** `DA3-SMALL`, `DA3-BASE`, `DA3MONO-LARGE`, `DA3METRIC-LARGE`
  - **CC BY-NC 4.0 - non-commercial only:** `DA3-LARGE`, `DA3-LARGE-1.1`, `DA3-GIANT`, `DA3-GIANT-1.1`,
    `DA3NESTED-GIANT-LARGE`, `DA3NESTED-GIANT-LARGE-1.1`
- When you choose Depth Anything 3 (`--depth-engine da3`) it uses `DA3-BASE` by default, which is fine for client work.
  Only switch to a Large / Giant / Nested model for tests or personal projects.

## MEMFOF (MSU Video Group)
- Code: https://github.com/msu-video-group/memfof - BSD 3-Clause
- Models: https://huggingface.co/egorchistov (`optical-flow-MEMFOF-...`), published with the
  same BSD 3-Clause licence. They download automatically - no login needed.
- Paper: Bargatin, Chistov, Yakovenko, Vatolin, "MEMFOF: High-Resolution Training for
  Memory-Efficient Multi-Frame Optical Flow Estimation", ICCV 2025
- MEMFOF's encoder starts from torchvision's ResNet-34 (BSD-style licence).

## WAFT (Princeton Vision & Learning Lab)
- Code and weights: https://github.com/princeton-vl/WAFT - BSD 3-Clause
- Paper: Wang & Deng, "WAFT: Warping-Alone Field Transforms for Optical Flow", ICLR 2026
- WAFT uses **Depth Anything V2 Small** inside it:
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small - Apache 2.0
  (the larger Depth Anything V2 models are non-commercial, but WAFT only uses Small)

## SAM 3 / SAM 3.1 (Meta)
- Code: https://github.com/facebookresearch/sam3
- Models: https://huggingface.co/facebook/sam3.1 - **gated**: each person must request access
  and accept the licence themselves, then log in with their own Hugging Face token.
- Licence: **SAM License** (Meta's own licence, not open source in the usual sense). In short:
  - commercial use is allowed, royalty-free
  - if you publish research that used it, you must say you used SAM
  - you must follow trade-control / sanctions law, and it can't be used for military,
    weapons, nuclear or espionage purposes
  - if you pass SAM itself (or anything built from it) on to others, the SAM License has to go with it
  - Full text: the `LICENSE` file in the SAM 3 repo

## Other things the toolkit relies on
- **ffmpeg** (frame extraction) - you install it yourself; LGPL/GPL depending on the build
- **PyTorch** - BSD-style licence
- **OpenEXR** (writing EXR files) - BSD 3-Clause
- **OpenCV** - Apache 2.0
- **NumPy** - BSD 3-Clause

## If you publish work made with these tools
It's good practice (and for SAM in research, required) to credit the models used, e.g.
"Depth: Video Depth Anything (ByteDance). Optical flow: MEMFOF (MSU Video Group). Mattes: SAM 3.1 (Meta)."
