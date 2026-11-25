# Installing the tools

Each tool gets its own folder and its own Python environment, so they can't break each other.
You only need the tools you'll use. Allow an hour or two in total, mostly downloading.

Use **Command Prompt** (not PowerShell). Type each command as **one line**.
Every time you see `(.venv)` mentioned, check your prompt starts with it before carrying on.

The examples put the toolkit in `C:\ML\ml-video-tools`. If yours is elsewhere, change that part.

## Before you start
```
winget install Python.Python.3.12
winget install Git.Git
winget install Gyan.FFmpeg
```
Close and reopen Command Prompt afterwards so it finds them.
Also update your NVIDIA driver (NVIDIA app or GeForce Experience).

---

## 1. Depth - Video Depth Anything (the default)
```
git clone https://github.com/DepthAnything/Video-Depth-Anything.git C:\ML\vda
cd C:\ML\vda
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install -r C:\ML\ml-video-tools\requirements\depth_vda.txt
```
**Don't** use Video Depth Anything's own `requirements.txt` - it pins an old PyTorch that
doesn't support newer graphics cards, and a numpy with no build for Python 3.12.

Then download the models (one line each):
```
mkdir checkpoints
curl -L -o checkpoints\video_depth_anything_vits.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
curl -L -o checkpoints\video_depth_anything_vitl.pth https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
```
Small (`vits`) is the default and fine for client work. Large (`vitl`, about 1.5GB) has more
detail but is **non-commercial only**, and needs more graphics memory - the Large download is optional. Which one is used is set by `vda_encoder` in `settings.json`.

### Optional: Depth Anything 3 (use with `--depth-engine da3`)
```
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git C:\ML\da3\Depth-Anything-3
cd C:\ML\da3\Depth-Anything-3
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install -r C:\ML\ml-video-tools\requirements\depth_da3.txt
pip install -e .
pip install -r C:\ML\ml-video-tools\requirements\depth_da3.txt
```
(The requirements go in twice on purpose: the second time puts back any version the
`-e .` step changed.) Its model downloads by itself the first time you use it.

## 2. Flow - MEMFOF (the default)
```
git clone https://github.com/msu-video-group/memfof.git C:\ML\memfof
cd C:\ML\memfof
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install -r C:\ML\ml-video-tools\requirements\flow_memfof.txt
pip install -e .
```
Its model downloads by itself the first time you run flow. No login needed.

### Optional: WAFT (use with `--flow-engine waft`)
```
git clone https://github.com/princeton-vl/WAFT.git C:\ML\waft
cd C:\ML\waft
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install -r C:\ML\ml-video-tools\requirements\flow_waft.txt
```
Then download two model files in your browser:

1. **The WAFT model** -
   https://drive.google.com/file/d/1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC/view
   Make a folder `C:\ML\waft\ckpts`, save it there and rename it to **`waft_a1.pth`**.
   (If Google Drive says it can't scan it for viruses, click Download anyway.)
2. **Depth Anything V2 Small** (WAFT uses it inside) -
   https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
   Make a folder `C:\ML\waft\depth-anything-ckpts` and save it there **without renaming**.

## 3. Mattes - SAM 3.1
First, in your browser:
1. Log in at huggingface.co (make an account if you need to)
2. Go to https://huggingface.co/facebook/sam3.1, fill in the form and accept the licence
3. Wait for the approval email
4. Make a **Read** token at https://huggingface.co/settings/tokens and keep it safe

Then:
```
git clone https://github.com/facebookresearch/sam3.git C:\ML\sam
cd C:\ML\sam
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install -r C:\ML\ml-video-tools\requirements\mattes.txt
pip install -e .
pip install -r C:\ML\ml-video-tools\requirements\mattes.txt
hf auth login
```
Paste your token when it asks (nothing shows as you paste - that's normal), press Enter,
and say `n` if it asks about git credentials.

The model (about 3GB) downloads by itself the first time you run mattes.

---

## Last step
Check `settings.json` points at the folders you used (it's made from `settings.example.json`
the first time you run anything), then:
```
cd C:\ML\ml-video-tools
python process_clip.py --check
```

## If an install step goes wrong
| You see | Do this |
|---|---|
| `requires numpy<2` or numpy errors | `pip install "numpy<2"` |
| `No module named 'xyz'` | `pip install xyz "numpy<2"` (but **not** `cc_torch` or `flash_attn` - those are Linux-only extras) |
| `CUDA error: no kernel image is available` | `pip uninstall -y torch torchvision xformers` then install the requirements file again |
| Running `activate` opens Notepad | you're in Command Prompt but typed the PowerShell version - use `.venv\Scripts\activate.bat` |
| `--index-url option requires 1 argument` | the line got split when pasting - make sure it's all on one line |
