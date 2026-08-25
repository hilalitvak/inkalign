# inkalign: depth/orientation sweep on a free Kaggle GPU.
#
# Push as a script kernel with GPU + internet enabled:
#   kaggle kernels push -p <folder with this file + kernel-metadata.json>
# or paste into a Kaggle notebook. Works on both T4 and P100 sessions
# (validated 25 Aug 2026 on a P100; full run ~6 minutes wall clock).
#
# Same two-interpreter layout as colab_sweep.ipynb: inkalign in the base env
# (zarr 2.x), villa's vesuvius in a uv-managed Python 3.14 venv. The extra
# packages on the venv line are deps infer.py needs but the lean vesuvius
# install does not declare (tifffile, imagecodecs, zarr, nest-asyncio, aiohttp,
# requests, pyyaml, einops, timm).
import json
import os
import subprocess
import sys

# torch.compile requires Triton (CUDA capability >= 7.0); Kaggle sometimes
# assigns a P100 (6.0). Force eager mode so inference runs on any GPU.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

def run(cmd, **kw):
    shown = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n===> {shown}", flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)

# --- 1. environment -------------------------------------------------------
# imagecodecs in the base env too: inkalign reads back the LZW-compressed
# TIFFs that infer.py writes.
run(f"{sys.executable} -m pip install -q "
    f"'git+https://github.com/hilalitvak/inkalign' uv huggingface_hub imagecodecs")
run("uv venv /tmp/venv --python 3.14")
VENV_PY = "/tmp/venv/bin/python"
# cu126 wheels keep sm_60 (P100) kernels that the default cu130 build drops;
# install torch+torchvision first (same index, matched versions) so the
# vesuvius resolution sees them already satisfied.
run(f"uv pip install -q -p {VENV_PY} torch==2.13.0 torchvision "
    f"--index-url https://download.pytorch.org/whl/cu126")
run(f"uv pip install -q -p {VENV_PY} "
    f"'git+https://github.com/ScrollPrize/villa#subdirectory=vesuvius' "
    f"numba tifffile imagecodecs zarr nest-asyncio aiohttp requests pyyaml "
    f"einops timm")
run(f'{VENV_PY} -c "import torch; print(\'CUDA available:\', torch.cuda.is_available())"')
run(f"{VENV_PY} -m vesuvius.ink_detection.inference.infer --help > /dev/null")
print("infer entry point OK", flush=True)

# --- 2. real-data ROI + CPU depth profile ---------------------------------
VOLUME = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/"
          "PHerc0139/segments/20250108000000-w025_2025010863/surface-volumes/"
          "9.362um-1.2m-113keV-volume-20250728140407.zarr")
run([sys.executable, "-m", "inkalign.cli", "extract-roi", VOLUME,
     "--size", "768", "--out", "roi.zarr"])
run([sys.executable, "-m", "inkalign.cli", "profile", "roi.zarr",
     "--outdir", "/kaggle/working/profile_out"])

# --- 3. official cross-scroll checkpoint ----------------------------------
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("scrollprize/ink_9um", "hybrid_3d2d-seed42/step-075000.pth")
print("checkpoint:", ckpt, flush=True)

# --- 3.5 self-healing probe ------------------------------------------------
# Run one real inference; auto-install any ModuleNotFoundError that still
# appears (guards against new undeclared deps as villa moves).
import re
PKG_FOR_MODULE = {"yaml": "pyyaml", "cv2": "opencv-python-headless",
                  "skimage": "scikit-image", "PIL": "pillow"}
probe = (f"{VENV_PY} -m vesuvius.ink_detection.inference.infer roi.zarr {ckpt} "
         f"/tmp/probe.tif --layer-start 7 --layer-end 21 --direction forward")
auto_installed = []
for attempt in range(15):
    print(f"\n===> probe attempt {attempt + 1}", flush=True)
    p = subprocess.run(probe, shell=True, capture_output=True, text=True)
    if p.returncode == 0:
        print("probe inference PASSED", flush=True)
        break
    m = re.search(r"No module named '([\w\.]+)'", p.stderr)
    if not m:
        print(p.stderr[-4000:], flush=True)
        raise SystemExit("probe failed with a non-module error — see stderr above")
    mod = m.group(1).split(".")[0]
    pkg = PKG_FOR_MODULE.get(mod, mod)
    print(f"missing module {mod!r} -> installing {pkg}", flush=True)
    run(f"uv pip install -q -p {VENV_PY} {pkg}")
    auto_installed.append(pkg)
else:
    raise SystemExit("probe still failing after 15 attempts")
print("auto-installed beyond declared deps:", auto_installed, flush=True)

# --- 4. the sweep ----------------------------------------------------------
TEMPLATE = (f"{VENV_PY} -m vesuvius.ink_detection.inference.infer "
            "{volume} {checkpoint} {output} "
            "--layer-start {layer_start} --layer-end {layer_end} "
            "--direction {direction}")
run([sys.executable, "-m", "inkalign.cli", "sweep", "roi.zarr", ckpt,
     "--um-per-px", "9.362", "--outdir", "/kaggle/working/sweep_out",
     "--infer-template", TEMPLATE])

print(json.dumps(
    json.load(open("/kaggle/working/sweep_out/sweep_summary.json")), indent=2))
print("\nDONE — outputs in /kaggle/working (profile_out, sweep_out)", flush=True)
