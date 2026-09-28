from __future__ import annotations
import json, platform, subprocess, sys
from pathlib import Path

mods = ["torch", "torchvision", "numpy", "pandas", "scipy", "statsmodels", "PIL", "yaml", "matplotlib"]
versions = {}
for name in mods:
    try:
        m = __import__(name)
        versions[name] = getattr(m, "__version__", "unknown")
    except Exception as exc:
        versions[name] = f"unavailable: {exc}"
try:
    import torch
    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception:
    cuda = {}
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = None
payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": versions,
    "cuda": cuda,
    "git_commit": commit,
}
out = Path("results/paper_suite/environment.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
