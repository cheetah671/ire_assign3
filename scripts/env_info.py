"""Record machine specs and library versions to results/environment.json."""
import json
import os
import platform
import subprocess
import sys

import faiss
import h5py
import matplotlib
import numpy
import pandas
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ann.results import RESULTS_DIR  # noqa: E402


def cpu_name():
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"], text=True)
            return out.strip()
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor()


info = {
    "cpu": cpu_name(),
    "logical_cores": os.cpu_count(),
    "physical_cores": psutil.cpu_count(logical=False),
    "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    "os": platform.platform(),
    "python": platform.python_version(),
    "faiss": faiss.__version__,
    "faiss_simd": faiss.get_compile_options(),
    "numpy": numpy.__version__,
    "h5py": h5py.__version__,
    "pandas": pandas.__version__,
    "matplotlib": matplotlib.__version__,
}
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / "environment.json", "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
