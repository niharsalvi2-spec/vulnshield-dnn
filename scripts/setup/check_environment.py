"""Runtime Environment Diagnostic & Verification Script for VulnShield-DNN.

Checks Python, PyTorch, CUDA GPU capabilities, and all required research libraries.
"""

import sys
import platform
import shutil
from pathlib import Path

def check_environment() -> bool:
    print("=" * 65)
    print("      VulnShield-DNN — Runtime Environment Diagnostics")
    print("=" * 65)
    all_passed = True

    # 1. Python Check
    py_ver = platform.python_version()
    py_major, py_minor = sys.version_info[:2]
    print(f"[*] Python Version: {py_ver} ({platform.architecture()[0]})")
    if (py_major, py_minor) < (3, 10):
        print(f"  [FAIL] Python 3.10+ required. Found {py_ver}")
        all_passed = False
    else:
        print("  [PASS] Python version compatible.")

    # 2. Operating System & Hardware
    print(f"\n[*] OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"[*] Processor: {platform.processor()}")
    
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        avail_ram = psutil.virtual_memory().available / (1024 ** 3)
        print(f"[*] System RAM: Total = {total_ram:.2f} GB, Available = {avail_ram:.2f} GB")
        print(f"[*] CPU Cores: Physical = {psutil.cpu_count(logical=False)}, Logical = {psutil.cpu_count(logical=True)}")
    except ImportError:
        print("  [WARN] psutil not installed. Skipping detailed RAM query.")

    # Disk Space Check
    repo_root = Path(__file__).resolve().parent.parent.parent
    total_disk, _, free_disk = shutil.disk_usage(repo_root)
    print(f"[*] Disk Space at Repo Root: Free = {free_disk / (1024**3):.2f} GB / Total = {total_disk / (1024**3):.2f} GB")

    # 3. Core Research Dependencies
    print("\n[*] Checking Core Machine Learning Dependencies:")
    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "TQDM"),
        ("yaml", "PyYAML"),
        ("sklearn", "Scikit-Learn")
    ]

    for pkg_name, display_name in required_packages:
        try:
            m = __import__(pkg_name)
            ver = getattr(m, "__version__", "Installed")
            print(f"  [PASS] {display_name:<15} : v{ver}")
        except ImportError:
            print(f"  [FAIL] {display_name:<15} : NOT INSTALLED")
            all_passed = False

    # 4. PyTorch & CUDA Diagnostics
    print("\n[*] PyTorch & GPU Hardware Diagnostics:")
    try:
        import torch
        print(f"  PyTorch Version : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"  CUDA Available  : {cuda_avail}")
        
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_ver = torch.version.cuda
            print(f"  Device 0        : {gpu_name} ({vram:.2f} GB VRAM)")
            print(f"  CUDA Runtime    : {cuda_ver}")
            
            # Allocation test
            try:
                t = torch.randn(100, 100, device="cuda")
                print("  GPU Memory Test : PASS (Allocated test tensor on CUDA device)")
            except Exception as e:
                print(f"  GPU Memory Test : FAIL ({e})")
                all_passed = False
        else:
            print("  [WARN] CUDA is not available. Execution will fall back to CPU (slower).")
    except Exception as e:
        print(f"  [FAIL] Error querying PyTorch: {e}")
        all_passed = False

    print("\n" + "=" * 65)
    if all_passed:
        print("      [SUMMARY] Runtime Environment Status: PASS")
    else:
        print("      [SUMMARY] Runtime Environment Status: FAIL")
    print("=" * 65)
    return all_passed

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
