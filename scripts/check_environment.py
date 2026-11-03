#!/usr/bin/env python3
"""PhytoTrace AI — Environment and Hardware Verification Script."""

import platform
import sys
import psutil
import torch

from phytotrace.device import get_device, run_device_smoke_test


def main():
    print("=" * 60)
    print("🌿 PHYTOTRACE AI — LOCAL ENVIRONMENT INSPECTION")
    print("=" * 60)

    # 1. OS & Architecture
    print(f"Operating System : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform / Arch  : {platform.machine()} ({platform.processor()})")
    print(f"Python Version   : {platform.python_version()} ({sys.executable})")

    # 2. RAM and Storage
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024 ** 3)
    avail_ram_gb = mem.available / (1024 ** 3)
    print(f"Total Unified RAM: {total_ram_gb:.2f} GB (Available: {avail_ram_gb:.2f} GB)")

    # 3. PyTorch & Metal Performance Shaders
    print("-" * 60)
    print(f"PyTorch Version  : {torch.__version__}")
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    print(f"Apple MPS Built  : {mps_built}")
    print(f"Apple MPS Avail  : {mps_avail}")

    device = get_device("auto")
    print(f"Active Device    : {device.type.upper()}")

    # 4. Forward / Backward Pass Smoke Test
    print("-" * 60)
    print("Running forward/backward tensor pass smoke test...")
    smoke = run_device_smoke_test()
    if smoke["test_forward_backward_pass"]:
        print("✅ SUCCESS: Forward and backward gradient pass verified on active device.")
    else:
        print(f"❌ ERROR: Smoke test failed: {smoke.get('error')}")

    print("=" * 60)


if __name__ == "__main__":
    main()

