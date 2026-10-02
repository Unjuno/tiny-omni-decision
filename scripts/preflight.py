from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")

    try:
        import torch
    except ImportError:
        print("torch=not-installed")
        return

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name}")
        print(f"vram_gib={props.total_memory / (1024**3):.2f}")


if __name__ == "__main__":
    main()
