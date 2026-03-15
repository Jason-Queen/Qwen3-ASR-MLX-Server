from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # Compatibility launcher: keep implementation in whisper_mlx_server.py
    # while exposing a clearer project-facing script name.
    sys.argv[0] = str(Path(__file__).name)
    from whisper_mlx_server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
