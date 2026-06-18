"""Refresh dashboard data and serve the interactive demo locally."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    data_generator = DASHBOARD_DIR / "generate_dashboard_data.py"
    data_bundle = DASHBOARD_DIR / "data.js"
    if data_generator.exists():
        subprocess.run([sys.executable, str(data_generator)], check=True)
    elif not data_bundle.exists():
        raise FileNotFoundError(
            "dashboard/data.js is missing and the local data generator is not available."
        )
    else:
        print("Using committed dashboard/data.js bundle.")
    os.chdir(DASHBOARD_DIR)
    server = ThreadingHTTPServer(("localhost", args.port), SimpleHTTPRequestHandler)
    print(f"Dashboard ready at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
