"""Reproducible entry point for the Windows desktop build."""

from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys


def main():
    if sys.platform != "win32":
        raise SystemExit("Build the Windows executable on Windows.")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Discard cached build analysis")
    parser.add_argument("--portable", action="store_true", help="Build a single-file executable (slower startup)")
    args = parser.parse_args()
    staging = root / "build" / "desktop-dist"
    destination = root / "dist" / "portable" if args.portable else root / "dist"
    env = dict(os.environ, TRADINGAGENTS_BUILD_ONEFILE="1" if args.portable else "0")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", *(["--clean"] if args.clean else []),
                    "--distpath", str(staging), "--workpath", str(root / "build"),
                    str(root / "packaging" / "windows.spec")], cwd=root, check=True, env=env)
    destination.mkdir(parents=True, exist_ok=True)
    if args.portable:
        shutil.copy2(staging / "AlpacaTradingAgent.exe", destination / "AlpacaTradingAgent.exe")
    else:
        shutil.copytree(staging / "AlpacaTradingAgent", destination, dirs_exist_ok=True)
    print(f"Ready: {destination / 'AlpacaTradingAgent.exe'}")


if __name__ == "__main__":
    main()
