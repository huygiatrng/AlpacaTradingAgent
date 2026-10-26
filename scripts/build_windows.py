"""Reproducible entry point for the Windows desktop build."""

from pathlib import Path
import argparse
import subprocess
import sys


def main():
    if sys.platform != "win32":
        raise SystemExit("Build the Windows executable on Windows.")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Discard cached build analysis")
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", *(["--clean"] if args.clean else []),
                    "--distpath", str(root / "dist"), "--workpath", str(root / "build"),
                    str(root / "packaging" / "windows.spec")], cwd=root, check=True)
    print(f"Ready: {root / 'dist' / 'AlpacaTradingAgent.exe'}")


if __name__ == "__main__":
    main()
