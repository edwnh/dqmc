import sys
from glob import glob

from . import analyze_hub


def main(argv=None):
    if argv is None:
        argv = sys.argv
    # wildcard path expansion on Windows
    for path in argv[1:]:
        paths = sorted(glob(path))
        if len(paths) == 0:
            print("No paths matching:", path)
            continue
        for p in paths:
            data = analyze_hub.get(p, "sign", "den")
            print(f"<sign> = {data['sign']}")
            print(f"<n> = {data['den']}")


if __name__ == "__main__":
    main()
