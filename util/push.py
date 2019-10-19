import os
import sys
from glob import glob

def main(argv):
    if len(argv) < 3:
        print("usage: {} stackfile a.h5 b.h5 ...".format(argv[0]))
        return
    stack = argv[1]
    os.symlink(stack, stack + "~")
    with open(stack, "a") as f:
        for x in argv[2:]:
            files = sorted(glob(x))
            if len(files) == 0:
                print("No files matching:"+x)
            else:
                for ff in files:
                    print(os.path.abspath(ff), file=f)
    os.remove(stack + "~")

if __name__ == "__main__":
    main(sys.argv)
