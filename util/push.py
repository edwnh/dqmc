import os
import sys

def main(argv):
    if len(argv) < 3:
        print("usage: {} stackfile a.h5 b.h5 ...".format(argv[0]))
        return
    stack = argv[1]
    os.symlink(stack, stack + "~")
    with open(stack, "a") as f:
        for x in argv[2:]:
            print(os.path.abspath(x), file=f)
    os.remove(stack + "~")

if __name__ == "__main__":
    main(sys.argv)
