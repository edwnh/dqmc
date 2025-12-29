"""CLI entry point for dqmc-util."""
import sys
import importlib


def main():
    dispatch = {
        "gen": ("gen_1band_hub", "main"),
        "gen-1band-hub": ("gen_1band_hub", "main"),
        "info": ("info", "main"),
        "summary": ("summary", "main"),
        "get-mu": ("get_mu", "main"),
        "print-n": ("print_n", "main"),
        "enqueue": ("queue", "enqueue_main"),
        "queue-status": ("queue", "status_main"),
        "dequeue": ("queue", "dequeue_main"),
        "worker": ("worker", "main"),
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: dqmc-util <command> [args...]")
        print("\nCommands:")
        print("  gen, gen-1band-hub  Generate HDF5 simulation files")
        print("  info                Print HDF5 file metadata")
        print("  summary             Quick simulation status")
        print("  get-mu              Find chemical potential for target filling")
        print("  print-n             Print sign and density")
        print("\nQueue commands (sharded directory queue):")
        print("  enqueue             Add .h5 files to queue")
        print("  queue-status        Show queue counts")
        print("  dequeue             Remove files from todo queue")
        print("  worker              Run worker loop processing queue")
        return

    cmd = sys.argv[1]
    if cmd not in dispatch:
        print(f"unknown command: {cmd}")
        print(f"available: {', '.join(sorted(dispatch.keys()))}")
        sys.exit(1)

    # Remove the subcommand from argv so the module sees correct args
    sys.argv = [f"dqmc-util {cmd}"] + sys.argv[2:]

    module_name, func_name = dispatch[cmd]

    try:
        module = importlib.import_module(f".{module_name}", package=__package__)
        func = getattr(module, func_name)
        func()
    except (ImportError, AttributeError) as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
