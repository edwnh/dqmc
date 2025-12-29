"""CLI entry point for dqmc-util."""
import sys


def main():
    commands = {
        "gen": "gen_1band_hub",
        "gen_1band_hub": "gen_1band_hub",
        "info": "info",
        "summary": "summary",
        "get-mu": "get_mu",
        "get_mu": "get_mu",
        "print-n": "print_n",
        "print_n": "print_n",
        # Queue commands
        "enqueue": "enqueue",
        "worker": "worker",
        "queue-status": "queue_status",
        "dequeue": "dequeue",
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: dqmc-util <command> [args...]")
        print("\nCommands:")
        print("  gen, gen_1band_hub  Generate HDF5 simulation files")
        print("  info                Print HDF5 file metadata")
        print("  summary             Quick simulation status")
        print("  get-mu, get_mu      Find chemical potential for target filling")
        print("  print-n, print_n    Print sign and density")
        print("\nQueue commands (sharded directory queue):")
        print("  enqueue             Add .h5 files to queue")
        print("  worker              Run worker loop processing queue")
        print("  queue-status        Show queue counts")
        print("  dequeue             Remove files from todo queue")
        return

    cmd = sys.argv[1]
    if cmd not in commands:
        print(f"unknown command: {cmd}")
        print(f"available: {', '.join(sorted(set(commands.values())))}")
        sys.exit(1)

    # Remove the subcommand from argv so the module sees correct args
    sys.argv = [f"dqmc-util {cmd}"] + sys.argv[2:]

    module_name = commands[cmd]
    if module_name == "gen_1band_hub":
        from . import gen_1band_hub
        gen_1band_hub.main()
    elif module_name == "info":
        from . import info
        info.main()
    elif module_name == "summary":
        from . import summary
        summary.main()
    elif module_name == "get_mu":
        from . import get_mu
        get_mu.main()
    elif module_name == "print_n":
        from . import print_n
        print_n.main()
    elif module_name == "enqueue":
        from . import queue
        queue.enqueue_main()
    elif module_name == "worker":
        from . import worker
        worker.main()
    elif module_name == "queue_status":
        from . import queue
        queue.status_main()
    elif module_name == "dequeue":
        from . import queue
        queue.dequeue_main()


if __name__ == "__main__":
    main()
