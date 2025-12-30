import os
import numpy as np
import matplotlib.pyplot as plt
from dqmc_util import analyze_hub


def dir_get_all(directory):
    data = []
    for subdir in sorted(os.listdir(directory)):
        path = os.path.join(directory, subdir)
        if os.path.isdir(path):
            data.append(analyze_hub.get(path + "/", "sign", "den"))
    return data


def main():
    directory = "data/"
    print("processing", directory)
    data = dir_get_all(directory)
    data_table = np.array([(d["mu"], d["beta"],
                   d["sign"][0], d["sign"][1],
                   d["den"][0], d["den"][1]) for d in data])

    # <n> vs mu
    for beta in np.unique(data_table[:, 1]):
        sub = data_table[data_table[:, 1] == beta]
        order = np.argsort(sub[:, 0])
        sub = sub[order]
        plt.errorbar(
            sub[:, 0], sub[:, 4],
            yerr=sub[:, 5],
            fmt='o-',
            label=f"$T/t={1/beta}$"
        )
    plt.xlabel(r"$\mu/t$")
    plt.ylabel(r"$\langle n \rangle$")
    plt.legend()
    plt.savefig("n_vs_mu.png", dpi=300)
    plt.clf()

    # <sign> vs <n>
    for beta in np.unique(data_table[:, 1]):
        sub = data_table[data_table[:, 1] == beta]
        order = np.argsort(sub[:, 4])
        sub = sub[order]
        plt.errorbar(
            sub[:, 4], sub[:, 2], xerr=sub[:, 5], yerr=sub[:, 3],
            fmt='o-',
            label=f"$T/t={1/beta}$"
        )
    plt.xlabel(r"$\langle n \rangle$")
    plt.ylabel(r"$\langle$sign$\rangle$")
    plt.legend()
    plt.savefig("sign_vs_n.png", dpi=300)
    print("plots saved: n_vs_mu.png, sign_vs_n.png")

if __name__ == "__main__":
    main()
