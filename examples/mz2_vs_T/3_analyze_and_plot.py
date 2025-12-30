import os
import numpy as np
import matplotlib.pyplot as plt
from dqmc_util import analyze_hub


def dir_get_all(directory):
    data = []
    for subdir in sorted(os.listdir(directory)):
        path = os.path.join(directory, subdir)
        if os.path.isdir(path):
            data.append(analyze_hub.get(path + "/", "zzr")) # get <m_z^2> from spin-z correlator
    return data


def main():
    directory = "data/"
    print("processing", directory)
    data = dir_get_all(directory)
    data_table = np.array([(d["U"], d["beta"],
                            4*d["zzr"][0, 0, 0], # 4x because Sz = 1/2 (n_up - n_down)
                            4*d["zzr"][1, 0, 0])
                           for d in data])
    for U in np.unique(data_table[:, 0]):
        sub = data_table[data_table[:, 0] == U]
        order = np.argsort(sub[:, 1])
        sub = sub[order]
        plt.errorbar(
            1/sub[:, 1], sub[:, 2],
            yerr=sub[:, 3],
            fmt='o-',
            label=f"$U/t={U}$"
        )
    plt.xlabel("$T/t$")
    plt.ylabel(r"$\langle m_z^2 \rangle$")
    plt.xscale('log')
    plt.legend()
    plt.savefig("mz2_vs_T.png", dpi=300)
    print("plot saved to mz2_vs_T.png")


if __name__ == "__main__":
    main()
