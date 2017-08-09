import sys
import time

import h5py
import numpy as np

from scipy.linalg.lapack import dgeqp3, dgeqrf, dormqr, dgetrf, dgetrs

np.seterr(over="ignore")


# http://xoroshiro.di.unimi.it/xorshift1024star.c
def rand_uint(rng):
    s0 = rng[rng[16]]
    p = (int(rng[16]) + 1) & 15
    rng[16] = p
    s1 = rng[p]
    s1 ^= s1 << np.uint64(31)
    rng[p] = s1 ^ s0 ^ (s1 >> np.uint64(11)) ^ (s0 >> np.uint64(30))
    return rng[p] * np.uint64(1181783497276652981)


# fisher-yates
def shuffle(rng, n):
    a = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # discard lowest 3 bits
        j = int((rand_uint(rng) >> np.uint64(3)) % np.uint64(i + 1))
        if i != j:
            a[i] = a[j]
        a[j] = i
    return a


def calcG(f, C):
    F = C.shape[0]
    order = (np.arange(F) + f) % F
    Q, jpvt, tau, work, info = dgeqp3(C[order[0]] if F % 2 == 1 else np.dot(C[order[1]], C[order[0]]))
    d = Q.diagonal().copy()
    d[d == 0.0] = 1.0
    T = ((np.triu(Q).T / d).T)[:, jpvt.argsort()]
    for l in range((1 if F % 2 == 1 else 2), F, 2):
        W, work, info = dormqr("R", "N", Q, tau, np.dot(C[order[l+1]], C[order[l]]), work.shape[0])
        W *= d
        jpvt = (W*W).sum(0).argsort()[::-1]
        Q, tau, work, info = dgeqrf(W[:, jpvt])
        d[...] = Q.diagonal()
        d[d == 0.0] = 1.0
        T = np.dot((np.triu(Q).T / d).T, T[jpvt, :])

    N = C.shape[1]
    invDb = np.zeros((N, N))
    for i in range(N):
        invDb[i, i] = 1.0 / d[i] if np.abs(d[i]) > 1.0 else 1.0
    invDbQT, work, info = dormqr("R", "T", Q, tau, invDb, work.shape[0])

    for i in range(N):
        if np.abs(d[i]) <= 1.0:
            T[i, :] *= d[i]

    T += invDbQT
    T_LU, piv, info = dgetrf(T)
    sign = 1
    for i in range(N):
        if (T_LU[i, i] < 0) ^ (piv[i] != i) ^ (invDb[i, i] < 0) ^ (tau[i] > 0):
            sign *= -1

    G, info = dgetrs(T_LU, piv, invDbQT)
    return G, sign


def update_shermor(N, delta, rng, hs, Gu, Gd):
    sign = 1
    c = np.empty(N)
    for i in shuffle(rng, N):
        delu = delta[hs[i], i]
        deld = delta[int(not hs[i]), i]
        pu = 1. + (1. - Gu[i, i]) * delu
        pd = 1. + (1. - Gd[i, i]) * deld
        p = pu*pd
        if rand_uint(rng) < (np.abs(p) * 2**64):
            c[...] = Gu[:, i]
            c[i] -= 1.
            Gu += delu / pu * c[None].T * Gu[i, :]

            c[...] = Gd[:, i]
            c[i] -= 1.
            Gd += deld / pd * c[None].T * Gd[i, :]

            if p < 0:
                sign *= -1
            hs[i] = not hs[i]
    return sign


# note: delayed and submatrix updates may be slower in python due to overhead
# def update_delayed(N, delta, rng, hs, Gu, Gd, q):
    # sign = 1
    # k = 0
    # au = np.zeros((N, N))
    # bu = np.zeros((N, N))
    # ad = np.zeros((N, N))
    # bd = np.zeros((N, N))
    # du = Gu.diagonal().copy()
    # dd = Gd.diagonal().copy()
    # for i in shuffle(rng, N):
        # delu = delta[hs[i], i]
        # deld = delta[int(not hs[i]), i]
        # ru = 1. + (1. - du[i]) * delu
        # rd = 1. + (1. - dd[i]) * deld
        # prob = ru * rd
        # if rand_uint(rng) < (np.abs(prob) * 2**64):
            # au[:, k] = Gu[:, i] + np.dot(au[:, :k], bu[:k, i])
            # ad[:, k] = Gd[:, i] + np.dot(ad[:, :k], bd[:k, i])
            # bu[k, :] = Gu[i, :] + np.dot(au[i, :k], bu[:k, :])
            # bd[k, :] = Gd[i, :] + np.dot(ad[i, :k], bd[:k, :])
            # au[i, k] -= 1.0
            # ad[i, k] -= 1.0
            # au[:, k] *= delu/ru
            # ad[:, k] *= deld/rd
            # du += au[:, k] * bu[k, :]
            # dd += ad[:, k] * bd[k, :]
            # k += 1
            # if prob < 0:
                # sign *= -1
            # hs[i] = not hs[i]
        # if k == q:
            # k = 0
            # Gu += np.dot(au[:, :q], bu[:q, :])
            # Gd += np.dot(ad[:, :q], bd[:q, :])
            # du[...] = Gu.diagonal()
            # dd[...] = Gd.diagonal()
    # Gu += np.dot(au[:, :k], bu[:k, :])
    # Gd += np.dot(ad[:, :k], bd[:k, :])
    # return sign


# from scipy.linalg import solve_triangular, lu_solve
# def update_submat(N, delta, rng, hs, Gu, Gd, q):
    # sign = 1
    # k = 0
    # arangeN=np.arange(N)
    # r = np.zeros(N, dtype=int)
    # gammau = np.zeros(N, dtype=np.float64)
    # gammad = np.zeros(N, dtype=np.float64)
    # Gr_u = np.zeros((N, N))
    # G_ru = np.zeros((N, N))
    # Gr_d = np.zeros((N, N))
    # G_rd = np.zeros((N, N))
    # LUu = np.zeros((N, N))
    # LUd = np.zeros((N, N))
    # for ii, i in enumerate(shuffle(rng, N)):
        # gammau[k] = delta[hs[i], i]
        # gammad[k] = delta[int(not hs[i]), i]
        # du = Gu[i, i] - (1. + gammau[k])/gammau[k]
        # dd = Gd[i, i] - (1. + gammad[k])/gammad[k]
        # if k > 0:
            # yu = solve_triangular(LUu[:k, :k], Gr_u[:k, i], 0, True, True)
            # xu = solve_triangular(LUu[:k, :k], G_ru[i, :k], 1, False)
            # du -= np.dot(xu, yu)
            # yd = solve_triangular(LUd[:k, :k], Gr_d[:k, i], 0, True, True)
            # xd = solve_triangular(LUd[:k, :k], G_rd[i, :k], 1, False)
            # dd -= np.dot(xd, yd)
        # probu = -du*gammau[k]
        # probd = -dd*gammad[k]
        # prob = probu*probd
        # if rand_uint(rng) < (np.abs(prob) * 2**64):
            # r[k] = i
            # Gr_u[k, :] = Gu[i, :]
            # G_ru[:, k] = Gu[:, i]
            # Gr_d[k, :] = Gd[i, :]
            # G_rd[:, k] = Gd[:, i]
            # if k > 0:
                # LUu[:k, k] = yu
                # LUu[k, :k] = xu
                # LUd[:k, k] = yd
                # LUd[k, :k] = xd
            # LUu[k, k] = du
            # LUd[k, k] = dd
            # k += 1
            # if prob < 0:
                # sign *= -1
            # hs[i] = not hs[i]
        # if k == q or (ii == N-1 and k > 0):
            # Gu -= np.dot(G_ru[:, :k], lu_solve((LUu[:k, :k], arangeN[:k]), Gr_u[:k, :]))
            # gammau += 1.0
            # Gu[r[:k], :] /= gammau[:k][None].T
            # Gd -= np.dot(G_rd[:, :k], lu_solve((LUd[:k, :k], arangeN[:k]), Gr_d[:k, :]))
            # gammad += 1.0
            # Gd[r[:k], :] /= gammad[:k][None].T
            # k = 0
    # return sign


def measure_eqlt(params, sign, Gu, Gd, meas):
    meas["n_sample"] += 1
    meas["sign"] += sign

    # 1-site measurements
    for i in range(params["N"]):
        r = params["map_i"][i]
        pre = sign / params["degen_i"][r]
        Guii = Gu[i, i]
        Gdii = Gd[i, i]
        meas["density"][r] += pre*(2. - Guii - Gdii)
        meas["double_occ"][r] += pre*(1. - Guii)*(1. - Gdii)

    # 2-site measurements (quite slow in python)
    # for j in range(params["N"]):
        # for i in range(params["N"]):
            # delta = int(i == j)
            # r = params["map_ij"][j, i]
            # pre = sign / params["degen_ij"][r]
            # Guii = Gu[i, i]
            # Guij = Gu[i, j]
            # Guji = Gu[j, i]
            # Gujj = Gu[j, j]
            # Gdii = Gd[i, i]
            # Gdij = Gd[i, j]
            # Gdji = Gd[j, i]
            # Gdjj = Gd[j, j]
            # meas["g00"][r] += 0.5*pre*(Guij + Gdij)
            # x = delta*(Guii + Gdii) - (Guji*Guij + Gdji*Gdij)
            # meas["nn"][r] += pre*((2. - Guii - Gdii)*(2. - Gujj - Gdjj) + x)
            # meas["xx"][r] += 0.25*pre*(delta*(Guii + Gdii) - (Guji*Gdij + Gdji*Guij))
            # meas["zz"][r] += 0.25*pre*((Gdii - Guii)*(Gdjj - Gujj) + x)
            # meas["pair_sw"][r] += pre*(delta*(1. - Guii - Gdii) + 2.*Guij*Gdij)


def measure_uneqlt(params, sign, Gu, Gd, meas):
    meas["n_sample"] += 1
    meas["sign"] += sign


def dqmc(params, state, meas_eqlt, meas_uneqlt):
    Bu = np.zeros((params["L"], params["N"], params["N"]))
    Bd = np.zeros((params["L"], params["N"], params["N"]))
    Cu = np.zeros((params["F"], params["N"], params["N"]))
    Cd = np.zeros((params["F"], params["N"], params["N"]))

    arangeN = np.arange(params["N"])

    for f in range(params["F"]):
        l = f*params["n_matmul"]
        exp_Vu = params["exp_lambda"][state["hs"][l, :], arangeN]
        Bu[l, :, :] = params["exp_K"] * exp_Vu
        Cu[f, :, :] = Bu[l, :, :]
        for l in range(f*params["n_matmul"] + 1, (f + 1)*params["n_matmul"]):
            exp_Vu = params["exp_lambda"][state["hs"][l, :], arangeN]
            Bu[l, :, :] = params["exp_K"] * exp_Vu
            Cu[f, :, :] = np.dot(Bu[l, :, :], Cu[f, :, :])

    for f in range(params["F"]):
        l = f*params["n_matmul"]
        exp_Vd = params["exp_lambda"][1-state["hs"][l, :], arangeN]
        Bd[l, :, :] = params["exp_K"] * exp_Vd
        Cd[f, :, :] = Bd[l, :, :]
        for l in range(f*params["n_matmul"] + 1, (f + 1)*params["n_matmul"]):
            exp_Vd = params["exp_lambda"][1-state["hs"][l, :], arangeN]
            Bd[l, :, :] = params["exp_K"] * exp_Vd
            Cd[f, :, :] = np.dot(Bd[l, :, :], Cd[f, :, :])

    Gu = np.zeros((params["N"], params["N"]))
    Gd = np.zeros((params["N"], params["N"]))
    Gu[...], signu = calcG(0, Cu)
    Gd[...], signd = calcG(0, Cd)
    sign = signu*signd

    for state["sweep"] in range(state["sweep"], params["n_sweep"]):
        warmed_up = state["sweep"] >= params["n_sweep_warm"]
        enabled_eqlt = warmed_up and params["period_eqlt"] > 0
        enabled_uneqlt = warmed_up and params["period_uneqlt"] > 0

        for l in range(params["L"]):
            # updates
            sign *= update_shermor(params["N"], params["del"], state["rng"],
                                   state["hs"][l], Gu, Gd)

            # recalculate C
            f = l // params["n_matmul"]

            exp_Vu = params["exp_lambda"][state["hs"][l, :], arangeN]
            Bu[l, :, :] = params["exp_K"] * exp_Vu
            if l % params["n_matmul"] == 0:
                Cu[f, :, :] = Bu[l, :, :]
            else:
                Cu[f, :, :] = np.dot(Bu[l, :, :], Cu[f, :, :])

            exp_Vd = params["exp_lambda"][1-state["hs"][l, :], arangeN]
            Bd[l, :, :] = params["exp_K"] * exp_Vd
            if l % params["n_matmul"] == 0:
                Cd[f, :, :] = Bd[l, :, :]
            else:
                Cd[f, :, :] = np.dot(Bd[l, :, :], Cd[f, :, :])

            # shift to next G(l)
            if (l + 1) % params["n_matmul"] == 0:  # recalculate
                # invBlu = exp_Vd[None].T * params["inv_exp_K"]
                # Guwrp = np.dot(np.dot(Bu[l, :, :], Gu), invBlu)
                # invBld = exp_Vu[None].T * params["inv_exp_K"]
                # Gdwrp = np.dot(np.dot(Bd[l, :, :], Gd), invBld)
                # Guacc, signu = calcG((l + 1) % params["L"], Bu)
                # Gdacc, signu = calcG((l + 1) % params["L"], Bd)
                Gu[...], signu = calcG((f + 1) % params["F"], Cu)
                Gd[...], signd = calcG((f + 1) % params["F"], Cd)
                sign = signu*signd
                # print("Gu - Guwrp:\tmax {:.3e}\tavg {:.3e}".format(np.max(np.abs(Gu-Guwrp)), np.mean(np.abs(Gu-Guwrp))))
                # print("Gd - Gdwrp:\tmax {:.3e}\tavg {:.3e}".format(np.max(np.abs(Gd-Gdwrp)), np.mean(np.abs(Gd-Gdwrp))))
                # print("Gu - Guacc:\tmax {:.3e}\tavg {:.3e}".format(np.max(np.abs(Gu-Guacc)), np.mean(np.abs(Gu-Guacc))))
                # print("Gd - Gdacc:\tmax {:.3e}\tavg {:.3e}".format(np.max(np.abs(Gd-Gdacc)), np.mean(np.abs(Gd-Gdacc))))
                # print("Guwrp - Guacc:\tmax {:.3e}\tavg {:.3e}".format(np.max(np.abs(Guwrp-Guacc)), np.mean(np.abs(Guwrp-Guacc))))
                # print("Gdwrp - Gdacc:\tmax {:.3e}\tavg {:.3e}".format(np.max(np.abs(Gdwrp-Gdacc)), np.mean(np.abs(Gdwrp-Gdacc))))
            else:  # wrap
                invBlu = exp_Vd[None].T * params["inv_exp_K"]
                np.dot(np.dot(Bu[l, :, :], Gu), invBlu, out=Gu)
                invBld = exp_Vu[None].T * params["inv_exp_K"]
                np.dot(np.dot(Bd[l, :, :], Gd), invBld, out=Gd)

            # eqlt meas
            if enabled_eqlt and (l + 1) % params["period_eqlt"] == 0:
                measure_eqlt(params, sign, Gu, Gd, meas_eqlt)
        # uneqlt meas
        if enabled_uneqlt and state["sweep"] % params["period_uneqlt"] == 0:
            measure_uneqlt(params, sign, Gu, Gd, meas_uneqlt)
    state["sweep"] += 1  # for i in range(...) doesn't increment at the end


def main(argv):
    start_time = time.time()

    print("opening {}".format(argv[1]))
    if len(argv) > 2:
        print("args: {}".format(argv[2:]))

    with h5py.File(argv[1], "r") as f:
        params = {k: v[...] for k, v in f["params"].items()}
        state = {k: v[...] for k, v in f["state"].items()}
        meas_eqlt = {k: v[...] for k, v in f["meas_eqlt"].items()}
        if params["period_uneqlt"] > 0:
            meas_uneqlt = {k: v[...] for k, v in f["meas_uneqlt"].items()}
        else:
            meas_uneqlt = None

    print("{}/{} sweeps completed".format(state["sweep"], params["n_sweep"]))
    if state["sweep"] >= params["n_sweep"]:
        print("already finished")
        return

    print("starting dqmc")
    dqmc(params, state, meas_eqlt, meas_uneqlt)
    print("{}/{} sweeps completed".format(state["sweep"], params["n_sweep"]))

    print("saving data")
    with h5py.File(argv[1], "r+") as f:
        for k, v in f["state"].items():
            v[...] = state[k]
        for k, v in f["meas_eqlt"].items():
            v[...] = meas_eqlt[k]
        if params["period_uneqlt"] > 0:
            for k, v in f["meas_uneqlt"].items():
                v[...] = meas_uneqlt[k]

    print("wall time:", time.time() - start_time)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv)
