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

# algorithm 3 of 10.1109/IPDPS.2012.37
# B = Q d X
def calc_QdX_first(B):
    Q, jpvt, tau, work, info = dgeqp3(B)
    d = Q.diagonal().copy()
    d[d == 0.0] = 1.0
    X = ((np.triu(Q).T / d).T)[:, jpvt.argsort()]
    return (Q, tau, d, X)

def calc_QdX(B, QdX):
    Q, tau, d, X = QdX
    W, work, info = dormqr("R", "N", Q, tau, B, B.size)
    W *= d
    jpvt = (W*W).sum(0).argsort()[::-1]
    Q_, tau_, work, info = dgeqrf(W[:, jpvt])
    d_ = Q_.diagonal().copy()
    d_[d_ == 0.0] = 1.0
    X_ = np.dot((np.triu(Q_).T / d_).T, X[jpvt, :])
    return (Q_, tau_, d_, X_)

# G = (1 + QdX)^-1 = (idb Q.T  + ds X)^-1 idb Q.T, where d = idb^-1 ds
# for XdQ:
# G = (1 + X.T d Q.T)^-1 = Q idb (Q idb + X.T ds)^-1, where d = idb^-1 ds
#   = ((1 + QdX)^-1).T
def calc_Gtt_last(QdX, transpose=False):
    Q, tau, d, X = QdX
    N = Q.shape[0]
    idb = np.zeros((N, N))
    for i in range(N):
        idb[i, i] = 1.0 / d[i] if np.abs(d[i]) > 1.0 else 1.0
    idbQT, work, info = dormqr("R", "T", Q, tau, idb, N*N)

    M = np.zeros((N, N))
    for i in range(N):
        M[i, :] = (d[i] if np.abs(d[i]) <= 1.0 else 1.0) * X[i, :]

    M += idbQT
    M_LU, piv, info = dgetrf(M)
    sign = 1
    for i in range(N):
        if (M_LU[i, i] < 0) ^ (piv[i] != i) ^ (idb[i, i] < 0) ^ (tau[i] > 0):
            sign *= -1

    G, info = dgetrs(M_LU, piv, idbQT)
    return (G if not transpose else G.T), sign

# QdX0 contains B_{t-1} ... B_0
# QdX1 contains B_{L-1} ... B_t
# G = (1 + Q0 d0 X0 X1.T d1 Q1.T)
#   = Q1 id1b (id0b Q0.T Q1 id1b + d0s X0 X1.T d1s)^-1 id0b Q0.T
def calc_Gtt(QdX0, QdX1):
    Q0, tau0, d0, X0 = QdX0
    Q1, tau1, d1, X1 = QdX1
    N = Q0.shape[0]
    id0b = np.zeros((N, N))
    id1b = np.zeros((N, N))
    for i in range(N):
        id0b[i, i] = 1.0 / d0[i] if np.abs(d0[i]) > 1.0 else 1.0
        id1b[i, i] = 1.0 / d1[i] if np.abs(d1[i]) > 1.0 else 1.0
    id0bQ0T, work, info = dormqr("R", "T", Q0, tau0, id0b, N*N)
    Q1id1b, work, info = dormqr("L", "N", Q1, tau1, id1b, N*N)

    M = np.dot(X0, X1.T)
    for i in range(N):
        M[i, :] *= (d0[i] if np.abs(d0[i]) < 1.0 else 1.0)
    for i in range(N):
        M[:, i] *= (d1[i] if np.abs(d1[i]) < 1.0 else 1.0)

    M += np.dot(id0bQ0T, Q1id1b)
    M_LU, piv, info = dgetrf(M)
    sign = 1
    for i in range(N):
        if (M_LU[i, i] < 0) ^ (piv[i] != i) ^ (id0b[i, i] < 0) ^ (id1b[i, i] < 0) ^ (tau0[i] > 0) ^ (tau1[i] > 0):
            sign *= -1

    G = np.dot(Q1id1b, dgetrs(M_LU, piv, id0bQ0T)[0])
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


def dqmc(params, state, meas_eqlt, meas_uneqlt):
    L = params["L"]
    F = params["F"]
    n_matmul = params["n_matmul"]

    Bu = np.zeros((L, params["N"], params["N"]))
    Bd = np.zeros((L, params["N"], params["N"]))
    iBu = np.zeros((L, params["N"], params["N"])) # inverse of B matrices
    iBd = np.zeros((L, params["N"], params["N"]))
    Cu = np.zeros((F, params["N"], params["N"]))  # product of B
    Cd = np.zeros((F, params["N"], params["N"]))

    arangeN = np.arange(params["N"])

    for f in range(F):
        l = f*n_matmul
        exp_Vu = params["exp_lambda"][state["hs"][l, :], arangeN]
        Bu[l] = params["exp_Ku"] * exp_Vu
        Cu[f] = Bu[l]
        for l in range(f*n_matmul + 1, (f + 1)*n_matmul):
            exp_Vu = params["exp_lambda"][state["hs"][l, :], arangeN]
            Bu[l] = params["exp_Ku"] * exp_Vu
            Cu[f] = np.dot(Bu[l], Cu[f])

    for f in range(F):
        l = f*n_matmul
        exp_Vd = params["exp_lambda"][1-state["hs"][l, :], arangeN]
        Bd[l] = params["exp_Kd"] * exp_Vd
        Cd[f] = Bd[l]
        for l in range(f*n_matmul + 1, (f + 1)*n_matmul):
            exp_Vd = params["exp_lambda"][1-state["hs"][l, :], arangeN]
            Bd[l] = params["exp_Kd"] * exp_Vd
            Cd[f] = np.dot(Bd[l], Cd[f])

    # QdX decomposition of products of C matrices are stored in these.
    QdX0u = [()]*int(F)  # product of C_f ... C_0
    QdX0d = [()]*int(F)
    QdXLu = [()]*int(F)  # product of C_{F-1} ... C_{f+1}
    QdXLd = [()]*int(F)

    # initial down sweep to initialize QdXL, so that the first actual up sweep works.
    QdXLu[F-1] = calc_QdX_first(Cu[F-1].T)
    for f in range(F-2, -1, -1):
        QdXLu[f] = calc_QdX(Cu[f].T, QdXLu[f+1])

    QdXLd[F-1] = calc_QdX_first(Cd[F-1].T)
    for f in range(F-2, -1, -1):
        QdXLd[f] = calc_QdX(Cd[f].T, QdXLd[f+1])

    # initialize Green's functions
    Gttu = np.zeros((params["N"], params["N"]))
    Gttd = np.zeros((params["N"], params["N"]))
    Gttu[...], signu = calc_Gtt_last(QdXLu[0], transpose=True)
    Gttd[...], signd = calc_Gtt_last(QdXLd[0], transpose=True)
    sign = signu*signd

    for state["sweep"] in range(state["sweep"], params["n_sweep"]):
        warmed_up = state["sweep"] >= params["n_sweep_warm"]
        enabled_eqlt = warmed_up and params["period_eqlt"] > 0

        if state["sweep"] % 2 == 0:  # up sweep from l=0 to l=L-1
            for l in range(L):
                f, m = l // n_matmul, l % n_matmul

                sign *= update_shermor(params["N"], params["del"], state["rng"], state["hs"][l], Gttu, Gttd)

                # update B, inverse of B, and C (simple products of B)
                exp_Vu = params["exp_lambda"][state["hs"][l, :], arangeN]
                exp_Vd = params["exp_lambda"][1-state["hs"][l, :], arangeN]
                Bu[l] = params["exp_Ku"] * exp_Vu
                Bd[l] = params["exp_Kd"] * exp_Vd
                iBu[l] = exp_Vd[None].T * params["inv_exp_Ku"] # note that exp_Vd = 1/exp_Vu
                iBd[l] = exp_Vu[None].T * params["inv_exp_Kd"]
                Cu[f] = Bu[l] if m == 0 else np.dot(Bu[l], Cu[f])
                Cd[f] = Bd[l] if m == 0 else np.dot(Bd[l], Cd[f])

                # here, Gtt = G(l, l)
                if m == n_matmul - 1:  # calculate accurately G(l+1, l+1)
                    QdX0u[f] = calc_QdX_first(Cu[f]) if f == 0 else calc_QdX(Cu[f], QdX0u[f-1])
                    Gttu[...], signu = calc_Gtt_last(QdX0u[f]) if f == F-1 else calc_Gtt(QdX0u[f], QdXLu[f+1])
                    QdX0d[f] = calc_QdX_first(Cd[f]) if f == 0 else calc_QdX(Cd[f], QdX0d[f-1])
                    Gttd[...], signd = calc_Gtt_last(QdX0d[f]) if f == F-1 else calc_Gtt(QdX0d[f], QdXLd[f+1])
                    sign = signu*signd
                else:  # wrap forward
                    np.dot(np.dot(Bu[l], Gttu), iBu[l], out=Gttu)
                    np.dot(np.dot(Bd[l], Gttd), iBd[l], out=Gttd)
                # here, Gtt is G(l+1, l+1)

                # eqlt meas
                if enabled_eqlt and (l + 1) % params["period_eqlt"] == 0:
                    Gu_half = np.dot(params["inv_exp_halfKu"], np.dot(Gttu, params["exp_halfKu"]))
                    Gd_half = np.dot(params["inv_exp_halfKd"], np.dot(Gttd, params["exp_halfKd"]))
                    measure_eqlt(params, sign, Gu_half, Gd_half, meas_eqlt)
        else:  # down sweep from l=L-1 to l=0
            for l in reversed(range(L)):
                f, m = l // n_matmul, l % n_matmul

                # here, Gtt is G(l+1, l+1). wrap backward
                np.dot(np.dot(iBu[l], Gttu), Bu[l], out=Gttu)
                np.dot(np.dot(iBd[l], Gttd), Bd[l], out=Gttd)
                # here, Gtt is G(l, l)

                sign *= update_shermor(params["N"], params["del"], state["rng"], state["hs"][l], Gttu, Gttd)

                # update B, inverse of B, and C (simple products of B)
                exp_Vu = params["exp_lambda"][state["hs"][l, :], arangeN]
                exp_Vd = params["exp_lambda"][1-state["hs"][l, :], arangeN]
                Bu[l] = params["exp_Ku"] * exp_Vu
                Bd[l] = params["exp_Kd"] * exp_Vd
                iBu[l] = exp_Vd[None].T * params["inv_exp_Ku"] # note that exp_Vd = 1/exp_Vu
                iBd[l] = exp_Vu[None].T * params["inv_exp_Kd"]
                Cu[f] = Bu[l] if m == n_matmul - 1 else np.dot(Cu[f], Bu[l])
                Cd[f] = Bd[l] if m == n_matmul - 1 else np.dot(Cd[f], Bd[l])

                if m == 0:  # recalculate accurately G(l, l)
                    QdXLu[f] = calc_QdX_first(Cu[f].T) if f == F-1 else calc_QdX(Cu[f].T, QdXLu[f+1])
                    Gttu_new, signu = calc_Gtt_last(QdXLu[f], transpose=True) if f == 0 else calc_Gtt(QdX0u[f-1], QdXLu[f])
                    QdXLd[f] = calc_QdX_first(Cd[f].T) if f == F-1 else calc_QdX(Cd[f].T, QdXLd[f+1])
                    Gttd_new, signd = calc_Gtt_last(QdXLd[f], transpose=True) if f == 0 else calc_Gtt(QdX0d[f-1], QdXLd[f])
                    max_diff = max(np.max(np.abs(Gttu - Gttu_new)), np.max(np.abs(Gttd - Gttd_new)))
                    if max_diff > 1e-7:
                        print(f"On sweep {state['sweep']}, G differed by {max_diff:g}. Consider increasing n_matmul.")
                    Gttu[...] = Gttu_new
                    Gttd[...] = Gttd_new
                    sign = signu*signd

                # eqlt meas
                if enabled_eqlt and l % params["period_eqlt"] == 0:
                    Gu_half = np.dot(params["inv_exp_halfKu"], np.dot(Gttu, params["exp_halfKu"]))
                    Gd_half = np.dot(params["inv_exp_halfKd"], np.dot(Gttd, params["exp_halfKd"]))
                    measure_eqlt(params, sign, Gu_half, Gd_half, meas_eqlt)
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

    print("wall time:", time.time() - start_time)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv)
