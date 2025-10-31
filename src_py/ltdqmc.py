import sys
import time

import h5py
import numpy as np

from scipy.linalg.lapack import zgeqp3, zgeqrf, zunmqr, zgetrf, zgetrs

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
# input: rng(random number) and n, which creates (1,2,...,n)
# output: change the order of (0,1,...,n-1)
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
# the strating point of the QR decomposition
# input: B (starting B)
# output: Q tau d X


def calc_QdX_first(B):
    Q, jpvt, tau, work, info = zgeqp3(B)
    d = Q.diagonal().copy()
    d[d == 0.0] = 1.0
    X = ((np.triu(Q).T / d).T)[:, jpvt.argsort()]
    return (Q, tau, d, X)

# input: B and Q tau d X
# output: new Q tau d X


def calc_QdX(B, QdX):
    Q, tau, d, X = QdX
    W, work, info = zunmqr("R", "N", Q, tau, B, B.size)
    W *= d
    jpvt = (W*W.conj()).sum(axis=0).argsort()[::-1]
    Q_, tau_, work, info = zgeqrf(W[:, jpvt])
    d_ = np.diag(Q_).copy()
    d_[d_ == 0.0] = 1.0
    X_ = np.dot((np.triu(Q_).T / d_).T, X[jpvt, :])
    return (Q_, tau_, d_, X_)

# G = (1 + QdX)^-1 = (idb Q.T  + ds X)^-1 idb Q.T, where d = idb^-1 ds
# for XdQ:
# G = (1 + X.T d Q.T)^-1 = Q idb (Q idb + X.T ds)^-1, where d = idb^-1 ds
#   = ((1 + QdX)^-1).T
# input: Q, tau, d, X (QR parameter)
# output: Green's function for measurement, sign


def calc_Gtt_last(QdX, transpose=False):
    Q, tau, d, X = QdX
    N = Q.shape[0]

    # Initialize idb and handle diagonal elements (complex safe division)
    idb = np.zeros((N, N), dtype=np.complex128)
    np.fill_diagonal(idb, [1.0 / di if abs(di) >
                     1.0 else 1.0 + 0j for di in d])

    # Query optimal lwork and apply Q^H
    _, lwork, _ = zunmqr("R", "C", Q, tau, idb, lwork=-1)
    idbQH, _, _ = zunmqr("R", "C", Q, tau, idb, lwork=int(lwork.real))

    # Construct M = idbQH + diag(d_mod) * X (optimized with broadcasting)
    d_mod = np.where(np.abs(d) <= 1.0, d, 1.0)
    M = idbQH + (d_mod[:, None] * X)

    # LU decomposition
    M_LU, piv, _ = zgetrf(M)

    # Solve linear system
    G, _ = zgetrs(M_LU, piv, idbQH)

    phase = 1

    # Return result (optionally conjugate transpose)
    return (G if not transpose else G.T.conj()), phase

# QdX0 contains B_{t-1} ... B_0
# QdX1 contains B_{L-1} ... B_t
# G = (1 + Q0 d0 X0 X1.T d1 Q1.T)
#   = Q1 id1b (id0b Q0.T Q1 id1b + d0s X0 X1.T d1s)^-1 id0b Q0.T
# input: Q, tau, d, X for two ranges
# output: Green's function during update and sign


def calc_Gtt(QdX0, QdX1):
    Q0, tau0, d0, X0 = QdX0
    Q1, tau1, d1, X1 = QdX1
    N = Q0.shape[0]

    # Initialize id0b and id1b (complex safe division)
    id0b = np.zeros((N, N), dtype=np.complex128)
    np.fill_diagonal(id0b, [1.0 / di if abs(di) >
                     1.0 else 1.0 + 0j for di in d0])
    id1b = np.zeros((N, N), dtype=np.complex128)
    np.fill_diagonal(id1b, [1.0 / di.conj() if abs(di)
                     > 1.0 else 1.0 + 0j for di in d1])

    # Apply Q0^H to the right side of id0b
    _, lwork0, _ = zunmqr("R", "C", Q0, tau0, id0b, lwork=-1)
    id0bQ0H, _, _ = zunmqr("R", "C", Q0, tau0, id0b,
                           lwork=int(lwork0.real), overwrite_c=1)

    # Apply Q1 to the left side of id1b
    _, lwork1, _ = zunmqr("L", "N", Q1, tau1, id1b, lwork=-1)
    Q1id1b, _, _ = zunmqr("L", "N", Q1, tau1, id1b,
                          lwork=int(lwork1.real), overwrite_c=1)

    # Construct matrix M
    d0_mod = np.diag(np.where(np.abs(d0) <= 1.0, d0, 1.0))
    d1_mod = np.diag(np.where(np.abs(d1) <= 1.0, d1.conj(), 1.0))
    M = np.dot(np.dot(d0_mod, X0 @ X1.conj().T), d1_mod)
    M += id0bQ0H @ Q1id1b

    # LU decomposition
    M_LU, piv, _ = zgetrf(M)

    # Solve linear system and compute G
    solution, _ = zgetrs(M_LU, piv, id0bQ0H)
    G = Q1id1b @ solution

    phase = 1

    return G, phase


def calc_expV(l, N, N_inter, bonds_inter, exp_lambda, exp_lambda_a, hs):
    exp_V = np.eye(N, N, dtype=np.complex128)
    for n_inter in range(N_inter):
        exp_V[bonds_inter[0, n_inter], bonds_inter[0, n_inter]
              ] *= exp_lambda[n_inter, hs[l, n_inter]]
        exp_V[bonds_inter[1, n_inter], bonds_inter[1, n_inter]
              ] *= exp_lambda_a[n_inter, hs[l, n_inter]]
    return (exp_V)


def update_shermor(N, N_inter, bonds_inter, rng, hs, Gu, Gd, all_state, a, weight_comparison_matrice, Delta_p, Delta_q):
    sign = 1+0j

    cu_p_column = np.empty(N, dtype=np.complex128)
    cu_q_column = np.empty(N, dtype=np.complex128)
    cd_p_column = np.empty(N, dtype=np.complex128)
    cd_q_column = np.empty(N, dtype=np.complex128)

    # all_state = np.array([-2 -1 1 2])

    for n_inter in shuffle(rng, N_inter):
        other_states = [s for s in all_state if s != hs[n_inter]]
        rand_jump_para = rand_uint(rng)/(2**64)
        if rand_jump_para < 1/3:
            after_jump = other_states[0]
        elif rand_jump_para < 2/3:
            after_jump = other_states[1]
        else:
            after_jump = other_states[2]

        p = bonds_inter[0, n_inter]
        q = bonds_inter[1, n_inter]

        Deltap = Delta_p[n_inter, after_jump, hs[n_inter]]
        Deltaq = Delta_q[n_inter, after_jump, hs[n_inter]]

        A_divided = weight_comparison_matrice[n_inter, after_jump, hs[n_inter]]
        pu = (1 + Deltap * (1-Gu[p, p]) + Deltaq * (1-Gu[q, q]) + Deltap * Deltaq * (
            1-Gu[p, p]) * (1-Gu[q, q]) - Deltap * Deltaq * Gu[p, q] * Gu[q, p])
        pd = (1 + Deltap * (1-Gd[p, p]) + Deltaq * (1-Gd[q, q]) + Deltap * Deltaq * (
            1-Gd[p, p]) * (1-Gd[q, q]) - Deltap * Deltaq * Gd[p, q] * Gd[q, p])
        prob = A_divided*pu*pd

        if rand_uint(rng) < (np.abs(prob) * 2**64):
            if p < q:
                Fu_11 = Deltaq * (1. + Deltaq * (1. - Gu[q, q]))
                Fu_12 = Deltap * Deltaq * Gu[p, q]
                Fu_21 = Deltap * Deltaq * Gu[q, p]
                Fu_22 = Deltap * (1. + Deltap * (1. - Gu[p, p]))
                Denou = Fu_11 * Fu_22 - Fu_12 * Fu_21

                Fu_renor_11 = Deltap**2 * Fu_11
                Fu_renor_12 = Deltap * Deltaq * Fu_12
                Fu_renor_21 = Deltap * Deltaq * Fu_21
                Fu_renor_22 = Deltaq**2 * Fu_22
                Fu_renor = np.array(
                    [[Fu_renor_11, Fu_renor_12], [Fu_renor_21, Fu_renor_22]])
                Fu_renor /= Denou

                cu_p_column[...] = Gu[:, p]
                cu_p_column[p] -= 1.
                cu_q_column[...] = Gu[:, q]
                cu_q_column[q] -= 1.

                cu_column = np.column_stack((cu_p_column, cu_q_column))
                cu_row = np.vstack((Gu[p, :], Gu[q, :]))

                Gu += cu_column @ Fu_renor @ cu_row

                Fd_11 = Deltaq * (1. + Deltaq * (1. - Gd[q, q]))
                Fd_12 = Deltap * Deltaq * Gd[p, q]
                Fd_21 = Deltap * Deltaq * Gd[q, p]
                Fd_22 = Deltap * (1. + Deltap * (1. - Gd[p, p]))
                Denod = Fd_11 * Fd_22 - Fd_12 * Fd_21

                Fd_renor_11 = Deltap**2 * Fd_11
                Fd_renor_12 = Deltap * Deltaq * Fd_12
                Fd_renor_21 = Deltap * Deltaq * Fd_21
                Fd_renor_22 = Deltaq**2 * Fd_22
                Fd_renor = np.array(
                    [[Fd_renor_11, Fd_renor_12], [Fd_renor_21, Fd_renor_22]])
                Fd_renor /= Denod

                cd_p_column[...] = Gd[:, p]
                cd_p_column[p] -= 1.
                cd_q_column[...] = Gd[:, q]
                cd_q_column[q] -= 1.

                cd_column = np.column_stack((cd_p_column, cd_q_column))
                cd_row = np.vstack((Gd[p, :], Gd[q, :]))

                Gd += cd_column @ Fd_renor @ cd_row

            else:
                Fu_11 = Deltap * (1. + Deltap * (1. - Gu[p, p]))
                Fu_12 = Deltaq * Deltap * Gu[q, p]
                Fu_21 = Deltaq * Deltap * Gu[p, q]
                Fu_22 = Deltaq * (1. + Deltaq * (1. - Gu[q, q]))
                Denou = Fu_11 * Fu_22 - Fu_12 * Fu_21

                Fu_renor_11 = Deltaq**2 * Fu_11
                Fu_renor_12 = Deltaq * Deltap * Fu_12
                Fu_renor_21 = Deltaq * Deltap * Fu_21
                Fu_renor_22 = Deltap**2 * Fu_22
                Fu_renor = np.array(
                    [[Fu_renor_11, Fu_renor_12], [Fu_renor_21, Fu_renor_22]])
                Fu_renor /= Denou

                cu_q_column[...] = Gu[:, q]
                cu_q_column[q] -= 1.
                cu_p_column[...] = Gu[:, p]
                cu_p_column[p] -= 1.

                cu_column = np.column_stack((cu_q_column, cu_p_column))
                cu_row = np.vstack((Gu[q, :], Gu[p, :]))

                Gu += cu_column @ Fu_renor @ cu_row

                Fd_11 = Deltap * (1. + Deltap * (1. - Gd[p, p]))
                Fd_12 = Deltaq * Deltap * Gd[q, p]
                Fd_21 = Deltaq * Deltap * Gd[p, q]
                Fd_22 = Deltaq * (1. + Deltaq * (1. - Gd[q, q]))
                Denod = Fd_11 * Fd_22 - Fd_12 * Fd_21

                Fd_renor_11 = Deltaq**2 * Fd_11
                Fd_renor_12 = Deltaq * Deltap * Fd_12
                Fd_renor_21 = Deltaq * Deltap * Fd_21
                Fd_renor_22 = Deltap**2 * Fd_22
                Fd_renor = np.array(
                    [[Fd_renor_11, Fd_renor_12], [Fd_renor_21, Fd_renor_22]])
                Fd_renor /= Denod

                cd_q_column[...] = Gd[:, q]
                cd_q_column[q] -= 1.
                cd_p_column[...] = Gd[:, p]
                cd_p_column[p] -= 1.

                cd_column = np.column_stack((cd_q_column, cd_p_column))
                cd_row = np.vstack((Gd[q, :], Gd[p, :]))

                Gd += cd_column @ Fd_renor @ cd_row

            sign = 1
            hs[n_inter] = after_jump
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


# update


def dqmc(params, state, meas_eqlt, meas_uneqlt):
    L = params["L"]
    F = params["F"]
    n_matmul = params["n_matmul"]

    Bu = np.zeros((L, params["N"], params["N"]), dtype=np.complex128)
    Bd = np.zeros((L, params["N"], params["N"]), dtype=np.complex128)
    iBu = np.zeros((L, params["N"], params["N"]),
                   dtype=np.complex128)  # inverse of B matrices
    iBd = np.zeros((L, params["N"], params["N"]), dtype=np.complex128)
    Cu = np.zeros((F, params["N"], params["N"]),
                  dtype=np.complex128)  # product of B
    Cd = np.zeros((F, params["N"], params["N"]), dtype=np.complex128)

    for f in range(F):
        l = f*n_matmul
        expV_temp = calc_expV(l, params["N"], params["N_inter"], params["bonds_inter"],
                              params["exp_lambda"], params["exp_lambda_a"], state["hs"])
        Bu[l] = np.dot(params["exp_Ku"], expV_temp)
        Cu[f] = Bu[l]
        Bd[l] = np.dot(params["exp_Kd"], expV_temp)
        Cd[f] = Bd[l]
        for l in range(f*n_matmul + 1, (f+1)*n_matmul):
            expV_temp = calc_expV(l, params["N"], params["N_inter"], params["bonds_inter"],
                                  params["exp_lambda"], params["exp_lambda_a"], state["hs"])
            Bu[l] = np.dot(params["exp_Ku"], expV_temp)
            Cu[f] = np.dot(Bu[l], Cu[f])
            Bd[l] = np.dot(params["exp_Kd"], expV_temp)
            Cd[f] = np.dot(Bd[l], Cd[f])

    # QdX decomposition of products of C matrices are stored in these.
    QdX0u = [()]*int(F)  # product of C_f ... C_0
    QdX0d = [()]*int(F)
    QdXLu = [()]*int(F)  # product of C_{F-1} ... C_{f+1}
    QdXLd = [()]*int(F)

    # initial down sweep to initialize QdXL, so that the first actual up sweep works.
    QdXLu[F-1] = calc_QdX_first(Cu[F-1].conj().T)
    for f in range(F-2, -1, -1):
        QdXLu[f] = calc_QdX(Cu[f].conj().T, QdXLu[f+1])

    QdXLd[F-1] = calc_QdX_first(Cd[F-1].conj().T)
    for f in range(F-2, -1, -1):
        QdXLd[f] = calc_QdX(Cd[f].conj().T, QdXLd[f+1])

    # initialize Green's functions
    Gttu = np.zeros((params["N"], params["N"]), dtype=np.complex128)
    Gttd = np.zeros((params["N"], params["N"]), dtype=np.complex128)
    Gttu[...], signu = calc_Gtt_last(QdXLu[0], transpose=True)
    Gttd[...], signd = calc_Gtt_last(QdXLd[0], transpose=True)
    sign = signu*signd

    for state["sweep"] in range(state["sweep"], params["n_sweep"]):
        warmed_up = state["sweep"] >= params["n_sweep_warm"]
        enabled_eqlt = warmed_up and params["period_eqlt"] > 0

        if state["sweep"] % 2 == 0:  # up sweep from l=0 to l=L-1
            for l in range(L):
                f, m = l // n_matmul, l % n_matmul
                sign *= update_shermor(params["N"], params["N_inter"], params["bonds_inter"], state["rng"], state["hs"][l],
                                       Gttu, Gttd, params["all_state"], params["a"], params["weight_comparison_matrice"], params["Delta_p"], params["Delta_q"])

                # update B, inverse of B, and C (simple products of B)
                exp_Vu = calc_expV(l, params["N"], params["N_inter"], params["bonds_inter"],
                                   params["exp_lambda"], params["exp_lambda_a"], state["hs"])
                exp_Vd = exp_Vu.copy()
                Bu[l] = np.dot(params["exp_Ku"], exp_Vu)
                Bd[l] = np.dot(params["exp_Kd"], exp_Vd)
                # note that exp_Vd is no longer equal to 1/exp_Vu
                inv_exp_Vu = np.linalg.inv(exp_Vu)
                inv_exp_Vd = inv_exp_Vu.copy()
                iBu[l] = np.dot(inv_exp_Vu, params["inv_exp_Ku"])
                iBd[l] = np.dot(inv_exp_Vd, params["inv_exp_Kd"])
                Cu[f] = Bu[l] if m == 0 else np.dot(Bu[l], Cu[f])
                Cd[f] = Bd[l] if m == 0 else np.dot(Bd[l], Cd[f])

                # here, Gtt = G(l, l)
                if m == n_matmul - 1:  # calculate accurately G(l+1, l+1)
                    QdX0u[f] = calc_QdX_first(
                        Cu[f]) if f == 0 else calc_QdX(Cu[f], QdX0u[f-1])
                    Gttu[...], signu = calc_Gtt_last(
                        QdX0u[f]) if f == F-1 else calc_Gtt(QdX0u[f], QdXLu[f+1])
                    QdX0d[f] = calc_QdX_first(
                        Cd[f]) if f == 0 else calc_QdX(Cd[f], QdX0d[f-1])
                    Gttd[...], signd = calc_Gtt_last(
                        QdX0d[f]) if f == F-1 else calc_Gtt(QdX0d[f], QdXLd[f+1])
                    sign = signu*signd
                else:  # wrap forward
                    np.dot(np.dot(Bu[l], Gttu), iBu[l], out=Gttu)
                    np.dot(np.dot(Bd[l], Gttd), iBd[l], out=Gttd)
                # here, Gtt is G(l+1, l+1)

                # eqlt meas
                if enabled_eqlt and (l + 1) % params["period_eqlt"] == 0:
                    Gu_half = np.dot(params["inv_exp_halfKu"], np.dot(
                        Gttu, params["exp_halfKu"]))
                    Gd_half = np.dot(params["inv_exp_halfKd"], np.dot(
                        Gttd, params["exp_halfKd"]))
                    measure_eqlt(params, sign, Gu_half, Gd_half, meas_eqlt)
        else:  # down sweep from l=L-1 to l=0
            for l in reversed(range(L)):
                f, m = l // n_matmul, l % n_matmul

                # here, Gtt is G(l+1, l+1). wrap backward
                np.dot(np.dot(iBu[l], Gttu), Bu[l], out=Gttu)
                np.dot(np.dot(iBd[l], Gttd), Bd[l], out=Gttd)
                # here, Gtt is G(l, l)

                sign *= update_shermor(params["N"], params["N_inter"], params["bonds_inter"], state["rng"], state["hs"][l],
                                       Gttu, Gttd, params["all_state"], params["a"], params["weight_comparison_matrice"], params["Delta_p"], params["Delta_q"])

                # update B, inverse of B, and C (simple products of B)
                exp_Vu = calc_expV(l, params["N"], params["N_inter"], params["bonds_inter"],
                                   params["exp_lambda"], params["exp_lambda_a"], state["hs"])
                exp_Vd = exp_Vu.copy()
                Bu[l] = np.dot(params["exp_Ku"], exp_Vu)
                Bd[l] = np.dot(params["exp_Kd"], exp_Vd)
                inv_exp_Vu = np.linalg.inv(exp_Vu)
                inv_exp_Vd = inv_exp_Vu.copy()
                iBu[l] = np.dot(inv_exp_Vu, params["inv_exp_Ku"])
                iBd[l] = np.dot(inv_exp_Vd, params["inv_exp_Kd"])
                Cu[f] = Bu[l] if m == n_matmul - 1 else np.dot(Cu[f], Bu[l])
                Cd[f] = Bd[l] if m == n_matmul - 1 else np.dot(Cd[f], Bd[l])

                if m == 0:  # recalculate accurately G(l, l)
                    QdXLu[f] = calc_QdX_first(
                        Cu[f].conj().T) if f == F-1 else calc_QdX(Cu[f].conj().T, QdXLu[f+1])
                    Gttu_new, signu = calc_Gtt_last(
                        QdXLu[f], transpose=True) if f == 0 else calc_Gtt(QdX0u[f-1], QdXLu[f])
                    QdXLd[f] = calc_QdX_first(
                        Cd[f].conj().T) if f == F-1 else calc_QdX(Cd[f].conj().T, QdXLd[f+1])
                    Gttd_new, signd = calc_Gtt_last(
                        QdXLd[f], transpose=True) if f == 0 else calc_Gtt(QdX0d[f-1], QdXLd[f])
                    max_diff = max(np.max(np.abs(Gttu - Gttu_new)),
                                   np.max(np.abs(Gttd - Gttd_new)))

                    if max_diff > 1e-7:
                        print(f"On sweep {state['sweep']}, G differed by {
                              max_diff:g}. Consider decreasing n_matmul.")
                    Gttu[...] = Gttu_new
                    Gttd[...] = Gttd_new
                    sign = signu*signd

                # eqlt meas
                if enabled_eqlt and l % params["period_eqlt"] == 0:
                    Gu_half = np.dot(params["inv_exp_halfKu"], np.dot(
                        Gttu, params["exp_halfKu"]))
                    Gd_half = np.dot(params["inv_exp_halfKd"], np.dot(
                        Gttd, params["exp_halfKd"]))
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
