import numpy as np
from Calibration import get_K

def triangulate_point(Rs, ts, xs):
    Rs = np.asarray(Rs, dtype=float)
    ts = np.asarray(ts, dtype=float)
    xs = np.asarray(xs, dtype=float)
    rows = []
    for R, t, (u, v) in zip(Rs, ts, xs):
        P = np.hstack([R, t.reshape(3, 1)])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.vstack(rows)
    X = np.linalg.svd(A)[2][-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_points(Ks, Rs, ts, obs):
    Ks, Rs, ts, obs = (np.asarray(a, dtype=float) for a in (Ks, Rs, ts, obs))
    N, M = obs.shape[:2]
    xn = np.full_like(obs, np.nan)
    for i in range(N):
        ones = np.ones((M, 1))
        h = np.linalg.inv(Ks[i]) @ np.hstack([obs[i], ones]).T
        xn[i] = (h[:2] / h[2]).T
    X = np.full((M, 3), np.nan)
    for j in range(M):
        vis = ~np.isnan(obs[:, j]).any(axis=1)
        if vis.sum() >= 2:
            X[j] = triangulate_point(Rs[vis], ts[vis], xn[vis, j])
    return X


def read():
    N, M = (int(x) for x in input().split())
    paths = [input().strip() for _ in range(N)]
    Ks = np.array([get_K(None if p == "-" else p) for p in paths])
    Rs = [np.eye(3)]
    ts = [np.zeros(3)]
    for _ in range(N - 1):
        Rs.append(np.array([[float(x) for x in input().split()] for _ in range(3)]))
        ts.append(np.array([float(x) for x in input().split()]))

    obs = np.empty((N, M, 2))
    for j in range(M):
        row = [float(x) for x in input().split()]
        obs[:, j, :] = np.array(row).reshape(N, 2)
    return Ks, np.array(Rs), np.array(ts), obs

def main():
    Ks, Rs, ts, obs = read()
    X = triangulate_points(Ks, Rs, ts, obs)
    for j, x in enumerate(X):
        print(f"точка {j + 1}: {x.tolist()}")


if __name__ == "__main__":
    main()
