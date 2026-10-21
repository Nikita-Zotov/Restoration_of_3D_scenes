import numpy as np


def triangulation(R, t, x1, x2):
    R = np.asarray(R).reshape(3,3)
    t = np.asarray(t).reshape(3,1)
    P1 = np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = np.hstack([R,t])
    x1 = np.asarray(x1).reshape(3)
    x2 = np.asarray(x2).reshape(3)
    x1 /= x1[2];
    x2 /= x2[2];
    u1 = x1[0];
    v1 = x1[1];
    u2 = x2[0];
    v2 = x2[1];
    A = np.vstack([u1*P1[2] - P1[0], v1*P1[2] - P1[1], u2*P2[2] - P2[0], v2*P2[2] - P2[1]])
    VT = np.linalg.svd(A)[2]
    X = VT[-1]
    X3d = [X[0]/X[3], X[1]/X[3], X[2]/X[3]]
    return X3d

def read():
    R = [[float(x) for x in input().split()] for i in range(3)]
    t = [float(x) for x in input().split()]
    x1 = [float(x) for x in input().split()]
    x2 = [float(x) for x in input().split()]
    return R, t, x1, x2

def main():
    R, t, x1, x2 = read()
    X = triangulation(R, t, x1, x2)
    print(X)


main()
