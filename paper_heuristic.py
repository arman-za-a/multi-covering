import glob
import json
import math
import os
import random
import sys
import time
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans

def make_circle(points):
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c

def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c
def _make_circle_two_points(points, p, q):
    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q
    for r in points:
        if is_in_circle(circ, r):
            continue
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
            left = c
        elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
            right = c
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right

def make_diameter(a, b):
    cx = (a[0] + b[0]) / 2
    cy = (a[1] + b[1]) / 2
    r0 = math.hypot(cx - a[0], cy - a[1])
    r1 = math.hypot(cx - b[0], cy - b[1])
    return (cx, cy, max(r0, r1))

def make_circumcircle(a, b, c):
    ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
    oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2
    ax = a[0] - ox
    ay = a[1] - oy
    bx = b[0] - ox
    by = b[1] - oy
    cx = c[0] - ox
    cy = c[1] - oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by)
              * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
    y = oy + ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by)
              * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
    ra = math.hypot(x - a[0], y - a[1])
    rb = math.hypot(x - b[0], y - b[1])
    rc = math.hypot(x - c[0], y - c[1])
    return (x, y, max(ra, rb, rc))

_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def is_in_circle(c, p):
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON

def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

def iterative_alg(X, m, k):
    tolerance = 1e-6
    if m > X.shape[0]:
        Y_opt = np.zeros((m, 2))
        r_opt = np.zeros(m)
        if np.sum(k) <= m:
            for ik in range(len(k)):
                if ik == 0:
                    Y_opt[:k[ik], :] = np.tile(X[ik, :], (k[ik], 1))
                else:
                    Y_opt[np.sum(k[:ik]):np.sum(k[:ik + 1]),
                          :] = np.tile(X[ik, :], (k[ik], 1))
        else:
            Y_opt[:X.shape[0], :] = X
            X_not_covered = X[np.sum(distance_matrix(
                X, Y_opt) <= r_opt[None, :], axis=1) < k, :]
            it = 0
            discs = 0
            while X_not_covered.shape[0] > 0:
                it += 1
                if it == 1:
                    nclusters = int((m - X.shape[0]) / (np.max(k) - 1))
                    kmeans = KMeans(nclusters, init='k-means++', n_init=3, max_iter=100, tol=1e-4,
                                    verbose=0, algorithm='lloyd', random_state=None).fit(X_not_covered)
                    for cluster in range(nclusters):
                        cluster_points = X_not_covered[kmeans.labels_ == cluster, :]
                        cx, cy, radius = make_circle(cluster_points)
                        Y_opt[cluster, :] = np.array([cx, cy])
                        r_opt[cluster] = radius + tolerance
                    discs += nclusters
                else:
                    nclusters = max(1, int(X_not_covered.shape[0]/2))
                    kmeans = KMeans(nclusters, init='k-means++', n_init=3, max_iter=100, tol=1e-4,
                                    verbose=0, algorithm='lloyd', random_state=None).fit(X_not_covered)
                    for cluster in range(nclusters):
                        cluster_points = X_not_covered[kmeans.labels_ == cluster, :]
                        cx, cy, radius = make_circle(cluster_points)
                        Y_opt[cluster + discs, :] = np.array([cx, cy])
                        r_opt[cluster + discs] = radius + tolerance
                    discs += nclusters
                X_not_covered = X[np.sum(distance_matrix(
                    X, Y_opt) <= r_opt[None, :], axis=1) < k, :]
        return r_opt, Y_opt
    best_cost = np.inf
    best_r_opt = None
    best_Y_opt = None
    for seed in range(5):
        np.random.seed(seed)
        r = np.zeros(m)
        Y = np.zeros((m, 2))
        num_permutations = 5
        for perm in range(num_permutations):
            perm_index = np.random.permutation(X.shape[0])
            kmeans = KMeans(n_clusters=m, init='k-means++', n_init=3, max_iter=100, tol=1e-4,
                            verbose=0, algorithm='lloyd', random_state=None).fit(X)
            idx = kmeans.labels_
            assignment = np.zeros((X.shape[0], m))
            for i in range(len(idx)):
                assignment[i, idx[i]] = 1
            for cluster in range(m):
                cluster_points = X[idx == cluster, :]
                cx, cy, radius = make_circle(cluster_points)
                Y[cluster, :] = np.array([cx, cy])
                r[cluster] = radius + tolerance
            X_not_covered = X[np.sum(distance_matrix(
                X, Y) <= r[None, :], axis=1) < k, :]
            cover_redundancy = 1
            while X_not_covered.shape[0] > 0:
                cover_redundancy += 1
                distances = distance_matrix(X_not_covered, Y)
                closest_idx = np.argsort(distances, axis=1)[
                    :, cover_redundancy - 1]
                for i, index_y in enumerate(closest_idx):
                    current_x = X_not_covered[i, :]
                    index_x = np.where((X == current_x).all(axis=1))[0][0]
                    assignment[index_x, index_y] = 1
                    cluster_points = X[assignment[:, index_y] == 1, :]
                    cx, cy, radius = make_circle(cluster_points)
                    Y[index_y, :] = np.array([cx, cy])
                    r[index_y] = radius + tolerance
                X_not_covered = X[np.sum(distance_matrix(
                    X, Y) <= r[None, :], axis=1) < k, :]
            for cluster in range(m):
                overcovered = np.sum(distance_matrix(
                    X, Y) <= r[None, :], axis=1) > k
                temp = assignment[overcovered, cluster]
                assignment[overcovered, cluster] = 0
                cluster_points = X[assignment[:, cluster] == 1, :]
                if cluster_points.shape[0] == 0:
                    continue
                cx, cy, radius = make_circle(cluster_points)
                Y[cluster, :] = np.array([cx, cy])
                r[cluster] = radius + tolerance
                if np.sum(np.sum(distance_matrix(X, Y) <= r[None, :], axis=1) < k) > 0:
                    assignment[overcovered, cluster] = temp
                    cluster_points = X[assignment[:, cluster] == 1, :]
                    cx, cy, radius = make_circle(cluster_points)
                    Y[cluster, :] = np.array([cx, cy])
                    r[cluster] = radius + tolerance
            current_cost = np.sum(np.pi * r ** 2)
            if current_cost < best_cost:
                best_cost = current_cost
                best_r_opt = r
                best_Y_opt = Y
    return best_r_opt, best_Y_opt

def check_solution(X, k, r, Y, tol=1e-9):
    dist = distance_matrix(X, Y)
    cover = np.sum(dist <= r[None, :] + tol, axis=1)
    return bool(np.all(cover >= k)), int(np.sum(cover < k))

def heuristic(X, k, m, seed=None):
    t0 = time.time()
    if X.ndim != 2 or X.shape[1] != 2 or len(X) != len(k):
        raise ValueError("points must be (n,2) and kappa must have length n")
    if k.max() > m:
        raise ValueError(f"max kappa {k.max()} > m={m}: not enough disks")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    r, Y = iterative_alg(X, m, k)
    elapsed = time.time() - t0
    r = np.asarray(r)
    Y = np.asarray(Y)
    ok, n_bad = check_solution(X, k, r, Y)
    return dict(centers=Y, radii=r, area=float(np.sum(np.pi * r ** 2)),
                time_s=elapsed, feasible=ok, undercovered_points=n_bad)

def solve_gmc(instance, problem):
    with open(f"instances/{instance}/{problem}", "r") as f:
        d = json.load(f)
    pts = d["points"]
    X = np.array([p["coord"] for p in pts], dtype=float)
    k = np.array([int(p["mult"]) for p in pts], dtype=int)
    m = int(d["num_disc"])
    res = heuristic(X, k, m)
    return { 'feasible': res['feasible'], 'area': res['area'], 'time_s': res['time_s'] }