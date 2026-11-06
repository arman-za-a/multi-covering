import random 
import math
import itertools
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.collections import PatchCollection

MULTIPLICATIVE_EPSILON = 1 + 1e-14  # تحمل ضربی برای بررسی داخل بودن نقطه در دایره
TOLERANCE = 1e-9  # تحمل عددی برای شرط پوشش
EPS_IMPROVE = 1e-9  # حداقل بهبود لازم برای پذیرش تغییر
EPS_LOCAL_IMPROVE = 1e-7  # حداقل بهبود لازم برای ادامه جستجوی محلی
EPS_ILS_IMPROVE = 1e-9  # حداقل بهبود لازم برای پذیرش در جستجوی تکرارشونده
ACCEPTANCE_THRESHOLD = 1.003  # آستانه پذیرش جواب کمی بدتر در جستجوی تکرارشونده
RNG = random.Random(0)  # مولد تصادفی سراسری با سید صفر

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

def is_in_circle(c, p):
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * MULTIPLICATIVE_EPSILON

def cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

def make_circle_two_points(points, p, q):
    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q
    for r in points:
        if is_in_circle(circ, r):
            continue
        cross = cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (left is None or cross_product(px, py, qx, qy, c[0], c[1]) > cross_product(px, py, qx, qy, left[0], left[1])):
            left = c
        elif cross < 0.0 and (right is None or cross_product(px, py, qx, qy, c[0], c[1]) < cross_product(px, py, qx, qy, right[0], right[1])):
            right = c
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right

def make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for i, q in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = make_circle_two_points(points[:i + 1], p, q)
    return c

def make_circle(points):
    shuffled = [(float(x), float(y)) for (x, y) in points]
    RNG.shuffle(shuffled)
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = make_circle_one_point(shuffled[: i + 1], p)
    return c

def candidate_disks(points):
    u = len(points)
    cen = [points.copy()]
    rad = [np.zeros(u)]
    if u >= 2:
        ia, ib = np.triu_indices(u, 1)
        cen.append((points[ia] + points[ib]) / 2.0)
        rad.append(np.sqrt(((points[ia] - points[ib]) ** 2).sum(axis=1)) / 2)
    if u >= 3:
        comb = np.array(list(itertools.combinations(range(u), 3)))
        A, B, C = points[comb[:, 0]], points[comb[:, 1]], points[comb[:, 2]]
        a2 = ((B - C) ** 2).sum(axis=1)
        b2 = ((C - A) ** 2).sum(axis=1)
        c2 = ((A - B) ** 2).sum(axis=1)
        acute = (a2 + b2 > c2) & (a2 + c2 > b2) & (b2 + c2 > a2)
        A, B, C = A[acute], B[acute], C[acute]
        if len(A):
            d = 2.0 * (A[:, 0] * (B[:, 1] - C[:, 1]) + B[:, 0] * (C[:, 1] - A[:, 1])
                       + C[:, 0] * (A[:, 1] - B[:, 1]))
            ok = np.abs(d) > 1e-12
            A, B, C, d = A[ok], B[ok], C[ok], d[ok]
            sa, sb, sc = (A ** 2).sum(1), (B ** 2).sum(1), (C ** 2).sum(1)
            ux = (sa * (B[:, 1] - C[:, 1]) + sb * (C[:, 1] - A[:, 1]) + sc * (A[:, 1] - B[:, 1])) / d
            uy = (sa * (C[:, 0] - B[:, 0]) + sb * (A[:, 0] - C[:, 0]) + sc * (B[:, 0] - A[:, 0])) / d
            cc = np.stack([ux, uy], axis=1)
            cen.append(cc)
            rad.append(np.sqrt(((A - cc) ** 2).sum(axis=1)))
    return np.concatenate(cen), np.concatenate(rad)

def minimum_enclosing_circle(points):
    # محاسبه کوچک‌ترین دایره محیطی برای آرایه نقاط
    if len(points) == 1:  # اگر فقط یک نقطه
        return points[0].copy(), 0  # مرکز همان نقطه، شعاع صفر
    if len(points) == 2:  # اگر دو نقطه
        c = (points[0] + points[1]) / 2  # مرکز وسط دو نقطه
        return c, float(np.sqrt(((points[0] - c) ** 2).sum()))  # شعاع فاصله تا مرکز است
    cx, cy, _ = make_circle([(float(x), float(y)) for x, y in points])  # الگوریتم دایره محیطی کمینه
    c = np.array([cx, cy])
    r = float(np.sqrt(((points - c) ** 2).sum(axis=1)).max())  # شعاع واقعی برابر بیشینه فاصله نقاط از مرکز است
    return c, r  # مرکز و شعاع

def distance(A, B):
    # ماتریس فاصله اقلیدسی بین هر سطر از دو ماتریس
    d = A[:, None, :] - B[None, :, :]  # اختلاف برداری همه زوج‌ها
    return np.sqrt((d * d).sum(axis=2))  # نرم اقلیدسی

class Solution:
    def __init__(self, points, kapa, centers, radii):
        self.points = points  # نقاط
        self.kapa = kapa  # تقاضای پوشش هر نقطه
        self.centers = np.array(centers)  # مرکز دیسک‌ها
        self.radii = np.array(radii)  # شعاع دیسک‌ها
        self.Distances = distance(self.points, self.centers)  # فاصله هر نقطه تا هر مرکز
        self.Covered = self.Distances <= self.radii[None, :] + TOLERANCE  # ماتریس بولی: آیا نقطه داخل دیسک است؟
        self.n_covers = self.Covered.sum(axis=1)  # تعداد دیسک‌هایی که هر نقطه را پوشش می‌دهند
        # مقدار بهبود مساحت در هر مرحله
        self.improvements = { "reduce_overcoverage": 0, "shrink_and_recover": 0, "optimize_pairs": 0, "iterated_local_search": 0 }

    def cost(self):
        # هزینه متناسب با شعاع بدون عدد پی
        return float(np.sum(self.radii ** 2))

    def feasible(self):
        # آیا همه نقاط حداقل به تعداد تقاضا پوشش دارند؟
        return bool(np.all(self.n_covers >= self.kapa))

    def copy(self):
        # کپی کامل جواب برای تست تغییرات و بازگردانی
        solution = Solution.__new__(Solution)
        solution.kapa, solution.points = self.kapa, self.points
        solution.centers, solution.radii = self.centers.copy(), self.radii.copy()
        solution.Distances, solution.Covered, solution.n_covers = self.Distances.copy(), self.Covered.copy(), self.n_covers.copy()
        solution.improvements = self.improvements.copy()
        return solution

    def set_disk(self, j, y, r):
        # به‌روزرسانی دیسک با مرکز و شعاع جدید
        self.n_covers -= self.Covered[:, j]  # پوشش قبلی دیسک را کم کن
        self.centers[j] = y  # مرکز جدید
        self.radii[j] = r  # شعاع جدید
        self.Distances[:, j] = np.sqrt(((self.points - self.centers[j]) ** 2).sum(axis=1))  # فاصله‌ها تا مرکز جدید
        self.Covered[:, j] = self.Distances[:, j] <= r + TOLERANCE  # عضویت جدید
        self.n_covers += self.Covered[:, j]  # پوشش جدید را اضافه کن

def initialize_centers(points, weights, n_centers, rng):
    # انتخاب چند مرکز اولیه به روش خوشه‌بندی وزن‌دار
    n = len(points)
    centers = [rng.choice(n, p=weights / weights.sum())]  # اولین مرکز با احتمال متناسب با وزن
    for _ in range(1, n_centers):
        point_nearest_center_distances = np.min(((points[:, None, :] - points[centers][None, :, :]) ** 2).sum(axis=2), axis=1)  # مربع فاصله هر نقطه تا نزدیک‌ترین مرکز انتخاب‌شده
        probabilities = point_nearest_center_distances * weights  # احتمال انتخاب متناسب با فاصله و وزن برای هر نقطه
        if probabilities.sum() == 0:  # اگر همه احتمال‌ها صفر شد
            centers.append(rng.integers(n))  # تصادفی انتخاب کن
        else:
            centers.append(rng.choice(n, p=probabilities / probabilities.sum()))  # انتخاب وزن‌دار بر اساس احتمال
    return centers

def repair_coverage(solution, rng, forbidden=None):
    # نقاط کم‌پوشش را با بزرگ‌کردن کم‌هزینه‌ترین دیسک‌ها ترمیم می‌کند
    cost_increment = 0  # افزایش هزینه ناشی از ترمیم
    while True:
        point_coverage_deficits = solution.kapa - solution.n_covers  # کمبود پوشش هر نقطه
        undercovered_points = np.where(point_coverage_deficits > 0)[0]  # نقاط کم‌پوشش
        if len(undercovered_points) == 0:  # اگر هیچ کمبود پوششی نیست
            return cost_increment
        max_coverage_deficit = point_coverage_deficits[undercovered_points].max()  # بیشینه کمبود پوشش
        most_undercovered_points = undercovered_points[point_coverage_deficits[undercovered_points] == max_coverage_deficit]  # نقاطی با بیشینه کمبود پوشش
        most_undercovered_point = rng.choice(most_undercovered_points)  # یکی از نقاط با بیشینه کمبود پوشش را تصادفی انتخاب کن
        disk_cost_increments = solution.Distances[most_undercovered_point] ** 2 - solution.radii ** 2  # افزایش هزینه اگر شعاع دیسک‌ها تا نقطه با بیشینه کمبود پوشش بزرگ شود
        disk_cost_increments[solution.Covered[most_undercovered_point]] = np.inf  # دیسک‌هایی که نقطه با بیشینه کمبود پوشش را پوشش می‌دهند نیاز نیست بزرگ شوند
        if forbidden is not None:
            disk_cost_increments[forbidden] = np.inf  # دیسک ممنوعه را در نظر نگیر
        min_cost_disk = np.argmin(disk_cost_increments)  # کم‌هزینه‌ترین دیسک
        if not np.isfinite(disk_cost_increments[min_cost_disk]):  # اگر هیچ دیسک مجازی نبود
            raise RuntimeError("cannot repair: not enough disks")
        cost_increment += disk_cost_increments[min_cost_disk]  # افزایش هزینه را اضافه کن
        solution.set_disk(min_cost_disk, solution.centers[min_cost_disk], solution.Distances[most_undercovered_point, min_cost_disk])  # شعاع کم‌هزینه‌ترین دیسک را تا پوشش نقطه با بیشینه کمبود پوشش بزرگ کن

def reduce_overcoverage(solution, rng):
    #  شعاع دیسک‌هایی که پوشش اصافی دارند را کم می‌کند
    m = len(solution.centers)  # تعداد دیسک‌ها
    disks = list(range(m))  # ترتیب دیسک‌ها
    rng.shuffle(disks)  # تصادفی برای تنوع
    for disk in disks:
        critical_points = solution.Covered[:, disk] & (solution.n_covers == solution.kapa)  # نقاط بحرانی: نقاطی که دیسک پوشش می‌دهد و پوشششان دقیقاً به تقاضا می‌رسد
        if not critical_points.any():  # اگر دیسک برای هیچ نقطه بحرانی نیست
            if solution.radii[disk] > 0:  # اگر شعاع مثبت است
                solution.improvements["reduce_overcoverage"] += round(np.pi * (solution.radii[disk] ** 2), 2)  # ثبت بهبود
                solution.set_disk(disk, solution.centers[disk], 0)  # شعاع دیسک را صفر کن
            continue
        center, radius = minimum_enclosing_circle(solution.points[critical_points])  # دایره محیطی کمینه نقاط بحرانی
        if radius < solution.radii[disk] - EPS_IMPROVE:  # اگر شعاع کوچک‌تر شد
            solution.improvements["reduce_overcoverage"] += round(np.pi * (solution.radii[disk] ** 2 - radius ** 2), 2)  # ثبت بهبود
            solution.set_disk(disk, center, radius)  # دیسک را به دایره محیطی کمینه جدید تغییر بده

def shrink_and_recover(solution, rng):
    # یک نقطه مرزی را از دیسک حذف می‌کند، دیسک را کوچک می‌کند و کمبود پوشش را با سایر دیسک‌ها ترمیم می‌کند
    m = len(solution.centers)  # تعداد دیسک‌ها
    disks = list(range(m))  # ترتیب دیسک‌ها
    rng.shuffle(disks)  # تصادفی برای تنوع
    for disk in disks:
        if solution.radii[disk] == 0:  # دیسک با شعاع صفر قابل کوچک‌شدن نیست
            continue
        critical_points = np.where(solution.Covered[:, disk] & (solution.n_covers <= solution.kapa))[0]  # نقاط بحرانی دیسک
        if len(critical_points) == 0:  # اگر نقطه بحرانی ندارد
            continue
        support_points = critical_points[solution.Distances[critical_points, disk] >= solution.radii[disk] - TOLERANCE]  # نقاط پشتیبان: نقاط بحرانی مرزی که شعاع دیسک را تعیین می‌کنند
        for support_point in support_points:  # هر نقطه پشتیبان را امتحان کن
            other_critical_points = critical_points[critical_points != support_point]  # نقاط بحرانی به‌جز نقطه پشتیبان
            if len(other_critical_points) == 0:  # اگر نقطه بحرانی‌ای نماند
                center, radius = solution.centers[disk], 0  # شعاع دیسک را صفر کن
            else:
                center, radius = minimum_enclosing_circle(solution.points[other_critical_points])  # دایره محیطی کمینه نقاط بحرانی به‌جز نقطه پشتیبان
            if radius >= solution.radii[disk] - TOLERANCE:  # اگر شعاع کوچک نشد
                continue
            new_solution = solution.copy()  # یک جواب آزمایشی
            new_solution.set_disk(disk, center, radius)  # دیسک را کوچک کن
            try:
                cost_increment = repair_coverage(new_solution, rng, forbidden=disk)  # کمبود پوشش دیسک را با دیسک‌های دیگر ترمیم کن
            except RuntimeError:
                continue
            if (radius ** 2 - solution.radii[disk] ** 2) + cost_increment < -EPS_IMPROVE:  # اگر هزینه کم شد
                solution.improvements["shrink_and_recover"] += round(np.pi * (solution.radii[disk] ** 2 - radius ** 2 - cost_increment), 2)  # ثبت بهبود
                solution.centers, solution.radii = new_solution.centers, new_solution.radii  # پذیرش جواب
                solution.Distances, solution.Covered, solution.n_covers = new_solution.Distances, new_solution.Covered, new_solution.n_covers
                break

def find_best_pair(points, point_residual_coverages, upper_bound):
    # برای مجموعه نقاط با نیاز باقی‌مانده صفر یا یک یا دو، بهترین دو دیسک از کاندیدها را می‌یابد
    centers1, radii1 = candidate_disks(points)  # دیسک‌های کاندید
    Distances = distance(centers1, points)  # فاصله هر دیسک کاندید تا هر نقطه
    Covered1 = Distances <= radii1[:, None] + TOLERANCE  # ماتریس بولی: آیا دیسک کاندید نقطه را پوشش می‌دهد؟
    points_depending_two_disks = point_residual_coverages == 2  # نقاطی که به هر دو دیسک نیاز دارند
    if points_depending_two_disks.any():  # اگر نقطه ای باشد که به هر دو دیسک نیاز داشته باشد
        promising_disks = radii1 * radii1 < upper_bound  # فقط دیسک‌های کاندیدی که هزینه‌شان از کران بالا کمتر است
        candidates_covered_all_points_depending_two_disks = promising_disks & Covered1[:, points_depending_two_disks].all(axis=1)  # دیسک‌های کاندیدی که همه نقاطی که به هر دو دیسک نیاز دارند را پوشش می‌دهند
        centers1, radii1, Covered1 = centers1[candidates_covered_all_points_depending_two_disks], radii1 [candidates_covered_all_points_depending_two_disks], Covered1[candidates_covered_all_points_depending_two_disks]  # مراکز، شعاع‌ها و ماتریس پوشش برای دیسک‌های کاندید اولی 
    if len(radii1) == 0:  # اگر دیسک کاندید اولی باقی نموند
        return None
    points_depending_one_disk = point_residual_coverages == 1  # نقاطی که فقط به یک دیسک نیاز دارند
    # برای هر کاندید به‌عنوان دیسک اول، مشخص کن دیسک دوم باید کدام نقاط را پوشش دهد
    Disk2_must_cover = points_depending_two_disks[None, :] | (points_depending_one_disk[None, :] & ~Covered1)  # ماتریس بولی: آیا دیسک دوم باید نقطه را علاوه بر دیسک کاندید 1 پوشش بده ؟
    Point_distances = distance(points, points)  # فاصله بین نقاط
    Disk2_point_pairs = Disk2_must_cover[:, :, None] & Disk2_must_cover[:, None, :]  # زوج نقاطی که هر دو باید توسط دیسک دوم پوشش داده شوند
    disk1_diameters = np.where(Disk2_point_pairs, Point_distances[None, :, :], 0).max(axis=(1, 2))  # قطر نقاط باقی‌مانده برای هر دیسک کاندید اولی
    lower_bounds = radii1 ** 2 + (disk1_diameters / 2) ** 2  # کران پایین هزینه دو دیسک
    disks = np.argsort(lower_bounds, kind="stable")  # مرتب‌سازی بر اساس کران پایین
    best_cost = upper_bound  # بهترین هزینه فعلی
    best_pair = None  # بهترین زوج
    seen = set()  # برای حذف تکراری
    for disk1 in disks:
        if lower_bounds[disk1] >= best_cost:  # اگر کران پایین از بهترین بدتر است، بقیه هم بدترند
            break
        key = Disk2_must_cover[disk1].tobytes()
        if key in seen:
            continue
        seen.add(key)
        center1 = centers1[disk1].copy()  # مرکز دیسک اول
        radius1 = radii1[disk1]  # شعاع دیسک اول
        if not Disk2_must_cover[disk1].any():  # اگر دیسک دوم لازم نیست
            cost = radius1 ** 2  # هزینه فقط دیسک اول
            if cost < best_cost:  # اگر هزینه از بهترین هزینه فعلی بهتر شد
                best_cost, best_pair = cost, (center1, radius1, None, 0) # نیازی به دیسک دوم نیست
            continue
        center2, radius2 = minimum_enclosing_circle(points[Disk2_must_cover[disk1]])  # دایره محیطی کمینه نقاط باقی‌مانده برای دیسک دوم
        cost = radius1 ** 2 + radius2 ** 2  # هزینه کل دو دیسک
        if cost < best_cost:  # اگر هزینه از بهترین هزینه فعلی بهتر شد
            best_cost, best_pair = cost, (center1, radius1, center2, radius2)  # بهترین هزینه و بهترین زوج مشخص شدند
    if best_pair is None:  # اگر زوج بهتری پیدا نشد
        return None
    return best_cost, best_pair  # هزینه و مشخصات دو دیسک

def optimize_pairs(solution, rng):
    # بهینه‌سازی دوبه‌دو دیسک‌های نزدیک
    m = len(solution.centers)  # تعداد دیسک‌ها
    n_pairs = 3 * m  # تعداد زوج‌های نزدیک برای بررسی
    Disk_distances = distance(solution.centers, solution.centers)  # فاصله مراکز دیسک‌ها
    rows, cols = np.triu_indices(m, 1)  # اندیس زوج‌های متمایز
    closests = np.argsort(Disk_distances[rows, cols], kind="stable")[:n_pairs]  # نزدیک‌ترین زوج‌ها
    pairs = [(rows[closest], cols[closest]) for closest in closests]  # لیست زوج‌ها
    rng.shuffle(pairs)  # ترتیب تصادفی
    for (disk_i, disk_j) in pairs:
        old_cost = solution.radii[disk_i] ** 2 + solution.radii[disk_j] ** 2  # هزینه قبلی این دو دیسک
        if old_cost <= EPS_IMPROVE:
            continue
        # نیاز باقی‌مانده هر نقطه به این دو دیسک، حداکثر ۲
        point_residual_coverages = np.clip(solution.kapa - (solution.n_covers - solution.Covered[:, disk_i] - solution.Covered[:, disk_j]), 0, 2)
        dependent_points = np.where(point_residual_coverages > 0)[0]  # نقاطی که حداقل به یکی از این دو دیسک نیاز دارند
        if len(dependent_points) == 0:  # اگر هیچ نقطه‌ای به این دو نیاز ندارد
            solution.improvements["optimize_pairs"] += round(np.pi * old_cost, 2)  # ثبت بهبود
            solution.set_disk(disk_i, solution.centers[disk_i], 0)  # شعاع هر دو را صفر کن
            solution.set_disk(disk_j, solution.centers[disk_j], 0)
            continue
        if len(dependent_points) > 28:  # اگر نقاط زیاد باشند، محاسبه سنگین است
            continue
        best_pair = find_best_pair(solution.points[dependent_points], point_residual_coverages[dependent_points], old_cost - EPS_IMPROVE)  # بهترین دو دیسک
        if best_pair is None:
            continue
        _, (center1, radius1, center2, radius2) = best_pair  # مراکز و شعاع‌های دو دیسک جدید
        new_solution = solution.copy()  # جواب آزمایشی
        new_solution.set_disk(disk_i, center1, radius1)  # دیسک اول را جایگزین کن
        if center2 is None:  # اگر دیسک دوم لازم نیست
            new_solution.set_disk(disk_j, new_solution.centers[disk_j], 0)  # شعاع دیسک 2 را صفر کن
        else:
            new_solution.set_disk(disk_j, center2, radius2)  # دیسک دوم را جایگزین کن
        if new_solution.feasible() and new_solution.cost() < solution.cost() - EPS_IMPROVE / 2:
            solution.improvements["optimize_pairs"] += round(np.pi * (solution.cost() - new_solution.cost()), 2)  # ثبت بهبود
            solution.centers, solution.radii = new_solution.centers, new_solution.radii  # پذیرش جواب
            solution.Distances, solution.Covered, solution.n_covers = new_solution.Distances, new_solution.Covered, new_solution.n_covers

def local_search(solution, rng):
    # جستجوی محلی
    while True:
        start_cost = solution.cost()  # هزینه ابتدایی
        reduce_overcoverage(solution, rng) # کاهش پوشش اضافی
        shrink_and_recover(solution, rng)  # تلاش برای کوچک‌کردن با ترمیم پوشش
        optimize_pairs(solution, rng)  # بهینه‌سازی دوبه‌دو
        if solution.cost() > start_cost - EPS_LOCAL_IMPROVE * max(1, start_cost):  # اگر بهبود کافی نبود
            break
    return solution

def perturb_solution(solution, rng):
    # جابه‌جایی یک دیسک و سپس شکستن بزرگترین دیسک به دو دیسک
    m = len(solution.centers)  # تعداد دیسک‌ها
    n_disks = min(m, 25)  # تعداد دیسک‌های تصادفی که باید برسی شوند
    disks = rng.choice(m, size=n_disks, replace=False) # دیسک‌های تصادفی
    disk_exclusion_costs = []  # هزینه حذف هر دیسک
    for disk in disks:
        new_solution = solution.copy()  # جواب آزمایشی
        new_solution.set_disk(disk, new_solution.centers[disk], 0)  # شعاع دیسک را صفر کن
        try:
            cost_increment = repair_coverage(new_solution, rng, forbidden=disk)  # کمبود پوشش دیسک را با دیسک‌های دیگر ترمیم کن
        except RuntimeError:
            continue
        disk_exclusion_costs.append((cost_increment - solution.radii[disk] ** 2, disk))  # هزینه حذف دیسک را اضافه کن
    if not disk_exclusion_costs:
        return solution
    disk_exclusion_costs.sort()  # هزینه های حذف دیسک‌ها را مرتب کن
    chosen_disk = disk_exclusion_costs[rng.integers(min(3, len(disk_exclusion_costs)))][1]  # یکی از بهترین‌ها را تصادفی انتخاب کن
    solution.set_disk(chosen_disk, solution.centers[chosen_disk], 0)  # شعاع دیسک انتخاب‌شده را صفر کن
    repair_coverage(solution, rng, forbidden=chosen_disk)  # ترمیم پوشش
    costs = solution.radii ** 2  # هزینه‌ها
    costs[chosen_disk] = -1  # دیسک  انتخاب‌شده را برای انتخاب بزرگ‌ترین دیسک کنار بگذار
    max_cost_disk = np.argmax(costs)  # بزرگ‌ترین دیسک
    critical_points = np.where(solution.Covered[:, max_cost_disk] & (solution.n_covers <= solution.kapa))[0]  # نقاط بحرانی بزرگترین دیسک
    if len(critical_points) < 2:
        return solution
    critical_points = solution.points[critical_points]  #  مختصات نقاط بحرانی
    Critical_point_distances = distance(critical_points, critical_points)  # فاصله بین نقاط بحرانی
    max_distance_pair = np.unravel_index(np.argmax(Critical_point_distances), Critical_point_distances.shape)  #  دورترین زوج نقاط بحرانی
    seeds = np.array([critical_points[max_distance_pair[0]], critical_points[max_distance_pair[1]]])  # دو سید اولیه
    for _ in range(3):
        labels = (distance(critical_points, seeds)).argmin(axis=1)  # تخصیص به نزدیک‌ترین سید
        for g in (0, 1):
            if (labels == g).any():
                seeds[g] = critical_points[labels == g].mean(axis=0)  # به‌روزرسانی سید
    labels = (distance(critical_points, seeds)).argmin(axis=1)  # تخصیص نهایی
    if (labels == 0).all() or (labels == 1).all():  # اگر تقسیم نشد
        return solution
    center1, radius1 = minimum_enclosing_circle(critical_points[labels == 0])  # دایره محیطی کمینه خوشه اول
    center2, radius2 = minimum_enclosing_circle(critical_points[labels == 1])  # دایره محیطی کمینه خوشه دوم
    solution.set_disk(max_cost_disk, center1, radius1)  # بزرگ‌ترین دیسک را با خوشه اول جایگزین کن
    solution.set_disk(chosen_disk, center2, radius2)  # دیسک انتخاب‌شده را با خوشه دوم جایگزین کن
    repair_coverage(solution, rng)  # ترمیم پوشش نهایی
    return solution

def iterated_local_search(points, kapa, m, best_solution, rng, n_iterations):
    # جستجوی محلی تکرارشونده: هر دور جواب را خراب می‌کند و دوباره بهینه می‌کند
    iteration = 0
    current_solution = best_solution.copy()  # جواب فعلی
    no_improvement = 0  # شمارنده عدم بهبود
    while iteration < n_iterations:
        iteration += 1
        solution = current_solution.copy()  # جواب جدید
        perturb_solution(solution, rng)  # جابه‌جایی و شکستن
        local_search(solution, rng)  # جستجوی محلی روی جواب جدید
        if solution.cost() < best_solution.cost() - EPS_ILS_IMPROVE:  # اگر جواب جدید از بهترین جواب بهتر بود
            solution.improvements["iterated_local_search"] += round(np.pi * (best_solution.cost() - solution.cost()), 2)  # ثبت بهبود
            best_solution = solution.copy()  # بهترین جواب رو به جواب جدید به‌روزرسانی کن
            no_improvement = 0
        else:
            no_improvement += 1
        if solution.cost() <= current_solution.cost() * ACCEPTANCE_THRESHOLD:
            current_solution = solution  # پذیرش جواب جاری حتی اگر کمی بدتر باشد
        if no_improvement >= 40:
            current_solution, no_improvement = best_solution.copy(), 0  # شروع دوباره از بهترین جواب
    return best_solution

def construct_start(points, kapa, m, rng):
    # ساخت جواب اولیه
    centers = initialize_centers(points, kapa, m, rng)  # انتخاب چند مرکز با وزن تقاضا
    solution = Solution(points, kapa, points[centers], np.zeros(m))  # جواب با شعاع صفر
    repair_coverage(solution, rng)  # بزرگ‌کردن دیسک‌ها برای پوشش
    return solution

def new_heuristic(points, kapa, m, n_starts = 3, n_iterations = 5, seed=0):
    # هیورستیک جدید
    t0 = time.time()  # زمان شروع
    RNG.seed(seed)  # سید مولد تصادفی سراسری
    rng = np.random.default_rng(seed)  # سید مولد تصادفی آرایه
    if kapa.max() > m:
        raise ValueError("infeasible")  # اگر تقاضای یک نقطه از تعداد دیسک‌ها بیشتر باشد ناشدنی است
    if kapa.sum() <= m:
        # حالت خاص: تعداد کل نیاز پوشش از تعداد دیسک‌ها کمتر است؛ با دیسک‌های شعاع صفر همه را پوشش بده
        centers = np.repeat(points, kapa, axis=0)  # هر نقطه را به تعداد تقاضا تکرار کن
        if len(centers) < m:
            centers = np.vstack([centers, np.repeat(points[:1], m - len(centers), axis=0)])  # تا رسیدن به تعداد دیسک‌ها با تکرار نقطه اول پر کن
        radii = np.zeros(m)  # همه شعاع‌ها صفر
        return dict(
            feasible=True,  # feasible بودن
            centers=centers,  # مراکز
            radii=radii,  # شعاع‌ها
            area=0,  # مساحت صفر
            improvements={},  # بدون نیاز به بهبود
            time_s=time.time() - t0  # زمان اجرا
        )
    best_solution = None  # بهترین جواب
    for _ in range(n_starts):  # چند شروع تصادفی
        start_solution = construct_start(points, kapa, m, rng)  # جواب اولیه
        local_search(start_solution, rng)  # جستجوی محلی
        if best_solution is None or start_solution.cost() < best_solution.cost():
            best_solution = start_solution.copy()  # به‌روزرسانی بهترین جواب
    best_solution = iterated_local_search(points, kapa, m, best_solution, rng, n_iterations)  # جستجوی محلی تکرارشونده 
    feasible, centers, radii, area, improvements = best_solution.feasible(), best_solution.centers, best_solution.radii, np.pi * best_solution.cost(), best_solution.improvements  # خروجی نهایی
    return dict(
        feasible=feasible,  # feasible بودن
        centers=centers,  # مراکز
        radii=radii,  # شعاع‌ها
        area= area,  # مساحت
        improvements=improvements,  # بهبود‌ها
        time_s=time.time() - t0  # زمان اجرا
    )

def plot_solution(points, kapa, centers, radii, instance, problem):
    # رسم نقاط به‌همراه تقاضای پوشش و دیسک‌های جواب
    fig, ax = plt.subplots(figsize=(10, 10))
    circles = [
        MplCircle((float(cx), float(cy)), float(r), fill=False)
        for (cx, cy), r in zip(centers, radii)
    ]
    if circles:
        collection = PatchCollection(circles, facecolor="blue", edgecolor="blue", alpha=0.2, linewidth=1)
        ax.add_collection(collection)
    ax.scatter(
        centers[:, 0], 
        centers[:, 1],
        marker="o", 
        c="red", 
        s=15, 
        label="centers", 
        zorder=3
    )
    ax.scatter(
        points[:, 0], 
        points[:, 1],
        color="black", 
        s=5, 
        label="points", 
        zorder=4
    )
    Distances = distance(points, centers)
    n_covers = (Distances <= radii[None, :] + TOLERANCE).sum(axis=1)
    for (x, y), k, n_cover in zip(points, kapa, n_covers):
        ax.annotate(
            int(k), 
            (x, y), 
            xytext=(3, 3),
            textcoords="offset points", 
            fontsize=8, 
            color="black"
        )
        if n_cover >= k:
            ax.annotate(
                "✓", 
                (x, y), 
                xytext=(6, 3),
                textcoords="offset points", 
                fontsize=8, 
                color="green"
            )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{instance} / {problem}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(f"plots/{instance}/{problem}.png", dpi=150)

def solve_gmc(instance, problem):
    # خواندن ‌دیتاست و اجرای هیوریستیک
    with open(f"instances/{instance}/{problem}", "r") as dc:
        dataset = json.load(dc)
    points = np.array([row["coord"] for row in dataset["points"]], dtype=float)  # مختصات نقاط
    kapa = np.array([int(row["mult"]) for row in dataset["points"]], dtype=int)  # تقاضای پوشش هر نقطه
    m = int(dataset["num_disc"])  # تعداد دیسک‌ها
    result = new_heuristic(points, kapa, m)  # اجرای هیوریستیک جدید
    plot_solution(points, kapa, result["centers"], result["radii"], instance, problem)
    disks = [
        {"center": (float(center[0]), float(center[1])), "radius": float(radius)}
        for center, radius in zip(result["centers"], result["radii"])
    ]
    return dict(
        feasible=result["feasible"],
        disks=disks,
        area=result["area"],
        time_s=result["time_s"],
        improvements=result["improvements"]
    )