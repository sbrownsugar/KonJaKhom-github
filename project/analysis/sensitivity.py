# -*- coding: utf-8 -*-
"""
sensitivity.py — "คำว่า n=25 พอ มันเปราะแค่ไหน?"
------------------------------------------------------------------------
power_sweep.py บอกว่าที่ n=25 เราจับสัญญาณเจอ 89% ของครั้ง
แต่นั่นอยู่บนสมมติฐานที่ *เราตั้งเอง* ว่าผลของพฤติกรรมแรงแค่ไหน

สคริปต์นี้ถามว่า: ถ้าความจริงโหดกว่าที่คิด power จะเหลือเท่าไหร่

ตัวแปรที่บิด:
  1. ผลจริงอ่อนกว่าที่จำลอง (x1.0 / x0.6 / x0.4)     <- น่าจะเป็นจริงที่สุด
  2. MediaPipe วัดมุมคลาดเคลื่อน (0 / 3 / 6 องศา)    <- งานวิจัยบอก markerless pose มี error ~2-5 องศา
  3. นักเรียนส่วนใหญ่ไม่ปวด (floor effect)             <- ถ้าเด็กสุขภาพดี NRS จะกองอยู่ที่ 0-1
  4. จำนวนฟีเจอร์ 6 vs 4 (ตัด interaction + stress)   <- ฟีเจอร์น้อย = power สูงขึ้น

คำถามที่ต้องตอบ: **n ขั้นต่ำที่ยัง >= 80% power ในแต่ละสถานการณ์คือเท่าไหร่**
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats

ALPHA = 0.05
N_SIMS = 500
N_GRID = (20, 25, 30, 40, 50, 70, 100, 150, 200)

FEAT_6 = ["posture_uc", "asym", "cervical_load", "recovery_def", "stress", "posture_x_load"]
FEAT_4 = ["posture_uc", "asym", "cervical_load", "recovery_def"]


def gen(n, rng, eff=1.0, ang_noise=0.0, healthy=False):
    """สร้างข้อมูล 1 ห้องเรียน. eff = ตัวคูณความแรงของผลจริง"""
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)

    # --- สรีระจริง (ที่ยังไม่ผ่านกล้อง) ---
    fha_true = rng.normal(18, 6, n)
    spa_true = rng.normal(10, 4, n)
    asym_true = np.abs(rng.normal(0, 2.5, n))

    # --- พฤติกรรม ---
    sit_h = np.clip(rng.normal(8, 2, n), 2, 14)
    phone_h = np.clip(rng.normal(4, 1.5, n), 0.5, 10)
    exercise_d = rng.poisson(2, n).astype(float)
    sleep_h = np.clip(rng.normal(6.5, 1.1, n), 3, 10)
    stress = rng.integers(0, 11, n).astype(float)
    bag_kg = np.clip(rng.normal(4, 1.5, n), 0.5, 10)

    # --- ความจริง: NRS ขึ้นกับท่าทาง "จริง" + พฤติกรรม ---
    load = 15 * sit_h + 25 * phone_h
    lin = eff * (0.28 * z(fha_true) + 0.12 * z(spa_true) + 0.10 * z(asym_true)
                 + 0.70 * z(load) + 0.55 * z(-sleep_h) + 0.35 * z(-exercise_d)
                 + 0.40 * z(stress) + 0.45 * z(fha_true) * z(load))
    prob = 1 / (1 + np.exp(-lin))

    base = -1.0 if healthy else 2.0          # healthy = เด็กส่วนใหญ่ไม่ปวด -> กองที่ 0
    nrs = np.clip(np.round(base + 6 * prob + rng.normal(0, 1.2, n)), 0, 10)

    # --- กล้อง/MediaPipe วัดมุมได้ "ไม่ตรงเป๊ะ" ---
    fha = fha_true + rng.normal(0, ang_noise, n)
    spa = spa_true + rng.normal(0, ang_noise, n)
    asym = np.abs(asym_true + rng.normal(0, ang_noise, n))

    cerv = 15.0 * sit_h + 25.0 * phone_h + 1.0 * bag_kg
    posture_uc = z(fha) + z(spa)
    recovery = z(-sleep_h) + z(-exercise_d)

    cols = {
        "posture_uc": posture_uc,
        "asym": z(asym),
        "cervical_load": z(cerv),
        "recovery_def": recovery,
        "stress": z(stress),
        "posture_x_load": z(posture_uc) * z(cerv),
    }
    return cols, nrs


def ftest_p(cols, nrs, feats):
    """partial F-test: พฤติกรรมเพิ่มข้อมูลเหนือท่าทางล้วนไหม"""
    n = len(nrs)
    posture = [f for f in feats if f in ("posture_uc", "asym")]
    if len(feats) - len(posture) < 1:
        return 1.0

    def rss(names):
        A = np.column_stack([np.ones(n)] + [cols[f] for f in names])
        beta, *_ = np.linalg.lstsq(A, nrs, rcond=None)
        r = nrs - A @ beta
        return float(r @ r)

    r0, r1 = rss(posture), rss(feats)
    df1 = len(feats) - len(posture)
    df2 = n - len(feats) - 1
    if df2 <= 0 or r1 <= 0:
        return 1.0
    f = ((r0 - r1) / df1) / (r1 / df2)
    return stats.f.sf(max(f, 0.0), df1, df2)


def power_of(n, eff, noise, healthy, feats):
    rng = np.random.default_rng(hash((n, int(eff * 100), int(noise), healthy, len(feats))) % (2**32))
    hits = zerofrac = 0
    for _ in range(N_SIMS):
        cols, nrs = gen(n, rng, eff, noise, healthy)
        if nrs.std() < 1e-6:
            continue
        zerofrac += (nrs == 0).mean()
        hits += ftest_p(cols, nrs, feats) < ALPHA
    return 100.0 * hits / N_SIMS, 100.0 * zerofrac / N_SIMS


def min_n(eff, noise, healthy, feats, target=80.0):
    for n in N_GRID:
        p, _ = power_of(n, eff, noise, healthy, feats)
        if p >= target:
            return n, p
    return None, None


SCENARIOS = [
    ("1. ตามที่จำลองไว้เดิม (มองโลกสวย)",        1.00, 0.0, False),
    ("2. มุมวัดคลาด 3 องศา (MediaPipe จริง)",     1.00, 3.0, False),
    ("3. ผลจริงอ่อนกว่า 40%",                     0.60, 0.0, False),
    ("4. ผลจริงอ่อนกว่า 60%",                     0.40, 0.0, False),
    ("5. ผลอ่อน 40% + มุมคลาด 3 องศา  <-- น่าจะจริงที่สุด", 0.60, 3.0, False),
    ("6. เด็กส่วนใหญ่ไม่ปวดเลย (floor effect)",    1.00, 0.0, True),
    ("7. แย่ทุกทาง: อ่อน 60% + คลาด 6 องศา + ไม่ค่อยปวด", 0.40, 6.0, True),
]

print("=" * 108)
print("SENSITIVITY: 'n=25 พอ' มันทนได้แค่ไหน?   (จำลอง %d ห้อง/จุด, ทดสอบ F-test บน NRS 0-10)" % N_SIMS)
print("=" * 108)
print("%-48s | %6s %6s %6s | %-22s" % ("สถานการณ์", "n=25", "n=30", "n=50", "n ขั้นต่ำที่ได้ 80%"))
print("-" * 108)

for name, eff, noise, healthy in SCENARIOS:
    p25, _ = power_of(25, eff, noise, healthy, FEAT_6)
    p30, _ = power_of(30, eff, noise, healthy, FEAT_6)
    p50, _ = power_of(50, eff, noise, healthy, FEAT_6)
    mn, mp = min_n(eff, noise, healthy, FEAT_6)
    tag = "%d คน (%.0f%%)" % (mn, mp) if mn else "เกิน 200 คน — สิ้นหวัง"
    flag = "" if (mn and mn <= 30) else ("  <-- อันตราย" if mn and mn <= 70 else "  <-- ตาย")
    print("%-48s | %5.0f%% %5.0f%% %5.0f%% | %s%s" % (name, p25, p30, p50, tag, flag))

print("-" * 108)
print("\n### ตัดฟีเจอร์เหลือ 4 ตัว (ตัด interaction + stress) ช่วยได้แค่ไหน? ###")
print("%-48s | %-18s | %-18s" % ("สถานการณ์", "6 ฟีเจอร์", "4 ฟีเจอร์"))
print("-" * 108)
for name, eff, noise, healthy in SCENARIOS:
    m6, _ = min_n(eff, noise, healthy, FEAT_6)
    m4, _ = min_n(eff, noise, healthy, FEAT_4)
    s6 = "%d คน" % m6 if m6 else ">200"
    s4 = "%d คน" % m4 if m4 else ">200"
    gain = ""
    if m6 and m4 and m4 < m6:
        gain = "  ประหยัด %d คน" % (m6 - m4)
    print("%-48s | %-18s | %-18s%s" % (name, s6, s4, gain))

print("-" * 108)
_, zf = power_of(30, 1.0, 0.0, True, FEAT_6)
print("\nหมายเหตุ: สถานการณ์ 'floor effect' -> นักเรียน %.0f%% ตอบ NRS = 0 (ไม่ปวดเลย)" % zf)
print("ถ้า pilot 20 คนพบว่าเกิน 60%% ตอบ 0 -> เราอยู่ในสถานการณ์นี้ ต้องเปลี่ยนแผนทันที")
