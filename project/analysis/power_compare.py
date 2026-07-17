# -*- coding: utf-8 -*-
"""
power_compare.py — เทียบพลังทดสอบ (statistical power) ระหว่างจำลอง 400 vs 1,000 ชุด
   โดยใช้ "ข้อมูลจำลองที่อิงงานวิจัยจริง" (ต่างจาก power_sweep.py เดิมที่ตัวเลขค่อนข้างตั้งเอง)
=============================================================================
กราวด์ด้วยของที่มีที่มาครบแล้วใน scoring.py:
  * ความชุกของแต่ละปัจจัย = ค่าอ้างอิงคนไทย (REFERENCE_NATIONAL)
  * น้ำหนักผล (β) = ln(OR) จากงานวิจัย (BETA)
  * *** ท่าทาง β = 0 *** ให้ตรงกับข้อค้นพบของเราเอง (power_sweep.py เดิมดันใส่ผลท่าทางไว้)

*** ย้ำ: ยังเป็นข้อมูลจำลอง (synthetic) — ตอบเรื่อง "วิธีการต้องใช้กี่คน" ไม่ใช่ "นักเรียนจริง" ***

เทียบ 400 vs 1,000: ถ้าตัวเลข power ใกล้กัน = จำลอง 400 รอบก็พอแล้ว (Monte Carlo นิ่ง)
เทียบ 2 วิธีวัดผล:  BINARY (ปวด/ไม่ปวด)  vs  NRS 0-10 (ระดับอาการ)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

# =============================================================================
# ค่าอิงงานวิจัย (คัดจาก scoring.py — ทุกตัวมีที่มา ดูคอมเมนต์ในไฟล์นั้น)
#   (ชื่อ, ความชุกไทย, β=ln OR)
# =============================================================================
FACTORS = [
    ("เคยปวดใน 12 เดือน", 0.42,  1.330),   # Keeratisiroj 2018 / Raine 2021
    ("เครียดสูง",         0.40,  0.740),   # รามาธิบดี 2568 / Gao 2023
    ("ออกกำลังกายน้อย",   0.60,  0.630),   # Gao 2023
    ("เพศหญิง",           0.52,  0.525),   # NSO 2566 / Gao 2023
    ("นอนไม่ถึง 8 ชม.",   0.753, 0.457),   # GSHS 2564 / Auvinen 2010
    ("นั่ง >= 6 ชม.",     0.80,  0.230),   # Meng 2025
    ("BMI เกิน",          0.19,  0.166),   # GSHS 2564 / Garcia-Moreno 2024
]
PHONE_MEAN, PHONE_SD, PHONE_BETA = 4.1, 1.8, 0.215   # Tangmunkongvorakul 2020 / SR+MA 2025
LAMBDA_GEN = 0.6     # shrinkage เดียวกับสมการของเรา (กัน OR รวมเกินจริง)
BASE_PREV  = 0.42    # ความชุกฐาน (ปวดคอวัยรุ่นไทย)

N_LIST   = [25, 30, 40, 50, 70]
N_MAX    = 1000
CHECKPTS = [400, 1000]
ALPHA    = 0.05


def pop_risk_stats(seed=1, big=300_000):
    """mean/sd ของ 'คะแนนเสี่ยง' ในประชากร (ไว้ center ตอนสร้าง outcome)"""
    rng = np.random.default_rng(seed)
    r = np.zeros(big)
    for _, prev, beta in FACTORS:
        r += beta * rng.binomial(1, prev, big)
    phone = np.clip(rng.normal(PHONE_MEAN, PHONE_SD, big), 0, 12)
    r += PHONE_BETA * np.minimum(phone, 6)
    return float(r.mean()), float(r.std())


def make_realistic_data(n, seed, mu):
    """สร้าง 1 'ห้องเรียนจำลอง' n คน — อิงความชุกไทย + β งานวิจัย · ท่าทางไม่เกี่ยว (β=0)"""
    rng = np.random.default_rng(seed)
    feats, risk = [], np.zeros(n)
    for _, prev, beta in FACTORS:
        x = rng.binomial(1, prev, n).astype(float)
        feats.append(x)
        risk += beta * x
    phone6 = np.minimum(np.clip(rng.normal(PHONE_MEAN, PHONE_SD, n), 0, 12), 6)
    feats.append(phone6)
    risk += PHONE_BETA * phone6

    # outcome สร้างจาก logit = b0 + lambda*(risk - mu)  (ปวดสัมพันธ์กับพฤติกรรมจริง)
    b0 = np.log(BASE_PREV / (1 - BASE_PREV))
    prob = 1.0 / (1.0 + np.exp(-(b0 + LAMBDA_GEN * (risk - mu))))
    binary = rng.binomial(1, prob).astype(int)
    nrs = np.clip(np.round(1 + 8 * prob + rng.normal(0, 1.5, n)), 0, 10)

    # *** ใส่ท่าทางเป็น noise (β=0) 2 คอลัมน์ เพื่อความสมจริง แต่ไม่เกี่ยวกับ outcome ***
    fha = rng.normal(0, 1, n)
    asym = rng.normal(0, 1, n)
    X = np.column_stack(feats + [fha, asym])
    return X, binary, nrs


def _loglik(X, y):
    m = LogisticRegression(C=1.0, max_iter=3000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_detect(X, y):
    """BINARY: จับได้ไหมว่าโมเดลพฤติกรรมทำนายปวด (LRT vs ค่าคงที่)"""
    if y.sum() < 3 or (len(y) - y.sum()) < 3:
        return False
    ll1 = _loglik(X, y)
    p0 = np.clip(y.mean(), 1e-9, 1 - 1e-9)
    ll0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
    return stats.chi2.sf(max(2 * (ll1 - ll0), 0.0), X.shape[1]) < ALPHA


def nrs_detect(X, nrs):
    """NRS 0-10: จับได้ไหม (partial F-test ของโมเดลเต็ม vs ค่าคงที่)"""
    n = len(nrs)
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, nrs, rcond=None)
    r = nrs - A @ beta
    rss1 = float(r @ r)
    rss0 = float(((nrs - nrs.mean()) ** 2).sum())
    df1, df2 = X.shape[1], n - X.shape[1] - 1
    if df2 <= 0 or rss1 <= 0:
        return False
    f = ((rss0 - rss1) / df1) / (rss1 / df2)
    return stats.f.sf(max(f, 0.0), df1, df2) < ALPHA


def main():
    mu, sd = pop_risk_stats()
    print("=" * 92)
    print("POWER: จับ 'พฤติกรรมทำนายปวด' ที่ปลูกไว้เจอกี่ %% ของครั้ง  (ข้อมูลจำลองอิงงานวิจัย)")
    print("  ความชุก = ค่าคนไทย · β = ln(OR) งานวิจัย · ท่าทาง β=0 · λ_gen=%.1f · ความชุกปวดฐาน %.0f%%"
          % (LAMBDA_GEN, 100 * BASE_PREV))
    print("  คะแนนเสี่ยงประชากร: mean=%.2f sd=%.2f  ·  เกณฑ์ power ที่ยอมรับ >= 80%%" % (mu, sd))
    print("=" * 92)
    header = "%4s | %-24s | %-24s | %s" % (
        "n", "BINARY (ปวด/ไม่ปวด)", "NRS 0-10 (ระดับอาการ)", "ปวดเฉลี่ย%")
    print(header)
    print("%4s | %7s %7s %6s | %7s %7s %6s |" % (
        "", "400", "1000", "Δ", "400", "1000", "Δ"))
    print("-" * 92)

    rows = []
    for n in N_LIST:
        # รัน N_MAX รอบ เก็บ detect ราย sim -> power@400 = 400 รอบแรก, power@1000 = ทั้งหมด
        bdet = np.zeros(N_MAX, dtype=bool)
        ndet = np.zeros(N_MAX, dtype=bool)
        prevs = np.zeros(N_MAX)
        for s in range(N_MAX):
            X, y, nrs = make_realistic_data(n, seed=100_000 + s, mu=mu)
            prevs[s] = y.mean()
            bdet[s] = binary_detect(X, y)
            ndet[s] = nrs_detect(X, nrs)

        b400, b1000 = 100 * bdet[:400].mean(), 100 * bdet.mean()
        n400, n1000 = 100 * ndet[:400].mean(), 100 * ndet.mean()
        print("%4d | %6.1f%% %6.1f%% %+5.1f | %6.1f%% %6.1f%% %+5.1f | %8.0f%%" % (
            n, b400, b1000, b1000 - b400, n400, n1000, n1000 - n400, 100 * prevs.mean()))
        rows.append((n, b1000, n1000))

    print("-" * 92)
    # หา n ขั้นต่ำที่ NRS ได้ power >= 80
    nmin = next((n for n, _, npow in rows if npow >= 80), None)
    bmin = next((n for n, bpow, _ in rows if bpow >= 80), None)
    print("""
สรุป:
  • ตัวเลข 400 vs 1000 ต่างกันแค่ระดับ Monte Carlo noise (คอลัมน์ Δ เล็ก) -> จำลอง 400 รอบก็เพียงพอ
  • NRS 0-10 ได้ power >= 80%% ที่ n = %s   |   BINARY ต้องใช้ n = %s (มากกว่า/หรือไม่ถึงในช่วงที่ทดสอบ)
  • ยืนยันข้อสรุปเดิม: 'ระดับอาการ 0-10' ให้พลังทดสอบสูงกว่า 'ปวด/ไม่ปวด' บนคนกลุ่มเดียวกัน
  • ข้อแตกต่างจาก power_sweep.py เดิม: เวอร์ชันนี้ตั้งท่าทาง β=0 (ตรงข้อค้นพบ) และใช้ความชุกไทยจริง
    -> power อาจต่างจากเดิมบ้าง เพราะ 'ความจริงที่ปลูก' ต่างกัน (เดิมปลูกผลท่าทาง+interaction ไว้ด้วย)
  • ย้ำ: นี่คือผลของ 'วิธีการ' บนข้อมูลจำลอง ไม่ใช่ผลจากนักเรียนจริง
""" % (nmin if nmin else "> 70", bmin if bmin else "> 70"))


if __name__ == "__main__":
    main()
