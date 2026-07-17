# -*- coding: utf-8 -*-
"""
power_lambda_sensitivity.py — power ของ NRS 0-10 ขึ้นกับ "ความแรงของผลจริง" แค่ไหน
=============================================================================
power_compare.py ใช้ λ_gen = 0.6 (ผลจริงระดับอนุรักษ์นิยม) แล้วได้ว่า NRS ต้องใช้ n=50
แต่ power_sweep.py เดิมได้ n=25 พอ -> เพราะเดิม "ปลูกผลไว้แรงกว่า" (มีท่าทาง+interaction)

ไฟล์นี้ตอบ: ถ้าผลจริงแรงเท่าไหร่ n=25/30/40/50 ถึงจะพอ (power>=80%)
วัด NRS อย่างเดียว (เร็ว ไม่ใช้ logistic) — สร้างข้อมูลอิงความชุกไทย + β งานวิจัย เหมือน power_compare
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats

FACTORS = [
    ("hx", 0.42, 1.330), ("stress", 0.40, 0.740), ("lowex", 0.60, 0.630),
    ("female", 0.52, 0.525), ("sleep", 0.753, 0.457), ("sit", 0.80, 0.230),
    ("bmi", 0.19, 0.166),
]
PHONE_MEAN, PHONE_SD, PHONE_BETA = 4.1, 1.8, 0.215
BASE_PREV = 0.42
N_LIST = [25, 30, 40, 50, 70]
LAMBDAS = [0.6, 0.8, 1.0, 1.2]
N_SIMS = 1000


def pop_mu(seed=1, big=300_000):
    rng = np.random.default_rng(seed)
    r = np.zeros(big)
    for _, p, b in FACTORS:
        r += b * rng.binomial(1, p, big)
    r += PHONE_BETA * np.minimum(np.clip(rng.normal(PHONE_MEAN, PHONE_SD, big), 0, 12), 6)
    return float(r.mean())


def nrs_power(n, lam, mu, n_sims=N_SIMS, seed0=200_000):
    hits = 0
    for s in range(n_sims):
        rng = np.random.default_rng(seed0 + s)
        feats, risk = [], np.zeros(n)
        for _, p, b in FACTORS:
            x = rng.binomial(1, p, n).astype(float)
            feats.append(x); risk += b * x
        phone6 = np.minimum(np.clip(rng.normal(PHONE_MEAN, PHONE_SD, n), 0, 12), 6)
        feats.append(phone6); risk += PHONE_BETA * phone6
        b0 = np.log(BASE_PREV / (1 - BASE_PREV))
        prob = 1.0 / (1.0 + np.exp(-(b0 + lam * (risk - mu))))
        nrs = np.clip(np.round(1 + 8 * prob + rng.normal(0, 1.5, n)), 0, 10)
        # noise 2 คอลัมน์ (ท่าทาง β=0)
        X = np.column_stack(feats + [rng.normal(0, 1, n), rng.normal(0, 1, n)])
        A = np.column_stack([np.ones(n), X])
        beta, *_ = np.linalg.lstsq(A, nrs, rcond=None)
        r = nrs - A @ beta
        rss1 = float(r @ r); rss0 = float(((nrs - nrs.mean()) ** 2).sum())
        df1, df2 = X.shape[1], n - X.shape[1] - 1
        if df2 <= 0 or rss1 <= 0:
            continue
        f = ((rss0 - rss1) / df1) / (rss1 / df2)
        hits += stats.f.sf(max(f, 0.0), df1, df2) < 0.05
    return 100.0 * hits / n_sims


def main():
    mu = pop_mu()
    print("=" * 78)
    print("power ของ NRS 0-10 (%%) ตาม 'ความแรงของผลจริง' (λ_gen) — จำลอง %d รอบ/ช่อง" % N_SIMS)
    print("λ สูง = ผลจริงแรง (พฤติกรรมทำนายปวดชัด) · λ ต่ำ = ผลจริงอ่อน (อนุรักษ์นิยม)")
    print("=" * 78)
    print("%5s | %s" % ("n", " ".join("λ=%.1f" % l for l in LAMBDAS)))
    print("-" * 78)
    grid = {}
    for n in N_LIST:
        vals = [nrs_power(n, l, mu) for l in LAMBDAS]
        grid[n] = vals
        print("%5d | %s" % (n, "  ".join("%5.0f%%" % v for v in vals)))
    print("-" * 78)
    print("\nอ่านว่า: ที่แต่ละ λ ต้องใช้ n เท่าไหร่ถึงได้ power>=80%")
    for i, l in enumerate(LAMBDAS):
        nmin = next((n for n in N_LIST if grid[n][i] >= 80), None)
        print("  λ=%.1f -> n>=%s" % (l, nmin if nmin else "> 70"))
    print("""
สรุปแบบซื่อสัตย์:
  * 'n=25 พอ' (power_sweep เดิม) จริงเฉพาะกรณีผลจริงแรง (λ สูง / มี interaction ท่าทาง)
  * ถ้าผลจริงอ่อนแบบอนุรักษ์นิยม (λ=0.6 = ตามสมการเราที่หดค่าไว้แล้ว) ต้องใช้ n~50
  * ข้อสรุปที่ปลอดภัยบนเวที: 'เป้า n=50 · ขั้นต่ำ 40' ไม่ใช่ 'n=25 พอ'
""")


if __name__ == "__main__":
    main()
