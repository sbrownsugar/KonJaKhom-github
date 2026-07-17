# -*- coding: utf-8 -*-
"""
power_sweep.py — "ที่ n เท่าไหร่ วิธีไหนพิสูจน์ได้จริง?"
------------------------------------------------------------------
รันหลายร้อยรอบต่อ n แล้วนับว่า "จับความจริงเจอกี่ % ของครั้ง" = POWER

ข้อมูลจำลองมีความจริงใส่ไว้แล้ว (พฤติกรรมมีผลจริง, interaction มีจริง)
ถ้าวิธีไหนจับไม่เจอ = วิธีนั้นใช้ไม่ได้ที่ n นั้น ไม่ใช่ว่าความจริงไม่มี

เทียบ 3 วิธี:
  A) BINARY  LRT (chi2)  : ปวด/ไม่ปวด — พฤติกรรมเพิ่มข้อมูลเหนือท่าทางไหม
  B) CONT    F-test      : NRS 0-10  — พฤติกรรมเพิ่มข้อมูลเหนือท่าทางไหม   <-- ตัวหวัง
  C) ML ชนะ rule ไหม     : เทียบ AUC (รู้อยู่แล้วว่า underpowered แต่วัดให้เห็นตัวเลข)

*** ใช้สูตรวิเคราะห์ (chi2 / F) ไม่ใช่ permutation loop -> เร็วกว่า 100 เท่า CPU แทบไม่ขยับ ***
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, r"D:\Obec\project\analysis")
from evaluate import build_features, rule_baseline, make_fake_data

ALPHA = 0.05
N_SIMS = 400                     # จำลองกี่ห้องเรียนต่อ n
N_LIST = (25, 30, 40, 50, 70, 100, 150)

POSTURE = [0, 1]                 # คอลัมน์ท่าทางล้วน
FULL    = [0, 1, 2, 3, 4]        # + พฤติกรรม
FULL_IX = [0, 1, 2, 3, 4, 5]     # + interaction


def loglik(X, y):
    m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=10000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lrt_p(X, y, small, big):
    """A) BINARY: likelihood-ratio test"""
    stat = 2 * (loglik(X[:, big], y) - loglik(X[:, small], y))
    return stats.chi2.sf(max(stat, 0.0), len(big) - len(small))


def ftest_p(X, nrs, small, big):
    """B) CONTINUOUS: partial F-test ของโมเดลซ้อน (นี่คือ LRT เวอร์ชันต่อเนื่อง)"""
    n = len(nrs)

    def rss(cols):
        A = np.column_stack([np.ones(n), X[:, cols]])
        beta, *_ = np.linalg.lstsq(A, nrs, rcond=None)
        r = nrs - A @ beta
        return float(r @ r)

    rss0, rss1 = rss(small), rss(big)
    df1 = len(big) - len(small)
    df2 = n - len(big) - 1
    if df2 <= 0 or rss1 <= 0:
        return 1.0
    f = ((rss0 - rss1) / df1) / (rss1 / df2)
    return stats.f.sf(max(f, 0.0), df1, df2)


def r2_oos_proxy(X, nrs, cols):
    """R² แบบปรับแล้ว (adjusted) — ประมาณค่าที่คาดหวังนอกกลุ่มตัวอย่าง"""
    n = len(nrs)
    A = np.column_stack([np.ones(n), X[:, cols]])
    beta, *_ = np.linalg.lstsq(A, nrs, rcond=None)
    r = nrs - A @ beta
    ss_res, ss_tot = float(r @ r), float(((nrs - nrs.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    k = len(cols)
    return 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)


print("=" * 100)
print("POWER = จับความจริงที่ใส่ไว้ในข้อมูลเจอกี่ %% ของครั้ง  (จำลอง %d ห้องต่อ n, alpha=%.2f)" % (N_SIMS, ALPHA))
print("เกณฑ์ที่ยอมรับได้ในงานวิจัย: power >= 80%%")
print("=" * 100)
print("%5s %7s %7s | %-26s | %-26s | %-16s" % (
    "n", "ปวด%", "EPV", "A) BINARY: พฤติกรรมเพิ่ม?", "B) NRS 0-10: พฤติกรรมเพิ่ม?", "C) ML ชนะ rule?"))
print("%5s %7s %7s | %8s %8s %8s | %8s %8s %8s | %7s %8s" % (
    "", "", "", "LRT", "+inter", "", "F-test", "+inter", "R2adj", "ชนะ%", "AUC diff"))
print("-" * 100)

for n in N_LIST:
    hit_lrt = hit_lrt_ix = hit_f = hit_f_ix = 0
    ml_wins = 0
    r2s, prevs, epvs, daucs = [], [], [], []
    used = 0

    for s in range(N_SIMS):
        d, y, nrs = make_fake_data(n=n, seed=10_000 + s)
        if y.sum() < 5 or (n - y.sum()) < 5:
            continue                      # ห้องที่ไม่มีความหลากหลายเลย ข้าม
        used += 1
        X, _, _ = build_features(d)
        rule = rule_baseline(d)

        prevs.append(y.mean())
        epvs.append(y.sum() / 6.0)

        # A) BINARY
        hit_lrt    += lrt_p(X, y, POSTURE, FULL) < ALPHA
        hit_lrt_ix += lrt_p(X, y, FULL, FULL_IX) < ALPHA

        # B) CONTINUOUS
        hit_f    += ftest_p(X, nrs, POSTURE, FULL) < ALPHA
        hit_f_ix += ftest_p(X, nrs, FULL, FULL_IX) < ALPHA
        r2s.append(r2_oos_proxy(X, nrs, FULL))

        # C) ML vs rule (in-sample AUC ของ logistic เต็ม — เข้าข้าง ML ด้วยซ้ำ)
        m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(X, y)
        a_ml = roc_auc_score(y, m.predict_proba(X)[:, 1])
        a_rl = roc_auc_score(y, rule)
        daucs.append(a_ml - a_rl)
        ml_wins += (a_ml > a_rl)

    pct = lambda c: 100.0 * c / used
    print("%5d %6.0f%% %7.1f | %7.0f%% %7.0f%% %8s | %7.0f%% %7.0f%% %8.2f | %6.0f%% %+8.3f" % (
        n, 100 * np.mean(prevs), np.mean(epvs),
        pct(hit_lrt), pct(hit_lrt_ix), "",
        pct(hit_f), pct(hit_f_ix), np.mean(r2s),
        pct(ml_wins), np.mean(daucs)))

print("-" * 100)
print("""
วิธีอ่าน:
  * คอลัมน์ A (BINARY)  = ถ้าใช้ label "ปวด/ไม่ปวด" จะพิสูจน์ได้กี่ % ของครั้ง
  * คอลัมน์ B (NRS 0-10) = ถ้าใช้ label "ระดับปวด 0-10" จะพิสูจน์ได้กี่ % ของครั้ง
  * คอลัมน์ C           = ML ชนะกฎ if-else กี่ % ของครั้ง (ค่าบวก = ชนะ)
  * แถวไหนที่คอลัมน์ B >= 80% -> นั่นคือ n ขั้นต่ำที่เราต้องการจริงๆ
""")
