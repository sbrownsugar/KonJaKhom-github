# -*- coding: utf-8 -*-
"""
evaluate.py — ชุดวิเคราะห์สำหรับ "ก่อนจะค่อม : กระจกอนาคต"  (n = 40-70)
--------------------------------------------------------------------------
รันได้เลย:  D:\\Obec\\.venv\\Scripts\\python.exe D:\\Obec\\project\\analysis\\evaluate.py

สิ่งที่สคริปต์นี้ทำ (เรียงตามลำดับความสำคัญของงานวิจัย):
  H1  LRT  : พฤติกรรม "เพิ่มข้อมูล" เหนือท่าทางเปล่าไหม   <-- PRIMARY (power ~0.90 ที่ n=50)
  H1b LRT  : interaction ท่าทาง x ภาระสะสม มีจริงไหม (df=1)
  H2  PERM : โมเดลมี signal จริงไหม (ไม่ใช่ฟลุก)          <-- permutation test, valid ที่ n ใดๆ
  H3  BOOT : ML ชนะ rule-based ไหม (paired bootstrap CI ของ "ผลต่าง") <-- underpowered, ต้องรายงานตามจริง
  H4  NB-t : paired t-test แบบแก้ correction (Nadeau-Bengio) บน fold scores
  H5  RIDGE: ทำนาย NRS (0-10) ต่อเนื่อง — ได้ข้อมูลต่อคนมากกว่า binary ~1.6 เท่า

*** กฎ 3 ข้อที่ห้ามแหก ***
 1) ทุกอย่างที่ "เรียนรู้จากข้อมูล" (scaler, การเลือก C) ต้องอยู่ใน Pipeline และอยู่ "ข้างใน" CV
 2) permute label แล้วต้องรัน CV ทั้งท่อใหม่ ไม่ใช่ permute แค่ตอนคิด AUC
 3) 1 คน = 1 แถวเสมอ  ถ้ามีหลายรูป/คน ต้อง groups=student_id และใช้ StratifiedGroupKFold
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     GridSearchCV, cross_val_predict)
from sklearn.metrics import roc_auc_score, r2_score

RNG = np.random.default_rng(42)
N_PERM = 1000      # permutation rounds  -> p ต่ำสุดที่เป็นไปได้ = 1/1001 = 0.000999
N_BOOT = 2000      # bootstrap rounds
CV_REPEATS = 20    # repeated CV: ลด variance ของค่า AUC ที่รายงาน (ไม่ได้เพิ่ม n)


# ============================================================
# 1) FEATURE ENGINEERING  — 6 ฟีเจอร์ ไม่เกินนี้
#    ยุบฟีเจอร์ "ก่อน" เห็น label เสมอ (น้ำหนักมาจากทฤษฎี ไม่ใช่จากข้อมูล) => ไม่ leak
# ============================================================
def build_features(d):
    """d = dict ของ numpy array ความยาว n (1 คน = 1 แถว)"""
    z = lambda v: (v - np.mean(v)) / (np.std(v) + 1e-9)

    # --- ภาระคอสะสม (kg-hour/วัน) จากโมเดลชีวกลศาสตร์ Hansraj 2014 ---
    #   นั่งเรียน/โต๊ะ ~ ก้ม 15-30 deg  -> แรงกดคอ ~ 12-18 kg   (ใช้ 15)
    #   ก้มมือถือ     ~ ก้ม 45-60 deg  -> แรงกดคอ ~ 22-27 kg   (ใช้ 25)
    #   กระเป๋า: น้ำหนักจริง เพิ่มโหลดตรง ๆ ~1 ชม./วัน
    cervical_load = 15.0*d["sit_h"] + 25.0*d["phone_h"] + 1.0*d["bag_kg"]

    # --- คะแนน upper-crossed (FHA และ SPA โหลดบนแกนเดียวกัน -> ยุบเป็น 1) ---
    posture_uc = z(d["fha"]) + z(d["spa"])

    # --- การฟื้นตัว (Raine Study: นอน + ออกกำลังกาย คือตัวทำนายจริง) ---
    recovery_def = z(-d["sleep_h"]) + z(-d["exercise_d"])

    X = np.column_stack([
        posture_uc,                       # f1 ท่าทาง
        z(d["asym"]),                     # f2 ท่าทาง (คนละแกน — ห้ามยุบรวม)
        z(cervical_load),                 # f3 พฤติกรรม: ภาระ
        recovery_def,                     # f4 พฤติกรรม: ฟื้นตัวไม่พอ
        z(d["stress"]),                   # f5 พฤติกรรม: จิตสังคม
        z(posture_uc) * z(cervical_load), # f6 *** INTERACTION หลัก: ท่าแย่ x กดนาน ***
    ])
    names = ["posture_uc", "asym", "cervical_load", "recovery_def", "stress", "posture_x_load"]
    posture_only_idx = [0, 1]             # คอลัมน์ที่เป็น "ท่าทางล้วน"
    return X, names, posture_only_idx


def rule_baseline(d):
    """Baseline #1 = กฎ if-else แบบที่แอปวัดท่าทางทั่วโลกใช้ (นับธงแดงจากมุมล้วน)
       *** นี่คือ 'state of the practice' ไม่ใช่หุ่นฟาง — และมันคือสิ่งที่เราต้องล้ม ***"""
    return ((d["fha"] > 22).astype(float)
          + (d["spa"] > 15).astype(float)
          + (d["asym"] > 4).astype(float))


# ============================================================
# 2) โมเดล  — ridge logistic เท่านั้น. ห้าม XGBoost/RF/SVM-RBF ที่ n=50
# ============================================================
def make_model():
    """ridge logistic + เลือก C ด้วย inner CV  => ทั้งก้อนนี้ถูก fit ใหม่ในทุก fold นอก"""
    base = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                  solver="lbfgs", max_iter=5000)),
    ])
    return GridSearchCV(
        base,
        {"lr__C": [0.01, 0.03, 0.1, 0.3, 1.0]},   # C เล็ก = ลงโทษแรง = ที่ n=50 มักชนะ
        scoring="roc_auc",
        cv=StratifiedKFold(4, shuffle=True, random_state=0),
        refit=True,
    )


def oof_predict(X, y, seed=0, repeats=CV_REPEATS, groups=None):
    """คืน out-of-fold probability เฉลี่ยข้าม repeats. 1 ค่า/คน -> เอาไป pair กับ baseline ได้"""
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for r in range(repeats):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed*1000 + r)
        p = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba", groups=groups)[:, 1]
        acc += p; cnt += 1
    return acc / cnt


# ============================================================
# H2) PERMUTATION TEST  — "โมเดลมี signal จริงหรือฟลุก"
#     สลับ y 1000 รอบ แล้ว "รัน CV ทั้งท่อใหม่" ทุกรอบ
#     p = (1 + #{AUC_perm >= AUC_obs}) / (B + 1)   <- exact, ไม่พึ่ง normality, valid ที่ n=50
# ============================================================
def permutation_test_auc(X, y, n_perm=N_PERM, seed=0, repeats=3):
    obs = roc_auc_score(y, oof_predict(X, y, seed=seed, repeats=repeats))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        yp = rng.permutation(y)                        # <-- สลับ label เท่านั้น
        null[b] = roc_auc_score(yp, oof_predict(X, yp, seed=seed + 7 + b, repeats=1))
    p = (1.0 + np.sum(null >= obs)) / (n_perm + 1.0)   # +1 ทั้งบนล่าง = ห้ามได้ p=0
    return obs, null, p


# ============================================================
# H3) PAIRED BOOTSTRAP CI ของ "ผลต่าง" (ML - rule)   *** หัวใจของข้อ 2 ***
#     resample "คน" (ไม่ใช่ resample คะแนน) แล้วคำนวณ AUC ทั้งสองบน "คนชุดเดียวกัน"
#     -> ความผันผวนจากการสุ่มคน ถูกหักล้างกันไป (common random numbers)
# ============================================================
def paired_bootstrap_delta_auc(y, score_ml, score_rule, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); idx_all = np.arange(len(y))
    d_obs = roc_auc_score(y, score_ml) - roc_auc_score(y, score_rule)

    d_boot, a_boot, b_boot = [], [], []
    for _ in range(n_boot):
        # stratified bootstrap: คุมจำนวน case/control ไม่ให้ fold ไหนว่าง
        i1 = rng.choice(idx_all[y == 1], size=(y == 1).sum(), replace=True)
        i0 = rng.choice(idx_all[y == 0], size=(y == 0).sum(), replace=True)
        i = np.concatenate([i1, i0])
        a = roc_auc_score(y[i], score_ml[i])          # <-- คนชุดเดียวกัน
        b = roc_auc_score(y[i], score_rule[i])        # <-- คนชุดเดียวกัน  => paired
        a_boot.append(a); b_boot.append(b); d_boot.append(a - b)

    d_boot = np.array(d_boot); a_boot = np.array(a_boot); b_boot = np.array(b_boot)
    ci_d   = np.percentile(d_boot, [2.5, 97.5])
    # p แบบ bootstrap ทางเดียว (H1: ML > rule) — pre-register ทิศทางไว้ก่อนได้ ถูกต้องตามหลัก
    p_one  = (1.0 + np.sum(d_boot <= 0)) / (n_boot + 1.0)

    # ---- ตัวเลขที่เอาไปขึ้นสไลด์: paired แคบกว่า unpaired เพราะ cov > 0 ----
    sd_a, sd_b, sd_d = a_boot.std(ddof=1), b_boot.std(ddof=1), d_boot.std(ddof=1)
    r = np.corrcoef(a_boot, b_boot)[0, 1]
    sd_unpaired = np.sqrt(sd_a**2 + sd_b**2)          # ถ้าแกล้งทำเป็นว่าไม่ paired
    return dict(delta=d_obs, ci=ci_d, p_one_sided=p_one,
                sd_paired=sd_d, sd_unpaired_if_ignored=sd_unpaired,
                corr_between_AUCs=r,
                width_paired=ci_d[1]-ci_d[0],
                width_unpaired=2*1.96*sd_unpaired,
                auc_ml=roc_auc_score(y, score_ml), auc_rule=roc_auc_score(y, score_rule),
                ci_ml=np.percentile(a_boot, [2.5, 97.5]), ci_rule=np.percentile(b_boot, [2.5, 97.5]))


# ============================================================
# H4) paired t-test บน fold scores + CORRECTION ของ Nadeau-Bengio
#     *** t-test ธรรมดาบน repeated-CV folds "ผิด" *** เพราะ train set ซ้อนกัน
#     -> variance ถูกประเมินต่ำ -> p เล็กเกินจริง -> Type-I error พุ่งถึง ~30-50%
#     แก้: var_corrected = (1/k + n_test/n_train) * s^2
# ============================================================
def corrected_paired_ttest(fold_diffs, n_test, n_train):
    d = np.asarray(fold_diffs, float); k = len(d)
    m, s2 = d.mean(), d.var(ddof=1)
    var_c = (1.0/k + n_test/n_train) * s2
    if var_c <= 0: return m, np.nan, np.nan
    t = m / np.sqrt(var_c)
    p = stats.t.sf(abs(t), df=k-1) * 2
    return m, t, p


# ============================================================
# H1) LIKELIHOOD-RATIO TEST (nested)  *** อาวุธที่แรงที่สุดที่ n=50 ***
#     ไม่เทียบ AUC (ทิ้งข้อมูล) แต่เทียบ log-likelihood (ใช้ข้อมูลเต็ม)
#     power ที่ n=50 ~0.90  vs  การเทียบ AUC ~0.35
# ============================================================
def _loglik(X, y):
    m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=10000).fit(X, y)  # C ใหญ่ = แทบไม่ลงโทษ
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1-1e-12)
    return float(np.sum(y*np.log(p) + (1-y)*np.log(1-p))), m


def lrt(X, y, cols_small, cols_big, label=""):
    ll0, _ = _loglik(X[:, cols_small], y)
    ll1, m1 = _loglik(X[:, cols_big], y)
    df = len(cols_big) - len(cols_small)
    stat = 2*(ll1 - ll0)
    p = stats.chi2.sf(stat, df)
    # pseudo-R2 (Cox-Snell) ที่เพิ่มขึ้น
    n = len(y)
    r2_add = 1 - np.exp(-2*(ll1-ll0)/n)
    return dict(label=label, chi2=stat, df=df, p=p, delta_R2cs=r2_add,
                coefs=m1.coef_[0], odds=np.exp(m1.coef_[0]))


# ============================================================
# H5) RIDGE บน NRS (0-10)  — outcome ต่อเนื่อง = ข้อมูลต่อคนมากกว่า binary
# ============================================================
def ridge_on_nrs(X, nrs, seed=0, n_perm=500):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("rd", GridSearchCV(Ridge(), {"alpha": np.logspace(-1, 3, 12)},
                                         cv=5, scoring="neg_mean_squared_error"))])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    strat = (nrs > np.median(nrs)).astype(int)   # stratify เพื่อให้ fold สมดุล
    oof = np.zeros(len(nrs)); cnt = np.zeros(len(nrs))
    for tr, te in cv.split(X, strat):
        m = clone(pipe).fit(X[tr], nrs[tr]); oof[te] += m.predict(X[te]); cnt[te] += 1
    oof /= cnt
    r2_obs = r2_score(nrs, oof)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(nrs)
        o = np.zeros(len(yp)); c = np.zeros(len(yp))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(X, (yp>np.median(yp)).astype(int)):
            m = clone(pipe).fit(X[tr], yp[tr]); o[te] += m.predict(X[te]); c[te] += 1
        null.append(r2_score(yp, o/np.maximum(c,1)))
    null = np.array(null)
    p = (1 + np.sum(null >= r2_obs)) / (n_perm + 1)
    return r2_obs, p, np.corrcoef(nrs, oof)[0,1]


# ============================================================
# DEMO / SELF-TEST  — สร้างข้อมูลจำลอง n=50 แล้วรันทุกอย่าง
# ตอนใช้จริง: แทน make_fake_data() ด้วยการอ่าน CSV ของตัวเอง
# ============================================================
def make_fake_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    d = dict(
        fha=rng.normal(18, 6, n), spa=rng.normal(10, 4, n), asym=np.abs(rng.normal(0, 2.5, n)),
        sit_h=np.clip(rng.normal(8, 2, n), 2, 14), phone_h=np.clip(rng.normal(4, 1.5, n), 0.5, 10),
        exercise_d=rng.poisson(2, n), sleep_h=np.clip(rng.normal(6.5, 1.1, n), 3, 10),
        stress=rng.integers(0, 11, n).astype(float), bag_kg=np.clip(rng.normal(4, 1.5, n), 0.5, 10),
    )
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    load = 15*d["sit_h"] + 25*d["phone_h"]
    lin = (-0.3 + 0.28*z(d["fha"]) + 0.12*z(d["spa"]) + 0.10*z(d["asym"])
           + 0.70*z(load) + 0.55*z(-d["sleep_h"]) + 0.35*z(-d["exercise_d"]) + 0.40*z(d["stress"])
           + 0.45*z(d["fha"])*z(load))                       # <-- interaction จริง
    prob = 1/(1+np.exp(-lin))
    y = rng.binomial(1, prob)
    nrs = np.clip(np.round(2 + 6*prob + rng.normal(0, 1.2, n)), 0, 10)
    return d, y.astype(int), nrs


if __name__ == "__main__":
    d, y, nrs = make_fake_data(n=50, seed=3)
    X, names, pos_idx = build_features(d)
    rule = rule_baseline(d)
    n = len(y)
    print(f"n={n}  events={y.sum()}  prevalence={y.mean():.2f}  EPV(6 feat)={y.sum()/6:.1f}")
    print("features:", names, "\n")

    # ---------- H1: LRT ----------
    print("### H1  LRT: พฤติกรรมเพิ่มข้อมูลเหนือท่าทางเปล่าไหม (PRIMARY) ###")
    r = lrt(X, y, cols_small=pos_idx, cols_big=[0,1,2,3,4], label="posture -> posture+behaviour")
    print(f"  chi2({r['df']}) = {r['chi2']:.2f}   p = {r['p']:.4f}   Delta R2_CoxSnell = {r['delta_R2cs']:.3f}")
    r2 = lrt(X, y, cols_small=[0,1,2,3,4], cols_big=[0,1,2,3,4,5], label="+ interaction")
    print(f"### H1b interaction (posture x load), df=1: chi2 = {r2['chi2']:.2f}  p = {r2['p']:.4f}\n")

    # ---------- H2: permutation ----------
    print("### H2  PERMUTATION TEST (B=200 ในเดโม, ใช้จริง B=1000) ###")
    obs, null, p = permutation_test_auc(X, y, n_perm=200, seed=1, repeats=3)
    print(f"  AUC(out-of-fold) = {obs:.3f}   null mean = {null.mean():.3f}   p = {p:.4f}")
    print(f"  (p ต่ำสุดที่ B=1000 ทำได้ = 1/1001 = 0.000999)\n")

    # ---------- H3: paired bootstrap ของผลต่าง ----------
    print("### H3  ML vs RULE-BASED: paired bootstrap CI ของ 'ผลต่าง' ###")
    ml = oof_predict(X, y, seed=1, repeats=CV_REPEATS)
    B = paired_bootstrap_delta_auc(y, ml, rule, n_boot=2000, seed=1)
    print(f"  AUC_ML   = {B['auc_ml']:.3f}  95%CI [{B['ci_ml'][0]:.3f}, {B['ci_ml'][1]:.3f}]")
    print(f"  AUC_RULE = {B['auc_rule']:.3f}  95%CI [{B['ci_rule'][0]:.3f}, {B['ci_rule'][1]:.3f}]   <- CI สองอันนี้ทับกัน!")
    print(f"  >>> dAUC = {B['delta']:+.3f}  95%CI [{B['ci'][0]:+.3f}, {B['ci'][1]:+.3f}]  p(1-sided) = {B['p_one_sided']:.4f}")
    print(f"      SD(paired)={B['sd_paired']:.4f}  vs  SD ถ้าไม่ paired={B['sd_unpaired_if_ignored']:.4f}"
          f"  (corr ระหว่าง AUC สองตัว = {B['corr_between_AUCs']:+.2f})")
    print(f"      ความกว้าง CI: paired={B['width_paired']:.3f}  unpaired={B['width_unpaired']:.3f}"
          f"  -> paired แคบกว่า {(1-B['width_paired']/B['width_unpaired'])*100:.0f}%\n")

    # ---------- H4: Nadeau-Bengio ----------
    print("### H4  paired t-test บน fold AUC + Nadeau-Bengio correction ###")
    diffs = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=7)
    for tr, te in cv.split(X, y):
        m = clone(make_model()).fit(X[tr], y[tr])
        if len(np.unique(y[te])) < 2:   continue
        diffs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:,1]) - roc_auc_score(y[te], rule[te]))
    n_te = n//5; n_tr = n - n_te
    m_, t_, p_ = corrected_paired_ttest(diffs, n_te, n_tr)
    naive = stats.ttest_1samp(diffs, 0)
    print(f"  mean dAUC ต่อ fold = {m_:+.3f}  ({len(diffs)} folds)")
    print(f"  t-test ธรรมดา (ผิด!) : t={naive.statistic:.2f}  p={naive.pvalue:.4f}  <-- อย่ารายงานอันนี้")
    print(f"  Nadeau-Bengio (ถูก) : t={t_:.2f}  p={p_:.4f}"
          f"   (p ถูกดันขึ้น {naive.pvalue and p_/naive.pvalue:.1f} เท่า)\n")

    # ---------- H5: NRS ----------
    print("### H5  RIDGE บน NRS 0-10 (outcome ต่อเนื่อง) ###")
    r2o, pr2, corr = ridge_on_nrs(X, nrs, seed=2, n_perm=200)
    print(f"  out-of-fold R2 = {r2o:.3f}   r = {corr:.3f}   permutation p = {pr2:.4f}")
