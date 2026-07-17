# -*- coding: utf-8 -*-
"""
run_analysis.py — กดปุ่มเดียว ได้ผลครบ ตามแผนที่ล็อกไว้ใน analysis_plan.md
=============================================================================
วิธีใช้ (วันพฤหัส):
    1. ดาวน์โหลดคำตอบจาก Google Forms เป็น .csv
    2. วางไว้ที่  D:\\Obec\\5-เก็บข้อมูล\\responses.csv
    3. รัน:  D:\\Obec\\.venv\\Scripts\\python.exe D:\\Obec\\project\\analysis\\run_analysis.py

*** สคริปต์นี้ทำ "เฉพาะ" สิ่งที่ประกาศไว้ใน analysis_plan.md ***
ไม่มีการลองหลายโมเดลแล้วเลือกอันสวย  ไม่มีการจูนพารามิเตอร์  ไม่มีการเปลี่ยน seed
ถ้าอยากลองอย่างอื่น -> ต้องติดป้ายว่า "exploratory" แยกออกไป
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = r"D:\Obec\5-เก็บข้อมูล\responses.csv"
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- ค่าที่ล็อกไว้ในแผน ห้ามแก้ ----
SEED = 20260714
N_PERM = 1000
ALPHA_RIDGE = 1.0
N_SPLITS, N_REPEATS = 5, 20
PERM_REPEATS = 5                  # ใช้เท่ากันทั้งค่าจริงและค่าเปรียบเทียบ (ห้ามต่างกัน!)
ZERO_FRACTION_TRIGGER = 0.40      # ถ้า NRS=0 เกิน 40% -> เปลี่ยนเป็น Spearman ตามแผน


# =============================================================================
# 1) อ่าน CSV จาก Google Forms แล้วจับคู่คอลัมน์ด้วย "คำสำคัญ"
#    (Google Forms ใช้ข้อความคำถามเต็มเป็นชื่อคอลัมน์ ซึ่งยาวและมีอักขระพิเศษ)
# =============================================================================
KEYS = {
    "code":       ["รหัสห้อง-เลขที่", "รหัสห้อง", "รหัสผู้เข้าร่วม"],   # ใช้จับคู่กับภาพ
    "sex":        ["เพศ"],
    "weight_kg":  ["น้ำหนัก (กก", "น้ำหนัก(กก"],
    "height_cm":  ["ส่วนสูง"],
    "sit_hr":     ["นั่ง"],
    "phone_hr":   ["มือถือ", "ก้มคอ"],
    "exercise_d": ["ออกกำลังกาย"],
    "sleep_hr":   ["นอนคืนละ", "นอน"],
    "stress":     ["ความเครียด"],
    "bag_kg":     ["กระเป๋านักเรียนของคุณหนัก", "กี่กิโลกรัม"],
    "nrs":        ["รุนแรงที่สุด"],
    "sym7":       ["7 วันที่ผ่านมา"],
    "sym12":      ["12 เดือนที่ผ่านมาคุณมีอาการ"],
    "hx_before":  ["ก่อนหน้า 12 เดือน", "นานกว่า 1 ปี"],
}
RED_FLAG_KEYS = ["ตื่นกลางดึก", "ไข้", "ชา", "อุบัติเหตุ", "วินิจฉัย", "3 เดือน"]

# =============================================================================
# *** บั๊กที่ทำให้ "ธงแดงตรวจไม่เจอเลย" แบบเงียบสนิท ***
# เดิมเราเช็ค  .eq("ใช่")  ตัวเดียว
# ถ้า Google Form ตั้งตัวเลือกเป็น "มี / ไม่มี" หรือ "เคย / ไม่เคย" (ซึ่งเป็นไปได้สูงมาก
# เพราะคำถามขึ้นต้นว่า "คุณมีอาการ...") -> จะไม่ match แม้แต่คนเดียว
# -> ธงแดง = 0 คน -> **เด็กที่ควรถูกส่งพบแพทย์ ถูกโยนเข้าโมเดลแทน**
# -> และเราจะไม่มีทางรู้ เพราะ 0 คนดูเหมือนข่าวดี
#
# แก้ 2 ชั้น:
#   1. รับคำตอบ "ใช่" ได้หลายรูปแบบ (และตัดกรณี "ไม่ใช่/ไม่มี/ไม่เคย" ออกก่อนเสมอ)
#   2. ถ้าเจอคอลัมน์ธงแดงแต่ไม่ match ใครเลย -> **เตือนเสียงดัง** ไม่ปล่อยผ่านเงียบ ๆ
# =============================================================================
YES_TOKENS = ("ใช่", "มี", "เคย", "yes", "y", "true", "1")


def is_yes(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    neg = t.str.startswith(("ไม่", "no", "false", "0", "-"))     # ไม่ใช่ / ไม่มี / ไม่เคย
    pos = pd.Series(False, index=t.index)
    for tok in YES_TOKENS:
        pos |= t.eq(tok.lower())
    return pos & ~neg


def find_col(df, needles):
    for n in needles:
        for c in df.columns:
            if n in str(c):
                return c
    return None


def to_num(s):
    """คำตอบอย่าง '7-8 ชม.' หรือ 'ประมาณ 8' -> ดึงตัวเลขตัวแรกออกมา"""
    return pd.to_numeric(
        s.astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


FAKE_MARKER = "__FAKE_TEST_DATA_DO_NOT_USE__"


def load():
    if not os.path.exists(CSV):
        print("!! ยังไม่มีไฟล์:", CSV)
        print("   ดาวน์โหลดคำตอบจาก Google Forms เป็น .csv แล้ววางไว้ที่นั่น")
        sys.exit(1)

    # ---- ยามเฝ้า: กันการรันทับข้อมูลปลอมโดยไม่รู้ตัว ----
    with open(CSV, encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    if FAKE_MARKER in first:
        print("!" * 78)
        print("!! หยุด — ไฟล์นี้ยังเป็นข้อมูลปลอมที่ใช้ทดสอบ ไม่ใช่ข้อมูลจริง")
        print("!! ถ้ารันต่อ คุณจะได้ผลลัพธ์จากตัวเลขที่ AI แต่งขึ้น แล้วเอาไปขึ้นเวที")
        print("!!")
        print("!! ให้ทับไฟล์นี้ด้วย CSV จริงจาก Google Forms ก่อน:")
        print("!!   %s" % CSV)
        print("!" * 78)
        sys.exit(1)

    df = pd.read_csv(CSV)
    print("อ่าน CSV: %d แถว, %d คอลัมน์\n" % (len(df), len(df.columns)))

    cols = {k: find_col(df, v) for k, v in KEYS.items()}
    print("จับคู่คอลัมน์:")
    for k, c in cols.items():
        mark = "OK " if c else "!! "
        print("  %s%-12s -> %s" % (mark, k, (str(c)[:62] if c else "หาไม่เจอ")))
    missing = [k for k, c in cols.items() if not c]
    if missing:
        print("\n!! หาคอลัมน์ไม่เจอ: %s" % missing)
        print("   คอลัมน์ที่มีจริงในไฟล์:")
        for c in df.columns:
            print("     -", str(c)[:88])
        sys.exit(1)

    d = pd.DataFrame()
    for k in ("weight_kg", "height_cm", "sit_hr", "phone_hr",
              "exercise_d", "sleep_hr", "stress", "bag_kg", "nrs"):
        d[k] = to_num(df[cols[k]])
    d["female"] = (df[cols["sex"]].astype(str).str.contains("หญิง")).astype(float)
    d["sym7"] = df[cols["sym7"]].astype(str)
    d["code"] = df[cols["code"]].astype(str).str.strip().str.upper()   # <-- ใช้จับคู่กับภาพ

    # ---- ตัดคนที่ติดธงแดง (ตามแผน ข้อ 7) ----
    rf = pd.Series(False, index=df.index)
    found, dead = [], []
    for needle in RED_FLAG_KEYS:
        c = find_col(df, [needle])
        if c is None:
            continue
        hit = is_yes(df[c])
        rf |= hit
        found.append(c)
        if hit.sum() == 0:
            dead.append((c, sorted(df[c].astype(str).str.strip().unique())[:4]))

    d["red_flag"] = rf.values
    print("\nพบคอลัมน์ธงแดง %d/%d ข้อ -> ติดธงแดง %d คน (ตัดออกจากโมเดล ส่งพบแพทย์)"
          % (len(found), len(RED_FLAG_KEYS), int(rf.sum())))

    # *** ถ้าคอลัมน์มีอยู่ แต่จับ "ใช่" ไม่ได้เลย = สงสัยว่าคำตอบใช้คำอื่น -> ห้ามปล่อยผ่าน ***
    if dead:
        print("\n" + "!" * 74)
        print("!! คอลัมน์ธงแดงต่อไปนี้ **ไม่ match ใครเลยสักคน** — ตรวจสอบด่วน")
        print("!! ถ้าฟอร์มใช้คำตอบแบบอื่น เด็กที่ควรส่งพบแพทย์จะถูกโยนเข้าโมเดลแทน")
        for c, vals in dead:
            print("!!   %-42s คำตอบที่พบจริง: %s" % (str(c)[:42], vals))
        print("!! คำที่สคริปต์รู้จักว่าแปลว่า 'ใช่': %s" % (", ".join(YES_TOKENS)))
        print("!" * 74)
    if len(found) < len(RED_FLAG_KEYS):
        print("!! เตือน: หาคอลัมน์ธงแดงได้ไม่ครบ (%d/%d) -> คัดกรองได้ไม่เต็มที่"
              % (len(found), len(RED_FLAG_KEYS)))
    return d


# =============================================================================
# 2) สร้างฟีเจอร์ตามที่ล็อกไว้ — และต้อง "ไม่ leak"
#
# *** บั๊กที่เกือบปล่อยผ่าน ***
# แผนบอกว่า  recovery_def = z(-ชม.นอน) + z(-วันออกกำลังกาย)
# ถ้าคำนวณ z() จากข้อมูล "ทั้งก้อน" ก่อนแบ่งชุดทดสอบ
#   -> ชุดทดสอบแอบเห็นค่าเฉลี่ย/ส่วนเบี่ยงเบนของตัวเองไปแล้ว = data leakage
#   -> ผลจะดูดีเกินจริง และนี่คือความผิดพลาดที่ทำให้งานวิจัยถูกถอนบ่อยที่สุด
#
# แก้: ทำเป็น Transformer ที่ *เรียนค่าเฉลี่ยจากชุดเทรนเท่านั้น* แล้ววางไว้ใน Pipeline
#      -> ทุก fold คำนวณ z ใหม่จากข้อมูลที่มันเห็นได้เท่านั้น
#
# หมายเหตุ: cervical_load = 15×นั่ง + 25×มือถือ ไม่ leak อยู่แล้ว
#           เพราะ 15/25 มาจากโมเดลชีวกลศาสตร์ (Hansraj 2014) ไม่ได้ fit จากข้อมูลเรา
# =============================================================================
RAW_BEH = ["sit_hr", "phone_hr", "sleep_hr", "exercise_d", "stress"]
# แก้แผนครั้งที่ 1 (14 ก.ค. ก่อนเห็นข้อมูล): ตัด spa ออก
# เพราะ MediaPipe วัด "มุมไหล่ห่อ" ไม่ได้จริง — สิ่งที่วัดได้คือ "มุมเอนลำตัว" ซึ่งคนละอย่าง
# และเพี้ยนตามการเอียงกล้อง -> ใช้เป็นฟีเจอร์ไม่ได้ (ดู analysis_plan.md §4)
RAW_POS = ["fha", "asym"]
POSE_CSV = r"D:\Obec\5-เก็บข้อมูล\pose_angles.csv"


class Composites(BaseEstimator, TransformerMixin):
    """สร้างฟีเจอร์รวมตามแผน โดยเรียนค่าเฉลี่ย/SD จากชุดเทรนเท่านั้น

    ลำดับคอลัมน์ของ X (ห้ามสลับ — transform อ้างตำแหน่งตายตัว):
      with_posture=True : [fha, asym, sit_hr, phone_hr, sleep_hr, exercise_d, stress]
      with_posture=False: [           sit_hr, phone_hr, sleep_hr, exercise_d, stress]
    """

    def __init__(self, with_posture=False):
        self.with_posture = with_posture

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        want = 7 if self.with_posture else 5
        if X.shape[1] != want:
            raise ValueError("Composites: คาด %d คอลัมน์ ได้ %d — ลำดับคอลัมน์ผิด"
                             % (want, X.shape[1]))
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-9
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        z = (X - self.mu_) / self.sd_
        if self.with_posture:
            #        0    1     2    3      4      5   6
            #      [fha, asym, sit, phone, sleep, ex, stress]
            sit, phone, stress = X[:, 2], X[:, 3], X[:, 6]
            return np.column_stack([
                z[:, 0],                        # f1 มุมศีรษะยื่นหน้า (z)
                z[:, 1],                        # f2 asym (z)
                15.0 * sit + 25.0 * phone,      # f3 cervical_load (น้ำหนักจากทฤษฎี ไม่ leak)
                (-z[:, 4]) + (-z[:, 5]),        # f4 recovery_def
                stress,                         # f5 stress
            ])
        #      [sit, phone, sleep, ex, stress]
        sit, phone, stress = X[:, 0], X[:, 1], X[:, 4]
        return np.column_stack([
            15.0 * sit + 25.0 * phone,
            (-z[:, 2]) + (-z[:, 3]),
            stress,
        ])


def build(d, posture: pd.DataFrame | None = None):
    """คืน 'ข้อมูลดิบ' — การรวมฟีเจอร์เกิดข้างใน Pipeline (จะได้ไม่ leak)"""
    if posture is None:
        has_pos = False
    else:
        # อย่าปล่อยให้ผิดแบบเงียบ ๆ — ถ้าจำนวนแถวไม่ตรง แปลว่า join พัง
        if len(posture) != len(d):
            raise ValueError("posture มี %d แถว แต่ d มี %d แถว — การจับคู่รหัสผิดพลาด"
                             % (len(posture), len(d)))
        has_pos = True

    if has_pos:
        X = np.column_stack([posture["fha"].values, posture["asym"].values]
                            + [d[c].values for c in RAW_BEH])
        names = ["posture_fha", "asym", "cervical_load", "recovery_def", "stress"]
        pos_idx, beh_idx = [0, 1], [2, 3, 4]
    else:
        X = np.column_stack([d[c].values for c in RAW_BEH])
        names = ["cervical_load", "recovery_def", "stress"]
        pos_idx, beh_idx = [], [0, 1, 2]
    return X.astype(float), names, pos_idx, beh_idx, has_pos


def model(has_pos=False):
    """ทุกอย่างที่ 'เรียนจากข้อมูล' อยู่ใน Pipeline ทั้งหมด -> fit ใหม่ทุก fold"""
    return Pipeline([
        ("comp", Composites(with_posture=has_pos)),
        ("sc", StandardScaler()),
        ("rd", Ridge(alpha=ALPHA_RIDGE)),
    ])


def oof(X, y, seed=SEED, has_pos=False, n_repeats=N_REPEATS):
    """out-of-fold prediction — ทุกอย่างที่เรียนจากข้อมูล fit ใหม่ทุก fold (ไม่ leak)"""
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=seed)
    for tr, te in cv.split(X):
        m = clone(model(has_pos)).fit(X[tr], y[tr])
        acc[te] += m.predict(X[te]); cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def spearman(y, p):
    return float(stats.spearmanr(y, p).statistic)


def partial_f(X, y, small_idx, big_idx):
    """พฤติกรรมเพิ่มข้อมูลเหนือท่าทางไหม (และกลับกัน)"""
    n = len(y)
    def rss(idx):
        A = np.column_stack([np.ones(n)] + ([X[:, idx]] if idx else []))
        A = np.column_stack([np.ones(n), X[:, idx]]) if idx else np.ones((n, 1))
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        return float(r @ r), A.shape[1]
    r0, k0 = rss(small_idx)
    r1, k1 = rss(big_idx)
    df1, df2 = k1 - k0, n - k1
    if df1 <= 0 or df2 <= 0 or r1 <= 0:
        return np.nan, np.nan, np.nan
    f = ((r0 - r1) / df1) / (r1 / df2)
    p = stats.f.sf(max(f, 0), df1, df2)
    dR2 = (r0 - r1) / float(((y - y.mean()) ** 2).sum())
    return f, p, dR2


# =============================================================================
def main():
    print("=" * 78)
    print("รันตามแผนที่ล็อกไว้ใน analysis_plan.md — seed=%d, ไม่จูน, ไม่เลือกผลที่สวย" % SEED)
    print("=" * 78)

    d = load()
    n_all = len(d)
    d = d[~d["red_flag"]].dropna(subset=["nrs", "sit_hr", "phone_hr", "sleep_hr",
                                         "exercise_d", "stress"]).reset_index(drop=True)
    print("\nเหลือใช้ได้ %d คน (จาก %d) หลังตัดธงแดงและคำตอบไม่ครบ" % (len(d), n_all))

    # ---------- จับคู่กับค่ามุมจากภาพ (ต้องทำ *ก่อน* ดึง y ออกมา) ----------
    pos = None
    if os.path.exists(POSE_CSV):
        p = pd.read_csv(POSE_CSV)
        p["code"] = p["code"].astype(str).str.strip().str.upper()
        if "code" not in d:
            print("!! CSV แบบสอบถามไม่มีคอลัมน์รหัส -> จับคู่กับภาพไม่ได้")
        else:
            m = d[["code"]].merge(p, on="code", how="left")
            keep = m[RAW_POS].notna().all(axis=1).values
            print("จับคู่รหัสกับภาพได้ %d/%d คน (ตัด %d คนที่ไม่มีภาพหรือภาพไม่ผ่านเกณฑ์)"
                  % (int(keep.sum()), len(d), int((~keep).sum())))
            d = d[keep].reset_index(drop=True)          # <-- กรองก่อน แล้วค่อยดึง y
            pos = m.loc[keep, RAW_POS].reset_index(drop=True)
    else:
        print("\n" + "!" * 70)
        print("!! ไม่พบ %s" % POSE_CSV)
        print("!! -> จะวิเคราะห์เฉพาะพฤติกรรม และ **ไม่ทดสอบ S2** ")
        print("!!    ซึ่ง S2 คือข้อค้นพบหลักของโครงงาน (ท่าทางเพิ่มข้อมูลไหม)")
        print("!! -> ให้รัน  python analysis/build_pose_csv.py  ก่อน")
        print("!" * 70)

    if len(d) < 35:
        print("\n!!! n < 35 -> ตามแผน ต้องเปลี่ยนกรอบเป็น 'การศึกษานำร่อง (feasibility)' !!!")
        print("    (ตัดสินใจข้อนี้ไว้ล่วงหน้าแล้ว ไม่ใช่เพราะผลไม่สวย)")

    y = d["nrs"].values.astype(float)

    # =========================================================================
    # *** บั๊ก: เดาสเกลจากค่าที่บังเอิญเจอ ***
    # เดิมเขียนว่า  if y.min() >= 1: y = y - 1   ("สงสัยว่าฟอร์มใช้สเกล 1-10")
    # ปัญหา: ถ้าฟอร์มใช้ 0-10 จริง (ซึ่งเป็นแบบที่เราออกแบบไว้) แต่บังเอิญ
    #        **ไม่มีใครตอบ 0 เลย** -> y.min() = 1 -> สคริปต์ลบ 1 ทิ้งทั้งชุด
    #        -> คนที่ปวดระดับ 1 กลายเป็น 0 = "ไม่ปวดเลย"
    #        -> zero_frac พองขึ้นเอง -> อาจไปทริกเกอร์การเปลี่ยน primary test โดยไม่มีเหตุ
    # (โชคดีที่ Spearman เป็นอันดับ การเลื่อนค่าคงที่ไม่กระทบ r — แต่กระทบ "การตัดสินใจ")
    #
    # แก้: ไม่เดา — แบบสอบถามของเราล็อกไว้แล้วว่าเป็น NRS 0-10 (ข้อ 25)
    #      หน้าที่ของสคริปต์คือ **ตรวจว่าข้อมูลตรงกับที่ล็อกไว้ไหม** ไม่ใช่เดาแล้วแก้ให้เอง
    # =========================================================================
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if lo < 0 or hi > 10:
        print("\n!! NRS อยู่นอกช่วง 0-10 ที่ล็อกไว้ (พบ %.1f ถึง %.1f) — หยุดก่อน" % (lo, hi))
        print("   ตรวจสอบว่าคอลัมน์ที่จับได้ถูกต้องไหม แล้วค่อยรันใหม่")
        sys.exit(1)
    print("NRS อยู่ในช่วง %.0f-%.0f (ตรงกับที่ล็อกไว้: 0-10) — ไม่มีการปรับสเกลใด ๆ" % (lo, hi))

    zero_frac = float((y == 0).mean())
    print("\nระดับอาการ (NRS): เฉลี่ย %.2f, SD %.2f, ตอบ 0 = %.0f%%"
          % (y.mean(), y.std(), 100 * zero_frac))

    if zero_frac > ZERO_FRACTION_TRIGGER:
        print("!! ตอบ 0 เกิน %.0f%% -> ตามแผน primary เปลี่ยนเป็น Spearman/ordinal"
              % (100 * ZERO_FRACTION_TRIGGER))
    if y.std() < 0.5:
        print("!! ระดับอาการแทบไม่มีความหลากหลาย -> โมเดลเรียนอะไรไม่ได้ ต้องรายงานตรงๆ")
        return

    X, names, pos_idx, beh_idx, has_pos = build(d, pos)
    print("\nฟีเจอร์ที่ใช้ (%d ตัว): %s" % (len(names), names))
    if not has_pos:
        print("  (ยังไม่มีข้อมูลภาพ -> วิเคราะห์เฉพาะพฤติกรรม · "
              "S2 'ท่าทางเพิ่มข้อมูลไหม' จะทดสอบเมื่อมีภาพ)")

    # ---------- PRIMARY: permutation test ----------
    print("\n" + "=" * 78)
    print("PRIMARY: โมเดลทำนายระดับอาการได้ดีกว่าความบังเอิญไหม (permutation test B=%d)" % N_PERM)
    print("=" * 78)
    print("  *** ค่าจริงและค่าเปรียบเทียบ ใช้เงื่อนไขเดียวกันเป๊ะ (repeats=%d ทั้งสองฝั่ง) ***"
          % PERM_REPEATS)
    print("      ถ้าไม่เท่ากัน = เหมือนแข่งวิ่งแล้วให้คนหนึ่งออกตัวก่อน -> p ต่ำเกินจริง")

    r_obs = spearman(y, oof(X, y, has_pos=has_pos, n_repeats=PERM_REPEATS))
    print("\n  รันโมเดลจริง: Spearman r = %+.3f" % r_obs)
    print("  กำลังสลับ label %d รอบ (แต่ละรอบรัน cross-validation ใหม่ทั้งท่อ)..." % N_PERM)

    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        yp = rng.permutation(y)
        null[b] = spearman(yp, oof(X, yp, seed=SEED + 1 + b,
                                   has_pos=has_pos, n_repeats=PERM_REPEATS))
        if (b + 1) % 200 == 0:
            print("    ... %d/%d" % (b + 1, N_PERM))

    p_val = (1.0 + np.sum(null >= r_obs)) / (N_PERM + 1.0)
    print("\n  >>> r = %+.3f   ค่าเปรียบเทียบเฉลี่ย = %+.3f   **p = %.4f**"
          % (r_obs, null.mean(), p_val))
    print("  >>> %s" % ("มีสัญญาณจริง (p < 0.05)" if p_val < 0.05 else
                        "ยังพิสูจน์ไม่ได้ (p >= 0.05) — รายงานตรงๆ ตามที่สัญญาไว้"))

    # ---------- SECONDARY (F-test บนฟีเจอร์รวม) ----------
    # การทดสอบเชิงอนุมาน (inference) ใช้ข้อมูลทั้งชุด — ไม่ใช่การประเมินความแม่นในการทำนาย
    Z = Composites(with_posture=has_pos).fit_transform(X)
    f1 = pp1 = dr1 = f2 = pp2 = dr2 = float("nan")
    if has_pos:
        print("\n" + "=" * 78)
        print("SECONDARY (ประกาศทั้งสองทิศไว้ล่วงหน้า — ผลออกทางไหนก็รายงาน)")
        print("=" * 78)
        f1, pp1, dr1 = partial_f(Z, y, pos_idx, pos_idx + beh_idx)
        f2, pp2, dr2 = partial_f(Z, y, beh_idx, pos_idx + beh_idx)
        print("  S1 พฤติกรรมเพิ่มข้อมูลเหนือท่าทางล้วน : F=%.2f  p=%.4f  ΔR²=%.3f" % (f1, pp1, dr1))
        print("  S2 ท่าทางเพิ่มข้อมูลเหนือพฤติกรรมล้วน : F=%.2f  p=%.4f  ΔR²=%.3f" % (f2, pp2, dr2))
        if pp2 >= 0.05:
            print("  >>> S2 เป็น null -> เรา replicate Raine Study สำเร็จในวัยรุ่นไทย")
            print("      นี่คือผลที่เราประกาศไว้ล่วงหน้าว่าจะรายงาน ไม่ว่าออกทางไหน")

    # ---------- BASELINE ----------
    print("\n" + "=" * 78)
    print("BASELINE (ต้องรายงานเสมอ — ห้ามเทียบกับหุ่นฟาง)")
    print("=" * 78)
    print("  B0 ค่าเฉลี่ย (null)        : r = 0.000 (โดยนิยาม)")
    print("  รวมทั้งหมด                : r = %+.3f" % r_obs)
    if has_pos:
        # *** บั๊กที่แก้แล้ว: เดิม B1/B2 ใช้ Z ที่ fit จากข้อมูลทั้งก้อน = leak ***
        # ทำให้ baseline "ได้เปรียบอย่างไม่เป็นธรรม" -> ถ้า baseline ชนะ ก็ไม่รู้ว่าชนะจริงหรือชนะเพราะ leak
        # แก้: ให้ Composites อยู่ใน Pipeline เหมือน model() แล้วค่อยเลือกคอลัมน์
        class PickCols(BaseEstimator, TransformerMixin):
            def __init__(self, cols, with_posture):
                self.cols, self.with_posture = cols, with_posture
            def fit(self, X, y=None):
                self.comp_ = Composites(with_posture=self.with_posture).fit(X, y)
                return self
            def transform(self, X):
                return self.comp_.transform(X)[:, self.cols]

        def oof_sub(cols):
            acc, cnt = np.zeros(len(y)), np.zeros(len(y))
            cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=PERM_REPEATS, random_state=SEED)
            for tr, te in cv.split(X):
                pipe = Pipeline([("pick", PickCols(cols, has_pos)),
                                 ("sc", StandardScaler()),
                                 ("rd", Ridge(alpha=ALPHA_RIDGE))])
                m = clone(pipe).fit(X[tr], y[tr])
                acc[te] += m.predict(X[te]); cnt[te] += 1
            return acc / np.maximum(cnt, 1)

        print("  B1 แบบสอบถามอย่างเดียว    : r = %+.3f" % spearman(y, oof_sub(beh_idx)))
        print("  B2 ท่าทางอย่างเดียว       : r = %+.3f   <-- สิ่งที่แอปคู่แข่งทุกตัวทำ"
              % spearman(y, oof_sub(pos_idx)))

    # ---------- บทเรียนความซื่อสัตย์: in-sample vs out-of-fold ----------
    # *** สไลด์ปิดท้ายของเราใช้ตัวเลขคู่นี้ ***
    # เดิมทีมเอาตัวเลข +0.297 / -0.076 จาก *คนละสคริปต์ คนละ n คนละข้อมูลจำลอง*
    # มาเสนอเหมือนเป็นการวัดชุดเดียวกันบนนักเรียนจริง = พูดไม่ตรงความจริง
    # -> คำนวณตรงนี้ จากข้อมูลชุดเดียวกัน เมตริกเดียวกัน ตอบกรรมการได้ทุกคำถาม
    print("\n" + "=" * 78)
    print("บทเรียนความซื่อสัตย์: วัดสองแบบบน 'ข้อมูลชุดเดียวกัน' (n=%d)" % len(y))
    print("=" * 78)

    m_full = clone(model(has_pos)).fit(X, y)
    r_in = spearman(y, m_full.predict(X))          # เอาข้อสอบเก่ามาสอบเอง
    r_out = r_obs                                   # ข้อมูลที่โมเดลไม่เคยเห็น (out-of-fold)

    print("  วัดแบบเอาข้อสอบเก่ามาสอบเอง (in-sample)     : r = %+.3f" % r_in)
    print("  วัดอย่างซื่อสัตย์ (out-of-fold, ไม่เคยเห็น)  : r = %+.3f" % r_out)
    print("  >>> ความต่าง (optimism) = %+.3f" % (r_in - r_out))
    print("\n  เมตริก: Spearman r ระหว่างคะแนนที่โมเดลทาย กับ ระดับอาการ NRS 0-10")
    print("  วิธี validate: RepeatedKFold %d-fold x %d repeats, seed=%d"
          % (N_SPLITS, PERM_REPEATS, SEED))
    print("  ข้อมูล: นักเรียนจริง n=%d คน (ชุดเดียวกันทั้งสองบรรทัด)" % len(y))
    print("\n  *** ตัวเลขคู่นี้เอาขึ้นสไลด์ได้ — ตอบได้ทุกคำถาม ***")

    # ---------- บันทึกผล ----------
    res = os.path.join(OUT, "results.txt")
    with open(res, "w", encoding="utf-8") as f:
        f.write("=== ผลการวิเคราะห์ ก่อนจะค่อม (รันตาม analysis_plan.md) ===\n")
        f.write("n = %d คน  (มีข้อมูลภาพ: %s)\n" % (len(y), "ใช่" if has_pos else "ไม่"))
        f.write("NRS: เฉลี่ย %.2f  SD %.2f  ตอบ 0 = %.0f%%\n\n"
                % (y.mean(), y.std(), 100 * zero_frac))
        f.write("PRIMARY  : Spearman r = %+.3f   p = %.4f  (permutation B=%d)\n"
                % (r_obs, p_val, N_PERM))
        if has_pos:
            f.write("S1 พฤติกรรมเพิ่มข้อมูล : F=%.2f  p=%.4f  dR2=%.3f\n" % (f1, pp1, dr1))
            f.write("S2 ท่าทางเพิ่มข้อมูล   : F=%.2f  p=%.4f  dR2=%.3f\n" % (f2, pp2, dr2))
        f.write("\nความซื่อสัตย์ (ข้อมูลชุดเดียวกัน เมตริกเดียวกัน):\n")
        f.write("  in-sample   r = %+.3f\n" % r_in)
        f.write("  out-of-fold r = %+.3f\n" % r_out)
        f.write("  optimism    = %+.3f\n" % (r_in - r_out))
        f.write("\nเมตริก: Spearman r  ·  validate: RepeatedKFold %dx%d  ·  seed=%d\n"
                % (N_SPLITS, PERM_REPEATS, SEED))
    print("\nบันทึกผลที่:", res)
    print("\n*** ผลนี้คือผลที่เราสัญญาไว้ว่าจะรายงาน ไม่ว่ามันจะออกมาดีหรือแย่ ***")


if __name__ == "__main__":
    main()
