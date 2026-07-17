# -*- coding: utf-8 -*-
"""
reliability.py — "เครื่องมือของเราเองมั่วแค่ไหน?"
=============================================================================
*** ทำไมชิ้นนี้สำคัญกว่าที่คิด ***

เราตัดสินใจแล้วว่า **จะไม่ตัดสินว่ามุมท่าทาง "ปกติ" หรือ "ผิดปกติ"**
เพราะเราไม่มีค่ามาตรฐานมาเทียบ (เราวัดจากไหล่ ไม่ใช่ C7)

แต่เรายัง **เทียบคนเดิมกับตัวเอง** ได้ — เช่น "ก่อน/หลังทำท่าบริหาร"
และเพื่อจะพูดว่า "ดีขึ้นจริง" ได้ เราต้องรู้ก่อนว่า **เครื่องมือเรามั่วกี่องศา**

  ถ้าคนเดิมยืนใหม่แล้วค่าต่างกัน 4 องศา
  -> การที่ใครสักคน "ดีขึ้น 3 องศา" ก็ *ไม่มีความหมายอะไรเลย*

ตัวเลขที่ต้องได้: **MDC95** (Minimal Detectable Change)
= การเปลี่ยนแปลงขั้นต่ำที่เชื่อได้ 95% ว่า "เปลี่ยนจริง ไม่ใช่ความมั่วของเครื่องมือ"

=============================================================================
วิธีเก็บข้อมูล (15 คน · 15 นาที · ทำพร้อมกับการถ่ายปกติได้)

  ถ่ายภาพ -> ให้เขา **เดินออกไป 5 ก้าว** -> **กลับมายืนใหม่** -> ถ่ายอีกครั้ง

  *** ห้ามเอารูปเดิมมารันโมเดลซ้ำ *** นั่นวัดแค่ "โปรแกรมนิ่งไหม" (ซึ่งนิ่ง 100% อยู่แล้ว)
  ไม่ได้วัด "การวัดของเรานิ่งไหม" ซึ่งรวมความผันผวนของการยืน การถ่าย และ AI เข้าด้วยกัน

รูปแบบไฟล์ที่ต้องเตรียม: D:\\Obec\\5-เก็บข้อมูล\\retest.csv
    code,fha_1,asym_1,fha_2,asym_2
    A-01,18.2,3.1,19.0,2.6
    A-02,25.5,4.8,24.1,5.2
    ...
(ได้จากการรัน src/pose.py กับภาพรอบ 1 และรอบ 2 — ดู make_retest_csv() ด้านล่าง)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

CSV = r"D:\Obec\5-เก็บข้อมูล\retest.csv"
IMG_DIR = r"D:\Obec\5-เก็บข้อมูล\ภาพดิบ"


def icc_2_1(x1: np.ndarray, x2: np.ndarray) -> tuple:
    """ICC(2,1) — two-way random effects, absolute agreement, single measure

    ใช้ (2,1) ไม่ใช่ (3,k) เพราะ:
      * (2,·) = ผู้วัดถูกสุ่มมา และเราสนใจ "ความตรงกันแบบสัมบูรณ์" ไม่ใช่แค่ความสอดคล้อง
      * (·,1) = แอปจริงใช้ภาพ *รูปเดียว* ไม่ได้เฉลี่ยหลายรูป -> ต้องรายงานแบบ single measure
    """
    Y = np.column_stack([x1, x2]).astype(float)
    n, k = Y.shape                      # n = จำนวนคน, k = จำนวนครั้งที่วัด (=2)

    grand = Y.mean()
    ms_r = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)          # ระหว่างคน
    ms_c = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)          # ระหว่างครั้ง
    ss_e = ((Y - Y.mean(axis=1, keepdims=True)
             - Y.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))                                    # ความคลาดเคลื่อน

    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    icc = (ms_r - ms_e) / denom if denom > 0 else 0.0

    # ช่วงความเชื่อมั่น 95% (Shrout & Fleiss 1979)
    f = ms_r / ms_e if ms_e > 0 else np.inf
    df1, df2 = n - 1, (n - 1) * (k - 1)
    fl = f / stats.f.ppf(0.975, df1, df2)
    fu = f * stats.f.ppf(0.975, df2, df1)
    lo = (fl - 1) / (fl + (k - 1))
    hi = (fu - 1) / (fu + (k - 1))
    return float(icc), float(max(lo, -1)), float(min(hi, 1)), float(ms_e)


def sem_mdc(x1: np.ndarray, x2: np.ndarray) -> tuple:
    """SEM = ความมั่วของเครื่องมือ (องศา) · MDC95 = เปลี่ยนกี่องศาถึงจะเชื่อได้"""
    d = np.asarray(x1, float) - np.asarray(x2, float)
    sem = float(np.std(d, ddof=1) / np.sqrt(2))     # จากความแปรปรวนของผลต่าง
    mdc95 = 1.96 * np.sqrt(2) * sem
    return sem, float(mdc95)


def interpret_icc(icc: float, lo: float) -> str:
    """Koo & Li 2016 — และต้องดู *ขอบล่าง* ของ CI ไม่ใช่ค่ากลาง (อนุรักษ์นิยม)"""
    v = lo
    if v < 0.5:
        return "แย่ (poor)"
    if v < 0.75:
        return "พอใช้ (moderate)"
    if v < 0.90:
        return "ดี (good)"
    return "ดีมาก (excellent)"


def make_retest_csv():
    """ช่วยสร้าง retest.csv จากภาพ (ต้องตั้งชื่อไฟล์ CODE_right.jpg และ CODE_right_r2.jpg)"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.pose import analyze_side, analyze_front, PoseError

    rows = []
    if not os.path.isdir(IMG_DIR):
        print("!! ยังไม่มีโฟลเดอร์ภาพ:", IMG_DIR)
        return
    codes = sorted({f.split("_")[0] for f in os.listdir(IMG_DIR) if "_" in f})
    for code in codes:
        try:
            s1 = analyze_side(os.path.join(IMG_DIR, "%s_right.jpg" % code))
            s2 = analyze_side(os.path.join(IMG_DIR, "%s_right_r2.jpg" % code))
            f1 = analyze_front(os.path.join(IMG_DIR, "%s_front.jpg" % code))
            f2 = analyze_front(os.path.join(IMG_DIR, "%s_front_r2.jpg" % code))
        except (PoseError, FileNotFoundError, Exception) as e:
            print("  ข้าม %s (%s)" % (code, str(e)[:48]))
            continue
        rows.append(dict(code=code, fha_1=s1.fha_deg, fha_2=s2.fha_deg,
                         asym_1=f1.asym_deg, asym_2=f2.asym_deg))
    if rows:
        pd.DataFrame(rows).to_csv(CSV, index=False, encoding="utf-8-sig")
        print("สร้าง %s (%d คน)" % (CSV, len(rows)))


def main():
    if not os.path.exists(CSV):
        print("!! ยังไม่มีไฟล์:", CSV)
        print("   ถ้ามีภาพรอบ 1 และรอบ 2 แล้ว ให้รัน:  python reliability.py --build")
        return

    d = pd.read_csv(CSV)
    n = len(d)
    print("=" * 74)
    print("ความน่าเชื่อถือของเครื่องมือวัด (test-retest, n = %d คน)" % n)
    print("=" * 74)
    if n < 10:
        print("!! คนน้อยเกินไป (ควรมีอย่างน้อย 15 คน) — ช่วงความเชื่อมั่นจะกว้างมาก")

    for name, c1, c2, unit in [("ศีรษะยื่นหน้า", "fha_1", "fha_2", "องศา"),
                               ("ความไม่สมมาตร", "asym_1", "asym_2", "องศา")]:
        if c1 not in d or c2 not in d:
            continue
        x1, x2 = d[c1].values, d[c2].values
        icc, lo, hi, _ = icc_2_1(x1, x2)
        sem, mdc = sem_mdc(x1, x2)

        print("\n### %s ###" % name)
        print("  ICC(2,1) = %.2f   95%% CI [%.2f, %.2f]   -> %s"
              % (icc, lo, hi, interpret_icc(icc, lo)))
        print("  SEM      = %.1f %s   (ความมั่วของเครื่องมือ)" % (sem, unit))
        print("  MDC95    = %.1f %s   <-- ต้องเปลี่ยนเกินนี้ ถึงจะเชื่อได้ว่าเปลี่ยนจริง" % (mdc, unit))
        print("  ค่าเฉลี่ยผลต่างรอบ1-รอบ2 = %+.1f %s (ถ้าห่างจาก 0 มาก = มีอคติเชิงระบบ)"
              % (np.mean(x1 - x2), unit))

        if lo < 0.5:
            print("  🔴 ขอบล่างของ CI < 0.50 -> **ต้องพูดเองบนเวทีว่าเครื่องมือส่วนนี้ยังไม่นิ่งพอ**")
            print("     และลดสถานะตัวแปรนี้เป็น 'สัญญาณให้ถ่ายซ้ำ' ไม่ใช่ตัวเลขบนหน้าจอ")
        if name == "ศีรษะยื่นหน้า":
            print("  📌 เอาไปใช้ที่: mirror.py -> ตั้ง MDC_DEG = %.1f (ตอนนี้ตั้งไว้ 3.0 แบบเดา)" % mdc)
            print("     กระจกจะได้ไม่ประกาศว่า 'ดีขึ้น' ถ้าเปลี่ยนน้อยกว่าความมั่วของตัวเอง")

    # ---------- รายคน vs รายกลุ่ม: คำถามที่สำคัญที่สุด ----------
    if "fha_1" in d and "fha_2" in d:
        sem, mdc = sem_mdc(d["fha_1"].values, d["fha_2"].values)
        print("\n" + "=" * 74)
        print("คำถามที่สำคัญที่สุด: แล้วเราวัด 'ผลของท่าบริหาร' ได้ไหม?")
        print("=" * 74)
        print("  ผลที่ RCT รายงาน (Heydari 2022, 8 สัปดาห์) = +6.9 องศา\n")

        print("  [ก] บอกนักเรียน *คนเดียว* ว่า 'คุณดีขึ้นแล้ว'")
        print("      ต้องเปลี่ยนเกิน MDC95 = %.1f องศา" % mdc)
        if mdc < 6.9:
            print("      -> ผล 6.9° มากกว่าเกณฑ์ = พอบอกได้ **แต่ยังพลาดได้บ่อย** (ต้องรายงานความไม่แน่นอน)")
        else:
            print("      -> 🔴 ผล 6.9° **เล็กกว่าเกณฑ์** = **บอกรายคนไม่ได้** ต้องพูดเองบนเวที")

        print("\n  [ข] บอกโรงเรียน ว่า 'ทั้งห้องดีขึ้น' (paired t-test)")
        sd_diff = np.sqrt(2) * sem
        for nn in (15, 25, 40):
            mdd = (stats.t.ppf(0.975, nn - 1) + stats.t.ppf(0.80, nn - 1)) * sd_diff / np.sqrt(nn)
            mark = "✅" if mdd < 6.9 else "🔴"
            print("      %s n=%2d คน -> จับผลที่เล็กถึง %.1f องศา ได้" % (mark, nn, mdd))
        print("\n      *** ความมั่วของแต่ละคนเป็นแบบสุ่ม -> พอเฉลี่ยหลายคน มันหักล้างกันเอง ***")
        print("      => นี่คือเหตุผลที่ Teacher Dashboard ต้องมีอยู่ ไม่ใช่ของแถม")

    print("\n" + "=" * 74)
    print("ประโยคที่พูดได้บนเวที (และมีตัวเลขรองรับ):")
    print('  "แอปเราจะไม่บอกนักเรียนคนไหนว่า \'คุณดีขึ้นแล้ว\' เพราะเรายังไม่แม่นพอ')
    print('   แต่เราบอกโรงเรียนได้ว่า กิจกรรมที่จัดไปได้ผลจริงหรือเปล่า"')
    print("\nประโยคที่ *ห้าม* พูด:")
    print('  "การเปลี่ยนแปลงนี้มีความหมายทางคลินิก"  <-- นั่นคือ MCID ซึ่งเราไม่มี')
    print('  "แอปบอกได้ว่าคุณดีขึ้น"                  <-- รายคนยังเชื่อไม่ได้')


if __name__ == "__main__":
    if "--build" in sys.argv:
        make_retest_csv()
    main()
