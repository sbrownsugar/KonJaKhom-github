# -*- coding: utf-8 -*-
"""
build_pose_csv.py — ภาพถ่ายทั้งโฟลเดอร์ -> ตารางมุม (1 คน = 1 แถว)
=============================================================================
*** สะพานที่หายไป ***
run_analysis.py ต้องการค่ามุมของแต่ละคน แต่เดิมไม่มีโค้ดไหนผลิตมันเลย
-> S2 ("ท่าทางเพิ่มข้อมูลเหนือพฤติกรรมไหม") ซึ่งเป็น *ข้อค้นพบหลัก* ของโครงงาน
   จะไม่มีวันถูกทดสอบ

วิธีใช้ (วันพฤหัส หลังถ่ายรูปเสร็จ):
    python analysis/build_pose_csv.py

ต้องการ: ภาพใน D:\\Obec\\5-เก็บข้อมูล\\ภาพดิบ\\
    ชื่อไฟล์:  <รหัส>_right.jpg   <รหัส>_left.jpg   <รหัส>_front.jpg
    เช่น       5101_right.jpg     5101_front.jpg
    (_left ไม่บังคับ — ถ้ามีจะเฉลี่ยกับ _right)

ผลลัพธ์: D:\\Obec\\5-เก็บข้อมูล\\pose_angles.csv   (code, fha, asym, n_side, quality)

*** กฎที่บังคับในโค้ด (ตามแผน analysis_plan.md) ***
  §6.1  1 คน = 1 แถว     -> เฉลี่ยภาพซ้าย/ขวาของคนเดียวกัน ห้ามนับเป็น 2 แถว
  §7    คุณภาพ < 0.5     -> ตัดทิ้ง (มองไม่เห็นจุดชัดพอ = อย่าเดา)
  §7    ภาพไม่ครบ        -> ตัดคนนั้นทั้งคน (ต้องมีทั้งด้านข้างและด้านหน้า)
  ห้ามเฉลี่ยมุมจากภาพ *คนละมุมกล้อง* เข้าด้วยกัน (fha มาจากด้านข้าง, asym มาจากด้านหน้า)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pose import analyze_side, analyze_front, PoseError   # noqa: E402

IMG_DIR = r"D:\Obec\5-เก็บข้อมูล\ภาพดิบ"
OUT_CSV = r"D:\Obec\5-เก็บข้อมูล\pose_angles.csv"
MIN_QUALITY = 0.5


def build(img_dir: str = IMG_DIR, out_csv: str = OUT_CSV) -> pd.DataFrame:
    if not os.path.isdir(img_dir):
        print("!! ยังไม่มีโฟลเดอร์ภาพ: %s" % img_dir)
        print("   สร้างโฟลเดอร์นี้ แล้วเอาภาพใส่ (ตั้งชื่อ <รหัส>_right.jpg ฯลฯ)")
        sys.exit(1)

    files = [f for f in os.listdir(img_dir) if "_" in f and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print("!! ไม่มีไฟล์ภาพใน %s" % img_dir)
        sys.exit(1)

    codes = sorted({f.split("_")[0].strip().upper() for f in files})
    print("พบภาพ %d ไฟล์ · รหัสนักเรียน %d คน\n" % (len(files), len(codes)))

    rows, dropped = [], []
    for code in codes:
        # ---------- ด้านข้าง (เฉลี่ยซ้าย/ขวา ถ้ามีทั้งคู่) ----------
        fhas, quals = [], []
        for side in ("right", "left"):
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(img_dir, "%s_%s%s" % (code, side, ext))
                if not os.path.exists(p):
                    continue
                try:
                    s = analyze_side(p)
                except PoseError as e:
                    print("  ⚠ %s_%s : %s" % (code, side, str(e).splitlines()[0][:56]))
                    break
                if s.quality < MIN_QUALITY:
                    print("  ⚠ %s_%s : คุณภาพต่ำ (%.2f) — ตัดทิ้งตามแผน §7" % (code, side, s.quality))
                    break
                fhas.append(s.fha_deg)
                quals.append(s.quality)
                break

        # ---------- ด้านหน้า ----------
        asym, q_front = None, None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(img_dir, "%s_front%s" % (code, ext))
            if not os.path.exists(p):
                continue
            try:
                f = analyze_front(p)
            except PoseError as e:
                print("  ⚠ %s_front : %s" % (code, str(e).splitlines()[0][:56]))
                break
            if f.quality < MIN_QUALITY:
                print("  ⚠ %s_front : คุณภาพต่ำ (%.2f) — ตัดทิ้ง" % (code, f.quality))
                break
            asym, q_front = f.asym_deg, f.quality
            break

        # ---------- ต้องมีครบทั้งสองมุม ----------
        if not fhas or asym is None:
            why = []
            if not fhas:
                why.append("ไม่มีภาพด้านข้างที่ใช้ได้")
            if asym is None:
                why.append("ไม่มีภาพด้านหน้าที่ใช้ได้")
            dropped.append((code, " + ".join(why)))
            continue

        rows.append(dict(
            code=code,
            fha=round(float(np.mean(fhas)), 2),     # เฉลี่ยเฉพาะภาพด้านข้างด้วยกัน
            asym=round(float(asym), 2),             # จากภาพด้านหน้าเท่านั้น
            n_side=len(fhas),
            quality=round(float(min(min(quals), q_front)), 2),
        ))

    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("ใช้ได้ %d คน · ตัดออก %d คน" % (len(df), len(dropped)))
    if dropped:
        print("\nคนที่ถูกตัดออก (ตามเกณฑ์ในแผน §7 — ใช้เกณฑ์เดียวกันทุกคน ไม่ดูผลลัพธ์ประกอบ):")
        for code, why in dropped:
            print("  %-10s %s" % (code, why))
    if len(df):
        print("\nสถิติมุมที่วัดได้:")
        print("  ศีรษะยื่นหน้า : เฉลี่ย %.1f°  SD %.1f°  ช่วง %.1f–%.1f°"
              % (df["fha"].mean(), df["fha"].std(), df["fha"].min(), df["fha"].max()))
        print("  ไม่สมมาตร     : เฉลี่ย %.1f°  SD %.1f°  ช่วง %.1f–%.1f°"
              % (df["asym"].mean(), df["asym"].std(), df["asym"].min(), df["asym"].max()))
        print("\nบันทึกที่: %s" % out_csv)
        print("\n>>> ขั้นต่อไป: python analysis/run_analysis.py")
    else:
        print("\n!! ไม่มีข้อมูลที่ใช้ได้เลย — ตรวจชื่อไฟล์และคุณภาพภาพ")
    return df


if __name__ == "__main__":
    build()
