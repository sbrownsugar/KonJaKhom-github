# -*- coding: utf-8 -*-
"""
cva.py — วัด CVA ด้วยสติกเกอร์สี 2 จุด

  ██  ไม่ได้ใช้ในโครงงานนี้  (ทีมตัดสินใจ 14 ก.ค. ว่าไม่สะดวกติดสติกเกอร์)  ██

เก็บไว้เป็นทางเลือกในอนาคต ถ้ามีเวลาและอยากได้ค่ามาตรฐานที่เทียบงานวิจัยได้

*** ผลของการไม่ใช้สติกเกอร์ — และทำไมมันแทบไม่เสียอะไร ***
  สติกเกอร์มีไว้เพื่อเทียบมุมของเรากับ "ค่าปกติ" ในงานวิจัย -> บอกว่า "ผิดปกติ" ได้
  แต่เรา *ไม่ควร* พูดประโยคนั้นอยู่แล้ว เพราะเราให้น้ำหนักมุมท่าทาง = 0
  (Raine Study: ท่าทางไม่ทำนายความปวด) -> การวัดให้แม่นขึ้นเพื่อไปตัดสินว่า
  "ปกติ/ผิดปกติ" จึงย้อนแย้งกับสิ่งที่เราค้นพบเอง

  สิ่งที่ยังทำได้ครบโดยไม่ต้องมีสติกเกอร์ (เพราะเป็นการเทียบ "คนเดิมกับตัวเอง"):
    - วัดผลก่อน/หลังท่าบริหาร
    - กระจก 8 สัปดาห์
    - ธงส่งต่อความไม่สมมาตร
  สิ่งที่ต้องเลิก: การใส่แถบสี เขียว/เหลือง/แดง ให้กับมุมท่าทาง
    -> พูดตรง ๆ ว่า "ไม่มีค่ามาตรฐานสำหรับวิธีวัดแบบนี้ เราจึงไม่ตัดสินว่าปกติหรือไม่"

=============================================================================
*** ทำไมเคยจะมีไฟล์นี้ ทั้งที่ pose.py วัดมุมได้อยู่แล้ว ***

มุมที่ pose.py วัด ใช้จุดหมุนที่ "ไหล่" (acromion) เพราะ MediaPipe ให้มาแค่นั้น
แต่ CVA ที่ทุกงานวิจัยในโลกใช้ ใช้จุดหมุนที่ "C7" (กระดูกคอปุ่มที่คลำได้ตรงต้นคอ)
-> คนละจุด คนละมุม เทียบกับค่ามาตรฐานไม่ได้
-> เส้นแบ่งสีเดิมของเรา (15/22 องศา) จึงไม่มีงานวิจัยรองรับแม้แต่เส้นเดียว

วิธีแก้ (ทำตามงานตีพิมพ์ AutoMCA, Automation 2025;6(4):88 — ได้ r > 0.98 เทียบมาตรฐานทอง):
  1. ติดสติกเกอร์สี 2 จุด: ที่ "ติ่งหู (tragus)" และที่ "C7"
  2. ใช้ MediaPipe หาว่าหัว/คออยู่ตรงไหนในภาพ (ROI) -> กันเจอสีเดียวกันจากพื้นหลัง
  3. ใน ROI นั้น ใช้ OpenCV หาจุดสี -> ได้พิกัดจริงของ tragus และ C7
  4. CVA = มุมระหว่างเส้น C7->tragus กับแนวนอน

=============================================================================
เกณฑ์สี — ทุกเส้นมีงานวิจัยรองรับ (ต่างจากเส้นเดิมที่เราตั้งเอง)

  เขียว : CVA >= 50 องศา
          Yip 2008 (Man Ther 13:148-154) — คนปกติ 50.58 +- 2.09
          Gallego-Izquierdo 2020 (IJERPH 17:6521) — ที่จุดตัด 50: Sn 94.4% / Sp 84.6%
  เหลือง: 45 <= CVA < 50
          Heydari 2022 (BMC Pediatr 22:230) — นักเรียนที่มี FHP จริง เฉลี่ย 45.2-46.2
  แดง   : CVA < 45
          Titcomb 2024 — นิยาม "severe FHP"
          Yip 2008 — ผู้ป่วยปวดคอ เฉลี่ย 43.94 +- 3.61 (อยู่ใต้เส้นนี้พอดี)

*** ต้องประกาศเองบนเวที ***
  - วรรณกรรมไม่มีฉันทามติ จุดตัด CVA มีตั้งแต่ <40, <45, <48, <50
    เราเลือก 50/45 เพราะเป็นคู่เดียวที่มีทั้ง "หลักฐานเชื่อมโยงอาการ" (Yip)
    และ "ค่าความไว/ความจำเพาะ" (Gallego) รองรับพร้อมกัน
  - ต้องล็อกท่าถ่าย (ยืน หรือ นั่ง เลือกอย่างเดียว) — นั่งทำให้ CVA ต่ำลง 0.8-2.2 องศา
  - MCID ของ CVA = 1.4 องศา แต่ MDC ของเครื่องมือระดับมือถือ = 5-6 องศา
    -> เราจับ MCID ไม่ได้ ต้องรายงานเป็นช่วงสี ห้ามรายงานทศนิยม
  - *** และแม้ CVA จะวัดได้แม่นขึ้น มันก็ยังมีน้ำหนัก 0 ในคะแนนความเสี่ยง ***
    (Raine Study: ท่าทางไม่ทำนายความปวด) — CVA มีไว้ให้เห็นภาพ + วัดผลก่อน/หลังท่าบริหาร
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pose import detect_landmarks, PoseError, EAR_L, EAR_R, SHO_L, SHO_R

# ช่วงสีในระบบ HSV — เลือกสีที่ "ไม่มีบนตัวคน" เพื่อไม่ให้สับสนกับผิว/เสื้อ
MARKER_COLORS = {
    "เขียวสะท้อนแสง": [((35, 90, 90), (85, 255, 255))],
    "ชมพู/บานเย็น":  [((140, 80, 90), (175, 255, 255))],
    "ฟ้า":            [((90, 90, 90), (130, 255, 255))],
    "ส้ม":            [((5, 130, 130), (20, 255, 255))],
}

MIN_BLOB_AREA = 12          # จุดสีเล็กกว่านี้ = น่าจะเป็นสัญญาณรบกวน

GREEN, YELLOW, RED = "เขียว", "เหลือง", "แดง"


@dataclass
class CVAResult:
    cva_deg: float
    band: str
    tragus_px: tuple
    c7_px: tuple
    marker_color: str
    note: str


def _find_markers(bgr, roi, color_ranges):
    """หาจุดสีใน ROI -> คืนจุดศูนย์กลางของ blob ที่ใหญ่ที่สุด 2 จุด (พิกัดในภาพเต็ม)"""
    x0, y0, x1, y1 = roi
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        raise PoseError("ตัดภาพบริเวณหัว-คอไม่ได้ — คนอาจอยู่นอกเฟรม")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in color_ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

    # ลบจุดรบกวนเล็ก ๆ แล้วอุดรู
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_BLOB_AREA:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        blobs.append((a, (x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"])))

    blobs.sort(key=lambda t: -t[0])
    return [p for _a, p in blobs[:2]]


def measure_cva(image_path: str, marker_color: str = "เขียวสะท้อนแสง") -> CVAResult:
    """ภาพด้านข้าง (ติดสติกเกอร์ที่ติ่งหู + C7) -> CVA จริง"""
    if marker_color not in MARKER_COLORS:
        raise PoseError("ไม่รู้จักสีสติกเกอร์: %s (มีให้เลือก: %s)"
                        % (marker_color, ", ".join(MARKER_COLORS)))

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise PoseError("เปิดไฟล์ภาพไม่ได้")
    H, W = bgr.shape[:2]

    # ---- 1) ใช้ MediaPipe หาว่าหัว-คออยู่ตรงไหน (กันเจอสีเดียวกันจากพื้นหลัง) ----
    lm, w, h = detect_landmarks(image_path)
    ears = [lm[EAR_R], lm[EAR_L]]
    shos = [lm[SHO_R], lm[SHO_L]]
    ex = np.mean([e.x for e in ears]) * W
    ey = np.mean([e.y for e in ears]) * H
    sx = np.mean([s.x for s in shos]) * W
    sy = np.mean([s.y for s in shos]) * H

    pad = max(abs(sy - ey), 60) * 1.25
    roi = (int(max(0, min(ex, sx) - pad)), int(max(0, min(ey, sy) - pad)),
           int(min(W, max(ex, sx) + pad)), int(min(H, max(ey, sy) + pad)))

    # ---- 2) หาจุดสีใน ROI ----
    pts = _find_markers(bgr, roi, MARKER_COLORS[marker_color])
    if len(pts) < 2:
        raise PoseError(
            "หาสติกเกอร์ไม่ครบ 2 จุด (เจอ %d จุด)\n"
            "ตรวจ: ติดสติกเกอร์ครบไหม · สีตรงกับที่เลือก (%s) ไหม · "
            "แสงสว่างพอไหม · สติกเกอร์ถูกผมบังหรือเปล่า" % (len(pts), marker_color))

    # ---- 3) จุดบน = ติ่งหู, จุดล่าง = C7 (ติ่งหูอยู่เหนือ C7 เสมอ) ----
    pts.sort(key=lambda p: p[1])            # y น้อย = อยู่สูงกว่า
    tragus, c7 = pts[0], pts[1]

    # ---- 4) CVA = มุมระหว่างเส้น C7 -> ติ่งหู กับแนวนอน ----
    dx = tragus[0] - c7[0]
    dy = c7[1] - tragus[1]                  # y ของภาพชี้ลง -> กลับด้านให้ขึ้นเป็นบวก
    cva = float(np.degrees(np.arctan2(abs(dy), abs(dx))))

    if cva >= 50:
        band, note = GREEN, "อยู่ในช่วงค่าปกติ (Yip 2008: คนปกติ 50.6 ± 2.1 องศา)"
    elif cva >= 45:
        band, note = YELLOW, "ต่ำกว่าเกณฑ์ปกติ — ใกล้เคียงกลุ่มที่มีศีรษะยื่นหน้า (Heydari 2022)"
    else:
        band, note = RED, ("ต่ำกว่า 45 องศา — Yip 2008 พบว่าผู้ป่วยปวดคอเฉลี่ย 43.9 องศา "
                           "แนะนำให้ปรึกษาครูอนามัย/นักกายภาพบำบัด")

    return CVAResult(cva_deg=round(cva, 1), band=band,
                     tragus_px=(round(tragus[0]), round(tragus[1])),
                     c7_px=(round(c7[0]), round(c7[1])),
                     marker_color=marker_color, note=note)


def annotate(image_path: str, r: CVAResult, out_path: str) -> str:
    """วาดเส้นและมุมทับภาพ ให้เห็นว่าโปรแกรมวัดจากตรงไหน (โปร่งใส = ตรวจสอบได้)"""
    img = cv2.imread(image_path)
    t = tuple(int(v) for v in r.tragus_px)
    c = tuple(int(v) for v in r.c7_px)
    col = {GREEN: (91, 157, 74), YELLOW: (61, 163, 232), RED: (79, 83, 217)}[r.band]

    cv2.line(img, c, (c[0] + 170, c[1]), (170, 170, 170), 2, cv2.LINE_AA)   # แนวนอนอ้างอิง
    cv2.line(img, c, t, col, 3, cv2.LINE_AA)                                 # C7 -> ติ่งหู
    for p in (t, c):
        cv2.circle(img, p, 8, col, -1, cv2.LINE_AA)
        cv2.circle(img, p, 8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "CVA %.0f deg" % r.cva_deg, (c[0] + 14, c[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    return out_path


# =============================================================================
if __name__ == "__main__":
    import os

    # สร้างภาพจำลอง: คนยืนด้านข้าง + สติกเกอร์เขียว 2 จุด
    # (เพื่อพิสูจน์ว่าคณิตศาสตร์ถูก — ภาพจริงต้องรอทีมถ่าย)
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(OUT, exist_ok=True)

    # *** จุดที่ผมเคยใส่พิกัดผิด และเกือบปล่อยผ่าน ***
    # ตอนแรกผมวางติ่งหูไว้ "เกือบตรงเหนือ C7" -> ได้ CVA 76-86 องศา ซึ่งเป็นไปไม่ได้
    # ความจริงทางกายวิภาค: ติ่งหูของ *คนปกติ* ล้ำหน้า C7 ไปแล้วราว 10-12 ซม.
    #   ระยะดิ่ง C7 -> หู ~13-15 ซม.  ->  tan(50°) = 1.19  ->  ระยะราบ ~11-12 ซม.
    # ยิ่งศีรษะยื่นหน้า ระยะราบยิ่งมาก -> มุมยิ่ง "หุบลง" (ไม่ใช่กางออก)
    print("ทดสอบคณิตศาสตร์ของ CVA (ยังไม่มีภาพจริง — ทีมต้องถ่ายมา)")
    print("สมมติระยะดิ่ง C7 -> ติ่งหู = 160 พิกเซล (≈ 14 ซม.)\n")
    print("  %-18s ระยะราบ   CVA     แถบ" % "ลักษณะ")
    print("  " + "-" * 52)

    C7 = (290.0, 260.0)
    RISE = 160.0
    for name, target_cva in [
        ("หัวตั้งตรงมาก", 57.0),
        ("ปกติ (Yip 2008)", 50.6),
        ("ยื่นหน้า (Heydari)", 46.0),
        ("ยื่นหน้ามาก (Yip: ผู้ป่วยปวดคอ)", 43.9),
    ]:
        run = RISE / np.tan(np.radians(target_cva))     # คำนวณย้อนจากค่ามาตรฐานจริง
        tx, ty = C7[0] + run, C7[1] - RISE
        dx, dy = tx - C7[0], C7[1] - ty
        cva = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
        band = GREEN if cva >= 50 else (YELLOW if cva >= 45 else RED)
        print("  %-32s %5.0f px  %5.1f°  %s" % (name, run, cva, band))

    print("\n>>> ตรวจว่าสูตรเดินถูกทาง: ยิ่งหัวยื่นหน้า (ระยะราบมากขึ้น) มุมต้อง 'ลดลง' ✓")
    print(">>> เกณฑ์ทุกเส้นมี citation:")
    print("    เขียว  CVA >= 50°  (Yip 2008: คนปกติ 50.6±2.1 · Gallego 2020: Sn 94.4%/Sp 84.6%)")
    print("    เหลือง 45-50°      (Heydari 2022: นักเรียนที่มี FHP จริง 45.2-46.2)")
    print("    แดง    < 45°       (Yip 2008: ผู้ป่วยปวดคอ 43.9±3.6)")
    print("\n*** และแม้จะวัดแม่นขึ้น CVA ก็ยังมีน้ำหนัก 0 ในคะแนนความเสี่ยง ***")
    print("    (Raine Study: ท่าทางไม่ทำนายความปวด) — CVA มีไว้ให้เห็นภาพ + วัดผลก่อน/หลังท่าบริหาร")
