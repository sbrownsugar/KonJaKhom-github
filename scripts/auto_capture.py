# -*- coding: utf-8 -*-
"""
auto_capture.py — ตั้งกล้องแล้วถ่ายเอง สำหรับคนที่อยู่คนเดียว
=====================================================================
ใช้:  D:\\Obec\\.venv\\Scripts\\python.exe D:\\Obec\\scripts\\auto_capture.py [รหัส]

วิธีทำงาน:
  1. เปิดกล้อง เดินไปยืนหน้ากล้องให้เห็นทั้งตัว
  2. พอ AI จับท่าได้ครบ (ไหล่+สะโพก+ข้อเท้า อยู่ในเฟรม) และ "นิ่ง" พอ
     -> ขึ้นนับถอยหลัง 3-2-1 แล้วถ่ายเอง
  3. ถ่าย 2 รูป: ด้านหน้า -> ด้านข้าง  (เซฟเป็น <รหัส>_front.jpg / <รหัส>_right.jpg)

ปุ่ม:  SPACE = ถ่ายเดี๋ยวนี้เลย (ไม่ต้องรอ)   ·   R = เริ่มรูปนี้ใหม่   ·   ESC = ออก

*** หมายเหตุ: สคริปต์นี้ต้องรันบนเครื่องที่ "มีเว็บแคม" — ทดสอบบนเครื่อง headless ไม่ได้ ***
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "project", "models", "pose_landmarker_lite.task")
OUT_DIR = os.path.join(ROOT, "5-เก็บข้อมูล", "ภาพดิบ")

# ดัชนี landmark (ตรงกับ src/pose.py)
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24
L_ANK, R_ANK = 27, 28
MIN_VIS = 0.5
STEADY_FRAMES = 12          # ต้อง "พร้อม" ติดกันกี่เฟรมก่อนเริ่มนับ
COUNTDOWN = 3.0             # วินาที

SHOTS = [("front", "ด้านหน้า — ยืนตรง หันหน้าเข้ากล้อง"),
         ("right", "ด้านข้าง — หันข้างขวาให้กล้อง")]


def make_landmarker():
    if not os.path.exists(MODEL):
        raise SystemExit("ไม่พบไฟล์โมเดล: %s" % MODEL)
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE)
    return vision.PoseLandmarker.create_from_options(opts)


PAIRS = ((L_SH, R_SH), (L_HIP, R_HIP), (L_ANK, R_ANK))


def body_ready(lms, w, h, tag="front"):
    """คนอยู่ในเฟรมครบตัวไหม (ไหล่+สะโพก+ข้อเท้า เห็นชัด และไม่ชิดขอบเกิน)

    *** บั๊ก: "รูปด้านข้างไม่มีวันถ่ายได้" (แก้ 16 ก.ค.) ***
    เดิมบังคับให้เห็นครบทั้ง 6 จุด แต่ตอนยืนหันข้าง จุดฝั่งไกลถูกลำตัวบังเสมอ
    -> visibility ตกต่ำกว่าเกณฑ์ -> เงื่อนไขนี้เป็นไปไม่ได้ทางกายภาพ กล้องรอตลอดกาล
    (ด้านหน้าผ่านฉลุยเพราะเห็นครบ — บั๊กเลยซ่อนอยู่จนมีคนลองยืนหันข้างจริง)
    """
    if tag != "front":          # tag ของรูปด้านข้างในไฟล์นี้คือ "right" ไม่ใช่ "side"
        idx = []
        for a, b in PAIRS:
            best = a if lms[a].visibility >= lms[b].visibility else b
            if lms[best].visibility < MIN_VIS:
                return False, "ยืนให้เห็นทั้งตัว (หัวถึงเท้า)"
            idx.append(best)
    else:
        idx = [i for pair in PAIRS for i in pair]
        for i in idx:
            if lms[i].visibility < MIN_VIS:
                return False, "ยืนให้เห็นทั้งตัว (หัวถึงเท้า)"

    ys = [lms[i].y for i in idx]
    xs = [lms[i].x for i in idx]
    if min(ys) < 0.03 or max(ys) > 0.99 or min(xs) < 0.02 or max(xs) > 0.98:
        return False, "ถอยห่างกล้องอีกนิด ให้เห็นทั้งตัว"
    # ต้องยืน (ไหล่อยู่สูงกว่าสะโพกพอควร)
    sh_y = (lms[L_SH].y + lms[R_SH].y) / 2
    hip_y = (lms[L_HIP].y + lms[R_HIP].y) / 2
    if hip_y - sh_y < 0.12:
        return False, "ยืนตรง ๆ"
    # กันถ่ายรูป "หันหน้า" ไปเก็บเป็นรูปด้านข้าง (มุมที่วัดได้จะเป็นขยะ)
    # หันข้าง -> ไหล่ซ้อนกัน ~0.02 · หันหน้า -> ห่าง ~0.16+  => ขีดที่ 0.12
    if tag != "front" and abs(lms[L_SH].x - lms[R_SH].x) > 0.12:
        return False, "หันข้างขวาให้กล้อง"
    return True, "พร้อม"


def draw_pose(frame, lms, w, h):
    for i in (L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK):
        if lms[i].visibility >= MIN_VIS:
            cv2.circle(frame, (int(lms[i].x * w), int(lms[i].y * h)), 6, (60, 200, 120), -1)


def banner(frame, text, color=(255, 255, 255), y=40, scale=0.8):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def big_number(frame, n, w, h):
    txt = str(n)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 6, 12)
    x, y = (w - tw) // 2, (h + th) // 2
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 20, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 6, (60, 200, 255), 12, cv2.LINE_AA)


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "auto"
    # arg 2 = โฟลเดอร์ปลายทาง (แอปส่ง temp dir มา เพราะกฎเหล็กคือ "ไม่เก็บไฟล์ภาพ"
    #         -> แอปอ่านภาพ แปลงเป็นมุม แล้วลบทันที ห้ามลงโฟลเดอร์วิจัย)
    #         ถ้าไม่ส่งมา = ใช้งานที่บูธเก็บข้อมูล -> ลง 5-เก็บข้อมูล/ภาพดิบ ตามเดิม
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    lm = make_landmarker()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("เปิดกล้องไม่ได้ — เครื่องนี้มีเว็บแคมไหม?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 60)
    print("ถ่ายอัตโนมัติ · รหัส: %s" % code)
    print("เซฟลง: %s" % out_dir)
    print("SPACE=ถ่ายเลย · R=เริ่มรูปนี้ใหม่ · ESC=ออก")
    print("=" * 60)

    shot_i = 0
    steady = 0
    counting_since = None
    win = "auto_capture (ก่อนจะค่อม)"
    # สร้างหน้าต่างไว้ก่อน เพื่อให้เช็คได้ว่าผู้ใช้กดกากบาทปิดไปหรือยัง
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    while shot_i < len(SHOTS):
        # *** ถ้าผู้ใช้กดกากบาทปิดหน้าต่าง ต้องออกให้ได้ ***
        # ไม่งั้นลูปจะวนต่อไปเรื่อย ๆ ไม่มีวันจบ -> แอปที่เรียกเราจะค้างรอตลอดกาล
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                print("ผู้ใช้ปิดหน้าต่าง")
                break
        except cv2.error:
            break

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                    # ภาพกระจก (selfie)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        tag, instr = SHOTS[shot_i]
        ready, why = False, "ไม่พบคนในภาพ"
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            draw_pose(frame, lms, w, h)
            ready, why = body_ready(lms, w, h, tag)   # เกณฑ์ต่างกันระหว่าง front / right

        banner(frame, "Shot %d/%d : %s" % (shot_i + 1, len(SHOTS), tag), (0, 220, 255), 40)
        banner(frame, instr, (255, 255, 255), 75, 0.7)

        do_capture = False
        if ready:
            steady += 1
            if steady >= STEADY_FRAMES:
                if counting_since is None:
                    counting_since = time.time()
                elapsed = time.time() - counting_since
                remain = COUNTDOWN - elapsed
                if remain > 0:
                    big_number(frame, int(remain) + 1, w, h)
                else:
                    do_capture = True
            else:
                banner(frame, "READY... hold still", (60, 220, 120), h - 30, 0.8)
        else:
            steady = 0
            counting_since = None
            banner(frame, why, (0, 165, 255), h - 30, 0.8)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                                 # ESC
            print("ยกเลิก")
            break
        if key in (ord("r"), ord("R")):
            steady = 0; counting_since = None
        if key == 32:                                 # SPACE = ถ่ายเลย
            do_capture = True

        if do_capture:
            # เซฟ "เฟรมสะอาด" (ไม่มีเส้น overlay) โดยถ่ายใหม่ 1 เฟรม
            ok2, clean = cap.read()
            clean = cv2.flip(clean, 1) if ok2 else frame
            path = os.path.join(out_dir, "%s_%s.jpg" % (code, tag))
            cv2.imwrite(path, clean)
            print("บันทึก: %s" % path)
            flash = np.full_like(frame, 255)
            cv2.imshow(win, flash); cv2.waitKey(120)
            shot_i += 1
            steady = 0; counting_since = None
            continue

        cv2.imshow(win, frame)

    cap.release()
    cv2.destroyAllWindows()
    if shot_i >= len(SHOTS):
        print("\nเสร็จครบ %d รูป -> โฟลเดอร์: %s" % (len(SHOTS), out_dir))
        if out_dir == OUT_DIR:
            print("ต่อไป: python project/analysis/build_pose_csv.py เพื่อแปลงเป็นตารางมุม")
    else:
        print("\nได้ %d/%d รูป (ยกเลิกกลางคัน)" % (shot_i, len(SHOTS)))
    return 0 if shot_i > 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
