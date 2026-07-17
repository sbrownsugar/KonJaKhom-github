# -*- coding: utf-8 -*-
"""
test_webcam.py — ทดสอบระบบกล้อง "แบบไม่โต้ตอบ" (ไม่เปิดหน้าต่าง ไม่ค้าง)
ไล่หากล้องทุกตัว (index 0-3) + รอปรับแสง -> เลือกตัวที่เห็นภาพจริง
-> ให้ MediaPipe หาจุดร่างกาย -> รายงาน · เซฟภาพไว้ดูเอง (ไม่ส่งออกนอกเครื่อง)
"""
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
OUT_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\d--Obec\86f2a295-74f4-4f12-8f17-14d469ee29d6\scratchpad"
os.makedirs(OUT_DIR, exist_ok=True)

NAMES = {0: "จมูก", 7: "หูซ้าย", 8: "หูขวา", 11: "ไหล่ซ้าย", 12: "ไหล่ขวา",
         23: "สะโพกซ้าย", 24: "สะโพกขวา", 25: "เข่าซ้าย", 26: "เข่าขวา",
         27: "ข้อเท้าซ้าย", 28: "ข้อเท้าขวา"}


def grab(index, warm=35):
    """เปิดกล้อง index นี้ วอร์มอัพให้ปรับแสง แล้วคืนเฟรม + ความสว่าง"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None, 0.0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame = None
    for i in range(warm):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
        time.sleep(0.03)          # ให้เวลา auto-exposure ทำงาน
    cap.release()
    if frame is None:
        return None, 0.0
    return frame, float(frame.mean())


print("=" * 60)
print("ทดสอบระบบกล้อง — ตรวจหาคนในทุกกล้อง")
print("=" * 60)

lm = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE))
key = [7, 8, 11, 12, 23, 24, 27, 28]

best = None      # (idx, frame, seen, lms)
for idx in range(4):
    frame, bright = grab(idx)
    if frame is None:
        print("  กล้อง index %d : เปิดไม่ได้/ไม่มี" % idx)
        continue
    h, w = frame.shape[:2]
    if bright < 25:
        print("  กล้อง index %d : %dx%d · สว่าง %.0f  <- ภาพดำ (ฝาปิด?)" % (idx, w, h, bright))
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        print("  กล้อง index %d : %dx%d · สว่าง %.0f · เห็นภาพแต่ไม่พบคน" % (idx, w, h, bright))
        continue
    lms = res.pose_landmarks[0]
    seen = sum(1 for i in key if lms[i].visibility >= 0.5)
    print("  กล้อง index %d : %dx%d · สว่าง %.0f · **พบคน! เห็นชัด %d/8 จุด**"
          % (idx, w, h, bright, seen))
    if best is None or seen > best[2]:
        best = (idx, frame, seen, lms)

if best is None:
    print("-" * 60)
    print("!! ไม่พบคนในกล้องไหนเลย — เช็ค: ฝาปิดเลนส์เปิดยัง? อยู่หน้ากล้องไหม? ถอยให้เห็นทั้งตัว")
    sys.exit(0)

idx, frame, seen, lms = best
h, w = frame.shape[:2]
print("-" * 60)
print("กล้องที่ดีที่สุด: index %d (พบคน เห็นชัด %d/8 จุด)" % (idx, seen))
raw_path = os.path.join(OUT_DIR, "webcam_test_raw.jpg")
cv2.imwrite(raw_path, frame)
for i in key:
    v = lms[i].visibility
    print("    %-11s %.2f  %s" % (NAMES[i], v, "เห็นชัด" if v >= 0.5 else "ไม่ชัด"))

for i, name in NAMES.items():
    if lms[i].visibility >= 0.3:
        x, y = int(lms[i].x * w), int(lms[i].y * h)
        cv2.circle(frame, (x, y), 7, (60, 200, 120), -1)
        cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)
annot_path = os.path.join(OUT_DIR, "webcam_test_annotated.jpg")
cv2.imwrite(annot_path, frame)

print("-" * 60)
sys.path.insert(0, os.path.join(ROOT, "project"))
try:
    from src.pose import analyze_front, PoseError
    try:
        fv = analyze_front(raw_path)
        print("OK  ท่อวัดมุมด้านหน้าใช้ได้: ไม่สมมาตร %.1f° · คุณภาพ %.0f%%"
              % (fv.asym_deg, 100 * fv.quality))
    except PoseError as e:
        print("ท่อด้านหน้า: %s (ปกติถ้าไม่ได้ยืนเต็มตัวตรงหน้ากล้อง)" % e)
except Exception as e:
    print("โหลดท่อวัดมุมไม่ได้: %s" % e)

print("=" * 60)
print("สรุป: ระบบกล้อง + AI หาจุดร่างกาย = ทำงานได้ (กล้อง index %d)" % idx)
print("ภาพให้ดูเอง: %s" % annot_path)
