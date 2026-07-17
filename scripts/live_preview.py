# -*- coding: utf-8 -*-
"""
live_preview.py — ดูกล้องสด + จุดร่างกายที่ AI จับ แบบ real-time
รัน:  python scripts/live_preview.py [camera_index]   (ค่าเริ่มต้น 0 = กล้องโน้ตบุ๊ก)
ปิด:  กด ESC  ·  ถ่ายเก็บภาพ: กด SPACE
"""
import os
import sys
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "project", "models", "pose_landmarker_lite.task")
IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SAVE_DIR = r"C:\Users\USER\AppData\Local\Temp\claude\d--Obec\86f2a295-74f4-4f12-8f17-14d469ee29d6\scratchpad"

# เส้นเชื่อมโครงร่าง (คู่ของ landmark)
CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
        (0, 7), (0, 8), (7, 11), (8, 12)]
KEY = [7, 8, 11, 12, 23, 24, 27, 28]
GREEN, WHITE, CYAN, AMBER = (90, 210, 130), (255, 255, 255), (255, 220, 60), (0, 170, 255)


def angle_at(b, a, c):
    """มุมภายในที่จุด b (ระหว่าง a-b-c) องศา"""
    v1 = (a[0] - b[0], a[1] - b[1]); v2 = (c[0] - b[0], c[1] - b[1])
    d = math.hypot(*v1) * math.hypot(*v2)
    if d == 0:
        return None
    cosv = max(-1, min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / d))
    return math.degrees(math.acos(cosv))


def text(img, s, xy, color=WHITE, scale=0.7, th=2):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def main():
    if not os.path.exists(MODEL):
        print("ไม่พบโมเดล:", MODEL); return
    lm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(IDX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("เปิดกล้อง index %d ไม่ได้" % IDX); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    win = "ก่อนจะค่อม - กล้องสด (ESC=ปิด, SPACE=ถ่าย)"
    print("เปิดกล้องสดแล้ว (index %d) · กด ESC ที่หน้าต่างเพื่อปิด" % IDX)

    ts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts += 33
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        status, scol = "ไม่พบคน - ยืนหน้ากล้อง", AMBER
        if res.pose_landmarks:
            L = res.pose_landmarks[0]
            P = lambda i: (int(L[i].x * w), int(L[i].y * h))
            # เส้นโครงร่าง
            for a, b in CONN:
                if L[a].visibility >= 0.3 and L[b].visibility >= 0.3:
                    cv2.line(frame, P(a), P(b), GREEN, 3, cv2.LINE_AA)
            # จุด
            for i in KEY:
                if L[i].visibility >= 0.3:
                    cv2.circle(frame, P(i), 7, CYAN, -1)
                    cv2.circle(frame, P(i), 7, WHITE, 2)
            seen = sum(1 for i in KEY if L[i].visibility >= 0.5)
            full = all(L[i].visibility >= 0.5 for i in (11, 12, 23, 24, 27, 28))
            # มุมไหล่สด (ทดลอง) เมื่อเห็น หู-ไหล่-สะโพก ข้างใดข้างหนึ่ง
            for ear, sh, hip, side in ((7, 11, 23, "L"), (8, 12, 24, "R")):
                if all(L[i].visibility >= 0.5 for i in (ear, sh, hip)):
                    ang = angle_at(P(sh), P(ear), P(hip))
                    if ang:
                        text(frame, "angle(ear-sh-hip) %s: %.0f deg" % (side, ang), (20, h - 25), CYAN, 0.7)
                    break
            if full:
                status, scol = "FULL BODY - พร้อมวัด (%d/8)" % seen, GREEN
            else:
                status, scol = "STAND BACK - ถอยให้เห็นสะโพก/ขา (%d/8)" % seen, AMBER

        cv2.rectangle(frame, (0, 0), (w, 46), (30, 30, 30), -1)
        text(frame, status, (20, 32), scol, 0.8)
        text(frame, "ESC=ปิด  SPACE=ถ่าย", (w - 300, 32), WHITE, 0.6, 1)
        cv2.imshow(win, frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 32:
            p = os.path.join(SAVE_DIR, "live_snapshot.jpg")
            cv2.imwrite(p, frame)
            print("บันทึกภาพ:", p)

    cap.release()
    cv2.destroyAllWindows()
    print("ปิดกล้องแล้ว")


if __name__ == "__main__":
    main()
