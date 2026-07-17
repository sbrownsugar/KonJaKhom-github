"""ทดสอบ MediaPipe PoseLandmarker (Tasks API) แบบครบวงจร
ดาวน์โหลดโมเดล -> ตรวจจับท่าทางจากภาพคน -> คำนวณมุมจริง
"""
import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_DIR = r"D:\Obec\project\models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("ดาวน์โหลดโมเดล pose_landmarker_lite.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
print("โมเดล: %s (%.1f MB)" % (MODEL_PATH, os.path.getsize(MODEL_PATH) / 1e6))

# ภาพทดสอบ: ดาวน์โหลดภาพคนยืนจริงจาก MediaPipe test assets
TEST_IMG = os.path.join(MODEL_DIR, "_test_person.jpg")
if not os.path.exists(TEST_IMG):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-assets/pose.jpg", TEST_IMG)
print("ภาพทดสอบ:", TEST_IMG)

options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    image = mp.Image.create_from_file(TEST_IMG)
    result = landmarker.detect(image)

if not result.pose_landmarks:
    print(">>> ไม่พบท่าทาง")
    raise SystemExit(1)

lm = result.pose_landmarks[0]
print("\nตรวจพบท่าทาง! landmark ทั้งหมด =", len(lm))

NAMES = {0: "จมูก", 7: "หูซ้าย", 8: "หูขวา", 11: "ไหล่ซ้าย",
         12: "ไหล่ขวา", 23: "สะโพกซ้าย", 24: "สะโพกขวา"}
for i, th in NAMES.items():
    p = lm[i]
    print("  [%2d] %-9s x=%.3f y=%.3f visibility=%.2f" % (i, th, p.x, p.y, p.visibility))


def angle_from_vertical(dx, dy):
    """มุมของเวกเตอร์เทียบแนวดิ่ง (องศา)"""
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


# --- คำนวณ 3 ฟีเจอร์หลักของเรา (พิสูจน์ว่าสูตรใช้ได้จริง) ---
h, w = cv2.imread(TEST_IMG).shape[:2]
P = lambda i: np.array([lm[i].x * w, lm[i].y * h])

ear_r, sh_r, hip_r = P(8), P(12), P(24)
ear_l, sh_l, hip_l = P(7), P(11), P(23)

fha = angle_from_vertical(ear_r[0] - sh_r[0], sh_r[1] - ear_r[1])
spa = angle_from_vertical(sh_r[0] - hip_r[0], sh_r[1] - hip_r[1])
shoulder_tilt = float(np.degrees(np.arctan2(abs(sh_l[1] - sh_r[1]),
                                            abs(sh_l[0] - sh_r[0]) + 1e-6)))
hip_tilt = float(np.degrees(np.arctan2(abs(hip_l[1] - hip_r[1]),
                                       abs(hip_l[0] - hip_r[0]) + 1e-6)))
asym = 0.6 * shoulder_tilt + 0.4 * hip_tilt

print("\n=== 3 ฟีเจอร์หลักของเรา (คำนวณจากภาพจริง) ===")
print("  1. ดัชนีศีรษะยื่นหน้า (FHA proxy) : %.1f องศา" % fha)
print("  2. แนวโน้มไหล่ห่อ (SPA proxy)     : %.1f องศา" % spa)
print("  3. ดัชนีความไม่สมมาตร (AI)        : %.1f องศา" % asym)
print("     (ไหล่เอียง %.1f°, สะโพกเอียง %.1f°)" % (shoulder_tilt, hip_tilt))
print("\n>>> ครบวงจร: MediaPipe -> landmark -> มุมสรีระ ใช้งานได้จริง")
