# -*- coding: utf-8 -*-
"""
pose.py — ภาพ -> จุดสำคัญบนร่างกาย -> มุมสรีระ 3 ค่า
=====================================================================
*** สำคัญ: MediaPipe บน Python 3.14 มีแค่ Tasks API ***
    `mp.solutions.pose` (ที่ทุก tutorial ใช้) ไม่มีในเครื่องนี้ ห้ามเรียก

หลักการวัดมุม — "เทียบแกนลำตัว ไม่ใช่เทียบกรอบภาพ"
---------------------------------------------------------------------
ถ้าวัดมุมเทียบกับขอบภาพ กล้องเอียง 5 องศา = ทุกมุมเพี้ยน 5 องศา
เราจึงวัดมุม "ภายในร่างกาย" ซึ่งหมุนภาพยังไงค่าก็ไม่เปลี่ยน:

  fha_deg  = 180 - มุมที่หัวไหล่ ในสามเหลี่ยม (หู - ไหล่ - สะโพก)
             -> หัวตั้งตรงบนลำตัว = 0 องศา   หัวยื่นไปข้างหน้า = ค่าเพิ่ม
             -> หมุนภาพ/กล้องเอียง ค่าไม่เปลี่ยน  (invariant)

  asym_deg = | มุมเอียงของแนวไหล่  -  มุมเอียงของแนวสะโพก |
             -> กล้องเอียง 3 องศา ทำให้ทั้งไหล่และสะโพกเอียง 3 องศาเท่ากัน
                พอลบกันก็หายไป  (invariant)

  trunk_deg = มุมเอนของลำตัวเทียบแนวดิ่งของภาพ
             -> อันนี้ *ไม่* invariant  ต้องตั้งกล้องให้ได้ระดับ (เปิดเส้นกริดในแอปกล้อง)
                รายงานไว้เพื่อคุมคุณภาพ ไม่ใช่ฟีเจอร์หลัก

ข้อจำกัดที่ต้องพูดเองบนเวที (ห้ามปิด)
---------------------------------------------------------------------
  * MediaPipe ไม่มีจุด C7 และ ASIS ซึ่งเป็นจุดที่คลินิกใช้วัดจริง
    -> ค่าที่ได้เป็น "ตัวแทน (proxy)" ไม่ใช่ craniovertebral angle ทางคลินิก
  * เป็นการวัดจากภาพ 2 มิติ ตัวเอียงเข้าหากล้องนิดเดียวก็คลาดเคลื่อน
  * ผมยาวปิดหู -> จุด 'หู' เพี้ยน -> fha เพี้ยน  (จึงต้องมัดผมขึ้นตอนถ่าย)
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

# ดัชนีจุด landmark ของ MediaPipe (มีทั้งหมด 33 จุด)
NOSE, EAR_L, EAR_R = 0, 7, 8
SHO_L, SHO_R = 11, 12
HIP_L, HIP_R = 23, 24

MIN_VISIBILITY = 0.5          # ต่ำกว่านี้ = มองไม่เห็นจุดนั้นจริง ๆ อย่าเดา


class PoseError(Exception):
    """ภาพนี้ใช้ไม่ได้ — ต้องบอกผู้ใช้ให้ถ่ายใหม่ ห้ามคืนค่ามั่ว"""


@dataclass
class SideView:
    """ผลจากภาพ 'ด้านข้าง'"""
    fha_deg: float          # ศีรษะยื่นหน้า เทียบแกนลำตัว (invariant) <-- ฟีเจอร์หลัก
    trunk_deg: float        # ลำตัวเอนจากแนวดิ่ง (ใช้คุมคุณภาพภาพ)
    side: str               # 'right' | 'left'
    quality: float          # ความมั่นใจเฉลี่ยของจุดที่ใช้ (0-1)


@dataclass
class FrontView:
    """ผลจากภาพ 'ด้านหน้า' หรือ 'ด้านหลัง'"""
    asym_deg: float         # |ไหล่เอียง - สะโพกเอียง| (invariant) <-- ฟีเจอร์หลัก
    shoulder_tilt_deg: float
    hip_tilt_deg: float
    quality: float


# --------------------------------------------------------------------------
def ensure_model() -> str:
    """โหลดโมเดลถ้ายังไม่มี (ครั้งแรกครั้งเดียว ~5.6 MB)"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


_LANDMARKER = None


def _landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        _LANDMARKER = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
        )
    return _LANDMARKER


def _imread_unicode(path: str):
    """อ่านภาพที่พาธมีตัวอักษรไทยได้ (คืน None ถ้าอ่านไม่ได้ เหมือน cv2.imread)

    *** บั๊กที่จะฆ่าท่อวิเคราะห์ภาพทั้งท่อในวันเก็บข้อมูล ***
    cv2.imread บน Windows ส่งพาธเป็น ANSI -> เจอตัวอักษรไทยแล้วเปิดไฟล์ไม่เจอ
    คืน None เงียบ ๆ ไม่โยน error  และโฟลเดอร์เก็บภาพของโครงงานคือ
        5-เก็บข้อมูล\\ภาพดิบ\\   <- ไทยทั้งสองชั้น
    -> ถ้าไม่แก้ ถ่ายรูปมากี่คนก็อ่านไม่ได้สักไฟล์ ตอนรัน build_pose_csv.py

    วิธีแก้: อ่านเป็น bytes ด้วย numpy (รองรับ unicode) แล้วให้ cv2 ถอดรหัสจาก buffer
    ตรวจแล้วว่า imdecode หมุนตาม EXIF เหมือน imread ทุกพิกเซล
    (ทดสอบด้วยภาพที่ฝัง Orientation=6) -> ไม่ทำให้บั๊ก EXIF ข้างล่างกลับมา
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: str, img) -> bool:
    """เขียนภาพลงพาธที่มีตัวอักษรไทยได้ (cv2.imwrite ก็ติดปัญหาเดียวกับ imread)"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def detect_landmarks(image_path: str):
    """คืน (landmarks, width, height) — ถ้าหาคนไม่เจอให้โยน PoseError

    *** บั๊กที่เกือบทำให้ทุกรูปจากมือถือวัดผิด ***
    รูปจากมือถือมีข้อมูล EXIF Orientation ฝังอยู่ (บอกว่า "ภาพนี้ต้องหมุนก่อนดู")
      - cv2.imread()               -> หมุนให้อัตโนมัติ  (ได้ w,h *หลัง* หมุน)
      - mp.Image.create_from_file() -> ไม่สนใจ EXIF     (ได้ landmark บนภาพ *ก่อน* หมุน)
    เดิมเราเอา w,h จาก cv2 ไปคูณกับพิกัดจาก MediaPipe = คนละภาพกัน -> มุมเพี้ยนหมด
    และรูปถ่ายแนวตั้งจากมือถือ = 100% ของรูปที่นักเรียนจะอัปโหลด

    แก้: อ่านภาพด้วย cv2 ตัวเดียว (ซึ่งหมุนตาม EXIF ให้แล้ว) แล้ว *สร้าง mp.Image
         จาก array นั้นโดยตรง* -> ทั้งสองฝั่งเห็นภาพเดียวกันแน่นอน
    """
    img = _imread_unicode(image_path)     # หมุนตาม EXIF ให้แล้ว + อ่านพาธไทยได้
    if img is None:
        raise PoseError("เปิดไฟล์ภาพไม่ได้: %s" % image_path)
    h, w = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = _landmarker().detect(mp_img)
    if not result.pose_landmarks:
        raise PoseError("ตรวจไม่พบคนในภาพ — ลองถ่ายให้เห็นเต็มตัว แสงสว่างพอ พื้นหลังเรียบ")
    return result.pose_landmarks[0], w, h


# --------------------------------------------------------------------------
def _pt(lm, i, w, h) -> np.ndarray:
    return np.array([lm[i].x * w, lm[i].y * h], dtype=float)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """มุมระหว่างเวกเตอร์สองตัว (0-180 องศา)"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        raise PoseError("จุดสองจุดทับกัน วัดมุมไม่ได้ — ภาพอาจเบลอหรือคนอยู่ไกลเกินไป")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _tilt_from_horizontal(p1: np.ndarray, p2: np.ndarray) -> float:
    """มุมเอียงของเส้นที่ลากผ่าน 2 จุด เทียบแนวนอนของภาพ (องศา, -90 ถึง +90)

    *** บั๊กที่เคยทำให้ได้ asym = 355 องศา ***
    arctan2 คืนค่า -180..+180  พอถ่าย "ด้านหลัง" จุดซ้าย-ขวาสลับกัน
    เวกเตอร์จะชี้ไปทาง -x -> ค่าไปเกาะขอบ ±180 -> พอเอามาลบกันเกิดการวนขอบ
    (ไหล่เอียงจริง 5° แต่คำนวณได้ 355°)

    แก้: ทำให้เป็น "เส้นตรง" ไม่ใช่ "เวกเตอร์มีทิศ"
         เส้นที่เอียง +170° กับ -10° คือเส้นเดียวกัน -> พับเข้าช่วง -90..+90
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ang = float(np.degrees(np.arctan2(dy, dx)))
    # พับให้อยู่ในช่วง (-90, 90]  — เส้นตรงไม่มีทิศ
    while ang > 90.0:
        ang -= 180.0
    while ang <= -90.0:
        ang += 180.0
    return ang


# --------------------------------------------------------------------------
def analyze_side(image_path: str) -> SideView:
    """ภาพด้านข้าง -> มุมศีรษะยื่นหน้า (เทียบแกนลำตัว)"""
    lm, w, h = detect_landmarks(image_path)

    # เลือกข้างที่กล้องเห็นชัดกว่า — ข้างที่หันเข้ากล้องจะมี visibility สูงกว่า
    vis_r = min(lm[EAR_R].visibility, lm[SHO_R].visibility, lm[HIP_R].visibility)
    vis_l = min(lm[EAR_L].visibility, lm[SHO_L].visibility, lm[HIP_L].visibility)
    if max(vis_r, vis_l) < MIN_VISIBILITY:
        raise PoseError(
            "มองไม่เห็น หู/ไหล่/สะโพก ชัดพอ (ความมั่นใจ %.2f)\n"
            "สาเหตุที่พบบ่อย: ผมยาวปิดหู · เสื้อหลวมบังแนวไหล่ · ยืนไกลเกินไป" % max(vis_r, vis_l))

    if vis_r >= vis_l:
        side, ear, sho, hip, q = "right", _pt(lm, EAR_R, w, h), _pt(lm, SHO_R, w, h), _pt(lm, HIP_R, w, h), vis_r
    else:
        side, ear, sho, hip, q = "left", _pt(lm, EAR_L, w, h), _pt(lm, SHO_L, w, h), _pt(lm, HIP_L, w, h), vis_l

    # --- มุมภายในที่หัวไหล่: (หู <- ไหล่ -> สะโพก) ---
    # หูอยู่ตรงเหนือไหล่พอดี และไหล่อยู่เหนือสะโพกพอดี -> สามจุดเรียงเป็นเส้นตรง = 180 องศา
    # หัวยื่นไปข้างหน้า -> มุมนี้หุบลง -> fha = 180 - มุม  จะเพิ่มขึ้น
    interior = _angle_between(ear - sho, hip - sho)
    fha = 180.0 - interior

    # --- ลำตัวเอนจากแนวดิ่งของภาพ (ไม่ invariant ใช้คุมคุณภาพ) ---
    trunk_vec = sho - hip
    trunk = _angle_between(trunk_vec, np.array([0.0, -1.0]))   # y ของภาพชี้ลง

    return SideView(fha_deg=round(fha, 1), trunk_deg=round(trunk, 1), side=side, quality=round(float(q), 2))


def analyze_front(image_path: str) -> FrontView:
    """ภาพด้านหน้า/หลัง -> ดัชนีความไม่สมมาตร (ผลต่างไหล่-สะโพก = ภูมิคุ้มกันกล้องเอียง)"""
    lm, w, h = detect_landmarks(image_path)

    need = [SHO_L, SHO_R, HIP_L, HIP_R]
    q = min(lm[i].visibility for i in need)
    if q < MIN_VISIBILITY:
        raise PoseError("มองไม่เห็น ไหล่/สะโพก ทั้งสองข้างชัดพอ (ความมั่นใจ %.2f)" % q)

    sl, sr = _pt(lm, SHO_L, w, h), _pt(lm, SHO_R, w, h)
    hl, hr = _pt(lm, HIP_L, w, h), _pt(lm, HIP_R, w, h)

    sho_tilt = _tilt_from_horizontal(sr, sl)
    hip_tilt = _tilt_from_horizontal(hr, hl)

    # กล้องเอียง k องศา -> ทั้งสองค่าบวก k เท่ากัน -> ผลต่างหักล้างกันหมด
    asym = abs(sho_tilt - hip_tilt)

    return FrontView(
        asym_deg=round(asym, 1),
        shoulder_tilt_deg=round(abs(sho_tilt), 1),
        hip_tilt_deg=round(abs(hip_tilt), 1),
        quality=round(float(q), 2),
    )


def analyze_person(side_image: str, front_image: str) -> dict:
    """รวมภาพ 2 มุม -> ฟีเจอร์ท่าทางของคน 1 คน"""
    s = analyze_side(side_image)
    f = analyze_front(front_image)
    out = {**asdict(s), **asdict(f)}
    out["quality"] = round(min(s.quality, f.quality), 2)
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    test = os.path.join(MODEL_DIR, "_test_person.jpg")
    if not os.path.exists(test):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-assets/pose.jpg", test)

    print("ทดสอบกับภาพ:", test)
    s = analyze_side(test)
    print("  ด้านข้าง : fha=%.1f องศา | ลำตัวเอน=%.1f องศา | ใช้ข้าง %s | คุณภาพ %.2f"
          % (s.fha_deg, s.trunk_deg, s.side, s.quality))
    f = analyze_front(test)
    print("  ด้านหน้า : asym=%.1f องศา (ไหล่ %.1f, สะโพก %.1f) | คุณภาพ %.2f"
          % (f.asym_deg, f.shoulder_tilt_deg, f.hip_tilt_deg, f.quality))

    # --- พิสูจน์ว่ามุมของเรา 'ทนต่อการเอียงกล้อง' จริง ---
    img = _imread_unicode(test)
    h, w = img.shape[:2]
    tilted = os.path.join(MODEL_DIR, "_test_tilted.jpg")
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0)          # หมุนภาพ 8 องศา
    _imwrite_unicode(tilted, cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255)))

    s2, f2 = analyze_side(tilted), analyze_front(tilted)
    print("\n>>> เอียงกล้อง 8 องศา แล้ววัดใหม่:")
    print("  fha  : %.1f -> %.1f  (เปลี่ยน %+.1f)   [ควรใกล้ 0 = ทนการเอียง]"
          % (s.fha_deg, s2.fha_deg, s2.fha_deg - s.fha_deg))
    print("  asym : %.1f -> %.1f  (เปลี่ยน %+.1f)   [ควรใกล้ 0 = ทนการเอียง]"
          % (f.asym_deg, f2.asym_deg, f2.asym_deg - f.asym_deg))
    print("  ลำตัวเอน: %.1f -> %.1f  (เปลี่ยน %+.1f)  [ตัวนี้ *จะ* เพี้ยน ~8 องศา = ตามคาด]"
          % (s.trunk_deg, s2.trunk_deg, s2.trunk_deg - s.trunk_deg))
