# -*- coding: utf-8 -*-
"""
livecam.py — กล้องสดในหน้าเว็บ + ถ่ายอัตโนมัติ
=============================================================================
เห็นภาพตัวเอง real-time ในหน้าเว็บเลย (ไม่มีหน้าต่างเด้งแยก)
จับท่าได้ครบ -> นิ่งพอ -> นับถอยหลัง 3-2-1 -> ถ่ายเอง (ด้านหน้า -> ด้านข้าง)

*** ข้อควรรู้ 3 ข้อ ***

1) ฟังก์ชัน __call__ ถูกเรียกจาก **เธรดของ WebRTC** ไม่ใช่เธรดของ Streamlit
   -> ห้ามแตะ st.session_state ในนั้นเด็ดขาด (จะพังเงียบ ๆ)
   -> ใช้ threading.Lock เก็บสถานะแทน แล้วให้ฝั่ง Streamlit มาอ่านทีหลัง

2) ตัวหนังสือบนวิดีโอต้องเป็น **อังกฤษ/ตัวเลขเท่านั้น**
   cv2.putText วาดภาษาไทยไม่ได้ (ออกมาเป็น ????) -> คำอธิบายภาษาไทยอยู่ใน UI รอบ ๆ แทน

3) กฎเหล็ก "ออฟไลน์ 100%": streamlit-webrtc ค่าเริ่มต้นจะไปคุย STUN ของ Google
   -> ต้องส่ง iceServers=[] เสมอ (ดู RTC_CONFIG) ไม่งั้นเราโกหกเรื่องออฟไลน์
   -> ใช้ได้เพราะเบราว์เซอร์กับเซิร์ฟเวอร์อยู่เครื่องเดียวกัน (host candidate พอ)
"""
from __future__ import annotations

import os
import threading
import time

import av
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "pose_landmarker_lite.task")

# ห้ามให้ไปคุยกับเซิร์ฟเวอร์ข้างนอก — เราประกาศว่าออฟไลน์ 100%
RTC_CONFIG = {"iceServers": []}

# ดัชนี landmark (ตรงกับ src/pose.py และ scripts/auto_capture.py)
L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24
L_ANK, R_ANK = 27, 28

MIN_VIS = 0.5
STEADY_FRAMES = 12          # ต้อง "พร้อม" ติดกันกี่เฟรมก่อนเริ่มนับ
COUNTDOWN = 3.0             # วินาที

SHOTS = [("front", "FRONT - face the camera"),
         ("side", "SIDE - turn your right side to the camera")]


PAIRS = ((L_SH, R_SH), (L_HIP, R_HIP), (L_ANK, R_ANK))


def _body_ready(lms, tag="front"):
    """คนอยู่ในเฟรมครบตัวไหม

    *** บั๊กที่ทำให้ "รูปด้านข้างไม่มีวันถ่ายได้" (แก้ 16 ก.ค.) ***
    เดิมบังคับให้ทั้ง 6 จุด (ไหล่/สะโพก/ข้อเท้า ทั้งซ้ายและขวา) มี visibility >= 0.5
    แต่ตอนยืน **หันข้าง** จุดฝั่งไกลถูกลำตัวบังเสมอ -> visibility ตกต่ำกว่าเกณฑ์
    -> เกณฑ์นี้จึงเป็นไปไม่ได้ทางกายภาพสำหรับรูปด้านข้าง กล้องจะรอไปตลอดกาล
       (ด้านหน้าเห็นครบเลยผ่านฉลุย — อาการตรงกับที่ผู้ใช้เจอ)

    แก้: ด้านข้างขอแค่ **ฝั่งใดฝั่งหนึ่ง** ของแต่ละคู่เห็นชัดพอ
         ด้านหน้ายังบังคับครบทั้งคู่เหมือนเดิม (เพราะต้องวัดความไม่สมมาตร)
    """
    if tag != "front":          # ทนกว่าเช็ค == "side" เผื่อวันหลังเปลี่ยนชื่อ tag
        idx = []
        for a, b in PAIRS:
            best = a if lms[a].visibility >= lms[b].visibility else b
            if lms[best].visibility < MIN_VIS:
                return False, "Stand so your whole body is visible"
            idx.append(best)
    else:
        idx = [i for pair in PAIRS for i in pair]
        for i in idx:
            if lms[i].visibility < MIN_VIS:
                return False, "Stand so your whole body is visible"

    ys = [lms[i].y for i in idx]
    xs = [lms[i].x for i in idx]
    if min(ys) < 0.03 or max(ys) > 0.99 or min(xs) < 0.02 or max(xs) > 0.98:
        return False, "Step back a bit"

    sh_y = (lms[L_SH].y + lms[R_SH].y) / 2
    hip_y = (lms[L_HIP].y + lms[R_HIP].y) / 2
    if hip_y - sh_y < 0.12:
        return False, "Stand up straight"

    # ด้านข้าง: เตือน (ไม่บล็อก) ถ้ายังหันหน้าเข้ากล้องอยู่ — ไหล่สองข้างห่างกันแนวนอน = ยังไม่ได้หันข้าง
    # ด้านข้าง: กันการถ่ายรูป "หันหน้า" ไปเก็บเป็นรูปด้านข้าง (จะทำให้มุมที่วัดได้เป็นขยะ)
    #
    # ⚠️ ค่า 0.28 นี้ "ตั้งเอง" จากการเดา ไม่ได้วัดจากคนจริง — เดิมตั้ง 0.12 แล้วสงสัยว่า
    #    มันไปบล็อกการถ่ายด้านข้าง เลยผ่อนให้หลวมมาก (เอาไว้กันแค่กรณีหันหน้าเต็ม ๆ)
    #    ตัวเลข dx จริงถูกพิมพ์บนวิดีโอแล้ว (แถบ DBG) -> ปรับให้ตรงความจริงได้เมื่อมีข้อมูล
    if tag != "front" and abs(lms[L_SH].x - lms[R_SH].x) > 0.28:
        return False, "Turn your RIGHT side to the camera"

    return True, "Ready"


def _debug_line(lms):
    """ตัวเลขจริงที่ใช้ตัดสิน — เอาขึ้นจอเลย จะได้ไม่ต้องเดากันว่าติดตรงไหน"""
    if lms is None:
        return "DBG: no landmarks"
    return "DBG vis sh %.2f/%.2f hip %.2f/%.2f ank %.2f/%.2f | dx %.3f" % (
        lms[L_SH].visibility, lms[R_SH].visibility,
        lms[L_HIP].visibility, lms[R_HIP].visibility,
        lms[L_ANK].visibility, lms[R_ANK].visibility,
        abs(lms[L_SH].x - lms[R_SH].x))


def _banner(img, text, color=(255, 255, 255), y=36, scale=0.75):
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _big_number(img, n):
    h, w = img.shape[:2]
    txt = str(n)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)
    x, y = (w - tw) // 2, (h + th) // 2
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 18, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (60, 200, 255), 10, cv2.LINE_AA)


class AutoShooter:
    """สถานะการถ่าย — ถูกเรียกทุกเฟรมจากเธรด WebRTC"""

    def __init__(self):
        self._lock = threading.Lock()
        self._lm = None
        self.shots = {}
        self.i = 0
        self.steady = 0
        self.t0 = None
        self.force = False          # ผู้ใช้กด "ถ่ายเดี๋ยวนี้" -> ถ่ายเฟรมถัดไปเลย ไม่สนเงื่อนไข
        self.why = "-"              # เหตุผลล่าสุดที่ยังไม่ถ่าย (ให้ฝั่ง UI เอาไปโชว์)

    # ---------- ฝั่ง Streamlit เรียก ----------
    def reset(self):
        with self._lock:
            self.shots = {}
            self.i = 0
            self.steady = 0
            self.t0 = None
            self.force = False

    def request_shot(self):
        """ถ่ายเฟรมถัดไปทันที ข้ามการตรวจท่าทั้งหมด

        *** ทำไมต้องมี ***
        การตรวจท่าอัตโนมัติเป็นของ "ดีถ้าได้" — แต่ถ้ามันไม่ยอมถ่าย ผู้ใช้ต้องมีทางออกเสมอ
        ไม่งั้นเครื่องมือทั้งตัวใช้ไม่ได้เพราะเกณฑ์ที่เราเดาเอาเอง
        """
        with self._lock:
            self.force = True

    def snapshot(self):
        """คืนสำเนาภาพที่ถ่ายได้ + ความคืบหน้า + เหตุผลที่ยังไม่ถ่าย (ปลอดภัยข้ามเธรด)"""
        with self._lock:
            return dict(self.shots), self.i, len(SHOTS), self.why

    # ---------- ฝั่ง WebRTC เรียก ----------
    def _landmarker(self):
        if self._lm is None:
            if not os.path.exists(MODEL):
                raise RuntimeError("ไม่พบโมเดล: %s" % MODEL)
            self._lm = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
                    running_mode=vision.RunningMode.IMAGE))
        return self._lm

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)                       # ภาพกระจก ให้ผู้ใช้ขยับตัวถูกด้าน
        clean = img.copy()                           # เก็บเฟรมสะอาดไว้ (ไม่มีเส้น overlay)

        with self._lock:
            done = self.i >= len(SHOTS)
        if done:
            _banner(img, "DONE - all shots captured", (60, 220, 120), 36)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        try:
            res = self._landmarker().detect(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        except Exception as e:                       # noqa: BLE001
            _banner(img, "Pose error: %s" % str(e)[:40], (0, 100, 255), 36)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        with self._lock:
            tag, instr = SHOTS[self.i]

        ready, why = False, "No person detected"
        lms = None
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            h, w = img.shape[:2]
            for i in (L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK):
                if lms[i].visibility >= MIN_VIS:
                    cv2.circle(img, (int(lms[i].x * w), int(lms[i].y * h)), 5, (60, 200, 120), -1)
            ready, why = _body_ready(lms, tag)      # เกณฑ์ต่างกันระหว่าง front / side

        with self._lock:
            _banner(img, "Shot %d/%d" % (self.i + 1, len(SHOTS)), (0, 220, 255), 34)
            _banner(img, instr, (255, 255, 255), 64, 0.6)
            # ตัวเลขจริงที่ใช้ตัดสิน — โชว์ไว้เลย จะได้ debug ได้โดยไม่ต้องเดา
            _banner(img, _debug_line(lms), (200, 200, 200), 90, 0.42)
            self.why = why

            take = self.force                      # กดปุ่ม "ถ่ายเดี๋ยวนี้" = ข้ามทุกเงื่อนไข
            if ready and not take:
                self.steady += 1
                if self.steady >= STEADY_FRAMES:
                    if self.t0 is None:
                        self.t0 = time.time()
                    remain = COUNTDOWN - (time.time() - self.t0)
                    if remain > 0:
                        _big_number(img, int(remain) + 1)
                    else:
                        take = True
                else:
                    _banner(img, "READY - hold still", (60, 220, 120), img.shape[0] - 20, 0.7)
            elif not ready:
                self.steady = 0
                self.t0 = None
                _banner(img, why, (0, 165, 255), img.shape[0] - 20, 0.7)

            if take:
                ok, buf = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    self.shots[tag] = buf.tobytes()
                    self.i += 1
                self.steady = 0
                self.t0 = None
                self.force = False
                img[:] = 255                       # แฟลชขาว บอกว่าถ่ายแล้ว

        return av.VideoFrame.from_ndarray(img, format="bgr24")
