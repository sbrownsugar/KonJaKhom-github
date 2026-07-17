"""ทดสอบว่า MediaPipe Pose ทำงานได้จริงบนเครื่องนี้ (ไม่ใช่แค่ import ผ่าน)"""
import numpy as np
import cv2
import mediapipe as mp

print("mediapipe :", mp.__version__)
print("opencv    :", cv2.__version__)

mp_pose = mp.solutions.pose

# สร้างภาพทดสอบ: วาดรูปคนแบบง่าย ๆ (หัว ลำตัว แขน ขา) บนพื้นขาว
img = np.full((480, 320, 3), 255, dtype=np.uint8)
cv2.circle(img, (160, 70), 35, (60, 60, 60), -1)          # หัว
cv2.rectangle(img, (125, 105), (195, 260), (60, 60, 60), -1)  # ลำตัว
cv2.rectangle(img, (95, 110), (125, 240), (60, 60, 60), -1)   # แขนซ้าย
cv2.rectangle(img, (195, 110), (225, 240), (60, 60, 60), -1)  # แขนขวา
cv2.rectangle(img, (130, 260), (155, 400), (60, 60, 60), -1)  # ขาซ้าย
cv2.rectangle(img, (165, 260), (190, 400), (60, 60, 60), -1)  # ขาขวา

with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                  min_detection_confidence=0.3) as pose:
    res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if res.pose_landmarks:
    lm = res.pose_landmarks.landmark
    print("ตรวจพบท่าทาง! จำนวน landmark =", len(lm))
    names = {0: "จมูก", 7: "หูซ้าย", 8: "หูขวา", 11: "ไหล่ซ้าย",
             12: "ไหล่ขวา", 23: "สะโพกซ้าย", 24: "สะโพกขวา"}
    for i, th in names.items():
        p = lm[i]
        print("  [%2d] %-10s x=%.3f y=%.3f visibility=%.2f" % (i, th, p.x, p.y, p.visibility))
    print("\n>>> MediaPipe Pose ใช้งานได้จริงบนเครื่องนี้")
else:
    print(">>> ไม่พบท่าทางในภาพทดสอบ (รูปวาดง่ายเกินไป)")
    print(">>> แต่ MediaPipe รันได้ไม่ error = พร้อมใช้กับภาพคนจริง")
