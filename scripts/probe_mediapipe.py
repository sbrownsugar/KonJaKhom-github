"""ตรวจว่า mediapipe build นี้มีอะไรให้ใช้บ้าง"""
import mediapipe as mp

print("version:", mp.__version__)
print("file   :", mp.__file__)
print("\nสิ่งที่ module มี (ไม่ขึ้นต้นด้วย _):")
attrs = [a for a in dir(mp) if not a.startswith("_")]
print(" ", attrs)

print("\nลองเข้า tasks API (ตัวใหม่):")
try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    print("  OK: mediapipe.tasks.python.vision ใช้ได้")
    print("  vision มี:", [a for a in dir(vision) if "Pose" in a or "Landmark" in a])
except Exception as e:
    print("  FAIL tasks:", type(e).__name__, e)

print("\nลองเข้า solutions API (ตัวเก่า):")
try:
    import mediapipe.python.solutions.pose as legacy_pose
    print("  OK: legacy solutions.pose ใช้ได้")
except Exception as e:
    print("  FAIL solutions:", type(e).__name__, e)
