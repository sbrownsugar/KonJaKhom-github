# -*- coding: utf-8 -*-
"""ครอปโลโก้จากไฟล์ต้นฉบับ 1024x559 -> badge สี่เหลี่ยมมน พื้นโปร่งใส"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SRC = r"C:\Users\USER\Downloads\messageImage_1784211553594.jpg"
OUT = r"D:\Obec\project\assets\logo.png"

im = Image.open(SRC).convert("RGB")
a = np.asarray(im).astype(np.int16)
h, w = a.shape[:2]
print("ต้นฉบับ: %dx%d" % (w, h))

lum = a.mean(axis=2)
sat = a.max(axis=2) - a.min(axis=2)

# badge = ขาวจัดและไม่มีสี · พื้นหลังเป็นฟ้าเทาเบลอ (มี sat และมืดกว่า)
white = (lum > 243) & (sat < 10)
print("พิกเซลขาวจัด: %d (%.1f%% ของภาพ)" % (white.sum(), 100 * white.mean()))

lab, n = ndimage.label(white)
print("ก้อนขาวทั้งหมด: %d ก้อน" % n)

# *** ห้ามเลือก "ก้อนใหญ่สุด" *** — พื้นหลังฝั่งซ้ายของภาพขาวจ้ากว่า badge และใหญ่กว่า
# badge อยู่กลางภาพเสมอ -> เลือกก้อนที่ "กรอบครอบจุดกึ่งกลางภาพ" และใหญ่สุดในบรรดานั้น
cx, cy = w // 2, h // 2
objs = ndimage.find_objects(lab)
cands = []
for i, sl in enumerate(objs, start=1):
    if sl is None:
        continue
    ys, xs = sl
    if xs.start <= cx < xs.stop and ys.start <= cy < ys.stop:
        area = int((lab[sl] == i).sum())
        cands.append((area, i, sl))
print("ก้อนที่ครอบจุดกึ่งกลาง (%d,%d): %d ก้อน" % (cx, cy, len(cands)))
assert cands, "หา badge ไม่เจอ (ไม่มีก้อนขาวไหนครอบจุดกึ่งกลาง)"

area, big, sl = max(cands)
badge = ndimage.binary_fill_holes(lab == big)      # เติมรูข้างใน (ตัวโลโก้เป็นสีเข้ม = รู)
print("เลือกก้อน #%d ขนาด %d px" % (big, area))

by, bx = np.where(badge)
x0, x1, y0, y1 = bx.min(), bx.max(), by.min(), by.max()
print("กรอบ badge: (%d,%d)-(%d,%d)  ขนาด %dx%d" % (x0, y0, x1, y1, x1 - x0, y1 - y0))
assert x0 > 20 and y0 > 5, "!! กรอบชิดขอบภาพ = ยังจับพื้นหลังอยู่"

# ต้องเป็นจัตุรัสคร่าว ๆ ไม่งั้นแปลว่าจับผิด
ar = (x1 - x0) / float(y1 - y0)
print("อัตราส่วน: %.3f  %s" % (ar, "OK (ใกล้จัตุรัส)" if 0.85 < ar < 1.18 else "!! ไม่ใช่จัตุรัส จับผิดแน่"))
assert 0.85 < ar < 1.18, "จับ badge ผิด"

crop = im.crop((x0, y0, x1 + 1, y1 + 1))
S = 512
crop = crop.resize((S, S), Image.LANCZOS)

# ตรวจมุม: ต้องเป็นขาวจริง (ถ้าเทา = ยังติดพื้นหลัง)
c = np.asarray(crop)
print("\nตรวจมุมหลังครอป (ควรขาว ~250):")
for nm, (py, px) in (("บนซ้าย", (30, 30)), ("บนขวา", (30, S - 31)),
                     ("ล่างซ้าย", (S - 31, 30)), ("ล่างขวา", (S - 31, S - 31))):
    print("  %-9s RGB=%s" % (nm, tuple(int(v) for v in c[py, px])))

# ล้างคราบเทาที่ขอบ: พื้น badge ควรขาวสนิท แต่ในไฟล์มีเงา/เบลอทำให้ออกเทาอมฟ้า
# -> พิกเซลที่ "สว่างพอและแทบไม่มีสี" ดันให้เป็นขาว 255 ไปเลย
# ปลอดภัยเพราะลายเส้นโลโก้เป็นฟ้า/เขียว/กรมท่า (sat สูง หรือ lum ต่ำ) ไม่เข้าเงื่อนไขนี้
cc = np.asarray(crop).astype(np.int16)
l2 = cc.mean(axis=2)
s2 = cc.max(axis=2) - cc.min(axis=2)
bg = (l2 > 206) & (s2 < 34)
print("\nล้างคราบเทา: %d px (%.1f%% ของ badge) -> ขาวสนิท" % (bg.sum(), 100 * bg.mean()))
cc[bg] = 255
crop = Image.fromarray(cc.astype(np.uint8), "RGB")

# ตรวจว่าไม่ได้ล้างโดนลายเส้นโลโก้: ต้องยังเหลือพิกเซลเข้ม/สีจัดอยู่พอสมควร
ink_left = ((l2 <= 206) | (s2 >= 34)).sum()
print("พิกเซลลายเส้นที่เหลือ: %d (%.1f%%)  %s"
      % (ink_left, 100.0 * ink_left / l2.size,
         "OK" if ink_left > 30000 else "!! ล้างโดนโลโก้แล้ว"))

# เก็บกวาดจุดเทาเล็ก ๆ ที่หลงเหลือ: ก้อนสีที่เล็กเกินกว่าจะเป็นส่วนของโลโก้ -> ล้างทิ้ง
cc = np.asarray(crop).astype(np.int16)
notwhite = cc.mean(axis=2) < 250
lab2, n2 = ndimage.label(notwhite)
sizes2 = ndimage.sum(notwhite, lab2, range(1, n2 + 1))
print("\nก้อนที่ไม่ใช่สีขาว: %d ก้อน" % n2)
allsz = sorted([int(s) for s in sizes2], reverse=True)
print("  ขนาดทุกก้อน: %s" % allsz)

# เกณฑ์ตั้งจากขนาดก้อนจริงของภาพนี้ (พิมพ์ออกมาดูแล้ว ไม่ได้เดา):
#   18230, 7365, 4436, 3205  = ตัวโลโก้ (คน · โล่ · ลูกศร · ป้ายเตือน)
#   1216 ... 655             = ตัวอักษร "KONCHA KHOM" 10 ตัว + ชิ้นส่วนย่อยของโลโก้
#   -------- 500 --------      <- ช่องว่างจริงอยู่ตรงนี้ (655 vs 367)
#   367, 210, 145, 136, 30...= จุดเปื้อนจากพื้นหลัง (ลบทิ้ง)
#
# บทเรียน: เคยตั้ง 900 (ลบตัวอักษรหมด เหลือ "N" กับ "M")
#          แล้วลองสูตร "ช่องว่างกว้างสุด" (ไปเจอ 18230->7365 เหลือก้อนเดียว)
#          ค่าคงที่ที่อ่านจากข้อมูลจริงชนะสูตรฉลาด ๆ ที่ไม่ได้ดูข้อมูล
#   -> ลอง 700 แล้วมันไปฆ่าตัว "C" (C = 655 px!) กลายเป็น "KON HA KHOM"
#      แปลว่า **จุดเปื้อนใหญ่กว่าตัวอักษร** -> แยกด้วยขนาดอย่างเดียวไม่ได้ ต้องใช้สีช่วย
MIN_BLOB = 500

# แยกด้วย "สี" — วัดค่าจริงของทุกก้อนแล้ว (blobinfo.py) ไม่ได้เดา:
#   ตัวโลโก้      lum  91-152 · sat  82-178   (ฟ้า/เขียว = สีจัด)
#   ตัวหนังสือ    lum  51- 59 · sat  33- 37   (กรมท่า = เข้มมาก)
#   จุดเปื้อน     lum 205     · sat  21       (สว่างสุด + ซีดสุด = โดดออกมาชัด)
# -> เกณฑ์ lum>170 & sat<28 แยกได้สะอาด มีระยะเผื่อทั้งสองทาง
#    (เคยตั้ง sat<18 พลาดไป 3 หน่วย จุดเปื้อนเลยรอด)
GREY_LUM, GREY_SAT = 170, 28
lum_all = cc.mean(axis=2)
sat_all = cc.max(axis=2) - cc.min(axis=2)
killed_grey = 0
for i, s in enumerate(sizes2, start=1):
    if s >= MIN_BLOB:
        m = lab2 == i
        L, St = lum_all[m].mean(), sat_all[m].mean()
        if L > GREY_LUM and St < GREY_SAT:
            print("     ลบก้อนเทา %5d px (lum %.0f, sat %.0f) <- พื้นหลังหลงมา" % (s, L, St))
            cc[m] = 255
            killed_grey += 1
print("  ลบก้อนสีเทา (ไม่สนขนาด): %d ก้อน" % killed_grey)

killed = 0
for i, s in enumerate(sizes2, start=1):
    if s < MIN_BLOB:
        cc[lab2 == i] = 255
        killed += 1
print("  ลบก้อนเล็กกว่า %d px: %d ก้อน" % (MIN_BLOB, killed))
crop = Image.fromarray(cc.astype(np.uint8), "RGB")

c = np.asarray(crop)
print("ตรวจมุมหลังล้าง:")
for nm, (py, px) in (("บนซ้าย", (30, 30)), ("ล่างขวา", (S - 31, S - 31))):
    print("  %-9s RGB=%s" % (nm, tuple(int(v) for v in c[py, px])))

# มุมมนโปร่งใส ให้เข้ากับพื้นขาวของแอป
r = int(S * 0.21)
alpha = Image.new("L", (S, S), 0)
ImageDraw.Draw(alpha).rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=255)
out = crop.convert("RGBA")
out.putalpha(alpha)
out.save(OUT)
out.resize((64, 64), Image.LANCZOS).save(r"D:\Obec\project\assets\logo_64.png")
print("\nบันทึก: logo.png (512x512) + logo_64.png")
