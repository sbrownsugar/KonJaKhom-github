# -*- coding: utf-8 -*-
"""
make_pose_examples.py — สร้างรูป "ตัวอย่างท่าถ่าย" แบบภาพเงา (ไม่ใช่คนจริง)
=====================================================================
ออกไฟล์ 2 รูปใน project/assets/ :
    pose_example_side.png   (ด้านข้าง)
    pose_example_front.png  (ด้านหน้า)
เป็นภาพเงา/หุ่นเส้น ไม่มีใบหน้า ไม่มีคนจริง -> ปลอดภัยด้าน PDPA และ พ.ร.บ.คอมพิวเตอร์
ให้ผู้ใช้ดูเป็นแนวทางว่า "ยืนยังไง กล้องอยู่ตรงไหน"
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "project", "assets")
os.makedirs(ASSET, exist_ok=True)

SIL = "#64748b"      # สีภาพเงา (เทาน้ำเงิน)
GUIDE = "#22a06b"    # เส้นไกด์เขียว
DOT = "#e8a33d"      # จุดสำคัญ (หู/ไหล่/สะโพก)
BG = "#f8fafc"


def _limb(ax, x, y, lw=14):
    ax.plot(x, y, color=SIL, lw=lw, solid_capstyle="round", zorder=2)


def make_side():
    fig, ax = plt.subplots(figsize=(3.4, 5.2), dpi=130)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # เส้นไกด์แนวดิ่ง (หู-ไหล่-สะโพก ควรเรียงใกล้แนวเดียว)
    ax.plot([0.52, 0.52], [0.08, 0.93], color=GUIDE, lw=1.5, ls="--", zorder=1)

    # ขา (โปรไฟล์ — ยืนตรง)
    _limb(ax, [0.52, 0.53], [0.08, 0.46])          # ขา
    # ลำตัว (ไหล่ -> สะโพก)
    _limb(ax, [0.52, 0.53], [0.46, 0.72], lw=22)   # ลำตัวหนา
    # แขนปล่อยข้าง (โปรไฟล์เห็นแขนทับลำตัวเล็กน้อย)
    _limb(ax, [0.54, 0.55], [0.70, 0.44], lw=10)
    # คอ
    _limb(ax, [0.52, 0.52], [0.72, 0.80], lw=12)
    # หัว (ไม่มีหน้า)
    ax.add_patch(Circle((0.52, 0.85), 0.055, color=SIL, zorder=3))

    # จุดสำคัญ: หู(หน้า) ไหล่ สะโพก + ป้ายกำกับ (AI ใช้ 3 จุดนี้วัดมุมคอ)
    for (x, y, t) in [(0.565, 0.85, "หู"), (0.52, 0.72, "ไหล่"), (0.525, 0.46, "สะโพก")]:
        ax.add_patch(Circle((x, y), 0.017, color=DOT, zorder=4))
        ax.text(x + 0.035, y, t, va="center", ha="left", fontsize=9,
                color="#8a5a10", fontname="Tahoma", fontweight="bold", zorder=5)
    ax.text(0.66, 0.63, "หู–ไหล่–สะโพก\nควรเรียงใกล้แนวเดียว", ha="left", va="center",
            fontsize=7.5, color=GUIDE, fontname="Tahoma")

    # พื้น
    ax.plot([0.2, 0.85], [0.08, 0.08], color="#cbd5e1", lw=2)
    # กล้อง
    ax.add_patch(FancyBboxPatch((0.05, 0.42), 0.08, 0.06, boxstyle="round,pad=0.005",
                                fc="#334155", ec="none", zorder=5))
    ax.annotate("", xy=(0.2, 0.5), xytext=(0.14, 0.47),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4))
    ax.text(0.09, 0.36, "กล้อง\nระดับไหล่", ha="center", va="top", fontsize=8,
            color="#334155", fontname="Tahoma")
    ax.text(0.52, 0.015, "ด้านข้าง", ha="center", fontsize=12, color=SIL,
            fontname="Tahoma", fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.savefig(os.path.join(ASSET, "pose_example_side.png"),
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def make_front():
    fig, ax = plt.subplots(figsize=(3.4, 5.2), dpi=130)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # เส้นไกด์แนวกลางลำตัว (ควรสมมาตรซ้าย-ขวา)
    ax.plot([0.5, 0.5], [0.08, 0.93], color=GUIDE, lw=1.5, ls="--", zorder=1)

    # ขา (แยกเท่าช่วงไหล่)
    _limb(ax, [0.5, 0.42], [0.46, 0.08])
    _limb(ax, [0.5, 0.58], [0.46, 0.08])
    # ลำตัว
    _limb(ax, [0.5, 0.5], [0.46, 0.72], lw=24)
    # ไหล่ (แนวนอน สมมาตร)
    _limb(ax, [0.36, 0.64], [0.71, 0.71], lw=14)
    # แขนปล่อยข้างลำตัว
    _limb(ax, [0.36, 0.33], [0.71, 0.45], lw=10)
    _limb(ax, [0.64, 0.67], [0.71, 0.45], lw=10)
    # คอ + หัว (ไม่มีหน้า)
    _limb(ax, [0.5, 0.5], [0.72, 0.80], lw=12)
    ax.add_patch(Circle((0.5, 0.85), 0.055, color=SIL, zorder=3))

    # จุดสมมาตร: ไหล่ซ้าย-ขวา, สะโพกซ้าย-ขวา
    for x in (0.36, 0.64):
        ax.add_patch(Circle((x, 0.71), 0.016, color=DOT, zorder=4))
    for x in (0.44, 0.56):
        ax.add_patch(Circle((x, 0.47), 0.014, color=DOT, zorder=4))
    ax.text(0.5, 0.66, "ไหล่ซ้าย–ขวา ควรอยู่ระดับเดียวกัน", ha="center", va="center",
            fontsize=7.5, color=GUIDE, fontname="Tahoma")

    ax.plot([0.2, 0.8], [0.08, 0.08], color="#cbd5e1", lw=2)
    ax.text(0.5, 0.015, "ด้านหน้า — ยืนตรง มองตรง", ha="center", fontsize=11, color=SIL,
            fontname="Tahoma", fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.savefig(os.path.join(ASSET, "pose_example_front.png"),
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    make_side()
    make_front()
    for f in ("pose_example_side.png", "pose_example_front.png"):
        p = os.path.join(ASSET, f)
        print("สร้าง: %s  (%.0f KB)" % (p, os.path.getsize(p) / 1024))
