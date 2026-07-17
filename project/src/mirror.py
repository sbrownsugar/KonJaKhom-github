# -*- coding: utf-8 -*-
"""
mirror.py — "กระจก": วาดภาพเงาจากมุมที่วัดได้จริง
=====================================================================
*** ทำไมไม่ใช่ "กระจกอนาคต 5 ปี" อีกต่อไป ***

เราตั้งใจจะฉายภาพ "อีก 5 ปี ถ้ายังทำแบบนี้ คุณจะค่อมแบบนี้"
แต่ค้นแล้วพบว่า **ไม่มีงานวิจัยไหนบอกได้ว่าท่าทางจะแย่ลงกี่องศาต่อปี**
ถ้าวาดออกมา = เราเสกตัวเลข = กรรมการถามคำถามเดียวก็จบ

และยังผิดกฎหมายด้วย ถ้าเอาภาพจริงของเด็กไป morph
(พ.ร.บ.คอมพิวเตอร์ ม.16 — ดัดแปลงภาพผู้อื่นให้น่าอับอาย)

*** สิ่งที่เราทำแทน — และมันดีกว่า ***
Heydari 2022 (RCT ในวัยรุ่น): ทำท่าบริหาร 8 สัปดาห์ -> มุมคอดีขึ้น +6.9 องศา *จริง*

กระจกเราจึงฉาย "อีก 8 สัปดาห์ ถ้าเริ่มวันนี้" ไม่ใช่ "อีก 5 ปี ถ้าไม่ทำอะไร"
=> เปลี่ยนจาก "ขู่ด้วยอนาคตที่เราแต่ง" เป็น "ให้ความหวังด้วยอนาคตที่พิสูจน์แล้ว"
=> ทุกเส้นบนภาพ มีที่มาบอกได้

*** กฎการวาด ***
  * ภาพเงา / เส้นโครง เท่านั้น  ห้ามใช้ภาพถ่ายจริงของนักเรียน
  * ต้องโชว์ตัวเลขและที่มาบนภาพเสมอ (โปร่งใส = ตรวจสอบได้)
  * ห้ามวาดกระดูกสันหลัง (เราไม่มีข้อมูลนั้น)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# --- ผลจาก RCT ที่เราอ้าง (ตัวเลขเดียวที่ใช้ฉายภาพ "อนาคต") ---
RCT_CVA_GAIN_8WK = 6.9      # องศา
RCT_CITE = "Heydari et al. 2022, BMC Pediatrics 22:230 — RCT ท่าบริหาร 8 สัปดาห์ในวัยรุ่น"

# =============================================================================
# 🔑 ข้อจำกัดสำคัญที่สุดของเครื่องมือ — และเหตุผลที่ Teacher Dashboard ต้องมีอยู่
#
# MDC95 = การเปลี่ยนแปลงขั้นต่ำที่เชื่อได้ว่า "เปลี่ยนจริง ไม่ใช่ความมั่วของเครื่องมือ"
# ถ้าเครื่องมือเรามั่ว ~2 องศา -> MDC95 ≈ 5.5 องศา
#
# จำลอง 4,000 รอบ (scratchpad/group_vs_individual.py) ได้ผลชัดเจน:
#
#   ❌ "นักเรียน *คนนี้* ดีขึ้นไหม"      -> จับได้แค่ 68% ของครั้ง = เชื่อไม่ได้
#   ✅ "ห้องนี้ *ทั้งห้อง* ดีขึ้นไหม" (n=25) -> จับได้ 100% และจับผลเล็กถึง 1.7 องศา
#
# ทำไมต่างกันขนาดนั้น? เพราะความมั่วของแต่ละคนเป็น *แบบสุ่ม*
# พอเฉลี่ย 25 คน มันหักล้างกันเองจนแทบหายไป
# (เหมือนวัดคนเดียวด้วยไม้บรรทัดยางยืด = มั่ว · แต่วัด 25 คนแล้วเฉลี่ย = แม่น)
#
# => แอปนักเรียน       : **ห้ามบอกใครว่า "คุณดีขึ้นแล้ว"** จนกว่าจะเปลี่ยนเกิน MDC95
# => Teacher Dashboard : **บอกได้ว่ากิจกรรมของโรงเรียนได้ผลจริงไหม** <-- นี่คือ 25 คะแนน
#
# ข้อจำกัดนี้ไม่ได้ฆ่าโครงงาน — มันคือ *เหตุผล* ที่ระบบต้องมีมุมมองระดับโรงเรียน
# =============================================================================
MDC_DEG = 5.5               # <-- ต้องแทนด้วยค่าจริงจาก analysis/reliability.py
MDC_IS_MEASURED = False     # <-- เปลี่ยนเป็น True เมื่อวัดจริงแล้ว


def _thai_font():
    """หาฟอนต์ไทยในเครื่อง Windows — ถ้าไม่เจอ matplotlib จะขึ้นสี่เหลี่ยม"""
    for name in ("Leelawadee UI", "Leelawadee", "Tahoma", "Angsana New", "Cordia New"):
        try:
            p = font_manager.findfont(font_manager.FontProperties(family=name),
                                      fallback_to_default=False)
            if p and os.path.exists(p):
                return font_manager.FontProperties(fname=p)
        except Exception:
            continue
    return None


FONT = _thai_font()


def _skeleton(ax, fha_deg: float, color: str, title: str, subtitle: str):
    """วาดเส้นโครงด้านข้าง จากมุมศีรษะยื่นหน้าที่วัดได้

    หุ่นนี้ *ไม่ใช่* ภาพกายวิภาคจริง มันคือแผนภาพที่สื่อ 'มุมเดียว' ที่เราวัดได้จริง
    เราจึงวาดแค่ 3 จุด: สะโพก -> ไหล่ -> หู  (จุดเดียวกับที่ MediaPipe ให้มา)
    """
    kw = dict(fontproperties=FONT) if FONT else {}
    hip = np.array([0.0, 0.0])
    shoulder = np.array([0.0, 1.15])
    th = np.radians(fha_deg)
    head_len = 0.62
    ear = shoulder + np.array([np.sin(th) * head_len, np.cos(th) * head_len])

    # เส้นดิ่งอ้างอิง (ถ้าหัวตั้งตรงเป๊ะ จะทับเส้นนี้พอดี)
    ax.plot([0, 0], [shoulder[1], shoulder[1] + head_len + 0.1],
            color="#B0B0B0", ls="--", lw=1.4, zorder=1)

    # ส่วนโค้งแสดงมุม
    arc = np.linspace(0, th, 40)
    r = 0.34
    ax.plot(np.sin(arc) * r, shoulder[1] + np.cos(arc) * r,
            color=color, lw=2.2, alpha=0.9, zorder=6)
    # ตัวเลของศา — วางไว้ซ้ายมือ ไม่ให้หัวบัง
    ax.text(-0.30, shoulder[1] + 0.52, "%.0f°" % fha_deg,
            color=color, fontsize=22, fontweight="bold",
            ha="right", va="center", zorder=7, **kw)

    # ลำตัว + คอ + หัว
    ax.plot([hip[0], shoulder[0]], [hip[1], shoulder[1]],
            color=color, lw=11, solid_capstyle="round", zorder=3)
    ax.plot([shoulder[0], ear[0]], [shoulder[1], ear[1]],
            color=color, lw=9, solid_capstyle="round", zorder=3)
    ax.add_patch(plt.Circle(ear, 0.23, color=color, zorder=4))
    ax.plot(*shoulder, "o", color="white", ms=7, zorder=5)

    ax.set_xlim(-0.75, 1.15)
    ax.set_ylim(-0.30, 2.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=15, color=color, fontweight="bold", pad=6, **kw)
    ax.text(0.2, -0.22, subtitle, ha="center", va="top", fontsize=10.5,
            color="#555", **kw)


def render_mirror(fha_now: float, out_path: str, doing_exercises: bool = True) -> dict:
    """วาดกระจก: 'ตอนนี้' เทียบกับ 'อีก 8 สัปดาห์ ถ้าเริ่มทำท่าบริหารวันนี้'"""
    gain = RCT_CVA_GAIN_8WK if doing_exercises else 0.0
    fha_future = max(fha_now - gain, 0.0)     # มุมยื่นหน้า *ลดลง* = ดีขึ้น

    kw = dict(fontproperties=FONT) if FONT else {}
    fig = plt.figure(figsize=(9.2, 6.8), dpi=130)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], hspace=0.30, wspace=0.05)
    axL, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axBar = fig.add_subplot(gs[1, :])

    _skeleton(axL, fha_now, "#D9534F", "ตอนนี้", "มุมที่วัดได้จากภาพของคุณ")
    _skeleton(axR, fha_future, "#4A9D5B", "อีก 8 สัปดาห์", "ถ้าทำท่าบริหารสม่ำเสมอ")

    # ---------- ไม้บรรทัด: "คุณอยู่ตรงไหน" — ไม่ใช่ "คุณปกติหรือไม่" ----------
    # *** ห้ามใส่แถบสี เขียว/เหลือง/แดง ***
    # เราไม่มีค่ามาตรฐานสำหรับวิธีวัดแบบนี้ (เราวัดจากไหล่ ไม่ใช่ C7 แบบที่งานวิจัยใช้)
    # การขีดเส้นว่า "เกินเท่านี้ = ผิดปกติ" คือการแต่งตัวเลขขึ้นมาเอง
    # -> ไม้บรรทัดนี้จึงเป็นสีเทากลาง ๆ แสดงแค่ตำแหน่ง ไม่ตัดสินอะไรทั้งสิ้น
    axBar.axhspan(0, 1, xmin=0.0, xmax=1.0, color="#D8DEE4", alpha=0.55)

    LO, HI = 0.0, 40.0
    pos = lambda v: (np.clip(v, LO, HI) - LO) / (HI - LO)

    axBar.annotate("ตอนนี้ %.0f°" % fha_now, xy=(pos(fha_now), 0.0),
                   xytext=(pos(fha_now), -1.15), ha="center", fontsize=11,
                   color="#D9534F", fontweight="bold",
                   arrowprops=dict(arrowstyle="-|>", color="#D9534F", lw=2), **kw)
    if gain > 0:
        axBar.annotate("อีก 8 สัปดาห์ %.0f°" % fha_future, xy=(pos(fha_future), 1.0),
                       xytext=(pos(fha_future), 2.15), ha="center", fontsize=11,
                       color="#4A9D5B", fontweight="bold",
                       arrowprops=dict(arrowstyle="-|>", color="#4A9D5B", lw=2), **kw)

    for v in (0, 10, 20, 30, 40):
        axBar.text(pos(v), 0.5, "%d°" % v, ha="center", va="center",
                   fontsize=9, color="#5a6472")
    axBar.set_xlim(0, 1); axBar.set_ylim(-2.2, 2.6); axBar.axis("off")
    axBar.text(0.5, -2.05, "ไม้บรรทัดนี้บอกแค่ว่าคุณอยู่ตรงไหน — ไม่ได้บอกว่าปกติหรือผิดปกติ",
               ha="center", va="bottom", fontsize=8.8, color="#5a6472",
               style="italic", **kw)

    fig.suptitle("กระจก 8 สัปดาห์ — ดีขึ้นได้ %.1f°" % gain,
                 fontsize=17, fontweight="bold", y=0.975, **kw)

    # ---- ข้อจำกัดที่ต้องพูดเอง ไม่รอให้ใครจับได้ ----
    note = (
        "ภาพนี้มาจากสูตรบรรทัดเดียว ไม่มีตัวเลขไหนที่เราเสกขึ้นเอง:  "
        "มุมอีก 8 สัปดาห์  =  มุมตอนนี้  −  %.1f°      (ที่มา: %s)\n"
        "\n"
        "ข้อจำกัดที่เราขอบอกเอง: เครื่องมือนี้ยัง \"ไม่แม่นพอ\" ที่จะบอกคุณคนเดียวว่าดีขึ้นแล้ว "
        "(ต้องเปลี่ยนเกิน %.1f° ถึงจะแยกออกจากความคลาดเคลื่อนของเครื่องมือ)\n"
        "แต่เมื่อวัดทั้งห้อง ความคลาดเคลื่อนของแต่ละคนหักล้างกัน "
        "จึงบอกโรงเรียนได้ว่ากิจกรรมได้ผลจริงหรือไม่\n"
        "ภาพขวาคือ \"สิ่งที่งานวิจัยบอกว่าจะเกิด\" ไม่ใช่ \"สิ่งที่เราสัญญาว่าจะวัดให้คุณเห็นเป็นรายคน\"\n"
        "\n"
        "เราไม่ฉายภาพ \"อีก 5 ปี\" เพราะไม่มีงานวิจัยใดบอกได้ว่าท่าทางแย่ลงกี่องศาต่อปี"
        % (RCT_CVA_GAIN_8WK, RCT_CITE, MDC_DEG)
    )
    if not MDC_IS_MEASURED:
        note += "   ·   (ค่า %.1f° ยังเป็นค่าประมาณ — จะแทนด้วยค่าจริงหลังถ่ายซ้ำ 15 คน)" % MDC_DEG

    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=7.6,
             color="#555", linespacing=1.75, **kw)

    fig.subplots_adjust(bottom=0.155, top=0.90)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"fha_now": round(fha_now, 1), "fha_8wk": round(fha_future, 1),
            "gain": gain, "cite": RCT_CITE, "path": out_path}


# =============================================================================
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs")
    os.makedirs(out, exist_ok=True)

    print("ฟอนต์ไทยที่ใช้:", FONT.get_name() if FONT else "!! ไม่เจอ — ภาษาไทยจะเป็นสี่เหลี่ยม")

    r = render_mirror(24.0, os.path.join(out, "mirror_demo.png"))
    print("\nวาดกระจกแล้ว:", r["path"])
    print("  ตอนนี้        : %.1f องศา" % r["fha_now"])
    print("  อีก 8 สัปดาห์ : %.1f องศา  (ดีขึ้น %.1f องศา)" % (r["fha_8wk"], r["gain"]))
    print("  ที่มา         :", r["cite"])
    print("\n>>> ทุกเส้นบนภาพมีที่มา ไม่มีตัวเลขไหนที่เราเสกขึ้นเอง")
