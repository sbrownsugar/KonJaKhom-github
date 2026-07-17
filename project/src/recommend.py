# -*- coding: utf-8 -*-
"""
recommend.py — เลือกคำแนะนำรายบุคคลจากคลัง 27 รายการ
=====================================================================
กฎที่คลังกำหนดไว้เอง (design_rules) และเราต้องเคารพ:
  * ธงแดง -> ระงับคำแนะนำทั้งหมด ส่งต่อผู้ใหญ่ (hard override)
  * ให้การ์ด 3-5 ใบ/ครั้ง  ไม่เกิน 1 ใบต่อหมวด  ท่าบริหารไม่เกิน 2 ใบ  ไลฟ์สไตล์อย่างน้อย 2 ใบ
  * never_surface_bmi = ห้ามพูดเรื่องน้ำหนัก/BMI กับวัยรุ่นเด็ดขาด (เสี่ยงกระตุ้นปัญหาการกิน)
  * ห้ามเขียนว่า "เพิ่มความเสี่ยง X เท่า" ต้องเขียนว่า "สัมพันธ์กับ"

*** ทำไมต้องมีไลฟ์สไตล์อย่างน้อย 2 ใบ ***
เพราะหลักฐานบอกว่าตัวทำนายจริงคือ การนอน/ออกกำลังกาย/ความเครียด ไม่ใช่มุมท่าทาง
ถ้าให้แต่ท่าบริหาร = เราสอนสิ่งที่หลักฐานบอกว่าไม่ช่วย
"""
from __future__ import annotations

import json
import os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "interventions_lifestyle.json")

with open(DATA, encoding="utf-8") as f:
    CATALOG = json.load(f)

RULES = CATALOG["design_rules"]
ITEMS = CATALOG["interventions"]

CATEGORY_TH = {
    "POS": "ท่าบริหาร", "EX": "ออกกำลังกาย", "BRK": "พักและขยับ",
    "SCR": "การใช้จอ", "ERG": "จัดโต๊ะ/จอ", "BAG": "กระเป๋า",
    "SLP": "การนอน", "STR": "ความเครียด", "NUT": "โภชนาการ",
}
POSTURE_CATS = {"POS"}
LIFESTYLE_CATS = {"EX", "SLP", "STR", "NUT", "BRK"}


def derive_flags(ans: dict, posture: dict | None = None) -> tuple[set, set]:
    """แปลงคำตอบดิบ -> ธงที่คลังคำแนะนำรู้จัก"""
    q = set()
    if ans.get("sit_hr", 0) >= 6:            q.add("study_hours_high")
    if ans.get("sit_hr", 0) >= 6:            q.add("sitting_bout_over_30min")
    if ans.get("phone_hr", 0) >= 2:          q.add("screen_over_2h")
    if ans.get("phone_hr", 0) >= 4:          q.add("screen_high")
    if ans.get("sleep_hr", 9) < 8:           q.add("sleep_under_8h")
    if ans.get("exercise_days_wk", 7) < 5:   q.add("mvpa_below_target")
    if ans.get("exercise_days_wk", 7) < 3:   q.add("strength_days_below_3")
    if ans.get("stress_0_10", 0) >= 7:       q.add("stress_high")
    if ans.get("bag_style") in ("บ่าเดียว", "สะพายข้าง"):  q.add("one_strap_bag")
    if ans.get("weight_kg") and ans.get("bag_kg"):
        if 100.0 * ans["bag_kg"] / ans["weight_kg"] >= 10:  q.add("bag_over_10pct")

    # --- ธงจากภาพถ่าย (ถ้ามี) ---
    p = set()
    if posture:
        # ใช้เกณฑ์เชิงสัมพัทธ์ ไม่ใช่เลขที่เราตั้งเอง — ดูหมายเหตุใน scoring.py
        if posture.get("fha_deg", 0) >= 20:   p.add("forward_head")
        if posture.get("trunk_deg", 0) >= 12: p.add("rounded_shoulders")
        if posture.get("asym_deg", 0) >= 4:   p.add("asymmetry")
    return q, p


# =============================================================================
# *** บั๊กเงียบที่ทำให้ตาข่ายนิรภัยชั้นที่ 2 มีรูโหว่ 4 รู ***
#
# ไฟล์นี้เคยเขียนรายการธงแดง "ของตัวเอง" แยกจาก scoring.py — แล้วทั้งสองไม่ตรงกัน
#   scoring.py ใช้คีย์ "chronic_6wk"   แต่ไฟล์นี้ดักคีย์ "chronic_3mo"  -> ดักไม่เจอ
#   และไฟล์นี้ไม่รู้จัก radiating / bladder_bowel / morning_stiff เลย (3 ข้อที่เพิ่มทีหลัง)
#
# แปลว่า: ถ้ามีใครเรียก recommend() โดยไม่ผ่านด่าน scoring.score() ก่อน
#         เด็กที่ตอบว่า "กลั้นปัสสาวะไม่อยู่" (cauda equina) จะได้ **ท่าบริหาร** กลับไป
#
# แก้ที่ต้นเหตุ: ไม่ให้มีรายการธงแดง 2 ชุดในโปรเจกต์อีกต่อไป — ดึงจาก scoring.py ชุดเดียว
#               ถ้าใครเพิ่มธงแดงใหม่ ไฟล์นี้จะรู้เองอัตโนมัติ
# =============================================================================
try:
    from .scoring import RED_FLAGS as _RED_FLAGS          # เรียกผ่านแพ็กเกจ (src.recommend)
except ImportError:                                        # รันไฟล์นี้ตรง ๆ
    from scoring import RED_FLAGS as _RED_FLAGS

RED_FLAG_KEYS = [k for k, _ in _RED_FLAGS]


def red_flag_block(ans: dict) -> str | None:
    """ธงแดง = ระงับคำแนะนำทั้งหมด (hard override ตามที่คลังกำหนด)

    รายการธงแดงมาจาก scoring.RED_FLAGS ชุดเดียว — ห้ามเขียนรายการซ้ำในไฟล์นี้
    """
    if any(ans.get(k) for k in RED_FLAG_KEYS):
        return CATALOG["referral"][0]["ui_text_th"]
    return None


# =============================================================================
# *** จุดที่เกือบทำให้ระบบขัดแย้งกับตัวเอง ***
#
# ตอนแรกเราจัดอันดับการ์ดจาก "จับคู่ธงได้กี่ตัว" ผลคือ:
#   - การ์ด "กระเป๋านักเรียน" ขึ้นอันดับ 1  ทั้งที่ scoring.py ให้น้ำหนักกระเป๋า = 0
#     (Calvo-Muñoz 2020, meta+IPD n=9,188 บอกว่าไม่สัมพันธ์กับอาการปวด)
#   - เด็กที่นอน 5.5 ชม. + เครียด 8/10 กลับ "ไม่ได้การ์ดเรื่องนอนและเครียดเลย"
#     ทั้งที่นั่นคือ 2 ใน 3 ปัจจัยที่หลักฐานบอกว่าสำคัญที่สุด
#
# => ระบบจะพูดไม่ตรงกับสมการของตัวเอง ซึ่งกรรมการจับได้แน่
#
# แก้: จัดอันดับการ์ดด้วย "น้ำหนักความเสี่ยงจริง (β) ของปัจจัยที่การ์ดนั้นแก้"
#      คือใช้ตัวเลขชุดเดียวกับ scoring.py -> ระบบพูดเป็นเสียงเดียวกัน
# =============================================================================
# หมวดการ์ด -> ปัจจัยเสี่ยงที่มันแก้ได้ (β จาก scoring.py)
CATEGORY_BETA = {
    "STR": 0.74,   # ความเครียด — Gao 2023 OR 2.09  <-- ปัจจัยที่ปรับได้ ที่แรงที่สุด
    "EX":  0.63,   # ออกกำลังกาย — Gao 2023 OR 1.88
    "SLP": 0.46,   # การนอน — Auvinen 2010 OR 1.58
    "SCR": 0.43,   # เวลาหน้าจอ — 0.215 × ~2 ชม. ที่ลดได้จริง
    "BRK": 0.23,   # ลุกขยับ (ลดเวลานั่งสะสม) — Meng 2025
    "ERG": 0.23,   # จัดโต๊ะ/จอ (ลดภาระสะสม)
    "POS": 0.50,   # ท่าบริหาร — *ไม่ได้มาจาก β ของมุม (ซึ่ง = 0)*
                   # แต่มาจาก RCT ตรงที่พบว่า "การฝึกกล้ามเนื้อคอลดอาการปวดได้จริง"
                   # (Andias 2022, RCT วัยรุ่นปวดคอเรื้อรัง n=127, คงผล 6 เดือน)
                   # => ท่าบริหารได้ผล *ถึงแม้* ท่าทางจะไม่ทำนายความปวด — คนละเรื่องกัน
    "NUT": 0.10,   # โภชนาการ — หลักฐานอ่อนต่ออาการปวดโดยตรง แต่คลังกำหนดให้เป็นการ์ดสากล
    "BAG": 0.00,   # *** กระเป๋า = 0 *** หลักฐานบอกว่าไม่สัมพันธ์กับอาการปวด
                   # เก็บไว้ในคลัง แสดงได้ถ้าถามหา แต่ห้ามขึ้นอันดับต้น ๆ
}


def _score_item(item: dict, qflags: set, pflags: set, symptoms: set) -> float:
    """คะแนน = (ปัญหานี้เป็นของคนนี้จริงไหม) × (แก้แล้วลดความเสี่ยงได้จริงแค่ไหน)"""
    t = item.get("target", {})
    cat = item["category"]

    # 1) ตรงกับปัญหาของคนนี้ไหม
    match = 0
    match += len(set(t.get("questionnaire", [])) & qflags)
    match += len(set(t.get("posture", [])) & pflags)
    match += len(set(t.get("symptom", [])) & symptoms)
    if match == 0:
        return 0.0

    # 2) แก้แล้วลดความเสี่ยงได้เท่าไหร่ (ตัวเลขชุดเดียวกับ scoring.py)
    beta = CATEGORY_BETA.get(cat, 0.0)

    # 3) หลักฐานแน่นแค่ไหน
    tier = {"A": 1.3, "B": 1.15, "C": 1.0, "D": 0.8}.get(item.get("evidence_tier", "D"), 0.8)

    # การ์ดที่ β = 0 (กระเป๋า) ได้คะแนนน้อยมาก แต่ไม่เป็นศูนย์ -> ยังโผล่ได้ถ้าไม่มีอะไรจะแนะนำ
    return (beta + 0.02) * (1.0 + 0.35 * (match - 1)) * tier


def recommend(ans: dict, posture: dict | None = None, symptoms: set | None = None) -> dict:
    block = red_flag_block(ans)
    if block:
        return {"referred": True, "message": block, "cards": []}

    qflags, pflags = derive_flags(ans, posture)
    symptoms = symptoms or set()

    scored = sorted(
        [(_score_item(i, qflags, pflags, symptoms), i) for i in ITEMS],
        key=lambda t: -t[0],
    )

    picked, used_cat = [], set()
    n_posture = n_life = 0

    def can_take(item) -> bool:
        c = item["category"]
        if RULES.get("max_cards_per_category", 1) <= len([1 for p in picked if p["category"] == c]):
            return False
        if c in POSTURE_CATS and n_posture >= RULES.get("max_posture_cards", 2):
            return False
        return True

    # รอบ 1: เอาที่ตรงปัญหาที่สุดก่อน
    for s, item in scored:
        if len(picked) >= RULES.get("max_cards_per_session", 5):
            break
        if s <= 0 or not can_take(item):
            continue
        picked.append(item); used_cat.add(item["category"])
        if item["category"] in POSTURE_CATS: n_posture += 1
        if item["category"] in LIFESTYLE_CATS: n_life += 1

    # รอบ 2: บังคับให้มีไลฟ์สไตล์อย่างน้อย 2 ใบ (กฎของคลัง)
    need_life = RULES.get("min_lifestyle_cards", 2) - n_life
    if need_life > 0:
        for s, item in scored:
            if need_life <= 0:
                break
            if item["category"] in LIFESTYLE_CATS and item not in picked and can_take(item):
                # ถ้าเต็มแล้ว ให้เตะการ์ดท่าบริหารใบท้ายสุดออก
                if len(picked) >= RULES.get("max_cards_per_session", 5):
                    for j in range(len(picked) - 1, -1, -1):
                        if picked[j]["category"] in POSTURE_CATS:
                            n_posture -= 1
                            picked.pop(j)
                            break
                    else:
                        break
                picked.append(item); used_cat.add(item["category"])
                n_life += 1; need_life -= 1

    # รอบ 3: เติมให้ครบขั้นต่ำ
    for s, item in scored:
        if len(picked) >= RULES.get("min_cards_per_session", 3):
            break
        if item not in picked and can_take(item):
            picked.append(item)

    return {"referred": False, "message": "", "cards": picked,
            "flags": {"questionnaire": sorted(qflags), "posture": sorted(pflags)}}


# =============================================================================
if __name__ == "__main__":
    demo = dict(sex="หญิง", weight_kg=52, height_cm=160, bag_kg=6, bag_style="บ่าเดียว",
                sit_hr=9, phone_hr=5, exercise_days_wk=0, sleep_hr=5.5, stress_0_10=8)
    posture = {"fha_deg": 24.0, "trunk_deg": 8.0, "asym_deg": 5.2}

    r = recommend(demo, posture, symptoms={"neck", "upper_back"})
    print("=" * 78)
    print("นักเรียนหญิง: นั่ง 9 ชม. · มือถือ 5 ชม. · ไม่ออกกำลังกาย · นอน 5.5 ชม. · เครียด 8/10")
    print("ภาพถ่าย: คอยื่น 24° · ไม่สมมาตร 5.2°")
    print("=" * 78)
    print("ธงที่ระบบจับได้:", len(r["flags"]["questionnaire"]), "จากแบบสอบถาม,",
          len(r["flags"]["posture"]), "จากภาพ")
    print("\nได้คำแนะนำ %d ข้อ (เรียงตามน้ำหนักความเสี่ยงจริงที่แก้ได้):\n" % len(r["cards"]))
    for i, c in enumerate(r["cards"], 1):
        print("─" * 78)
        print("%d. [%s · β=%.2f] %s" % (i, CATEGORY_TH.get(c["category"], c["category"]),
                                        CATEGORY_BETA.get(c["category"], 0), c["title_th"]))
        print("   ขนาด : %s" % c["dose_th"][:96])
        print("   ทำไม : %s..." % c["reason_th"][:130])
        print("   หลักฐานระดับ %s · %d อ้างอิง" % (c["evidence_tier"], len(c.get("cite", []))))

    print("\n" + "=" * 78)
    print("ทดสอบธงแดง (ปวดจนตื่นกลางดึก):")
    r2 = recommend({**demo, "night_pain": True})
    print("  ให้คำแนะนำ %d ข้อ (ต้องเป็น 0)" % len(r2["cards"]))
    print("  ข้อความ:", r2["message"][:110], "...")
