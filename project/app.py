# -*- coding: utf-8 -*-
"""
app.py — "ก่อนจะค่อม"
=============================================================================
รัน:  D:\\Obec\\.venv\\Scripts\\streamlit.exe run D:\\Obec\\project\\app.py

โครงหน้าจอ (แยกเป็น 3 ขั้น — ผู้ใช้ขอให้แยกหน้า):
  ขั้น 1  คำถาม  (ธงแดง -> ติด = หยุด · แบบสอบถามพฤติกรรม)
  ขั้น 2  ถ่ายรูป (ไม่บังคับ — มีรูปตัวอย่างท่าถ่าย)
  ขั้น 3  ผลลัพธ์ + คำแนะนำ + กระจก 8 สัปดาห์ (+ ดาวน์โหลด PDF)

กฎเหล็กที่ฝังไว้ในโค้ด:
  * ห้ามใช้คำว่า "วินิจฉัย" / "คุณเป็นออฟฟิศซินโดรม"
  * ห้ามโชว์ "โอกาสเป็นโรค %" -> โชว์ "ดัชนีเทียบกลุ่มอ้างอิง" แทน
  * "กลุ่มอาการที่สัมพันธ์" พูดได้ (เชิงกลุ่ม + มีที่มา) แต่ห้ามตัดสินว่าเป็น "โรค" รายบุคคล
  * ทุกตัวเลขต้องมีที่มาให้กดดูได้
  * ภาพถูกแปลงเป็นตัวเลของศาแล้วทิ้ง ไม่เก็บไฟล์ภาพ
"""
from __future__ import annotations

import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import scoring, recommend, mirror, dashboard, theme   # noqa: E402
from src.pose import analyze_side, analyze_front, PoseError    # noqa: E402

_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_64.png")
st.set_page_config(page_title="ก่อนจะค่อม", layout="wide",
                   page_icon=_ICON if os.path.exists(_ICON) else "🪞")
theme.inject_css()      # ธีมทั้งแอป — ต้องเรียกทันทีหลัง set_page_config

BAND_COLOR = {"เขียว": "#4A9D5B", "เหลือง": "#E8A33D", "แดง": "#D9534F"}
BAND_EMOJI = {"เขียว": "🟢", "เหลือง": "🟡", "แดง": "🔴"}
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# ==========================================================================
def header():
    st.markdown(
        "<h1 style='margin-bottom:0'>ก่อนจะค่อม</h1>"
        "<p style='color:#666;margin-top:4px;font-size:1.05em'>"
        "ระบบ<b>คัดกรองความเสี่ยง</b>เชิงท่าทางและพฤติกรรม — ใช้ได้กับทุกคน (นักเรียน · วัยทำงาน)</p>",
        unsafe_allow_html=True)
    st.info(
        "**นี่ไม่ใช่การวินิจฉัยโรค** ระบบนี้ประเมินความเสี่ยงเบื้องต้นจากพฤติกรรมเท่านั้น "
        "ไม่สามารถใช้แทนความเห็นแพทย์ได้ · หากมีอาการผิดปกติ กรุณาปรึกษาแพทย์หรือนักกายภาพบำบัด",
        icon="⚕️")


# ==========================================================================
# ตัวช่วยของ "เครื่องมือคัดกรองส่วนบุคคล" (ทุกคนใช้เหมือนกัน — ไม่มีโหมดครู/นักเรียนแล้ว)
# ==========================================================================
def _step_header(step: int):
    names = {1: "คำถาม", 2: "ถ่ายรูป (ไม่บังคับ)", 3: "ผลลัพธ์ + คำแนะนำ"}
    cols = st.columns(3)
    for i in (1, 2, 3):
        mark = "✅" if i < step else ("🔵" if i == step else "⚪")
        weight = "700" if i == step else "400"
        color = "#0f172a" if i == step else "#94a3b8"
        cols[i - 1].markdown(
            "<div style='font-weight:%s;color:%s'>%s ขั้น %d · %s</div>"
            % (weight, color, mark, i, names[i]), unsafe_allow_html=True)
    st.divider()


def _reset_flow():
    for k in ("pstep", "pans", "pposture"):
        st.session_state.pop(k, None)
    try:
        _shooter().reset()      # ล้างภาพจากกล้องด้วย ไม่ให้ค้างไปหาคนถัดไป
    except Exception:           # noqa: BLE001  (ยังไม่เคยเปิดกล้อง = ไม่มีอะไรต้องล้าง)
        pass


# ---- กลุ่มอาการที่ "สัมพันธ์" กับพฤติกรรม (เวอร์ชันปลอดภัย — ไม่ใช่การวินิจฉัยโรค) ----
# ผู้ใช้ขอ "บอกว่าเสี่ยงโรคอะไร" แต่กฎเหล็กของเราคือ 'คัดกรอง ไม่วินิจฉัย'
# เราจึงพูดถึง "กลุ่มอาการ (symptom group)" ที่งานวิจัยเชื่อมโยงกับพฤติกรรมเหล่านี้
# ไม่ใช่ตัดสินว่าใครเป็น "โรค" ชื่อเฉพาะ และย้ำเสมอว่าไม่ได้แปลว่าเป็นหรือจะเป็น
SYMPTOM_INTRO = ("กลุ่มอาการเหล่านี้ในทางการแพทย์อาจครอบคลุมได้หลายภาวะที่เกี่ยวกับ"
                 "กล้ามเนื้อและท่าทาง (กลุ่มไม่จำเพาะ ไม่ร้ายแรง) — ยกมาเป็นความรู้เท่านั้น")
# Q19 (16 ก.ค.): เวอร์ชันสำหรับผู้เยาว์ — "ไม่" ลิสต์ชื่อภาวะ กันการวินิจฉัยโดยนัย
SYMPTOM_INTRO_MINOR = ("ด้านล่างคือ “กลุ่มอาการ” ที่พฤติกรรมของคุณสัมพันธ์ด้วยตามงานวิจัย "
                       "— บอกแค่ว่าปัจจัยไหนของคุณเกี่ยวข้อง ไม่ได้แปลว่าคุณเป็นหรือจะเป็นอะไร")
SYMPTOM_DISCLAIMER = ("⚠️ **นี่ไม่ใช่การวินิจฉัย** ระบบคัดกรองจากพฤติกรรมเท่านั้น และ"
                      "**บอกไม่ได้ว่าคุณเป็นภาวะใดหรือเป็นหรือไม่** — การจะรู้ต้องให้แพทย์ตรวจ "
                      "ส่วนภาวะร้ายแรง/ปวดผิดปกติ ระบบดักด้วยคำถามความปลอดภัยตอนต้นไปแล้ว")
# ตัวอย่างภาวะต่อกลุ่มอาการ — *ผ่านด่านตรวจความปลอดภัยทางการแพทย์แบบปรปักษ์แล้ว (safe)*
# ทั้งหมดเป็นกลุ่ม "ไม่จำเพาะ/กล้ามเนื้อ" (benign, mechanical) ตรงหลักฐาน ไม่ใช่โรคโครงสร้าง
# โรคโครงสร้างร้ายแรง (หมอนรองกระดูกทับเส้น/รากประสาท/cauda equina ฯลฯ) เราดักด้วยธงแดง -> ส่งแพทย์
# จึงห้ามลิสต์ที่นี่ (จะทำให้ตกใจ + ผิดหลักฐาน)
SYMPTOM_EXAMPLES = {
    "อาการปวดคอ–บ่า (neck–shoulder pain)":
        "อาการปวดคอแบบไม่จำเพาะ (non-specific neck pain), กล้ามเนื้อคอ–บ่าตึงจากท่าทาง "
        "(postural neck pain), กลุ่มอาการปวดกล้ามเนื้อและพังผืด/จุดกดเจ็บ (myofascial pain syndrome) "
        "และ “ออฟฟิศซินโดรม” (office syndrome — เป็นคำเรียกรวมของกลุ่มอาการ ไม่ใช่โรคเดี่ยว)",
    "อาการปวดหลังส่วนล่าง (low back pain)":
        "อาการปวดหลังส่วนล่างแบบไม่จำเพาะ (non-specific low back pain), กล้ามเนื้อหลังตึงล้า "
        "(lumbar muscle strain), ปวดหลังจากท่าทางเชิงกลไก (mechanical low back pain)",
    "อาการปวดศีรษะจากความตึงตัว/ความเครียด (tension-type)":
        "ปวดศีรษะจากความตึงตัว/ความเครียด (tension-type headache), ปวดจากกล้ามเนื้อและพังผืด"
        "รอบคอ-บ่าตึงตัว/จุดกดเจ็บ (myofascial pain / trigger points)",
}


def _symptom_groups(ans: dict) -> list:
    sit = ans.get("sit_hr", 0) or 0
    phone = ans.get("phone_hr", 0) or 0
    ex = ans.get("exercise_days_wk", 7)
    ex = 7 if ex is None else ex
    sleep = ans.get("sleep_hr", 8) or 8
    stress = ans.get("stress_0_10", 0) or 0
    hx = ans.get("hx_pain_12mo")

    groups = []
    r = []
    if sit >= 6: r.append("นั่งต่อเนื่องนาน")
    if phone >= 2: r.append("ก้มคอใช้มือถือ")
    if ex < 2: r.append("ออกกำลังกายน้อย")
    if hx: r.append("เคยมีอาการมาก่อน")
    if r:
        groups.append(("อาการปวดคอ–บ่า (neck–shoulder pain)", r))

    r = []
    if sit >= 6: r.append("นั่งต่อเนื่องนาน")
    if ex < 2: r.append("ออกกำลังกายน้อย")
    if r:
        groups.append(("อาการปวดหลังส่วนล่าง (low back pain)", r))

    r = []
    if stress >= 7: r.append("ความเครียดสูง")
    if sleep < 7: r.append("นอนไม่พอ")
    if r:
        groups.append(("อาการปวดศีรษะจากความตึงตัว/ความเครียด (tension-type)", r))
    # แนบ "ตัวอย่างภาวะ" ที่ผ่านการตรวจความปลอดภัยแล้วเข้ากับแต่ละกลุ่ม
    return [(n, rs, SYMPTOM_EXAMPLES.get(n, "")) for n, rs in groups]


# หมายเหตุ: เคยมี _auto_capture() ที่เปิดหน้าต่าง OpenCV แยกผ่าน subprocess — ลบทิ้งแล้ว
# เพราะเจ้าของงานต้องการ "เห็นกล้องสดในหน้าเว็บ" ไม่ใช่หน้าต่างเด้ง (ดู _live_camera_ui)
# ส่วน scripts/auto_capture.py ยังอยู่ ใช้ที่บูธเก็บข้อมูลจริงเหมือนเดิม (เซฟลง 5-เก็บข้อมูล/ภาพดิบ)


@st.cache_resource
def _shooter():
    """ตัวจับท่า/นับถอยหลัง — ต้องเป็นตัวเดิมข้ามการ rerun ของ Streamlit
    (cache_resource = สร้างครั้งเดียว ไม่งั้นสถานะนับถอยหลังจะรีเซ็ตทุกเฟรม)"""
    from src.livecam import AutoShooter
    return AutoShooter()


def _live_camera_ui() -> dict:
    """กล้องสดในหน้าเว็บ + ถ่ายอัตโนมัติ -> คืน {"front": bytes, "side": bytes}

    ถ้าไม่มี streamlit-webrtc จะถอยไปใช้ st.camera_input (กดชัตเตอร์เอง แต่ยังเห็นภาพสด)
    """
    st.markdown("##### 📸 กล้องสด — ถ่ายอัตโนมัติ ไม่ต้องมีคนช่วยกด")

    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        from src import livecam
    except Exception as e:                           # noqa: BLE001
        st.info("โหมดกล้องสดใช้ไม่ได้ (%s) — ใช้แบบกดชัตเตอร์เองแทน" % e)
        return _camera_input_fallback()

    st.caption("กด **START** แล้วจะเห็นภาพตัวเองสด ๆ ตรงนี้เลย → เดินไปยืนให้เห็น**ทั้งตัว** "
               "(หัวถึงเท้า) → พอ AI จับท่าได้ครบและคุณนิ่งพอ จะ**นับถอยหลัง 3-2-1 แล้วถ่ายเอง** "
               "ถ่าย 2 รูป: ด้านหน้า → ด้านข้าง")

    shooter = _shooter()
    try:
        ctx = webrtc_streamer(
            key="bz_livecam",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=shooter,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=livecam.RTC_CONFIG,  # iceServers=[] -> ไม่คุยกับเซิร์ฟเวอร์ข้างนอก
            async_processing=True,
        )
    except Exception as e:                       # noqa: BLE001
        # *** กล้องพังต้องไม่ลากทั้งหน้าไปด้วย ***
        # ถ้าปล่อยให้ exception ทะลุขึ้นไป ทั้งสเต็ปจะพัง = ใช้เครื่องมือคัดกรองไม่ได้เลย
        # ทั้งที่รูปเป็นของ "ไม่บังคับ" ตั้งแต่แรก -> ถอยไปใช้กล้องแบบกดชัตเตอร์เอง
        st.info("โหมดกล้องสดใช้ไม่ได้ในบริบทนี้ (%s) — ใช้แบบกดชัตเตอร์เองแทน" % str(e)[:80])
        return _camera_input_fallback()

    shots, done, total, why = shooter.snapshot()

    # ปุ่มถ่ายเอง — ต้องมีเสมอ ไม่งั้นถ้าตัวตรวจท่าไม่ยอมถ่าย ผู้ใช้จะไปต่อไม่ได้เลย
    bc = st.columns([1.3, 1])
    if bc[0].button("📸 ถ่ายเดี๋ยวนี้เลย (ไม่ต้องรอ AI จับท่า)",
                    use_container_width=True, key="bz_forceshot",
                    disabled=not ctx.state.playing):
        shooter.request_shot()
        st.toast("ถ่ายแล้ว — ถ้ายังไม่ขึ้นให้รอ 1 วินาทีแล้วกดปุ่มรีเฟรชด้านขวา")
    if bc[1].button("🔄 รีเฟรชผล", use_container_width=True, key="bz_refresh",
                    disabled=not ctx.state.playing):
        st.rerun()

    if ctx.state.playing and not shots:
        st.info("กล้องทำงานแล้ว — ไปยืนหน้ากล้องให้เห็นทั้งตัว (หัวถึง**ข้อเท้า**) ได้เลย  \n"
                "ตอนนี้ระบบบอกว่า: **%s**" % why)
        st.caption("ถ้ามันไม่ยอมนับถอยหลังสักที ใช้ปุ่ม **ถ่ายเดี๋ยวนี้เลย** ได้ — "
                   "ผลลัพธ์เหมือนกันทุกอย่าง แค่ไม่ต้องรอ AI ตัดสินว่าท่าพร้อม")
    if shots:
        st.progress(done / total, text="ถ่ายแล้ว %d / %d รูป" % (done, total))
        st.success("ภาพอยู่ใน**หน่วยความจำเท่านั้น** ยังไม่เคยถูกเขียนลงดิสก์ "
                   "และจะถูกลบทิ้งทันทีที่แปลงเป็นตัวเลขมุมเสร็จ")
        pc = st.columns(2)
        for i, (tag, th) in enumerate((("front", "ด้านหน้า"), ("side", "ด้านข้าง"))):
            if shots.get(tag):
                pc[i].image(shots[tag], caption=th, use_container_width=True)
        if st.button("🔄 ถ่ายใหม่ทั้งหมด", key="bz_recap"):
            shooter.reset()
            st.rerun()

    return shots


def _camera_input_fallback() -> dict:
    """ทางถอย: กล้องสดของ Streamlit เอง — เห็นภาพสด แต่ต้องกดชัตเตอร์เอง"""
    out = {}
    c = st.columns(2)
    for col, tag, th in ((c[0], "front", "ด้านหน้า"), (c[1], "side", "ด้านข้าง")):
        with col:
            shot = st.camera_input(th, key="bz_ci_%s" % tag)
            if shot:
                out[tag] = shot.getvalue()
    return out


def _unused_pose_examples():
    """(เลิกใช้แล้ว 16 ก.ค. — เจ้าของงานสั่งเอารูปคนตัวอย่างออก)

    ไฟล์ assets/pose_example_*.png ยังอยู่บนดิสก์ ไม่ได้ลบ
    ถ้าอยากได้กลับ: เรียกฟังก์ชันนี้ใน _step_photo() และเปลี่ยนชื่อกลับ
    """
    side = os.path.join(ASSET_DIR, "pose_example_side.png")
    front = os.path.join(ASSET_DIR, "pose_example_front.png")
    if os.path.exists(side) or os.path.exists(front):
        st.markdown("**ตัวอย่างท่าถ่าย — ยืนแบบนี้** (ภาพนี้เป็นภาพเงา ไม่ใช่คนจริง)")
        ec = st.columns(2)
        if os.path.exists(side):
            ec[0].image(side, caption="ด้านข้าง — เห็นตั้งแต่หัวถึงสะโพก แขนปล่อยข้างลำตัว")
        if os.path.exists(front):
            ec[1].image(front, caption="ด้านหน้า — ยืนตรง เท้าห่างเท่าช่วงไหล่ มองตรง")


# ==========================================================================
def _step_questions():
    st.subheader("1️⃣ คำถามคัดกรองความปลอดภัย")
    # *** ไม่บอก "บทลงโทษ" ล่วงหน้า — ถามเป็นกลาง (กันการโกหกในคำถามที่สำคัญที่สุด) ***
    st.caption("ก่อนอื่น ขอถามเรื่องความปลอดภัยสั้น ๆ — **ตอบตามจริงนะครับ** "
               "ไม่มีข้อไหนถูกหรือผิด และคำตอบนี้ไม่ถูกส่งให้ใครทั้งนั้น")

    flags = {}
    c1, c2 = st.columns(2)
    for i, (key, label) in enumerate(scoring.RED_FLAGS):
        with (c1 if i % 2 == 0 else c2):
            flags[key] = st.checkbox(label, key="rf_" + key)

    if any(flags.values()):
        hit = [lbl for k, lbl in scoring.RED_FLAGS if flags.get(k)]
        urgent = any(k in scoring.URGENT_FLAGS for k, _ in scoring.RED_FLAGS if flags.get(k))
        if urgent:
            st.error("### 🚑 ข้อนี้ควรให้แพทย์ดู **โดยเร็ว** ไม่ใช่รอไว้ก่อน")
            st.markdown("**บอกผู้ปกครอง/คนใกล้ตัว หรือไปพบแพทย์วันนี้เลยนะครับ**")
        else:
            st.warning("### 🩺 สิ่งที่คุณตอบมา ต้องใช้คนที่ตรวจร่างกายได้")
        st.markdown(
            "อาการที่คุณติ๊กไว้:\n" + "\n".join("- %s" % h for h in hit) +
            "\n\nอาการกลุ่มนี้ **ไม่ได้เกิดจากท่านั่งหรือการก้มมือถือ** "
            "การให้ท่าบริหารกับมัน นอกจากจะไม่ช่วยแล้ว **อาจทำให้แย่ลง** "
            "ระบบจึงไม่ให้ท่าบริหารและไม่ให้คะแนนความเสี่ยงในรอบนี้ — "
            "ไม่ใช่เพราะกันคุณออก แต่เพราะคำแนะนำที่ผิดฝาผิดตัวคือสิ่งที่อันตรายที่สุด")
        st.success(
            "#### 📋 เอาสิ่งนี้ไปใช้ได้เลย — ลอกไปบอกแพทย์/คนดูแล\n\n"
            "> “มีอาการ: %s\n"
            "> เป็นมา ___ สัปดาห์ · เป็นตอน ___ (เช่น ตอนเช้า / ตอนก้ม / ตลอดเวลา)\n"
            "> อยากให้ช่วยดูให้หน่อยครับ/ค่ะ”\n\n"
            "**คุยกับใครได้บ้าง:** ห้องพยาบาล/คลินิก · แพทย์ · นักกายภาพบำบัด"
            % ", ".join(hit))
        st.stop()

    # ---------------- แบบสอบถาม ----------------
    st.divider()
    st.subheader("2️⃣ พฤติกรรมประจำวัน")
    st.caption("ตอบตามความจริงในช่วง 1 เดือนที่ผ่านมา — ไม่มีคำตอบถูกผิด")

    # ---- ใครกำลังใช้: อายุ + สถานะ (แทนโหมดครู/นักเรียน) ----
    who1, who2 = st.columns([1, 2])
    with who1:
        age = st.number_input("อายุ (ปี)", 10, 80, 16, 1, key="q_age")
    with who2:
        status = st.selectbox("ตอนนี้ทำอะไรอยู่",
                              ["กำลังเรียน", "ทำงาน", "ทั้งเรียนและทำงาน", "อื่น ๆ"],
                              key="q_status")
    work_type = ""
    if status in ("ทำงาน", "ทั้งเรียนและทำงาน"):
        work_type = st.selectbox(
            "ลักษณะงานหลัก (ช่วยให้คำแนะนำตรงขึ้น)",
            ["นั่งโต๊ะ/คอมพิวเตอร์เป็นหลัก", "ยืนนาน", "ยกของหนัก/ใช้แรง",
             "ขับรถ/เดินทางนาน", "งานผสม", "อื่น ๆ"], key="q_work")
    elif status == "อื่น ๆ":
        work_type = st.text_input("ระบุเพิ่มเติม (ถ้าต้องการ)", "", key="q_workother")

    a, b, c = st.columns(3)
    with a:
        sex = st.radio("เพศ", ["ชาย", "หญิง", "ไม่ระบุ"], horizontal=True, key="q_sex")
        weight = st.number_input("น้ำหนัก (กก.)", 25.0, 200.0, 55.0, 0.5, key="q_w")
        height = st.number_input("ส่วนสูง (ซม.)", 120.0, 210.0, 165.0, 0.5, key="q_h")
    with b:
        sit = st.slider("นั่งเรียน/ทำงาน/อ่านหนังสือ (ชม./วัน)", 0.0, 16.0, 8.0, 0.5, key="q_sit")
        # ปลดเพดานให้กรอกได้เกิน 12 ชม. (สมการนับผลสูงสุดที่ 6 ชม.อยู่แล้ว จึงไม่พองเกินหลักฐาน)
        phone = st.slider("ก้มมือถือ/แท็บเล็ต (ชม./วัน)", 0.0, 24.0, 4.0, 0.5, key="q_phone")
        sleep = st.slider("นอน (ชม./คืน)", 3.0, 12.0, 7.0, 0.5, key="q_sleep")
    with c:
        ex = st.slider("ออกกำลังกายจนหอบเหนื่อย (วัน/สัปดาห์)", 0, 7, 2, key="q_ex")
        stress = st.slider("ความเครียดช่วงนี้ (0 = ไม่เครียดเลย)", 0, 10, 5, key="q_stress")

    # ---- กระเป๋า: ถามก่อนว่าสะพายไหม แล้วค่อยให้ใส่น้ำหนัก ----
    carries_bag = st.checkbox("สะพายกระเป๋า / เป้ เป็นประจำ", key="q_hasbag")
    bag, bag_style = 0.0, "ไม่สะพาย"
    if carries_bag:
        bg1, bg2 = st.columns([1, 2])
        with bg1:
            bag = st.number_input("น้ำหนักกระเป๋า (กก.)", 0.0, 20.0, 4.0, 0.5, key="q_bag")
        with bg2:
            bag_style = st.radio("สะพายแบบไหน", ["สองบ่า", "บ่าเดียว", "สะพายข้าง", "ลาก"],
                                 horizontal=True, key="q_bagstyle")

    # ---- ปัจจัยเพิ่มเติมที่มีหลักฐาน OR จริง (เพิ่ม 15 ก.ค. ตามที่ผู้ใช้เลือก) ----
    st.markdown("**อีกนิด — ปัจจัยที่งานวิจัยพบว่าเกี่ยวข้อง**")
    e1, e2 = st.columns(2)
    with e1:
        break_freq = st.selectbox(
            "ระหว่างนั่งนาน ๆ คุณลุก/เปลี่ยนท่าบ่อยแค่ไหน",
            ["ลุกบ่อย (ราวทุก 30 นาที)", "นาน ๆ ครั้ง", "แทบไม่ได้ลุกเลย"], key="q_break")
        smoker = st.checkbox("สูบบุหรี่เป็นประจำ", key="q_smoke")
    with e2:
        heavy_work = st.checkbox("งาน/กิจวัตร ต้องยก-แบกของหนัก หรือก้มบิดตัวซ้ำ ๆ เป็นประจำ",
                                 key="q_heavy")
        job_strain = st.checkbox("งาน/การเรียน กดดันหรือภาระหนักเป็นประจำ", key="q_strain")
    prior_tx = st.checkbox("เคยรับการรักษา / ทำกายภาพบำบัด เรื่องปวดคอ-บ่า-หลัง มาก่อน",
                           key="q_priortx")
    sit_bout_long = break_freq in ("นาน ๆ ครั้ง", "แทบไม่ได้ลุกเลย")

    # *** คำถามน้ำหนักมากที่สุดในสมการ (β = 1.330) — ต้องตรงนิยาม Raine = "ใน 12 เดือน" ***
    hx = st.checkbox(
        "**ใน 12 เดือนที่ผ่านมา** เคยมีอาการปวด / ชา / อ่อนแรง ที่คอ ไหล่ หรือหลัง หรือไม่",
        key="q_hx")
    st.caption("↳ นับรวมอาการที่หายไปแล้วด้วย · นี่คือปัจจัยที่งานวิจัยพบว่า"
               "**ทำนายอาการในอนาคตได้แม่นที่สุด** (Raine Study, n=686, ติดตาม 5 ปี)")

    ans = dict(sex=sex, age=age, status=status, work_type=work_type,
               weight_kg=weight, height_cm=height, bag_kg=bag, bag_style=bag_style,
               sit_hr=sit, phone_hr=phone, exercise_days_wk=ex, sleep_hr=sleep,
               stress_0_10=stress, hx_pain_12mo=hx,
               sit_bout_long=sit_bout_long, smoker=smoker, heavy_work=heavy_work,
               job_strain=job_strain, prior_treatment=prior_tx, **flags)

    st.divider()
    if st.button("ต่อไป → ถ่ายรูป (ไม่บังคับ)", type="primary", use_container_width=True):
        st.session_state.pans = ans
        st.session_state.pstep = 2
        st.rerun()


# ==========================================================================
def _step_photo():
    st.subheader("3️⃣ ภาพถ่าย (ไม่บังคับ)")
    st.caption("ไม่มีรูปก็ใช้ระบบได้ — เพราะ**ความเสี่ยงคำนวณจากพฤติกรรม ไม่ใช่จากรูป** "
               "รูปมีไว้ 2 อย่าง: ตรวจความไม่สมมาตร และวาดกระจก 8 สัปดาห์")

    with st.expander("📸 ถ่ายยังไงให้ใช้ได้ (สำคัญ)"):
        st.markdown(
            "- **มัดผมขึ้นให้พ้นหู** — AI หาจุด “หู” เพื่อวัดมุมคอ ผมบัง = ค่าเพี้ยน\n"
            "- ใส่เสื้อรัดรูป/เสื้อพละ ไม่ใช่เสื้อคลุมหลวม\n"
            "- ยืน **สบาย ๆ แบบที่ยืนปกติ** อย่าจัดท่า อย่า “ยืดตัวตรง”\n"
            "- ยืนห่างกล้อง ~2 เมตร กล้องระดับไหล่ เห็นตั้งแต่หัวถึงสะโพก")

    # ---------- กล้องสดในหน้าเว็บ + ถ่ายอัตโนมัติ ----------
    cap = _live_camera_ui()

    # ---------- ทางสำรอง: อัปโหลดไฟล์เอง ----------
    with st.expander("หรือมีไฟล์ภาพอยู่แล้ว? อัปโหลดเองได้"):
        ci, cf = st.columns(2)
        side_img = ci.file_uploader("ภาพ **ด้านข้าง**", type=["jpg", "jpeg", "png"], key="up_side")
        front_img = cf.file_uploader("ภาพ **ด้านหน้า**", type=["jpg", "jpeg", "png"], key="up_front")

    st.divider()
    b1, b2, b3 = st.columns([1, 1, 1.4])
    if b1.button("← ย้อนกลับ", use_container_width=True):
        st.session_state.pstep = 1
        st.rerun()

    go_skip = b2.button("ข้ามการถ่ายรูป", use_container_width=True)
    go_use = b3.button("ดูผลลัพธ์ →", type="primary", use_container_width=True)

    if go_skip or go_use:
        posture = None
        if go_use:
            # ภาพจากกล้องอัตโนมัติมาก่อน ถ้าไม่มีค่อยใช้ไฟล์ที่อัปโหลดเอง
            src = []
            if cap.get("side"):
                src.append((cap["side"], analyze_side, "ด้านข้าง"))
            elif side_img:
                src.append((side_img.getbuffer(), analyze_side, "ด้านข้าง"))
            if cap.get("front"):
                src.append((cap["front"], analyze_front, "ด้านหน้า"))
            elif front_img:
                src.append((front_img.getbuffer(), analyze_front, "ด้านหน้า"))

            if src:
                posture = {}
                for buf, fn, tag in src:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                        tf.write(buf)
                        path = tf.name
                    try:
                        r = fn(path)
                        posture.update(r.__dict__)
                    except PoseError as e:
                        st.warning("ภาพ%s ใช้ไม่ได้: %s" % (tag, e))
                    finally:
                        os.unlink(path)      # ลบภาพทันทีหลังแปลงเป็นตัวเลข (PDPA)
                if not posture:
                    posture = None
        st.session_state.pposture = posture
        _shooter().reset()      # ทิ้งภาพออกจากหน่วยความจำทันทีที่แปลงเป็นมุมเสร็จ (กฎเหล็ก)
        st.session_state.pstep = 3
        st.rerun()


# ==========================================================================
def _step_results():
    ans = st.session_state.get("pans")
    posture = st.session_state.get("pposture")
    if not ans:
        st.error("ข้อมูลหาย — กรุณาเริ่มใหม่")
        if st.button("← กลับไปกรอกคำถาม"):
            st.session_state.pstep = 1
            st.rerun()
        return

    is_adult = (ans.get("age") or 0) >= 20
    ref_label = "ค่าเฉลี่ยกลุ่มอ้างอิง"

    res = scoring.score(ans)
    st.header("ผลการคัดกรอง")

    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.markdown(
            "<div style='background:%s22;border-left:6px solid %s;padding:18px;border-radius:8px'>"
            "<div style='font-size:2.6em;font-weight:700;color:%s'>%s %s</div>"
            "<div style='font-size:1.05em;margin-top:8px;line-height:1.5'>"
            "ดัชนีความเสี่ยงของคุณ = <b>%.1f</b><br>"
            "<span style='color:#666;font-size:0.92em'>(%s = 1.0)</span>"
            "</div></div>"
            % (BAND_COLOR[res.band], BAND_COLOR[res.band], BAND_COLOR[res.band],
               BAND_EMOJI[res.band], res.band, res.odds_ratio, ref_label),
            unsafe_allow_html=True)
        st.caption(
            "ℹ️ ตัวเลขนี้เป็น **ดัชนีสำหรับจัดลำดับ** (odds ratio) ไม่ใช่ “จะปวดมากกว่าคนอื่น X เท่า” "
            "และไม่ใช่ “% โอกาสเป็นโรค” — สิ่งที่หลักฐานรองรับคือ **การเรียงลำดับ** ไม่ใช่ตัวเลขสัมบูรณ์")
        if is_adult:
            st.caption("⚠️ **หมายเหตุสำหรับวัยทำงาน:** ค่าอ้างอิงที่ใช้เทียบยังอิงข้อมูล"
                       "วัยรุ่นไทยเป็นหลัก สำหรับผู้ใหญ่ให้ถือเป็น **การประมาณ** "
                       "(กำลังหาค่าอ้างอิงวัยทำงานมาแทน)")
        st.write(res.advice_headline)

        rob = scoring.band_robustness(ans)
        if not rob["stable"]:
            st.warning(
                "**สีของคุณอยู่ใกล้เส้นแบ่ง — อย่าไปยึดกับสี**\n\n"
                "เส้นแบ่ง “เขียว/เหลือง/แดง” เป็นเส้นที่**เราขีดเอง** ไม่มีงานวิจัยกำหนดไว้ "
                "ถ้าขยับสมมติฐานที่เราตั้งเอง (ในช่วงที่สมเหตุสมผล) คุณจะได้ตั้งแต่ **%s** ถึง **%s**\n\n"
                "**สิ่งที่เชื่อได้จริงคือรายการด้านขวา** — ปัจจัยไหนที่ดันความเสี่ยงขึ้น และแก้ได้ไหม"
                % (rob["low"], rob["high"]), icon="🎚️")
        else:
            st.caption("✅ สีนี้**ไม่เปลี่ยน**เลย ถึงเราจะขยับสมมติฐานที่เราตั้งเองทุกค่าที่เป็นไปได้")

    with col2:
        st.markdown("**อะไรดันดัชนีขึ้น และอะไรดึงลง** — เทียบกับ%s" % ref_label)
        pos = [c for c in res.contributions if c[1] > 0]
        neg = [c for c in res.contributions if c[1] < 0]
        if pos:
            st.markdown("###### 🔺 สิ่งที่ทำให้เสี่ยงขึ้น")
            for label, pts, cite in pos:
                st.markdown("**+%.2f** &nbsp; %s" % (pts, label))
                st.caption("↳ %s" % cite)
        if neg:
            st.markdown("###### 🟢 สิ่งที่คุณทำได้ดีอยู่แล้ว (ดึงความเสี่ยงลง)")
            for label, pts, cite in neg:
                st.markdown("**%.2f** &nbsp; ไม่เข้าข่าย “%s”" % (pts, label))
        if not pos and not neg:
            st.info("พฤติกรรมของคุณใกล้เคียง%sในทุกด้าน" % ref_label)

    # ---- กลุ่มอาการที่ "สัมพันธ์" (เวอร์ชันปลอดภัย ไม่ใช่การวินิจฉัย) ----
    # Q19 (16 ก.ค.): "ชื่อภาวะ/โรคตัวอย่าง" ถือเป็น "วินิจฉัยโดยนัย" ในหน้าเด็ก
    # -> โชว์ชื่อภาวะเฉพาะผู้ใหญ่ (อายุ ≥ 20) เท่านั้น · ผู้เยาว์เห็นแค่กลุ่มอาการ + ปัจจัยที่สัมพันธ์
    show_conditions = is_adult
    groups = _symptom_groups(ans)
    if groups:
        with st.container(border=True):
            st.markdown("#### 🔎 กลุ่มอาการที่พฤติกรรมของคุณ “สัมพันธ์” ด้วย (ตามงานวิจัย)")
            st.caption(SYMPTOM_INTRO if show_conditions else SYMPTOM_INTRO_MINOR)
            for name, reasons, examples in groups:
                st.markdown("**%s**" % name)
                st.caption("↳ สัมพันธ์กับ: %s" % " · ".join(reasons))
                if examples and show_conditions:
                    st.markdown("&nbsp;&nbsp;อาจครอบคลุมภาวะ เช่น %s" % examples)
            st.warning(SYMPTOM_DISCLAIMER)
            st.info("ข่าวดีคือปัจจัยเหล่านี้ **ปรับได้** — ดูคำแนะนำด้านล่าง")

    if res.zero_weight_notes:
        with st.expander("🧐 สิ่งที่เราวัดได้ แต่ **จงใจให้ 0 คะแนน** (จุดที่เราต่างจากทีมอื่น)"):
            for label, cite in res.zero_weight_notes:
                st.markdown("**%s** → น้ำหนัก **0**" % label)
                st.caption(cite)
            st.info("เราเก็บข้อมูลพวกนี้ และเรารายงานมัน — **แต่เราไม่ให้คะแนน เพราะหลักฐานไม่ให้**")

    # ---- ธงเหลือง "สุขภาพใจ" ----
    alert = scoring.wellbeing_alert(ans)
    if alert:
        st.divider()
        with st.container(border=True):
            st.markdown("## 💛 %s" % alert["title"])
            st.markdown(alert["body"])
            st.info(alert["help"])
            st.caption("นี่ **ไม่ใช่การวินิจฉัย** และไม่ได้แปลว่าคุณ “เป็นอะไร” · ไม่มีการส่งข้อมูลนี้ให้ใคร · "
                       "เกณฑ์ที่ใช้ (เครียด ≥ %d/10 หรือ นอน ≤ %.0f ชม.) **เราตั้งเอง**"
                       % (scoring.WELLBEING_STRESS_CUT, scoring.WELLBEING_SLEEP_CUT))

    # ---- ภาพถ่าย ----
    if posture:
        st.divider()
        st.subheader("สิ่งที่กล้องเห็น")
        m = st.columns(3)
        if "fha_deg" in posture:
            m[0].metric("ศีรษะยื่นหน้า", "%.0f°" % posture["fha_deg"],
                        help="วัดจาก หู–ไหล่–สะโพก ซึ่งเป็นจุดที่ AI หาเองได้")
        if "asym_deg" in posture:
            m[1].metric("ความไม่สมมาตร ไหล่–สะโพก", "%.1f°" % posture["asym_deg"])
        m[2].metric("คุณภาพภาพ", "%.0f%%" % (100 * posture.get("quality", 0)))
        # เดิมเป็นกล่องเตือนสีเหลืองก้อนใหญ่ — เจ้าของงานว่ารก จึงยุบเหลือบรรทัดเดียว
        # แต่ **ห้ามตัดทิ้งทั้งหมด**: ถ้าโชว์ "ศีรษะยื่นหน้า 22°" ลอย ๆ โดยไม่มีคำกำกับ
        # ผู้ใช้จะแปลเองว่า 22° ปกติหรือผิดปกติ = การวินิจฉัยที่เราห้ามไว้ในกฎเหล็ก
        # -> เหลือคำกำกับสั้นที่สุดที่ยังกันการตีความผิด + เหตุผลเต็มอยู่ใน expander
        st.caption("ตัวเลขเหล่านี้มี**น้ำหนัก 0 ในคะแนน** และมีไว้อย่างเดียวคือ "
                   "**เทียบกับตัวคุณเองในอนาคต** — ระบบไม่บอกว่า “ปกติ” หรือ “ผิดปกติ”")
        with st.expander("ทำไมไม่บอกว่าปกติหรือผิดปกติ"):
            st.markdown(
                "**1 — ไม่มีค่ามาตรฐานมาเทียบ** งานวิจัยทั้งโลกวัดมุมคอจากจุด **C7** ซึ่ง AI "
                "มองไม่เห็น เราวัดจากจุดที่ AI หาได้ (หู–ไหล่) → คนละจุด เทียบกันไม่ได้\n\n"
                "**2 — ต่อให้วัดแม่น มันก็ไม่ได้ทำนายความปวดอยู่ดี** "
                "(Raine Study n=686, OR 0.24 — ค่อมกลับปวดน้อยกว่า)")
        if "fha_deg" in posture:
            out = os.path.join(tempfile.gettempdir(), "mirror_%d.png" % os.getpid())
            mirror.render_mirror(posture["fha_deg"], out)
            st.image(out, caption="กระจก 8 สัปดาห์ — ทุกเส้นมีที่มา ไม่มีตัวเลขที่เสกขึ้นเอง")

    # ---- คำแนะนำ ----
    st.divider()
    st.subheader("สิ่งที่ทำได้ตั้งแต่วันนี้")
    rec = recommend.recommend(ans, posture, set())
    st.caption("เรียงตาม**น้ำหนักความเสี่ยงจริงที่แก้ได้** — ไม่ใช่ตามความง่าย")
    if ans.get("prior_treatment"):
        st.info("💬 คุณเคยรับการรักษา/ทำกายภาพมาก่อน — ถ้ามีท่าหรือคำแนะนำที่นักกายภาพให้ไว้ "
                "ให้ทำต่อเนื่อง และปรึกษาเขาได้ว่าคำแนะนำด้านล่างเหมาะกับคุณไหม")

    # ปุ่มดาวน์โหลด PDF คำแนะนำ (ถ้าตัวแปลงพร้อม)
    # Q19: PDF ของผู้เยาว์ก็ต้องตัดชื่อภาวะออกด้วย (ไม่งั้นชื่อหลุดทางไฟล์ดาวน์โหลด)
    pdf_groups = groups if show_conditions else [(n, rs, "") for n, rs, _ in groups]
    _recommend_pdf_button(ans, rec, pdf_groups)

    for i, card in enumerate(rec["cards"], 1):
        cat = recommend.CATEGORY_TH.get(card["category"], card["category"])
        with st.container(border=True):
            st.markdown("### %d. %s" % (i, card["title_th"]))
            st.markdown("**หมวด:** %s &nbsp;·&nbsp; **หลักฐานระดับ %s**"
                        % (cat, card["evidence_tier"]))
            st.markdown("**ทำเท่าไหร่:** %s" % card["dose_th"])
            if card.get("how_th"):
                st.markdown("**ทำยังไง:** %s" % card["how_th"])
            st.success("**ทำไปทำไม:** %s" % card["reason_th"])
            with st.expander("📚 อ้างอิง (%d)" % len(card.get("cite", []))):
                for c in card.get("cite", []):
                    st.markdown("- %s  \n  %s" % (c.get("label", ""), c.get("url", "")))

    st.divider()
    st.caption("ระบบนี้เป็นเครื่องมือคัดกรอง ไม่ใช่การวินิจฉัย · "
               "ภาพถ่ายของคุณถูกแปลงเป็นตัวเลของศาแล้ว**ลบทิ้งทันที** ไม่มีการเก็บไฟล์ภาพ")
    if st.button("🔄 เริ่มประเมินใหม่", use_container_width=True):
        _reset_flow()
        st.rerun()


def _recommend_pdf_button(ans, rec, groups):
    """สร้าง PDF คำแนะนำรายบุคคลให้ดาวน์โหลด (จะเติมเต็มใน Chunk ถัดไป)"""
    # ตัวสร้าง PDF อยู่ในโมดูลแยก — ถ้ายังไม่มีก็ข้ามไปเงียบ ๆ ไม่ให้แอปพัง
    try:
        from src.report import build_recommendation_pdf
    except Exception:
        return
    try:
        pdf_bytes = build_recommendation_pdf(ans, rec, groups, SYMPTOM_INTRO, SYMPTOM_DISCLAIMER)
    except Exception as e:
        st.caption("（ยังสร้าง PDF ไม่ได้: %s）" % e)
        return
    st.download_button("📄 ดาวน์โหลดคำแนะนำเป็น PDF", data=pdf_bytes,
                       file_name="คำแนะนำ-ก่อนจะค่อม.pdf", mime="application/pdf",
                       use_container_width=True)


def student_mode():
    """เครื่องมือคัดกรองส่วนบุคคล (3 ขั้น) — ทุกคนใช้เหมือนกัน"""
    st.session_state.setdefault("pstep", 1)
    step = st.session_state.pstep
    _step_header(step)
    if step == 1:
        _step_questions()
    elif step == 2:
        _step_photo()
    else:
        _step_results()


# ==========================================================================
def teacher_mode():
    st.subheader("📊 แดชบอร์ดครู")

    st.success(
        "**หลักคิดเดียวที่ทำให้หน้านี้มีค่า**\n\n"
        "ในบรรดาปัจจัยเสี่ยงที่ “ปรับได้” **4 ใน 6 ตัวอยู่ในมือโรงเรียน ไม่ใช่ในมือเด็ก** — "
        "ตารางเรียน · ปริมาณการบ้าน · ตารางสอบ · คาบพละ\n\n"
        "การบอกเด็กคนเดียวให้ “นั่งตัวตรง” แก้อะไรไม่ได้ ถ้าตารางเรียนบังคับให้นั่งติดกัน 4 คาบ\n\n"
        "หน้านี้จึงไม่ได้มีไว้ **จับเด็ก** แต่มีไว้ให้โรงเรียน **เห็นสิ่งที่โรงเรียนแก้ได้**",
        icon="🎯")

    st.info("🔒 **ไม่มีชื่อ ไม่มีใบหน้า ไม่มีคะแนนรายบุคคล** · ไม่แสดง BMI ให้ครูเห็น\n\n"
            "**กฎปิดข้อมูล 3 ชั้น** (บังคับในโค้ด ไม่ใช่แค่สัญญา)\n"
            "1. ผู้ตอบทั้งไฟล์ < %d คน → ไม่แสดงอะไรเลย แม้แต่ภาพรวม\n"
            "2. ห้องที่มีผู้ตอบ < %d คน → ปิดทั้งแถว\n"
            "3. ถ้าปิดไปแค่ห้องเดียว → **ปิดห้องเล็กสุดที่เหลือเพิ่มอีกหนึ่ง** "
            "เพราะไม่งั้นครูเอา “ภาพรวม ลบ ห้องที่โชว์” ก็ได้ห้องที่ปิดไว้กลับมาทั้งดุ้น"
            % (dashboard.MIN_TOTAL, dashboard.MIN_CELL))

    up = st.file_uploader("อัปโหลดไฟล์คำตอบจาก Google Forms (.csv)", type=["csv"])

    # ---- ชุดข้อมูลจำลองสำหรับสาธิต ----
    # ทำไมต้องมี: ถ้าไม่มีไฟล์ หน้านี้จะว่างเปล่า -> คนเปิดดูครั้งแรกไม่มีทางเห็นว่า
    #             "กฎปิดข้อมูล 3 ชั้น" ทำงานยังไง ทั้งที่มันคือของดีที่สุดของหน้านี้
    # ปลอดภัยยังไง: ไฟล์นี้มีคอลัมน์ __FAKE_TEST_DATA_DO_NOT_USE__ เป็นคอลัมน์แรก
    #             -> ถ้าใครเอาไปทับ responses.csv แล้วรัน run_analysis.py ยามเฝ้าจะหยุดให้เอง
    #             -> และหน้านี้จะขึ้นแบนเนอร์แดงประกาศตัวตลอดเวลาที่แสดงข้อมูลชุดนี้
    demo_path = os.path.join(ASSET_DIR, "demo_synthetic_responses.csv")
    st.session_state.setdefault("bz_demo", False)
    if not up and os.path.exists(demo_path):
        st.caption("ยังไม่มีไฟล์? กดปุ่มนี้เพื่อดูว่าหน้านี้ทำงานยังไง โดยไม่ต้องใช้ข้อมูลของใครจริง")
        if st.button("🧪  ดูตัวอย่างด้วยชุดข้อมูลจำลอง", key="bz_demo_btn"):
            st.session_state.bz_demo = True
            st.rerun()

    if up:
        src, st.session_state.bz_demo = up, False
    elif st.session_state.bz_demo and os.path.exists(demo_path):
        src = demo_path
    else:
        st.caption("รอไฟล์ข้อมูล — ดาวน์โหลดจาก Google Forms แล้วลากมาวางได้เลย")
        return

    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
    import run_analysis as RA

    raw = pd.read_csv(src)

    # ประกาศตัวเสียงดังทุกครั้งที่กำลังโชว์ข้อมูลจำลอง — ห้ามให้ใครเข้าใจผิดแม้แต่วินาทีเดียว
    if any(str(c).startswith("__FAKE_TEST_DATA") for c in raw.columns):
        st.error(
            "### 🧪 นี่คือข้อมูลจำลอง ไม่ใช่นักเรียนจริง\n\n"
            "ตัวเลขทุกตัวด้านล่างถูกสร้างด้วยสคริปต์ (`scripts/make_demo_data.py`) "
            "โดยอิงความชุกของประชากรไทยจากงานวิจัย **มีไว้สาธิตกลไกของหน้านี้เท่านั้น "
            "ห้ามนำไปอ้างเป็นผลการทดลอง**\n\n"
            "ขณะนี้โครงงานยังไม่มีข้อมูลผู้เข้าร่วมจริงแม้แต่รายเดียว — "
            "เมื่อได้ข้อมูลจริงแล้วจะวิเคราะห์ตามแผนที่ลงนามล็อกไว้ก่อนเก็บข้อมูล")
        if st.button("← ออกจากโหมดสาธิต", key="bz_demo_exit"):
            st.session_state.bz_demo = False
            st.rerun()
    cols = {k: RA.find_col(raw, v) for k, v in RA.KEYS.items()}
    missing = [k for k, c in cols.items() if not c]
    if missing:
        st.error("หาคอลัมน์ไม่เจอ: %s — ไฟล์อาจไม่ใช่ผลจากฟอร์มของเรา" % missing)
        st.write("คอลัมน์ที่มีในไฟล์:", list(raw.columns))
        return

    d = pd.DataFrame()
    for k in ("sit_hr", "phone_hr", "exercise_d", "sleep_hr", "stress", "nrs"):
        d[k] = RA.to_num(raw[cols[k]])
    rf = pd.Series(False, index=raw.index)
    dead = []
    for needle in RA.RED_FLAG_KEYS:
        c = RA.find_col(raw, [needle])
        if c is None:
            continue
        hit = RA.is_yes(raw[c])
        rf |= hit
        if hit.sum() == 0:
            dead.append(str(c)[:40])
    d["red_flag"] = rf.values
    if dead:
        st.warning("⚠️ คอลัมน์ธงแดง %d ข้อไม่ match ใครเลย — ถ้าฟอร์มใช้คำตอบแบบอื่น "
                   "(เช่น “มี/ไม่มี”) ตัวเลข “ควรพบผู้เชี่ยวชาญ” จะต่ำกว่าความจริง: %s"
                   % (len(dead), ", ".join(dead)))

    room_col = RA.find_col(raw, ["รหัสห้อง", "เลขที่"])
    if room_col:
        d[room_col] = raw[room_col]

    f = dashboard.prepare(d, room_col)
    f = f.dropna(subset=["sleep_hr"] if "sleep_hr" in f else [])
    summary = dashboard.school_summary(f)

    if summary.get("too_small"):
        st.error(
            "### 🔒 ผู้ตอบยังน้อยเกินกว่าจะแสดงได้อย่างปลอดภัย\n\n"
            "ไฟล์นี้มีผู้ตอบ **%d คน** แต่แดชบอร์ดต้องมีอย่างน้อย **%d คน**\n\n"
            "ไม่ใช่ข้อจำกัดทางเทคนิค แต่เป็น**การปกป้องนักเรียน** — "
            "ในกลุ่มเล็ก ตัวเลข “เครียดสูง 80%%” แปลว่า 4 ใน 5 คน ซึ่งเดาตัวได้ทันที"
            % (summary["n"], dashboard.MIN_TOTAL))
        st.caption("เก็บข้อมูลเพิ่มแล้วอัปโหลดใหม่ได้เลยครับ")
        return

    st.divider()
    st.markdown("### ภาพรวมทั้งโรงเรียน")
    k = st.columns(4)
    k[0].metric("ผู้ตอบทั้งหมด", "%d คน" % summary["n"])
    k[1].metric("ระดับอาการเฉลี่ย", "%.1f / 10" % summary["nrs_mean"])
    k[2].metric("ควรพบผู้เชี่ยวชาญ", "%.0f%%" % summary["red_flag_pct"],
                help="ติดคำถามคัดกรองความปลอดภัยอย่างน้อย 1 ข้อ — ระบบไม่ประเมินความเสี่ยงให้คนกลุ่มนี้")
    k[3].metric("ปิดข้อมูล", "ห้อง < %d คน" % dashboard.MIN_CELL)

    st.markdown("### 🎯 3 สิ่งที่โรงเรียนแก้ได้ทันที (เรียงตามที่พบมากที่สุด)")
    for i, (label, pct, who) in enumerate(dashboard.top_actions(summary), 1):
        with st.container(border=True):
            st.markdown("#### %d. %s — พบใน **%.0f%%** ของผู้ตอบ" % (i, label, pct))
            st.progress(min(pct / 100, 1.0))
            st.markdown("**ใครคุมได้:** %s" % who)
            st.info("**ทำอะไรได้:** %s" % dashboard.ACTION_TEXT.get(label, "—"))

    st.markdown("### ปัจจัยทั้งหมด")
    st.dataframe(
        pd.DataFrame([{"ปัจจัย": l, "พบใน (%)": round(p, 1), "ใครคุมได้": w}
                      for l, p, w in summary["drivers"]]),
        use_container_width=True, hide_index=True)

    if room_col:
        st.markdown("### รายห้อง")
        tbl = dashboard.by_room(f)
        blocked = int(tbl["_ปิดข้อมูล"].sum())
        show = tbl.drop(columns=["_ปิดข้อมูล"])
        st.dataframe(show, use_container_width=True, hide_index=True)
        if blocked:
            st.caption("🔒 %d ห้องถูกปิดข้อมูล — **ไม่ใช่ข้อผิดพลาด** แต่เป็นการปกป้องความเป็นส่วนตัว\n\n"
                       "ห้องที่ผู้ตอบน้อยกว่า %d คนถูกปิดโดยตรง และถ้าเหลือห้องที่ถูกปิด "
                       "เพียงห้องเดียว ระบบจะปิดห้องเล็กสุดที่เหลือเพิ่มอีกหนึ่งห้องเสมอ "
                       "(complementary suppression)"
                       % (blocked, dashboard.MIN_CELL))

    st.divider()
    st.warning(dashboard.HONESTY_BOX, icon="🔍")


# ==========================================================================
# เส้นทางหลักของแอป:   ปก  ->  เลือกบทบาท  ->  เครื่องมือ
#
# ปก (theme.cover_page) = ตารางน้ำหนัก β ของโมเดลจริง อ่านสดจาก scoring.BETA
#   ไม่ใช่หน้าโฆษณา — ตารางบนปกคือสมการเดียวกับที่ปุ่มบนปกจะรัน
#
# เลือกบทบาท (theme.role_picker) ดึง "แดชบอร์ดกลุ่ม" ขึ้นมาอยู่ระดับเดียวกับเครื่องมือหลัก
#   เดิมมันซ่อนอยู่ใน checkbox ท้าย sidebar -> คนเปิดแอปครั้งแรกมีสิทธิ์ไม่เจอเลย
#   ทั้งที่กฎปิดข้อมูล 3 ชั้นในนั้นคือของดีที่สุดชิ้นหนึ่งของโครงงาน
#
# หมายเหตุ: "อายุ" ยังถามในสเต็ป 1 เหมือนเดิม — บทบาทที่เลือกคือ *ใช้เครื่องมือไหน*
#           ไม่ใช่ *ให้คะแนนยังไง* (คนละเรื่องกับโหมดครู/นักเรียนเดิมที่ถอดออกไปแล้ว)
# ==========================================================================
st.session_state.setdefault("bz_started", False)    # ผ่านหน้าปกแล้วหรือยัง
st.session_state.setdefault("bz_role", None)        # None = ยังไม่เลือกบทบาท


def _leave_cover():
    st.session_state.bz_started = True
    st.rerun()


def _pick_role(role: str):
    st.session_state.bz_role = role
    st.rerun()


def _back_to_roles():
    st.session_state.bz_role = None
    _reset_flow()               # ล้างคำตอบเดิม ไม่ให้ค้างข้ามบทบาท
    st.rerun()


def _sidebar():
    st.sidebar.markdown(
        "**AI ที่ใช้**  \n"
        "MediaPipe BlazePose (Google) — โครงข่ายประสาทเทียมที่หา 33 จุดบนร่างกายจากภาพ\n\n"
        "**น้ำหนักความเสี่ยง**  \n"
        "มาจากงานวิจัยที่ศึกษาคนหลักหมื่น **ไม่ได้เทรนจากข้อมูลของเรา** "
        "เพราะข้อมูลไม่กี่สิบคนเทรนโมเดลไม่ได้ — เราพิสูจน์ด้วยการจำลอง 400 รอบแล้ว")
    st.sidebar.divider()
    if st.sidebar.button("← เปลี่ยนบทบาท", use_container_width=True, key="bz_backrole"):
        _back_to_roles()
    st.sidebar.caption("โครงงาน OBEC AI Spark · โรงเรียนวิทยาศาสตร์จุฬาภรณราชวิทยาลัย ปทุมธานี")


if not st.session_state.bz_started:
    theme.cover_page(on_start=_leave_cover)
elif st.session_state.bz_role is None:
    theme.role_picker(on_self=lambda: _pick_role("self"),
                      on_org=lambda: _pick_role("org"))
else:
    _sidebar()
    header()
    if st.session_state.bz_role == "org":
        teacher_mode()
    else:
        student_mode()
