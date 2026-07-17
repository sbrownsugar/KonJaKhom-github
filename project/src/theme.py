# -*- coding: utf-8 -*-
"""
theme.py — ธีม + หน้าปกของ "ก่อนจะค่อม"
=============================================================================
ทิศทาง: **"ศูนย์ที่เราขีดเอง"** — หน้าปกคือ *ตารางน้ำหนัก β ของโมเดลจริง*
ไม่ใช่ภาพประกอบ ไม่ใช่สไลด์ที่มีปุ่ม Start

กฎที่ฝังไว้ในไฟล์นี้ (อย่าแก้โดยไม่อ่านเหตุผล):
  * ทุกแถวในตารางบนหน้าปก **อ่านสดจาก scoring.BETA ตอน render**
    -> แก้ scoring.py แล้วหน้าปกเปลี่ยนตามเอง  ไม่มีตัวเลขไหนพิมพ์ทับไว้ในไฟล์นี้
  * แถวที่โชว์ = **เรียงจาก β มากไปน้อยจริง ๆ** (ไม่ใช่ curated list)
    และบอกตรง ๆ ว่ากำลังโชว์กี่แถวจากทั้งหมดกี่แถว -> กรรมการเปิด scoring.py แล้วนับตรง
  * แถว β = 0 ไม่ได้ "ทิ้ง" — โค้ดเขียนว่า "เก็บไว้ รายงาน แต่ไม่ให้คะแนน"
    ป้ายบนแถวจึงต้องพูดแบบเดียวกับที่โค้ดทำ  (เส้นทับ + ป้าย "วัด · รายงาน · ไม่ให้คะแนน")
  * ห้ามคำว่า "วินิจฉัย" ในเชิงอ้างความสามารถ · ห้าม % โอกาสเป็นโรค
  * ห้ามอวดตัวเลขที่ยังไม่มี (ผู้ใช้ N คน / ความแม่นยำ %) — ยังไม่มีผู้เข้าร่วมสักแถว

ออฟไลน์ 100%: ไม่มี CDN · ไม่มีฟอนต์นอก · ไม่มี JS · selector ไหนไม่ match = หายเงียบ
=============================================================================
"""
from __future__ import annotations

import html
import os

import streamlit as st

# scoring อยู่แพ็กเกจเดียวกัน — แต่ app.py เรียกแบบ `from src import theme`
# จึงรองรับทั้งสองทาง และถ้าหาไม่เจอจริง ๆ หน้าปกยังต้องขึ้นได้ (ไม่พังทั้งแอป)
try:
    from . import scoring                      # type: ignore
except Exception:                              # pragma: no cover
    try:
        import scoring                         # type: ignore
    except Exception:
        scoring = None                         # type: ignore

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

# แถวบนสุดของตารางบนปก — โชว์กี่แถว (นอกเหนือจากแถวศูนย์ที่โชว์ครบทุกแถวเสมอ)
COVER_TOP_N = 5


# =============================================================================
# 1) CSS ทั้งแอป
# =============================================================================
_CSS = """
<style>
/* ---------- TOKENS ---------- */
/* ---------- สเกลเทา (ไล่เฉด 9 ขั้น) ----------
   เจ้าของงานขอ "โทนขาวดำ เทาไล่เฉด" -> ทั้ง UI ใช้สีเทาล้วน ไม่มีสีอื่นเลย
   ผลพลอยได้: โลโก้ (ฟ้า-เขียว) กลายเป็น *สีเดียวของทั้งหน้า* -> ตาไปหยุดที่โลโก้เอง
   ถ้าจะเติมสีอื่นวันหลัง ให้คิดก่อนว่ามันจะไปแย่งความเด่นของโลโก้ไหม */
:root{
  --g0:#FFFFFF;  --g1:#FAFAFA;  --g2:#F4F4F5;  --g3:#E8E8EA;
  --g4:#D2D2D6;  --g5:#A1A1AA;  --g6:#71717A;  --g7:#3F3F46;  --g8:#1C1C1F;

  --bz-bg:var(--g0);       --bz-surface:var(--g1);
  --bz-ink:var(--g8);      --bz-muted:var(--g6);
  --bz-accent:var(--g7);   --bz-accent-ink:var(--g0);
  --bz-rule:var(--g2);
  --bz-hair:var(--g3);
  --bz-tint:var(--g2);
  --bz-radius:14px;                /* ความมนมาตรฐาน — ปุ่ม/การ์ดใช้ค่านี้ทั้งหมด */
  --bz-font:"Leelawadee UI","Sarabun","Tahoma",sans-serif;
  --bz-mono:"Consolas","Cascadia Mono","Courier New",monospace;
}
/* *** จงใจไม่มี @media (prefers-color-scheme: dark) *** — อ่านก่อนจะเติมกลับ
   เจ้าของงานเลือก "พื้นขาว" เป็นทางเดียวของแอปนี้ และเรา pin ธีม light ไว้ที่
   .streamlit/config.toml แล้ว

   บล็อก media query เดิมเป็น **บั๊ก** ไม่ใช่ของดี:
     prefers-color-scheme ตาม *ระบบปฏิบัติการ* เสมอ — ไม่สนว่า Streamlit pin อะไรไว้
     -> เครื่องที่ตั้ง dark mode จะได้ "โครง Streamlit สีขาว + token ของเราสีดำ" = หน้าพัง
     (คอมเมนต์เดิมเขียนว่า "pin แล้วบล็อกนี้จะไม่มีผล" ซึ่งไม่จริง)
   ถ้าวันหลังอยากได้ธีมมืดจริง ๆ ต้องเอา config.toml ออกด้วย ไม่ใช่เติมแค่ media query */

/* กันเหนียว: บางเวอร์ชัน Streamlit ติด data-theme ไว้ที่ .stApp
   (เดิมเคยเขียน `[data-theme="dark"] :root` ซึ่งเป็น selector ที่ไม่มีวัน match — ตัดทิ้งแล้ว) */
/* ธีมถูก pin เป็น light ที่ .streamlit/config.toml แล้ว — ไม่ต้อง override ต่อ */

/* ---------- พื้นหลัง: ขาวล้วน ----------
   เจ้าของงานเลือก "พื้นขาว" — ตัดลายเส้นบรรทัด/เส้นขอบกระดาษเดิมออกทั้งหมด
   โครงสร้างของหน้าจึงมาจาก "เส้นขอบผมเดียว (hair) + ที่ว่าง" ล้วน ๆ ไม่ใช่จากลายพื้น
   ผลพลอยได้: ตาราง β เด่นขึ้น เพราะไม่มีลายอะไรมาแย่งสายตาอีกแล้ว */
.stApp{ background: var(--bz-bg); }
[data-testid="stAppViewContainer"], [data-testid="stHeader"]{
  background: transparent; position: relative; z-index: 1;
}
[data-testid="stHeader"]{ height: 0; }
.block-container{ max-width: 980px; padding-top: 2.0rem; padding-bottom: 3rem; }

/* ---------- ตัวอักษร ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"]{
  font-family: var(--bz-font);
  color: var(--bz-ink);
  -webkit-font-smoothing: antialiased;
}
/* line-height เผื่อวรรณยุกต์ไทย — ห้ามต่ำกว่า 1.5 · ห้าม weight 300 กับไทย */
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown{ line-height:1.62; }
.stApp h1,.stApp h2,.stApp h3,.stApp h4{ line-height:1.38; padding-top:2px; letter-spacing:0; }

/* ---------- ปก ---------- */
.bz-eyebrow{
  font-size:12.5px; font-weight:600; color:var(--bz-muted);
  margin:0 0 12px; line-height:1.6;   /* ไม่ใส่ letter-spacing/uppercase กับข้อความไทย */
}
/* ---------- ปก: ขาวล้วน · โลโก้กลางหน้า · ปุ่มเริ่ม (เจ้าของงานเลือกแบบนี้ 16 ก.ค.) ---------- */
.bz-cover{ display:flex; flex-direction:column; align-items:center; text-align:center; }
/* ชื่อบนปกตัวเล็กและบาง — เพราะ *โลโก้มีชื่อ KONCHA KHOM อยู่แล้ว*
   ถ้าทำตัวใหญ่จะกลายเป็นบอกชื่อซ้ำสองรอบ = รกโดยไม่ได้อะไร */
.bz-cover-title{
  font-size:22px; font-weight:600; line-height:1.5; padding-top:2px;
  margin:0 0 6px; color:var(--bz-ink); letter-spacing:.2px;
}
.bz-cover-sub{ font-size:13px; line-height:1.7; color:var(--g5); margin:0 0 26px; }

/* โลโก้คู่กับชื่อ (ใช้ในหน้าอื่น) — โลโก้ต้องนั่งเสมอกับตัวอักษร ไม่ใช่ลอยอยู่ข้างบน */
.bz-lockup{ display:flex; align-items:center; gap:13px; margin:0 0 6px; }
.bz-lockup .bz-title{ margin:0; }
.bz-logo{ flex:0 0 auto; color:var(--bz-ink); }   /* currentColor ของกรอบ = สีหมึกของธีม */
.bz-title{
  font-size:34px; font-weight:700; line-height:1.38; padding-top:2px;
  margin:0 0 6px; color:var(--bz-ink);
}
.bz-sub{ font-size:15px; line-height:1.62; color:var(--bz-muted); margin:0 0 22px; max-width:74ch; }
.bz-sub b{ color:var(--bz-ink); }

/* --- ตาราง β --- */
.bz-ledger{
  background:var(--bz-surface);
  border:1px solid var(--bz-hair);
  border-radius:3px 3px 0 0;          /* ล่างเหลี่ยม -> ต่อกับปุ่ม */
  padding:4px 0 0; overflow:hidden;
}
.bz-head, .bz-row{
  display:grid;
  grid-template-columns: minmax(0,1.35fr) 78px 84px minmax(0,1.35fr);
  align-items:center; gap:12px; padding:0 16px;
}
.bz-head{
  height:30px; font-size:11px; font-weight:600; color:var(--bz-muted);
  border-bottom:1px solid var(--bz-hair);
}
.bz-head span:nth-child(3){ text-align:right; }
.bz-row{ min-height:34px; padding-top:5px; padding-bottom:5px; position:relative; font-size:14.5px; }
.bz-row + .bz-row{ border-top:1px solid var(--bz-rule); }
.bz-name{ color:var(--bz-ink); line-height:1.5; }   /* ห้าม nowrap/ellipsis — จะตัดชื่อแถวทิ้ง */
.bz-cite{ font-size:11px; line-height:1.55; color:var(--bz-muted); }  /* ฟอนต์ไทยปกติ ไม่ใช่ mono */
.bz-beta{
  font-family:var(--bz-mono); font-variant-numeric:tabular-nums;
  font-size:14.5px; text-align:right; color:var(--bz-ink); white-space:nowrap;
}
.bz-bar{ height:7px; background:var(--bz-rule); border-radius:1px; position:relative; }
.bz-bar > i{ display:block; height:100%; background:var(--bz-ink); opacity:.5; border-radius:1px; }

/* --- หัวข้อคั่นก่อนแถวศูนย์: บอกว่า AI ของเรา "ทำงาน" ตรงนี้ --- */
.bz-band{
  border-top:1px solid var(--bz-hair); border-bottom:1px solid var(--bz-hair);
  background:var(--bz-tint);        /* พื้นขาวแล้ว ต้องใช้ tint ไม่ใช่ --bz-bg (จะกลืนหาย) */
  padding:8px 16px; font-size:12.5px; line-height:1.6; color:var(--bz-ink);
}
.bz-band b{ font-weight:700; }
.bz-more{ padding:7px 16px; font-size:12px; color:var(--bz-muted); line-height:1.6; }

/* --- แถวศูนย์ = พระเอก --- */
.bz-row.is-zero{ padding-top:9px; padding-bottom:9px; }
.bz-row.is-zero::after{                 /* เส้นทับพาดทั้งแถว */
  content:""; position:absolute; left:16px; right:16px; top:50%;
  height:2px; background:var(--bz-accent); opacity:.8; z-index:0;
}
.bz-row.is-zero > *{ position:relative; z-index:1; }
.bz-row.is-zero .bz-name{ color:var(--bz-ink); font-weight:600; }
.bz-row.is-zero .bz-bar{ background:transparent; }
.bz-row.is-zero .bz-bar::before{        /* บาร์ยาว 0 -> เหลือขีดที่เส้นฐาน */
  content:""; position:absolute; left:0; top:-3px; width:3px; height:13px;
  background:var(--bz-accent);
}
.bz-row.is-zero .bz-beta{               /* chip ที่เส้นทับผ่านไม่ได้ — สีเดียวของทั้งหน้า */
  background:var(--bz-surface); color:var(--bz-accent);
  font-size:17px; font-weight:700;
  border:1.5px solid var(--bz-accent); border-radius:2px;
  padding:2px 5px; margin-right:-5px;
}
.bz-tag{                                /* "วัด · รายงาน · ไม่ให้คะแนน" — พูดตรงกับที่โค้ดทำ */
  display:inline-block; margin-top:3px; padding:1px 6px;
  border:1px solid var(--bz-accent); border-radius:2px;
  background:var(--bz-surface);
  font-size:10.5px; font-weight:600; color:var(--bz-accent); line-height:1.6;
  white-space:nowrap;
}
.bz-note{
  border-top:1px solid var(--bz-hair);
  padding:11px 16px 13px; font-size:13px; line-height:1.62; color:var(--bz-muted);
}
.bz-note b{ color:var(--bz-ink); }

/* --- ปุ่มหลัก: อ่านเป็น "แถวสุดท้ายของตาราง" ---
   *** สีปุ่ม = หมึก (ink) ไม่ใช่ accent ***
   เพราะ accent คือสีเดียวของทั้งหน้า และมันถูกจองไว้ให้ "เลข 0.000" เท่านั้น
   ถ้าปุ่มเป็น accent ด้วย = สอง accent สู้กัน แล้วปุ่มชนะพระเอกของตัวเอง
   selector เผื่อไว้หลายชั้น (Streamlit เปลี่ยน attribute นี้มาแล้วหลายรอบ) */
/* ปุ่มหลัก — มน เทาเข้ม ไม่ใช่กล่องดำเหลี่ยม */
.stApp .stButton > button[kind="primary"],
.stApp .stButton > button[data-testid="baseButton-primary"],
.stApp .stButton > button[data-testid="stBaseButton-primary"]{
  background:var(--g8) !important; color:var(--g0) !important;
  border:1px solid var(--g8) !important;
  border-radius:var(--bz-radius) !important;
  font-family:var(--bz-font) !important; font-size:15.5px !important; font-weight:600 !important;
  min-height:52px !important; box-shadow:none !important;
  transition:background .14s ease, transform .05s ease;
}
.stApp .stButton > button[kind="primary"]:hover,
.stApp .stButton > button[data-testid="baseButton-primary"]:hover,
.stApp .stButton > button[data-testid="stBaseButton-primary"]:hover{
  background:var(--g7) !important; border-color:var(--g7) !important; color:var(--g0) !important;
}
.stApp .stButton > button[kind="primary"]:active{ transform:translateY(1px); }
.stApp .stButton > button:focus-visible{ outline:2px solid var(--g5) !important; outline-offset:3px; }

/* ปุ่มรอง — โปร่ง ขอบเทา มนเท่ากัน */
.stApp .stButton > button[kind="secondary"],
.stApp .stButton > button[data-testid="stBaseButton-secondary"]{
  background:var(--g0) !important; color:var(--g7) !important;
  border:1px solid var(--g3) !important;
  border-radius:var(--bz-radius) !important;
  font-family:var(--bz-font) !important; font-weight:600 !important;
  min-height:46px !important; box-shadow:none !important;
}
.stApp .stButton > button[kind="secondary"]:hover{
  border-color:var(--g5) !important; background:var(--g1) !important; color:var(--g8) !important;
}

/* ช่องกรอก/กล่องต่าง ๆ ให้มนเข้าชุดกัน */
.stApp [data-baseweb="input"], .stApp [data-baseweb="select"] > div,
.stApp [data-testid="stFileUploaderDropzone"], .stApp [data-testid="stExpander"] details{
  border-radius:var(--bz-radius) !important;
}

/* ซ่อนแถบเครื่องมือของ Streamlit (ปุ่ม Deploy โผล่มุมขวาบน — ไม่ใช่ของเรา) */
[data-testid="stToolbar"], [data-testid="stDecoration"], .stAppDeployButton,
[data-testid="stStatusWidget"]{ display:none !important; }

/* จัดทั้งหน้าให้อยู่กลางจอแนวตั้ง — ใส่ <div class="bz-vcenter"> ในหน้าที่ต้องการ
   *** ทำไมต้องจัดที่ "ลูก" ไม่ใช่ที่ .block-container ***
   ลูกโดยตรงของ block-container คือ [data-testid="stVerticalBlock"] ซึ่ง Streamlit
   ตั้ง flex:1 1 0% ไว้ -> มันยืดเต็มความสูงเสมอ แล้วเรียงเนื้อหาจากบน (justify:start)
   => ต่อให้ block-container เป็น flex + justify-content:center ก็ไม่มีผล เพราะลูกเต็มพอดี
   (ตรวจ DOM จริงแล้ว: bc สูง 960 · ลูกสูง 880 top=92 grow=1)
   -> ต้องสั่ง justify-content:center ที่ "ลูก" ตัวนั้นแทน */
/* จัดหน้าให้อยู่กลางจอแนวตั้ง — ใส่ <div class="bz-vcenter"> ในหน้านั้น
   *** สั่งที่ "ลูก" ตัวเดียวจบ อย่าไปยุ่งกับ .block-container ***
   ประวัติที่พลาดมา (อย่าทำซ้ำ):
     - ใส่ flex+justify ที่ .block-container -> ลูก (stVerticalBlock, flex:1 1 0%) ยืดเต็ม
       แล้วเรียงเนื้อหาจากบน = ไม่กลาง
     - ย้าย justify ไปที่ลูกอย่างเดียว -> ลูกสูงเท่าเนื้อหาเอง (452px) = ไม่มีที่ว่างให้จัด
     - จะให้แม่เป็น flex ด้วยก็ไม่สำเร็จ: .block-container เป็น display:block และ
       display:flex ที่เราสั่งไม่เข้า (ตรวจ DOM แล้ว)
   => ทางที่ได้ผล: ให้ "ลูก" สูง 86vh ไปเลย แล้วมันจัดกลางในตัวเองได้ ไม่ต้องพึ่งแม่ */
.block-container:has(.bz-vcenter) > [data-testid="stVerticalBlock"]{
  min-height:86vh !important;
  justify-content:center !important;
}
/* marker เองต้องไม่กินที่ (มันเป็นแค่ธง ไม่ใช่เนื้อหา) */
.bz-vcenter{ display:none; }
[data-testid="stMarkdown"]:has(.bz-vcenter){ display:none; }

/* --- แถบธงแดง --- */
.bz-flag{
  margin:20px 0 0; padding:9px 0 9px 14px;
  border-left:3px solid var(--bz-accent);
  font-size:13.5px; line-height:1.62; color:var(--bz-ink);
}

/* --- footer เทคนิค --- */
.bz-foot{
  margin-top:22px; padding-top:12px; border-top:1px solid var(--bz-hair);
  font-size:11.5px; line-height:1.75; color:var(--bz-muted);
}

/* --- ตารางเต็มใน expander --- */
.bz-full{ font-size:13px; }
.bz-full .bz-row{ font-size:13px; }

/* ---------- ตัวเลือกบทบาท (Start -> เลือกบทบาท) ----------
   ใช้ภาษาเดียวกับตาราง β: การ์ดพื้น surface ขอบ hair มุมล่างเหลี่ยม -> ปุ่มเชื่อมใต้การ์ด
   ไม่ใช้ accent ที่นี่ — accent ถูกจองไว้ให้ "เลข 0.000" บนปกเท่านั้น */
.bz-rolehead{ font-size:20px; font-weight:600; line-height:1.5; padding-top:2px;
              margin:0 0 4px; text-align:center; }
.bz-rolesub{ font-size:13px; line-height:1.7; color:var(--g5); margin:0 0 22px; text-align:center; }
.bz-role{
  background:var(--g1); border:1px solid var(--g3);
  border-radius:var(--bz-radius); padding:18px 18px 16px;
  min-height:168px;                    /* ให้การ์ดสองใบสูงเท่ากัน ปุ่มจะได้เรียงตรงกัน */
  display:flex; flex-direction:column; margin-bottom:10px;
}
.bz-role h4{ font-size:16px; font-weight:600; line-height:1.5; padding-top:1px;
             margin:0 0 8px; color:var(--bz-ink); }
.bz-role .k{ font-size:12.5px; line-height:1.85; color:var(--g6); }
.bz-role .hint{
  margin-top:auto; padding-top:10px; border-top:1px solid var(--g3);
  font-size:11.5px; line-height:1.65; color:var(--g5);
}
.bz-role .hint b{ color:var(--g7); font-weight:600; }
@media (max-width:760px){ .bz-role{ min-height:0; } }

/* ---------- ขั้นตอน (step rail) ---------- */
.bz-rail{ display:flex; gap:18px; flex-wrap:wrap; align-items:center;
          border-bottom:1px solid var(--bz-hair); padding-bottom:9px; margin-bottom:14px; }
.bz-rail span{ font-size:13.5px; color:var(--bz-muted); line-height:1.6; }
.bz-rail span.on{ color:var(--bz-ink); font-weight:700; }
.bz-rail span.done{ color:var(--bz-muted); }

/* ---------- เก็บกวาด ---------- */
[data-testid="stSidebar"]{ background:var(--bz-surface); border-right:1px solid var(--bz-hair); }
.stApp [data-testid="stExpander"] details{
  background:transparent; border:1px solid var(--bz-hair); border-radius:3px;
}
#MainMenu, footer{ visibility:hidden; }
@media (max-width:760px){
  .bz-head span:nth-child(4), .bz-row .bz-cite{ display:none; }   /* จอแคบ: ตัดคอลัมน์ที่มา */
  .bz-head, .bz-row{ grid-template-columns: minmax(0,1fr) 60px 84px; }
  .bz-title{ font-size:26px; }
  .block-container{ padding-left:1rem; padding-right:1rem; }
}
</style>
"""


def inject_css() -> None:
    """ใส่ CSS ของธีมทั้งแอป — เรียก **ครั้งเดียว** ทันทีหลัง st.set_page_config()"""
    st.markdown(_CSS, unsafe_allow_html=True)


# =============================================================================
# 2) ตาราง β — อ่านสดจาก scoring.BETA (ไม่มีตัวเลขไหนพิมพ์ไว้ในไฟล์นี้)
# =============================================================================
def _logo_svg(px: int = 46) -> str:
    """โลโก้แบบ inline SVG (badge + โล่ + คนนั่ง + ลูกศรอนาคต)

    ต้องตรงกับ assets/logo.svg เสมอ — ถ้าแก้ที่นี่ ให้แก้ไฟล์นั้นด้วย
    (ไฟล์ .svg ใช้กับ Word/สไลด์/favicon · ตัวนี้ใช้ในแอป)

    *** อย่าเติมสามเหลี่ยมเตือน/จุดแดงบนสันหลัง *** — เหตุผลเต็มอยู่ใน assets/logo.svg
    สรุป: มันคือข้ออ้างที่สมการเราเองตั้งเป็น 0 (Raine 2021 OR 0.24)

    id ของ gradient ตั้งเป็น bzLogoGrad — ถ้าวางโลโก้หลายตัวในหน้าเดียว id จะซ้ำ
    แต่ SVG ใช้ id แรกที่เจอ ซึ่งเหมือนกันทุกตัวอยู่แล้ว จึงไม่มีปัญหา
    """
    return (
        '<svg class="bz-logo" width="%d" height="%d" viewBox="0 0 128 128" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="โลโก้ ก่อนจะค่อม">'
        '<defs><linearGradient id="bzLogoGrad" x1="0" y1="0" x2="0.9" y2="1">'
        '<stop offset="0" stop-color="#2FB4CC"/>'
        '<stop offset="0.55" stop-color="#1A5A94"/>'
        '<stop offset="1" stop-color="#0E2C57"/>'
        '</linearGradient></defs>'
        '<rect x="4" y="4" width="120" height="120" rx="28" fill="url(#bzLogoGrad)"/>'
        '<path d="M64 22 L96 33 V63 C96 84 81 99 64 107 C47 99 32 84 32 63 V33 Z" '
        'fill="rgba(255,255,255,.10)" stroke="#EAF6FA" stroke-width="3.2" '
        'stroke-linejoin="round"/>'
        '<g stroke="#EAF6FA" stroke-width="5.4" fill="none" stroke-linecap="round">'
        '<path d="M50 55 C48 65 54 69 53 77"/>'
        '<path d="M53 79 H70"/>'
        '<path d="M70 79 V93"/>'
        '</g>'
        '<circle cx="50" cy="45" r="7.6" fill="#EAF6FA"/>'
        '<path d="M76 60 H104 M96 52 L104 60 L96 68" stroke="#6FE0F2" stroke-width="4.6" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>' % (px, px)
    )


def _beta_rows():
    """คืน (rows_nonzero, rows_zero) เรียงจาก β มาก -> น้อย ตามค่าจริงใน scoring.BETA"""
    if scoring is None or not getattr(scoring, "BETA", None):
        return [], []
    items = [(k, b, label, cite) for k, (b, label, cite) in scoring.BETA.items()]
    items.sort(key=lambda t: -t[1])
    nonzero = [t for t in items if t[1] > 0]
    zeros = [t for t in items if t[1] == 0]
    return nonzero, zeros


def _clip(s: str, n: int) -> str:
    """ตัดข้อความยาวให้พอดีคอลัมน์ (ตัวเต็มยังอ่านได้จาก title= ตอน hover)"""
    s = " ".join((s or "").split())
    s = s.replace("*** ไม่มีผล ***", "").strip()
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = cut.rfind(" ")
    if sp > n * 0.6:
        cut = cut[:sp]
    return cut.rstrip(" ,·—-") + "…"


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _row_html(label: str, beta: float, cite: str, bmax: float, clip: int = 62) -> str:
    """แถวปกติ 1 แถว: ชื่อ · แท่งบาร์ตามค่า β · ตัวเลข β · ที่มา"""
    w = max(2.0, 100.0 * beta / bmax) if bmax else 0.0
    return (
        '<div class="bz-row">'
        '<div class="bz-name">%s</div>'
        '<div class="bz-bar"><i style="width:%.0f%%"></i></div>'
        '<div class="bz-beta">%.3f</div>'
        '<div class="bz-cite" title="%s">%s</div>'
        '</div>'
        % (_e(label), w, beta, _e(cite), _e(_clip(cite, clip)))
    )


def _zero_row_html(label: str, beta: float, cite: str) -> str:
    """แถวศูนย์: เส้นทับพาดทั้งแถว แต่ chip ของเลข 0.000 ตัดเส้นขาด
    + ป้าย "วัด · รายงาน · ไม่ให้คะแนน" ให้ตรงกับสิ่งที่ scoring.py ทำจริง
      (โค้ดเขียนว่า "เก็บไว้ รายงาน แต่ไม่ให้คะแนน" — ไม่ใช่ "ทิ้ง")"""
    return (
        '<div class="bz-row is-zero">'
        '<div class="bz-name">%s<br><span class="bz-tag">วัด · รายงาน · ไม่ให้คะแนน</span></div>'
        '<div class="bz-bar"></div>'
        '<div class="bz-beta">%.3f</div>'
        '<div class="bz-cite" title="%s">%s</div>'
        '</div>'
        % (_e(label), beta, _e(cite), _e(_clip(cite, 96)))
    )


def _ledger_html() -> str:
    nonzero, zeros = _beta_rows()
    if not nonzero and not zeros:
        return ""
    bmax = nonzero[0][1] if nonzero else 1.0
    top = nonzero[:COVER_TOP_N]
    rest = len(nonzero) - len(top)

    parts = ['<div class="bz-ledger">',
             '<div class="bz-head"><span>ปัจจัย</span><span></span>'
             '<span>น้ำหนัก β</span><span>ที่มา</span></div>']
    for _k, b, label, cite in top:
        parts.append(_row_html(label, b, cite, bmax))
    if rest > 0:
        parts.append('<div class="bz-more">↓ แสดง %d อันดับแรกจาก %d ปัจจัยที่มีน้ำหนัก '
                     '(ทั้งสมการมี %d ตัวแปร — กางดูครบทุกบรรทัดได้ด้านล่าง)</div>'
                     % (len(top), len(nonzero), len(nonzero) + len(zeros)))
    if zeros:
        parts.append('<div class="bz-band">ต่อไปนี้คือสิ่งที่ <b>AI ของเราวัดได้เอง</b> '
                     'จากภาพถ่าย (MediaPipe BlazePose — 33 จุดบนร่างกาย) '
                     'เราวัดมัน เรารายงานมัน <b>แต่เราตั้งน้ำหนักเป็นศูนย์ด้วยมือตัวเอง</b> '
                     'เพราะงานวิจัยบอกว่ามันทำนายความปวดไม่ได้</div>')
        for _k, b, label, cite in zeros:
            parts.append(_zero_row_html(label, b, cite))
    if zeros:
        parts.append('<div class="bz-note">%d บรรทัดที่ถูกทับด้วยเส้นคือฟีเจอร์ที่ทำยากที่สุด'
                     'ของโครงงาน — เราเก็บค่ามันไว้ และแสดงให้ผู้ใช้ดูเทียบกับตัวเองในอนาคต '
                     '<b>แต่มันไม่มีสิทธิ์ขยับคะแนนของใครแม้แต่ทศนิยมเดียว</b> '
                     '(Raine Study n=686: ท่าค่อม OR 0.24 — ค่อมกลับปวด <b>น้อยกว่า</b>)</div>'
                     % len(zeros))
    parts.append('</div><div class="bz-weld"></div>')
    return "".join(parts)


def _full_table_html() -> str:
    """ตารางเต็มทุกตัวแปรใน BETA (ใช้ใน expander) — เรียงจากมากไปน้อย"""
    nonzero, zeros = _beta_rows()
    rows = nonzero + zeros
    if not rows:
        return "_(อ่าน scoring.BETA ไม่ได้ในบริบทนี้)_"
    bmax = nonzero[0][1] if nonzero else 1.0
    parts = ['<div class="bz-ledger bz-full" style="border-radius:3px">',
             '<div class="bz-head"><span>ปัจจัย</span><span></span>'
             '<span>น้ำหนัก β</span><span>ที่มา</span></div>']
    for _k, b, label, cite in rows:
        if b == 0.0:
            parts.append(_zero_row_html(label, b, cite))
        else:
            parts.append(_row_html(label, b, cite, bmax, clip=86))
    parts.append('</div>')
    return "".join(parts)


# =============================================================================
# 3) หน้าปก
# =============================================================================
def logo_path() -> str | None:
    """ไฟล์โลโก้จริง (ครอปจากไฟล์ต้นฉบับที่เจ้าของงานส่งมา ด้วย scripts/make_logo.py)"""
    p = os.path.join(ASSET_DIR, "logo.png")
    return p if os.path.exists(p) else None


def cover_page(on_start) -> None:
    """หน้าปก — ขาวล้วน · โลโก้กลางหน้า · ปุ่มเริ่ม  เท่านั้น

    เจ้าของงานเลือกแบบนี้เอง (16 ก.ค.): "หน้าสีขาว ตรงกลางเป็นโลโก้ แล้วมีปุ่ม start แค่นี้"

    ตาราง β / แถบธงแดง / footer เทคนิค ที่เคยอยู่บนปก **ถูกย้ายออก ไม่ได้ลบทิ้ง**
    -> _ledger_html() และ _full_table_html() ยังอยู่ครบ เอาไปวางหน้าไหนก็ได้
       (ถ้าจะเอากลับขึ้นปก ให้ st.markdown(_ledger_html(), unsafe_allow_html=True) ก่อนปุ่ม)
    """
    st.markdown('<div class="bz-vcenter"></div>', unsafe_allow_html=True)  # จัดทั้งหน้ากลางจอ

    # โลโก้จริงมาก่อน · ถ้าหาไฟล์ไม่เจอค่อยถอยไปใช้ SVG ที่วาดเอง (แอปจะได้ไม่หน้าเปล่า)
    lp = logo_path()
    col = st.columns([1, 1, 1])
    with col[1]:
        if lp:
            st.image(lp, use_container_width=True)
        else:
            st.markdown('<div class="bz-cover">' + _logo_svg(150) + '</div>',
                        unsafe_allow_html=True)

    st.markdown(
        '<div class="bz-cover">'
        '<div class="bz-cover-title">ก่อนจะค่อม</div>'
        '<div class="bz-cover-sub">คัดกรองความเสี่ยงจากพฤติกรรม · ไม่ใช่การวินิจฉัย</div>'
        '</div>',
        unsafe_allow_html=True)

    # ปุ่มอยู่ในคอลัมน์กลาง — ไม่ให้ยาวเต็มจอ จะได้อยู่ใต้โลโก้พอดี
    col = st.columns([1, 1.1, 1])
    with col[1]:
        if st.button("เริ่มใช้งาน", type="primary",
                     use_container_width=True, key="bz_start"):
            on_start()


# =============================================================================
# 4) เลือกบทบาท — กด "เริ่มคัดกรอง" แล้วมาที่หน้านี้
#
# ทำไมต้องมีหน้านี้ (ไม่ใช่ยิงเข้าแบบสอบถามเลย):
#   แดชบอร์ดครูมี "กฎปิดข้อมูล 3 ชั้น" ซึ่งเป็นของดีที่สุดชิ้นหนึ่งของโครงงาน
#   แต่เดิมมันซ่อนอยู่ใน checkbox ท้าย sidebar -> กรรมการมีสิทธิ์ไม่เจอเลยตลอดการตัดสิน
#   หน้านี้ดึงมันขึ้นมาอยู่ระดับเดียวกับเครื่องมือหลัก
# ทั้งสองทาง = เครื่องมือจริงที่กดแล้วทำงาน ไม่ใช่หน้าอธิบาย (ไม่งั้นก็กลายเป็นสไลด์)
# =============================================================================
def role_picker(on_self, on_org) -> None:
    """หน้าเลือกบทบาท · on_self / on_org = callback เมื่อเลือกแต่ละทาง"""
    st.markdown('<div class="bz-vcenter"></div>', unsafe_allow_html=True)  # จัดทั้งหน้ากลางจอ
    st.markdown(
        '<div class="bz-rolehead">คุณกำลังเปิดในบทบาทไหน</div>'
        '<div class="bz-rolesub">เลือกผิดกดย้อนกลับได้ตลอด</div>',
        unsafe_allow_html=True)

    # คอลัมน์ริมสองข้างบีบให้การ์ดอยู่กลางจอแนวนอน ไม่กางเต็มความกว้าง
    pad = st.columns([1, 6, 6, 1], gap="medium")

    with pad[1]:
        st.markdown(
            '<div class="bz-role">'
            '<h4>ประเมินตัวเอง</h4>'
            '<div class="k">'
            'ตอบคำถามพฤติกรรม ~2 นาที<br>'
            'ถ่ายรูปท่าทาง (ข้ามได้)<br>'
            'ได้ดัชนีความเสี่ยง + คำแนะนำ + ไฟล์ PDF'
            '</div>'
            '<div class="hint"><b>ลองใส่อายุ 16 แล้วทำซ้ำด้วยอายุ 30</b> — '
            'ระบบพูดกับผู้เยาว์กับผู้ใหญ่ไม่เหมือนกัน โดยตั้งใจ</div>'
            '</div>',
            unsafe_allow_html=True)
        if st.button("เริ่มประเมินตัวเอง", type="primary",
                     use_container_width=True, key="bz_role_self"):
            on_self()

    with pad[2]:
        st.markdown(
            '<div class="bz-role">'
            '<h4>ครู / องค์กร</h4>'
            '<div class="k">'
            'ดูภาพรวมกลุ่มแบบไม่ระบุตัวตน<br>'
            'ชี้สิ่งที่โรงเรียนแก้ได้ ไม่ใช่สิ่งที่เด็กต้องแก้เอง<br>'
            'ไม่มีชื่อ ไม่มีใบหน้า ไม่มีคะแนนรายบุคคล'
            '</div>'
            '<div class="hint"><b>ไม่ต้องมีไฟล์ก็กดดูได้</b> — '
            'มีชุดข้อมูลจำลองให้เห็นกฎปิดข้อมูลทำงานจริง</div>'
            '</div>',
            unsafe_allow_html=True)
        if st.button("เปิดแดชบอร์ดกลุ่ม", type="primary",
                     use_container_width=True, key="bz_role_org"):
            on_org()


def assumptions_expander() -> None:
    """กล่อง "สิ่งที่เราตั้งเอง" — ประกาศสมมติฐานของตัวเองก่อนถูกถาม

    *** ยังไม่ถูกเรียกจากที่ไหน *** — เดิมโค้ดก้อนนี้อยู่ท้าย cover_page ของ agent
    แล้วหล่นไปอยู่ใน role_picker ตอนผมแทรกฟังก์ชัน ทำให้แอป crash
    ย้ายออกมาเป็นฟังก์ชันเดี่ยว เพราะเนื้อหาดี (ตรวจกับ scoring.py แล้วว่าจริงทุกข้อ)
    แต่ยังไม่รู้ว่าเจ้าของงานอยากให้ไปอยู่หน้าไหน -> เรียกใช้ได้บรรทัดเดียวเมื่อตัดสินใจแล้ว
    """
    with st.expander("ดูสมการเต็มทุกตัวแปร · λ ที่เราตั้งเอง · ที่มาทุกบรรทัด"):
        st.markdown(_full_table_html(), unsafe_allow_html=True)
        lam = getattr(scoring, "LAMBDA", None) if scoring else None
        lam_txt = "%.1f" % lam if isinstance(lam, (int, float)) else "—"
        # ห้ามใช้ %-format กับข้อความก้อนนี้: มี "25.1%" ลอยอยู่ -> Python อ่านเป็น format spec
        # แล้วพังด้วย "not enough arguments for format string" (บั๊กเดิมของโค้ดก้อนนี้)
        st.markdown(
            "**สิ่งที่เราตั้งเอง และไม่มีงานวิจัยรองรับ — เราประกาศเองก่อนถูกถาม**\n\n"
            "- **λ = " + lam_txt + "** (ตัวหด shrinkage) — เราตั้งเอง เพราะ OR แต่ละตัวมาจากคนละงาน "
            "คนละชุดตัวแปรที่ปรับ การบวก ln(OR) ดิบ ๆ จะประเมินเกินจริง\n"
            "- **เส้นแบ่งสี เขียว/เหลือง/แดง** — เราขีดเอง ไม่มีงานวิจัยกำหนด "
            "เรากวาดโปรไฟล์ที่เป็นไปได้ 1,728 แบบแล้วพบว่า **25.1% เปลี่ยนแถบสีได้** "
            "ถ้าขยับสมมติฐานที่เราตั้งเอง — ระบบจึงบอกผู้ใช้ตรง ๆ เมื่อสีของเขาไม่นิ่ง "
            "(รันตรวจเอง: `python analysis/band_stability.py`)\n"
            "- **β ทุกตัวแปลงจาก Odds Ratio ในงานตีพิมพ์** ไม่ได้เทรนจากข้อมูลของเรา\n"
            "- บางตัวแปรอยู่ในสมการแต่**ไม่แสดงผลรายบุคคล**ด้วยเหตุผลทางจริยธรรม "
            "(กรองที่ต้นทางใน `scoring.HIDDEN_FROM_STUDENT`)")


# =============================================================================
# 4) ตัวช่วยอื่น
# =============================================================================
def step_rail(step: int) -> None:
    """แถบขั้นตอนแบบเส้นเดียว (ใช้แทน _step_header เดิมได้ ถ้าต้องการให้เข้าธีม)"""
    names = {1: "คำถาม", 2: "ถ่ายรูป (ไม่บังคับ)", 3: "ผลลัพธ์ + คำแนะนำ"}
    out = ['<div class="bz-rail">']
    for i in (1, 2, 3):
        cls = "on" if i == step else ("done" if i < step else "")
        mark = "✓" if i < step else "%d" % i
        out.append('<span class="%s">%s · ขั้น %d — %s</span>' % (cls, mark, i, names[i]))
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def cover_asset(name: str) -> str | None:
    """คืน path ของไฟล์ใน assets/ ถ้ามีจริง — ไม่มีก็คืน None (ห้ามพึ่งรูปที่ไม่มี)"""
    p = os.path.normpath(os.path.join(ASSET_DIR, name))
    return p if os.path.exists(p) else None
