# -*- coding: utf-8 -*-
"""
report.py — สร้าง PDF "คำแนะนำรายบุคคล" ให้ผู้ใช้ดาวน์โหลด
=====================================================================
คืนค่าเป็น bytes ของไฟล์ PDF (เอาไปใส่ st.download_button ได้เลย)
ใช้ Microsoft Edge headless แปลง HTML -> PDF (มีในทุกเครื่อง Windows ไม่ต้องลงอะไรเพิ่ม)

*** ยึดกฎเหล็กเดียวกับแอป ***
  - ไม่มีคำว่า "วินิจฉัย" · ไม่โชว์ % โอกาสเป็นโรค · ไม่โชว์ BMI
  - "กลุ่มอาการที่สัมพันธ์" พูดเชิงกลุ่ม + ย้ำว่าไม่ใช่การวินิจฉัย
"""
from __future__ import annotations

import html
import os
import subprocess
import tempfile

CSS = """
@page { size: A4; margin: 15mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "Leelawadee UI","Tahoma",sans-serif; font-size: 11pt;
       line-height: 1.6; color: #1c1c1e; }
h1 { font-size: 19pt; color: #0f172a; margin: 0 0 2px; }
.sub { color: #64748b; font-size: 10pt; margin: 0 0 14px; }
h2 { font-size: 13pt; color: #0f172a; border-bottom: 2px solid #e2e8f0;
     padding-bottom: 4px; margin: 18px 0 8px; }
.card { border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 13px;
        margin: 9px 0; page-break-inside: avoid; }
.card h3 { margin: 0 0 4px; font-size: 12pt; color: #0f172a; }
.meta { color: #475569; font-size: 9.5pt; margin: 0 0 5px; }
.reason { background: #ecfdf5; border-left: 4px solid #10b981;
          padding: 7px 10px; border-radius: 0 6px 6px 0; margin-top: 6px; font-size: 10pt; }
.group { background: #fff7ed; border-left: 4px solid #f59e0b;
         padding: 8px 11px; border-radius: 0 6px 6px 0; margin: 6px 0; }
.small { color: #64748b; font-size: 9pt; }
ul { margin: 4px 0 4px 18px; padding: 0; }
li { margin: 2px 0; }
.foot { margin-top: 18px; padding-top: 8px; border-top: 1px solid #e2e8f0;
        color: #64748b; font-size: 8.5pt; }
"""


def _edge() -> str:
    for c in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(c):
            return c
    raise RuntimeError("หา Microsoft Edge ไม่เจอ")


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


CATEGORY_TH = {
    "POS": "ท่าบริหาร", "EX": "ออกกำลังกาย", "BRK": "พักและขยับ",
    "SCR": "การใช้จอ", "ERG": "จัดโต๊ะ/จอ", "BAG": "กระเป๋า",
    "SLP": "การนอน", "STR": "ความเครียด", "NUT": "โภชนาการ",
}


def _html(ans: dict, rec: dict, groups: list, intro: str = "", disclaimer: str = "") -> str:
    parts = []
    parts.append("<h1>คำแนะนำรายบุคคล — ก่อนจะค่อม</h1>")

    who = []
    if ans.get("age"):
        who.append("อายุ %s ปี" % _esc(int(ans["age"])))
    if ans.get("status"):
        who.append(_esc(ans["status"]))
    if ans.get("work_type"):
        who.append(_esc(ans["work_type"]))
    parts.append("<p class='sub'>%s · เอกสารนี้เป็นการ<b>คัดกรอง</b> ไม่ใช่การวินิจฉัยโรค</p>"
                 % (" · ".join(who) if who else "ผู้ใช้ทั่วไป"))

    if groups:
        parts.append("<h2>กลุ่มอาการที่พฤติกรรมของคุณ “สัมพันธ์” ด้วย (ตามงานวิจัย)</h2>")
        if intro:
            parts.append("<p class='small'>%s</p>" % _esc(intro))
        for g in groups:
            name, reasons = g[0], g[1]
            examples = g[2] if len(g) > 2 else ""
            block = "<b>%s</b><br>สัมพันธ์กับ: %s" % (_esc(name), _esc(" · ".join(reasons)))
            if examples:
                block += ("<br><span class='small'>อาจครอบคลุมภาวะ เช่น %s</span>"
                          % _esc(examples))
            parts.append("<div class='group'>%s</div>" % block)
        if disclaimer:
            parts.append("<p class='small'>%s</p>" % _esc(disclaimer.replace("**", "")))

    parts.append("<h2>สิ่งที่ทำได้ตั้งแต่วันนี้ (เรียงตามน้ำหนักความเสี่ยงจริงที่แก้ได้)</h2>")
    cards = rec.get("cards", [])
    if not cards:
        parts.append("<p>%s</p>" % _esc(rec.get("message", "ไม่มีคำแนะนำ")))
    for i, card in enumerate(cards, 1):
        cat = CATEGORY_TH.get(card.get("category"), card.get("category", ""))
        parts.append("<div class='card'>")
        parts.append("<h3>%d. %s</h3>" % (i, _esc(card.get("title_th"))))
        parts.append("<div class='meta'>หมวด: %s · หลักฐานระดับ %s</div>"
                     % (_esc(cat), _esc(card.get("evidence_tier"))))
        parts.append("<div><b>ทำเท่าไหร่:</b> %s</div>" % _esc(card.get("dose_th")))
        if card.get("how_th"):
            parts.append("<div><b>ทำยังไง:</b> %s</div>" % _esc(card.get("how_th")))
        parts.append("<div class='reason'><b>ทำไปทำไม:</b> %s</div>" % _esc(card.get("reason_th")))
        cites = card.get("cite", [])
        if cites:
            lis = "".join("<li>%s</li>" % _esc(c.get("label", "")) for c in cites)
            parts.append("<div class='small'>อ้างอิง:<ul>%s</ul></div>" % lis)
        parts.append("</div>")

    parts.append("<div class='foot'>ระบบนี้เป็นเครื่องมือคัดกรอง ไม่ใช่การวินิจฉัย · "
                 "ค่าน้ำหนักความเสี่ยงมาจากงานวิจัยที่ตีพิมพ์แล้ว ไม่ได้เทรนจากข้อมูลผู้ใช้ · "
                 "หากมีอาการผิดปกติ กรุณาปรึกษาแพทย์หรือนักกายภาพบำบัด</div>")

    return ("<!doctype html><html lang='th'><head><meta charset='utf-8'>"
            "<style>%s</style></head><body>%s</body></html>"
            % (CSS, "".join(parts)))


def build_recommendation_pdf(ans: dict, rec: dict, groups: list | None = None,
                             intro: str = "", disclaimer: str = "") -> bytes:
    groups = groups or []
    edge = _edge()
    tmp_html = os.path.join(tempfile.gettempdir(), "_rec_%d.html" % os.getpid())
    tmp_pdf = os.path.join(tempfile.gettempdir(), "_rec_%d.pdf" % os.getpid())
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(_html(ans, rec, groups, intro, disclaimer))
    try:
        subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--print-to-pdf=" + tmp_pdf,
                        "file:///" + tmp_html.replace("\\", "/")],
                       check=True, capture_output=True, timeout=90)
        with open(tmp_pdf, "rb") as f:
            return f.read()
    finally:
        for p in (tmp_html, tmp_pdf):
            try:
                os.unlink(p)
            except OSError:
                pass
