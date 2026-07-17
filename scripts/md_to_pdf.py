# -*- coding: utf-8 -*-
"""
md_to_pdf.py — แปลง Markdown -> PDF สวย ๆ ภาษาไทย (ใช้ Edge headless)
=====================================================================
ใช้:
    python md_to_pdf.py  "ไฟล์เข้า.md"  "ไฟล์ออก.pdf"  ["หัวเรื่อง"]

ทำไมใช้ Edge? เพราะมีอยู่ในทุกเครื่อง Windows อยู่แล้ว ไม่ต้องลง LaTeX/wkhtmltopdf
"""
import os
import subprocess
import sys
import tempfile

import markdown

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
* { box-sizing: border-box; }
body {
  font-family: "Leelawadee UI", "Tahoma", sans-serif;
  font-size: 10.5pt; line-height: 1.62; color: #1c1c1e; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 20pt; margin: 0 0 4px; color: #0f172a; letter-spacing: -.3px; }
h2 { font-size: 14pt; margin: 22px 0 8px; padding-bottom: 5px;
     border-bottom: 2px solid #e2e8f0; color: #0f172a; }
h3 { font-size: 11.5pt; margin: 16px 0 6px; color: #334155; }
p  { margin: 7px 0; }
ul, ol { margin: 7px 0 7px 20px; padding: 0; }
li { margin: 3px 0; }
strong { color: #0f172a; }
code { background: #f1f5f9; padding: 1px 5px; border-radius: 3px;
       font-family: Consolas, monospace; font-size: 9.5pt; color: #b91c1c; }
pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
      padding: 10px 12px; overflow-x: auto; }
pre code { background: none; color: #1e293b; padding: 0; }
blockquote {
  margin: 12px 0; padding: 11px 15px; background: #fffbeb;
  border-left: 4px solid #f59e0b; border-radius: 0 6px 6px 0;
}
blockquote > :first-child { margin-top: 0; }
blockquote > :last-child { margin-bottom: 0; }
table { border-collapse: collapse; width: 100%; margin: 11px 0; font-size: 9.6pt; }
th { background: #f1f5f9; text-align: left; font-weight: 600; color: #0f172a; }
th, td { border: 1px solid #cbd5e1; padding: 6px 9px; vertical-align: top; }
tr:nth-child(even) td { background: #fafafa; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }
h2, h3, table, blockquote, pre { page-break-inside: avoid; }
h2, h3 { page-break-after: avoid; }
.sig { margin-top: 26px; padding: 16px; border: 2px dashed #94a3b8; border-radius: 8px;
       background: #f8fafc; }
"""


def convert(md_path: str, pdf_path: str, title: str = "") -> str:
    with open(md_path, encoding="utf-8") as f:
        text = f.read()

    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "nl2br"])

    html = """<!doctype html><html lang="th"><head><meta charset="utf-8">
<title>%s</title><style>%s</style></head><body>%s</body></html>""" % (
        title or os.path.basename(md_path), CSS, html_body)

    tmp = os.path.join(tempfile.gettempdir(), "_md2pdf.html")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)

    edge = None
    for c in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if os.path.exists(c):
            edge = c
            break
    if not edge:
        raise SystemExit("หา Microsoft Edge ไม่เจอ")

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf=" + pdf_path, "file:///" + tmp.replace("\\", "/")],
                   check=True, capture_output=True, timeout=120)
    return pdf_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    out = convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    print("สร้าง PDF: %s  (%.0f KB)" % (out, os.path.getsize(out) / 1024))
