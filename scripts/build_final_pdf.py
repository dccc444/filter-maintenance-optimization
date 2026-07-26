from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "最终论文正文.md"
OUTPUT = ROOT / "output" / "pdf" / "过滤设备透水率退化建模与维护策略优化.pdf"
EQUATION_DIR = ROOT / "tmp" / "paper_equations"


def register_fonts():
    pdfmetrics.registerFont(TTFont("Song", r"C:\Windows\Fonts\simsun.ttc", subfontIndex=0))
    pdfmetrics.registerFont(TTFont("Hei", r"C:\Windows\Fonts\simhei.ttf"))
    pdfmetrics.registerFont(TTFont("Cambria", r"C:\Windows\Fonts\cambria.ttc", subfontIndex=0))


def clean_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.*?)`", r"<font name='Cambria'>\1</font>", text)
    text = re.sub(r"\\\((.*?)\\\)", lambda m: f"<font name='Cambria'>{latex_plain(m.group(1))}</font>", text)
    return text


def latex_plain(text: str) -> str:
    replacements = {
        r"\quad": "  ", r"\,": " ", r"\;": " ", r"\!": "",
        r"\sum": "∑", r"\prod": "∏", r"\min": "min", r"\max": "max",
        r"\arg": "arg", r"\in": "∈", r"\le": "≤", r"\ge": "≥",
        r"\lt": "<", r"\gt": ">", r"\approx": "≈", r"\sim": "∼",
        r"\times": "×", r"\pm": "±", r"\cdot": "·", r"\Delta": "Δ",
        r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
        r"\varepsilon": "ε", r"\eta": "η", r"\mu": "μ", r"\rho": "ρ",
        r"\sigma": "σ", r"\pi": "π", r"\tau": "τ", r"\widehat": "",
        r"\overline": "", r"\operatorname": "", r"\mathrm": "", r"\mathbb": "",
        r"\text": "", r"\left": "", r"\right": "", r"\Big": "", r"\big": "",
        r"\begin{aligned}": "", r"\end{aligned}": "", r"\begin{cases}": "",
        r"\end{cases}": "", r"\tag": "",
    }
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", text)
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.replace("&=", "=").replace("&", " ")
    text = text.replace(r"\\", "  ")
    text = re.sub(r"[_^]\{([^{}]+)\}", lambda m: ("_" if m.group(0)[0] == "_" else "^") + m.group(1), text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return html.escape(text)


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
            title="过滤设备透水率退化建模与维护策略优化",
            author="",
            subject="全国大学生数学建模竞赛论文",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=self.footer))

    @staticmethod
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Cambria", 9)
        canvas.drawCentredString(A4[0] / 2, 1.25 * cm, str(doc.page))
        canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN", parent=base["Title"], fontName="Hei", fontSize=22,
            leading=30, alignment=TA_CENTER, spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "H1CN", parent=base["Heading1"], fontName="Hei", fontSize=15,
            leading=21, spaceBefore=11, spaceAfter=6, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2CN", parent=base["Heading2"], fontName="Hei", fontSize=13,
            leading=18, spaceBefore=8, spaceAfter=4, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3CN", parent=base["Heading3"], fontName="Hei", fontSize=11,
            leading=16, spaceBefore=6, spaceAfter=3, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN", parent=base["BodyText"], fontName="Song", fontSize=9.2,
            leading=13.5, firstLineIndent=2 * 9.2, alignment=TA_JUSTIFY,
            wordWrap="CJK", spaceAfter=2,
        ),
        "list": ParagraphStyle(
            "ListCN", parent=base["BodyText"], fontName="Song", fontSize=9.2,
            leading=13.5, leftIndent=18, firstLineIndent=-18, alignment=TA_JUSTIFY,
            wordWrap="CJK", spaceAfter=2,
        ),
        "equation": ParagraphStyle(
            "Equation", parent=base["BodyText"], fontName="Cambria", fontSize=9.2,
            leading=13, alignment=TA_CENTER, spaceBefore=3, spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Song", fontSize=8.5,
            leading=12, alignment=TA_CENTER, spaceBefore=2, spaceAfter=5,
        ),
        "table": ParagraphStyle(
            "TableText", parent=base["BodyText"], fontName="Song", fontSize=7.2,
            leading=9.5, alignment=TA_CENTER, wordWrap="CJK",
        ),
    }


def fit_image(path: Path, max_w=15.2 * cm, max_h=7.5 * cm):
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def build():
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    st = styles()
    md = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    eq = []
    in_eq = False
    equation_no = 0
    fig_no = 0
    image_budget = 12
    while i < len(md):
        line = md[i].strip()
        if in_eq:
            if line == r"\]":
                equation_no += 1
                rendered = EQUATION_DIR / f"eq-{equation_no:02d}.png"
                if rendered.exists():
                    story.append(fit_image(rendered, max_w=15.2 * cm, max_h=1.5 * cm))
                    story.append(Spacer(1, 2))
                else:
                    formula = latex_plain(" ".join(eq))
                    story.append(Paragraph(formula, st["equation"]))
                eq = []
                in_eq = False
            else:
                eq.append(line)
            i += 1
            continue
        if line == r"\[":
            in_eq = True
            i += 1
            continue
        if not line:
            i += 1
            continue
        if line == r"\newpage":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(md) and md[i].strip().startswith("|"):
                rows.append([c.strip() for c in md[i].strip().strip("|").split("|")])
                i += 1
            if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in rows[1]):
                rows.pop(1)
            data = [[Paragraph(clean_inline(c), st["table"]) for c in row] for row in rows]
            col_width = 16 * cm / max(len(r) for r in rows)
            tbl = Table(data, colWidths=[col_width] * max(len(r) for r in rows), repeatRows=1, hAlign="CENTER")
            tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#607080")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EEF6")),
                ("FONTNAME", (0, 0), (-1, 0), "Hei"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.extend([Spacer(1, 3), tbl, Spacer(1, 5)])
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            fig_no += 1
            if fig_no <= image_budget:
                target = image_match.group(2).replace("?", "")
                path = (SOURCE.parent / target).resolve()
                if not path.exists():
                    candidates = list(path.parent.glob(path.stem.split(".")[0] + "*"))
                    if candidates:
                        path = candidates[0]
                if path.exists():
                    label = re.sub(r"^图\s*\d+\s*", "", image_match.group(1)).strip()
                    story.append(KeepTogether([
                        fit_image(path),
                        Paragraph(f"图 {fig_no}　{clean_inline(label)}", st["caption"]),
                    ]))
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            text = clean_inline(heading.group(2))
            if level == 1 and not story:
                story.append(Paragraph(text, st["title"]))
            else:
                story.append(Paragraph(text, st[f"h{level}"]))
            i += 1
            continue
        if line.startswith("**表") and line.endswith("**"):
            story.append(Paragraph(clean_inline(line[2:-2]), st["caption"]))
        elif re.match(r"^\d+\.\s+", line):
            story.append(Paragraph(clean_inline(line), st["list"]))
        else:
            story.append(Paragraph(clean_inline(line), st["body"]))
        i += 1
    PaperDocTemplate(str(OUTPUT)).build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
