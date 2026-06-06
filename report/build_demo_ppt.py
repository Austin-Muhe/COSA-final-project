from __future__ import annotations

import html
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).with_name("cosa_presentation_demo.pptx")

EMU = 914400
SLIDE_W = int(13.333333 * EMU)
SLIDE_H = int(7.5 * EMU)

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


def e(text: str) -> str:
    return html.escape(str(text), quote=True)


def emu(inches: float) -> int:
    return int(inches * EMU)


def color(hex_color: str) -> str:
    return hex_color.replace("#", "").upper()


class Slide:
    def __init__(self, idx: int, bg: str = "F8FAFC"):
        self.idx = idx
        self.bg = color(bg)
        self.parts: list[str] = []
        self.next_id = 2

    def _id(self) -> int:
        val = self.next_id
        self.next_id += 1
        return val

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "FFFFFF",
        line: str = "CBD5E1",
        radius: str = "roundRect",
        alpha: int | None = None,
    ) -> None:
        sid = self._id()
        fill_xml = f'<a:solidFill><a:srgbClr val="{color(fill)}"'
        if alpha is not None:
            fill_xml += f'><a:alpha val="{alpha}"/></a:srgbClr></a:solidFill>'
        else:
            fill_xml += "/></a:solidFill>"
        self.parts.append(
            f"""
            <p:sp>
              <p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
              <p:spPr>
                <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
                <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>
                {fill_xml}
                <a:ln w="12700"><a:solidFill><a:srgbClr val="{color(line)}"/></a:solidFill></a:ln>
              </p:spPr>
              <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
            </p:sp>
            """
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, col: str = "334155", width: int = 26000, dash: bool = False) -> None:
        sid = self._id()
        flip_h = " flipH=\"1\"" if x2 < x1 else ""
        flip_v = " flipV=\"1\"" if y2 < y1 else ""
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        dash_xml = '<a:prstDash val="dash"/>' if dash else ""
        self.parts.append(
            f"""
            <p:cxnSp>
              <p:nvCxnSpPr><p:cNvPr id="{sid}" name="Line {sid}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
              <p:spPr>
                <a:xfrm{flip_h}{flip_v}><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
                <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
                <a:ln w="{width}"><a:solidFill><a:srgbClr val="{color(col)}"/></a:solidFill>{dash_xml}</a:ln>
              </p:spPr>
            </p:cxnSp>
            """
        )

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        size: int = 28,
        bold: bool = False,
        col: str = "0F172A",
        align: str = "l",
        valign: str = "t",
        fill: str | None = None,
    ) -> None:
        sid = self._id()
        paras = []
        for raw in text.split("\n"):
            if raw == "":
                paras.append("<a:p/>")
                continue
            paras.append(
                f"""
                <a:p>
                  <a:pPr algn="{align}"/>
                  <a:r><a:rPr lang="en-US" sz="{size * 100}" {'b="1"' if bold else ''}>
                    <a:solidFill><a:srgbClr val="{color(col)}"/></a:solidFill>
                  </a:rPr><a:t>{e(raw)}</a:t></a:r>
                </a:p>
                """
            )
        fill_xml = '<a:noFill/>' if fill is None else f'<a:solidFill><a:srgbClr val="{color(fill)}"/></a:solidFill>'
        self.parts.append(
            f"""
            <p:sp>
              <p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
              <p:spPr>
                <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                {fill_xml}
                <a:ln><a:noFill/></a:ln>
              </p:spPr>
              <p:txBody><a:bodyPr anchor="{valign}" wrap="square"><a:spAutoFit/></a:bodyPr><a:lstStyle/>{''.join(paras)}</p:txBody>
            </p:sp>
            """
        )

    def pill(self, x: float, y: float, w: float, h: float, label: str, fill: str, col: str = "FFFFFF", size: int = 18) -> None:
        self.rect(x, y, w, h, fill=fill, line=fill, radius="roundRect")
        self.text(x + 0.05, y + 0.07, w - 0.1, h - 0.1, label, size=size, bold=True, col=col, align="ctr", valign="mid")

    def title(self, title: str, subtitle: str | None = None) -> None:
        self.text(0.55, 0.28, 11.7, 0.55, title, size=32, bold=True, col="0F172A")
        if subtitle:
            self.text(0.58, 0.88, 11.5, 0.35, subtitle, size=15, col="475569")
        self.line(0.55, 1.22, 12.75, 1.22, col="CBD5E1", width=11000)

    def xml(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{self.bg}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(self.parts)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def chart_axes(slide: Slide, x: float, y: float, w: float, h: float) -> None:
    slide.line(x, y + h, x + w, y + h, "94A3B8", width=14000)
    slide.line(x, y, x, y + h, "94A3B8", width=14000)


def mini_shift_chart(slide: Slide, x: float, y: float, w: float, h: float, sev: int, col: str, label: str) -> None:
    slide.rect(x, y, w, h, fill="FFFFFF", line="CBD5E1", radius="roundRect")
    slide.text(x + 0.15, y + 0.12, w - 0.3, 0.25, label, size=15, bold=True, col="0F172A", align="ctr")
    px, py, pw, ph = x + 0.25, y + 0.55, w - 0.5, h - 0.85
    chart_axes(slide, px, py, pw, ph)
    mid = px + pw * 0.5
    slide.line(mid, py, mid, py + ph, "334155", width=12000, dash=True)
    points = []
    for i in range(16):
        xx = px + pw * i / 15
        base = 0.54 + 0.08 * math.sin(i * 0.9) + 0.04 * math.sin(i * 2.1)
        if i >= 8:
            base += sev * 0.035
        yy = py + ph * (1 - min(max(base, 0.1), 0.95))
        points.append((xx, yy))
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        slide.line(x1, y1, x2, y2, col, width=22000)


def bar_chart(slide: Slide, x: float, y: float, w: float, h: float) -> None:
    labels = ["0σ", "5σ", "10σ"]
    vals = [1.27, -0.27, -0.21]
    cols = ["16A34A", "DC2626", "DC2626"]
    slide.rect(x, y, w, h, fill="FFFFFF", line="CBD5E1", radius="roundRect")
    slide.text(x + 0.25, y + 0.15, w - 0.5, 0.3, "Improvement%: positive means COSA helps", size=15, bold=True)
    axis_y = y + h * 0.55
    slide.line(x + 0.55, axis_y, x + w - 0.35, axis_y, "94A3B8", width=12000)
    max_abs = 1.4
    for i, val in enumerate(vals):
        cx = x + 1.0 + i * 1.35
        bh = abs(val) / max_abs * 1.4
        top = axis_y - bh if val >= 0 else axis_y
        slide.rect(cx, top, 0.65, bh, fill=cols[i], line=cols[i], radius="rect")
        slide.text(cx - 0.15, axis_y + 0.12, 0.95, 0.25, labels[i], size=13, bold=True, align="ctr")
        slide.text(cx - 0.25, top - 0.34 if val >= 0 else top + bh + 0.05, 1.15, 0.25, f"{val:+.2f}%", size=13, bold=True, col=cols[i], align="ctr")


def nar_chart(slide: Slide, x: float, y: float, w: float, h: float) -> None:
    labels = ["0σ", "5σ", "10σ"]
    vals = [37.13, 53.66, 56.40]
    slide.rect(x, y, w, h, fill="FFFFFF", line="CBD5E1", radius="roundRect")
    slide.text(x + 0.25, y + 0.15, w - 0.5, 0.3, "NAR: windows where COSA is worse", size=15, bold=True)
    base_y = y + h - 0.55
    slide.line(x + 0.55, base_y, x + w - 0.35, base_y, "94A3B8", width=12000)
    for i, val in enumerate(vals):
        cx = x + 1.0 + i * 1.35
        bh = val / 60 * 1.7
        col = "F59E0B" if val < 50 else "DC2626"
        slide.rect(cx, base_y - bh, 0.65, bh, fill=col, line=col, radius="rect")
        slide.text(cx - 0.15, base_y + 0.12, 0.95, 0.25, labels[i], size=13, bold=True, align="ctr")
        slide.text(cx - 0.27, base_y - bh - 0.32, 1.2, 0.25, f"{val:.1f}%", size=13, bold=True, col=col, align="ctr")
    slide.line(x + 0.55, base_y - 1.42, x + w - 0.35, base_y - 1.42, "DC2626", width=10000, dash=True)
    slide.text(x + w - 1.15, base_y - 1.75, 0.8, 0.25, "50%", size=12, bold=True, col="DC2626", align="r")


def make_slides() -> list[Slide]:
    slides: list[Slide] = []

    s = Slide(1)
    s.text(0.65, 0.45, 10.8, 0.75, "Can COSA Survive Sudden Exchange-Rate Shifts?", size=33, bold=True)
    s.text(0.68, 1.25, 10.8, 0.45, "Black-swan robustness test for time-series test-time adaptation", size=18, col="475569")
    s.rect(0.75, 2.05, 3.55, 2.45, fill="FFFFFF", line="CBD5E1")
    s.text(1.0, 2.28, 3.05, 0.35, "Real-world shock", size=20, bold=True, col="B91C1C", align="ctr")
    s.text(1.02, 2.88, 3.0, 1.15, "Exchange rates can jump after policy decisions or political events.", size=22, bold=True, col="0F172A", align="ctr", valign="mid")
    mini_shift_chart(s, 4.75, 2.05, 3.7, 2.45, 0, "2563EB", "normal stream")
    mini_shift_chart(s, 8.75, 2.05, 3.7, 2.45, 10, "DC2626", "abrupt shift")
    s.rect(1.0, 5.15, 11.2, 1.0, fill="E0F2FE", line="38BDF8")
    s.text(1.35, 5.38, 10.5, 0.45, "Research question: does COSA remain effective when the test stream suddenly changes?", size=23, bold=True, col="0C4A6E", align="ctr")
    s.text(0.78, 6.78, 11.8, 0.28, "5-minute group presentation demo", size=12, col="64748B")
    slides.append(s)

    s = Slide(2)
    s.title("What Is COSA?", "A lightweight test-time adaptation method for forecasting")
    labels = [("Historical input", "DBEAFE", "1D4ED8"), ("Base forecaster", "F1F5F9", "334155"), ("COSA adapter", "FEF3C7", "B45309"), ("Adapted forecast", "DCFCE7", "15803D")]
    xs = [0.8, 3.6, 6.45, 9.35]
    for i, (lab, fill, col) in enumerate(labels):
        s.rect(xs[i], 2.15, 2.1, 1.15, fill=fill, line=col)
        s.text(xs[i] + 0.15, 2.48, 1.8, 0.35, lab, size=20, bold=True, col=col, align="ctr", valign="mid")
        if i < 3:
            s.line(xs[i] + 2.1, 2.72, xs[i + 1] - 0.15, 2.72, "64748B", width=22000)
    s.rect(0.95, 4.35, 5.2, 1.35, fill="FFFFFF", line="CBD5E1")
    s.text(1.25, 4.62, 4.6, 0.65, "Baseline: fixed model, no test-time adaptation", size=22, bold=True, col="334155", align="ctr", valign="mid")
    s.rect(7.05, 4.35, 5.2, 1.35, fill="FFF7ED", line="FDBA74")
    s.text(7.35, 4.62, 4.6, 0.65, "COSA: uses recent test context to adjust outputs", size=22, bold=True, col="9A3412", align="ctr", valign="mid")
    s.pill(4.9, 6.25, 3.55, 0.55, "Key idea: adapt at test time", "0F172A", size=18)
    slides.append(s)

    s = Slide(3)
    s.title("Black-Swan Dataset Construction", "Exchange Rate test stream with synthetic abrupt level shocks")
    mini_shift_chart(s, 0.65, 1.65, 3.85, 2.15, 0, "2563EB", "0σ clean")
    mini_shift_chart(s, 4.75, 1.65, 3.85, 2.15, 5, "F97316", "5σ shift")
    mini_shift_chart(s, 8.85, 1.65, 3.85, 2.15, 10, "DC2626", "10σ shift")
    s.rect(0.9, 4.35, 5.1, 1.25, fill="FFFFFF", line="CBD5E1")
    s.text(1.15, 4.72, 4.6, 0.45, "x'_t = x_t,     t < T_shift", size=24, bold=True, col="0F172A", align="ctr")
    s.rect(7.3, 4.35, 5.1, 1.25, fill="FFFFFF", line="CBD5E1")
    s.text(7.55, 4.72, 4.6, 0.45, "x'_t = x_t + α · σ,     t ≥ T_shift", size=24, bold=True, col="0F172A", align="ctr")
    s.line(6.25, 4.95, 7.05, 4.95, "334155", width=26000)
    s.rect(1.0, 6.15, 11.1, 0.65, fill="E0F2FE", line="38BDF8")
    s.text(1.3, 6.32, 10.5, 0.3, "First half unchanged; second half receives a severity-scaled shift.", size=19, bold=True, col="0C4A6E", align="ctr")
    slides.append(s)

    s = Slide(4)
    s.title("How We Measure Robustness", "Overall accuracy plus window-level negative adaptation")
    s.rect(0.8, 1.7, 5.55, 2.0, fill="FFFFFF", line="CBD5E1")
    s.text(1.05, 1.98, 5.05, 0.35, "Improvement%", size=25, bold=True, col="15803D", align="ctr")
    s.text(1.05, 2.55, 5.05, 0.4, "(Baseline MSE - COSA MSE) / Baseline MSE", size=20, bold=True, col="0F172A", align="ctr")
    s.text(1.1, 3.17, 4.95, 0.25, "positive = COSA helps    negative = COSA hurts", size=15, col="475569", align="ctr")
    s.rect(7.0, 1.7, 5.55, 2.0, fill="FFFFFF", line="CBD5E1")
    s.text(7.25, 1.98, 5.05, 0.35, "NAR", size=25, bold=True, col="B91C1C", align="ctr")
    s.text(7.25, 2.55, 5.05, 0.4, "worse windows / total windows", size=20, bold=True, col="0F172A", align="ctr")
    s.text(7.3, 3.17, 4.95, 0.25, "higher = more negative adaptation", size=15, col="475569", align="ctr")
    for i in range(10):
        x = 1.35 + i * 0.78
        is_bad = i >= 5
        s.rect(x, 4.75, 0.45, 0.45, fill="FEE2E2" if is_bad else "DCFCE7", line="DC2626" if is_bad else "16A34A")
        s.text(x + 0.07, 4.79, 0.3, 0.25, "×" if is_bad else "✓", size=20, bold=True, col="DC2626" if is_bad else "15803D", align="ctr")
    s.text(1.35, 5.45, 3.4, 0.3, "COSA better or equal", size=17, bold=True, col="15803D")
    s.text(5.1, 5.45, 3.4, 0.3, "COSA worse", size=17, bold=True, col="DC2626")
    s.rect(8.75, 4.55, 3.0, 1.05, fill="FEF3C7", line="F59E0B")
    s.text(9.0, 4.83, 2.5, 0.35, "Example: 5 / 10", size=19, bold=True, col="92400E", align="ctr")
    s.text(9.0, 5.18, 2.5, 0.25, "NAR = 50%", size=18, bold=True, col="92400E", align="ctr")
    slides.append(s)

    s = Slide(5)
    s.title("Main Results", "COSA helps slightly on clean data, but weakens under abrupt shifts")
    bar_chart(s, 0.75, 1.55, 5.65, 3.35)
    nar_chart(s, 6.95, 1.55, 5.65, 3.35)
    rows = [
        ("0σ clean", "0.083", "0.082", "+1.27%", "37.13%"),
        ("5σ shift", "0.551", "0.552", "-0.27%", "53.66%"),
        ("10σ shift", "2.163", "2.167", "-0.21%", "56.40%"),
    ]
    s.rect(0.85, 5.35, 11.65, 1.25, fill="FFFFFF", line="CBD5E1", radius="roundRect")
    headers = ["Shift", "Baseline MSE", "COSA MSE", "Improvement", "NAR"]
    xs = [1.05, 3.2, 5.45, 7.65, 9.95]
    for x, h in zip(xs, headers):
        s.text(x, 5.52, 1.8, 0.22, h, size=12, bold=True, col="475569", align="ctr")
    for r, row in enumerate(rows):
        yy = 5.88 + r * 0.24
        for x, val in zip(xs, row):
            c = "15803D" if val.startswith("+") else ("DC2626" if val.startswith("-") or val.startswith("5") and "%" in val else "0F172A")
            s.text(x, yy, 1.8, 0.2, val, size=12, bold=True, col=c, align="ctr")
    s.pill(3.8, 6.82, 5.7, 0.45, "Under shifts, NAR rises above 50%", "B91C1C", size=16)
    slides.append(s)

    s = Slide(6)
    s.title("Analysis and Takeaway", "What we learned from the stress test")
    cards = [
        ("What worked", "COSA gives small gains on clean streams.", "DCFCE7", "15803D"),
        ("What failed", "Abrupt shifts make recent context misleading.", "FEE2E2", "B91C1C"),
        ("Future work", "Add shift detection or robust context filtering.", "E0F2FE", "0369A1"),
    ]
    for i, (head, body, fill, col) in enumerate(cards):
        x = 0.75 + i * 4.15
        s.rect(x, 1.85, 3.55, 2.0, fill=fill, line=col)
        s.text(x + 0.2, 2.12, 3.15, 0.35, head, size=23, bold=True, col=col, align="ctr")
        s.text(x + 0.32, 2.75, 2.9, 0.55, body, size=20, bold=True, col="0F172A", align="ctr", valign="mid")
    s.rect(1.0, 4.75, 11.15, 1.15, fill="0F172A", line="0F172A")
    s.text(1.35, 5.05, 10.45, 0.45, "Takeaway: test-time adaptation should be evaluated under realistic non-IID stream changes, not only clean test splits.", size=24, bold=True, col="FFFFFF", align="ctr")
    s.text(1.1, 6.45, 11.0, 0.3, "Suggested ending: COSA is useful, but black-swan robustness needs extra mechanisms.", size=16, col="475569", align="ctr")
    slides.append(s)

    return slides


def rels_xml(slide_count: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    ]
    for i in range(slide_count):
        rels.append(
            f'<Relationship Id="rId{i+3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>"""


def content_types(slide_count: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(slide_count):
        overrides.append(
            f'<Override PartName="/ppt/slides/slide{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {''.join(overrides)}
</Types>"""


def presentation_xml(slide_count: int) -> str:
    sld_ids = "".join(
        f'<p:sldId id="{256+i}" r:id="rId{i+3}"/>' for i in range(slide_count)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def minimal_theme() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="{NS['a']}" name="COSA Demo">
  <a:themeElements>
    <a:clrScheme name="COSA"><a:dk1><a:srgbClr val="0F172A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="334155"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="16A34A"/></a:accent2><a:accent3><a:srgbClr val="DC2626"/></a:accent3><a:accent4><a:srgbClr val="F59E0B"/></a:accent4><a:accent5><a:srgbClr val="0891B2"/></a:accent5><a:accent6><a:srgbClr val="7C3AED"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Default"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def slide_master() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>"""


def slide_layout() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="{NS['a']}" xmlns:r="{NS['r']}" xmlns:p="{NS['p']}" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def write_pptx() -> None:
    slides = make_slides()
    now = datetime.now(timezone.utc).isoformat()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>""")
        z.writestr("docProps/core.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>COSA Presentation Demo</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>""")
        z.writestr("docProps/app.xml", f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(slides)}</Slides></Properties>""")
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", rels_xml(len(slides)))
        z.writestr("ppt/theme/theme1.xml", minimal_theme())
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""")
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""")
        for i, slide in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide.xml())
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""")
    print(OUT)


if __name__ == "__main__":
    write_pptx()
