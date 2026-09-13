#!/usr/bin/env python3
# fig1_method_overview.py
# Generate Figure 1: LOWO method overview diagram as SVG.
# Usage: python tools/fig1_method_overview.py
# Output: fig1_method_overview.svg (in repo root, or pass --out)
#
# v3 (2026-09-10) — fix two caption/figure mismatches:
#   - drop-in sites named in words (no cryptic ↓ / ↑)
#   - PR is a property of the DWT/IDWT pair; mixer is drawn as
#     the lossy 8C → Cout step that is *not* in that identity
#
# v2 (2026-09-09) — compact single-column design for ICASSP.

import argparse
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

SVG_NS = "http://www.w3.org/2000/svg"

INK = "#2C2C2A"
MUTED = "#5F5E5A"
PAPER = "#F1EFE8"

BLUE_BG = "#E6F1FB"; BLUE_ST = "#185FA5"; BLUE_TX = "#0C447C"
PURP_BG = "#EEEDFE"; PURP_ST = "#534AB7"; PURP_TX = "#3C3489"
GREEN_BG = "#EAF3DE"; GREEN_ST = "#3B6D11"; GREEN_TX = "#27500A"
CORAL_BG = "#FAECE7"; CORAL_ST = "#D85A30"; CORAL_TX = "#993C1D"
TEAL_BG = "#E1F5EE"; TEAL_ST = "#0F6E56"; TEAL_TX = "#085041"


def _add_rect(parent, x, y, w, h, rx, fill, stroke, sw=1.0, dash=None):
    attrs = {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "rx": str(rx), "fill": fill, "stroke": stroke, "stroke-width": str(sw),
    }
    if dash:
        attrs["stroke-dasharray"] = dash
    return ET.SubElement(parent, "rect", attrs)


def _line(parent, x1, y1, x2, y2, color=MUTED, w=1.5):
    return ET.SubElement(
        parent, "line",
        {"x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
         "stroke": color, "stroke-width": str(w), "marker-end": "url(#arrow)"},
    )


def _polyline(parent, points, color=GREEN_ST, w=1.2, dash="5 3"):
    return ET.SubElement(
        parent, "polyline",
        {"points": points, "fill": "none", "stroke": color,
         "stroke-width": str(w), "stroke-dasharray": dash},
    )


def _text(parent, x, y, text, size, fill, anchor="middle", weight=None):
    attrs = {"x": str(x), "y": str(y), "font-size": str(size),
             "fill": fill, "text-anchor": anchor}
    if weight:
        attrs["font-weight"] = weight
    el = ET.SubElement(parent, "text", attrs)
    el.text = text
    return el


def make_svg() -> str:
    ET.register_namespace("", SVG_NS)

    svg = ET.Element(
        "svg",
        {"viewBox": "0 0 680 380", "width": "100%", "role": "img",
         "font-family": "DejaVuSans, sans-serif"},
    )

    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(
        defs, "marker",
        {"id": "arrow", "viewBox": "0 0 10 10", "refX": "8", "refY": "5",
         "markerWidth": "6", "markerHeight": "6",
         "orient": "auto-start-reverse"},
    )
    ET.SubElement(
        marker, "path",
        {"d": "M2 1L8 5L2 9", "fill": "none", "stroke": "context-stroke",
         "stroke-width": "1.5", "stroke-linecap": "round",
         "stroke-linejoin": "round"},
    )

    ET.SubElement(svg, "title").text = "LOWO method overview figure"
    ET.SubElement(svg, "desc").text = (
        "Fig.1: LOWO drops in at four sites (stem, downsample, upsample, "
        "equal-channel 3x3x3 convolution). An orthogonal analysis bank "
        "is followed by a lossy 1x1 mixer (8C to Cout). Perfect "
        "reconstruction is a property of the DWT/IDWT pair only."
    )

    # borrowed backbone
    _add_rect(svg, 20, 14, 640, 352, 20, "none", MUTED, 1.0, "6 4")
    _text(svg, 36, 34, "SegMamba backbone (unchanged)", 12, "#444441", "start")

    _add_rect(svg, 248, 42, 184, 36, 8, PAPER, MUTED)
    _text(svg, 340, 56, "3D input volume", 13, INK, weight="500")
    _text(svg, 340, 70, "4 modalities", 11, MUTED)
    _line(svg, 340, 78, 340, 96)

    # LOWO container
    _add_rect(svg, 24, 98, 632, 214, 14, PURP_BG, PURP_ST)
    _text(svg, 40, 116, "LOWO (proposed, drop-in)", 12, PURP_TX, "start")

    sites = (
        ("(i) stem", 40),
        ("(ii) downsample", 194),
        ("(iii) upsample", 348),
        ("(iv) 3\u00d73\u00d73", 502),
    )
    for label, x in sites:
        _add_rect(svg, x, 124, 144, 26, 6, "#F7F6F2", MUTED)
        _text(svg, x + 72, 142, label, 11, INK)

    # analysis: Q lives in the kernel, not as a later stage
    _add_rect(svg, 38, 164, 160, 96, 8, BLUE_BG, BLUE_ST)
    _text(svg, 118, 184, "analysis", 13, BLUE_TX, weight="500")
    _text(svg, 118, 204, "DWT  \u00b7  K = Q K_haar", 11, BLUE_ST)
    _text(svg, 118, 220, "Q = exp(A\u2212A^T)", 11, BLUE_ST)
    _text(svg, 118, 236, "\u2208 SO(8)", 11, BLUE_ST)

    # mixer is the lossy, identifiability-critical step
    _add_rect(svg, 218, 160, 256, 104, 8, "#F7F6F2", MUTED)
    _text(svg, 346, 178, "1\u00d71 mixer (lossy)", 12, INK, weight="500")
    _text(svg, 346, 194, "8C \u2192 C_out", 11, MUTED)
    _add_rect(svg, 230, 204, 232, 26, 5, CORAL_BG, CORAL_ST, 1.0, "4 3")
    _text(svg, 346, 222, "groups=1  \u00b7  Q absorbed", 11, CORAL_TX)
    _add_rect(svg, 230, 234, 232, 26, 5, TEAL_BG, TEAL_ST)
    _text(svg, 346, 252, "groups=8  \u00b7  Q load-bearing", 11, TEAL_TX)

    _add_rect(svg, 494, 164, 146, 96, 8, GREEN_BG, GREEN_ST)
    _text(svg, 567, 184, "synthesis", 13, GREEN_TX, weight="500")
    _text(svg, 567, 204, "IDWT", 11, GREEN_ST)
    _text(svg, 567, 220, "sites (iii), (iv)", 11, GREEN_ST)

    _line(svg, 198, 212, 218, 212)
    _line(svg, 474, 212, 494, 212)

    # PR identity skips the mixer
    _polyline(svg, "118,260 118,274 567,274 567,260")
    _text(svg, 340, 292,
          "PR of the pair only: IDWT(DWT(x)) = x for every Q \u2208 SO(8).  "
          "Mixer remains lossy.",
          10, GREEN_TX)

    _line(svg, 340, 312, 340, 330)
    _add_rect(svg, 248, 332, 184, 28, 8, PAPER, MUTED)
    _text(svg, 340, 351, "segmentation map", 13, INK, weight="500")

    raw = ET.tostring(svg, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Fig.1 LOWO method overview SVG")
    parser.add_argument(
        "--out", "-o",
        default=os.path.join(os.path.dirname(__file__), "..",
                             "fig1_method_overview.svg"),
        help="Output SVG path (default: fig1_method_overview.svg in repo root)",
    )
    args = parser.parse_args()

    svg_str = make_svg()
    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    print(f"[fig1] wrote {out_path} ({len(svg_str)} bytes)")


if __name__ == "__main__":
    main()
