import fitz
from collections import Counter

figs = [
    ("Fig1 (absorption fold)",       "G:/material/天枢/mamba/v13/figs/fig1_absorption.pdf", 3.386),
    ("Fig2 (subband freqresp grid)", "G:/material/天枢/mamba/v13/figs/subband_freqresp_so8_vs_haar_grid.pdf", 3.386),
    ("Fig3 (drift vs rho)",          "G:/material/天枢/mamba/v13/figs/drift_vs_rho.pdf", 3.386),
]

for name, path, disp in figs:
    doc = fitz.open(path)
    page = doc[0]
    nat_w = page.rect.width / 72.0
    scale = disp / nat_w
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = s["text"].strip()
                if t:
                    spans.append((s["size"], t))
    print(f"\n=== {name} ===  nat={nat_w:.3f}in disp={disp:.3f}in scale={scale:.3f}")
    # histogram of on-page sizes, in 0.5pt buckets
    buckets = Counter()
    ex = {}
    for sz, t in spans:
        eff = round(sz * scale * 2) / 2.0
        buckets[eff] += 1
        ex.setdefault(eff, t[:14])
    print("  ON-PAGE size histogram (pt : #spans : example):")
    for eff in sorted(buckets, reverse=True):
        print(f"    {eff:5.1f}pt  x{buckets[eff]:3d}   {ex[eff]!r}")
    small = sorted((round(sz * scale, 2), t) for sz, t in spans
                   if sz * scale < 8.5)
    if small:
        print("  spans below 8.5pt (all should be math subscripts):")
        for eff, t in small:
            print(f"    {eff:5.2f}pt  {t[:26]!r}")
    doc.close()
