"""
Read a PDF: list its figures, save them as PNG, grab captions and the sentences that mention each figure.
For each PDF: list figures (raster images + vector drawing clusters), save them as PNG, and collect
the caption + every body-text sentence that references each figure. This is what Agent 00 receives.

usage: python pdf_reader.py paper.pdf out_dir/
"""

import json
import pathlib
import re
import sys

import pymupdf

from src.tools.legend_markers import pdf_vector_marks

FIG_RE = re.compile(r"\bFig(?:ure|\.)\s*(\d+)\s*([a-h])?", re.I)
CAPTION_START_RE = re.compile(r"(?m)^Fig(?:ure|\.)\s*(\d+)(?=\s|[|.:])", re.I)


def caption_blocks(text, page_number):
    """Return caption blocks, including journals that use ``Fig. 10 The``."""
    matches = list(CAPTION_START_RE.finditer(text or ""))
    captions = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end].strip()
        captions.append(
            {
                "page": page_number,
                "num": match.group(1),
                "text": block[:1500],
            }
        )
    return captions


def inspect(pdf_path, out_dir, min_px=400, render_dpi=300, discovery_dpi=120):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    text_pages = [p.get_text() for p in doc]
    full = "\n".join(text_pages)
    # sentences mentioning "Fig. N"
    body = re.sub(
        r"\bFig\.\s*", "Fig~", full.replace("\n", " ")
    )  # protect "Fig. 2a" from the sentence splitter
    sentences = [x.replace("Fig~", "Fig. ") for x in re.split(r"(?<=[.!?])\s+", body)]
    refs = {}
    for s in sentences:
        for m in FIG_RE.finditer(s):
            refs.setdefault(m.group(1), []).append(s.strip())
    figures = []
    pages = []
    seen_xrefs = set()
    for pno, page in enumerate(doc):
        page_preview = out / f"page_{pno + 1:04d}.png"
        try:
            page.get_pixmap(dpi=discovery_dpi, alpha=False).save(page_preview)
            page_preview_value = str(page_preview)
        except (AttributeError, TypeError):
            # Lightweight test doubles and unusual PDF pages may not render.
            page_preview_value = None
        pages.append(
            {
                "page_number": pno + 1,
                "text": text_pages[pno],
                "preview": page_preview_value,
            }
        )
        # raster figures
        for im in page.get_images(full=True):
            xref = im[0]
            info = doc.extract_image(xref)
            if xref in seen_xrefs:
                continue
            if info["width"] < min_px or info["height"] < min_px:
                continue
            seen_xrefs.add(xref)
            f = out / f"p{pno + 1}_img{xref}.png"
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            pix.save(f)
            figures.append(
                {
                    "page": pno + 1,
                    "kind": "raster",
                    "file": str(f),
                    "source_id": f"xref:{xref}",
                    "width": info["width"],
                    "height": info["height"],
                }
            )
        # vector figures: many drawing commands on the page -> render the page region at high DPI
        dr = page.get_drawings()
        if len(dr) > 200:
            rect = pymupdf.Rect()
            for d in dr:
                rect |= d["rect"]
            clip = rect & page.rect
            f = out / f"p{pno + 1}_vector.png"
            page.get_pixmap(dpi=render_dpi, clip=clip).save(f)
            # exact marker paths (small drawings) in the rendered image's pixels
            marks_file = out / f"p{pno + 1}_vector.marks.json"
            marks_file.write_text(json.dumps(pdf_vector_marks(page, clip, render_dpi)), encoding="utf-8")
            figures.append(
                {
                    "page": pno + 1,
                    "kind": "vector",
                    "file": str(f),
                    "vector_marks": str(marks_file),
                    "n_paths": len(dr),
                    "clip": list(clip),
                }
            )
    # captions: "Fig. N |" blocks anywhere in the document, with their page
    captions = []
    for pno, text in enumerate(text_pages):
        captions.extend(caption_blocks(text, pno + 1))
    figures_per_page = {
        page_number: sum(figure["page"] == page_number for figure in figures)
        for page_number in range(1, len(doc) + 1)
    }
    captions_per_page = {
        page_number: [caption for caption in captions if caption["page"] == page_number]
        for page_number in range(1, len(doc) + 1)
    }
    for fig in figures:
        candidates = captions_per_page.get(fig["page"], [])
        fig["caption_candidates"] = [
            {"figure_number": item["num"], "caption": item["text"]}
            for item in candidates
        ]
        # Image XREF order is not visual/caption order. Bind automatically only
        # when the page has exactly one image candidate and one caption.
        if figures_per_page.get(fig["page"]) == 1 and len(candidates) == 1:
            fig["caption"] = candidates[0]["text"]
            fig["figure_number"] = candidates[0]["num"]
        else:
            fig["caption"] = None
            fig["figure_number"] = None
        number = str(fig.get("figure_number") or "")
        fig["references"] = (
            [
                reference
                for reference in refs.get(number, [])
                if not re.match(
                    rf"^Fig(?:ure|\.)\s*{re.escape(number)}\b", reference, re.I
                )
            ][:20]
            if number
            else []
        )
    meta = {
        "pdf": pdf_path,
        "title": doc.metadata.get("title"),
        "producer": doc.metadata.get("producer"),
        "page_count": len(doc),
        "pages": pages,
        "figures": figures,
        "captions": captions,
        "references_by_figure": refs,
        "document_text": full,
    }
    with open(out / "inspect.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)
    close = getattr(doc, "close", None)
    if close is not None:
        close()
    return meta


if __name__ == "__main__":
    m = inspect(sys.argv[1], sys.argv[2])
    print(f"{m['title']}\n{len(m['figures'])} figure(s):")
    for f in m["figures"]:
        print(
            f"  p{f['page']} {f['kind']} fig {f.get('figure_number')} -> {f['file']}  refs={len(f.get('references', []))}"
        )
