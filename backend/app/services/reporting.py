"""Premium diagnostic PDF report generator for NeuroScan-AI.

Redesign of the segmentation report generator. The public function,
``generate_segmentation_report``, keeps its existing signature and
integration contract; only the document layout, typography, image
handling, and robustness are improved.

The report is intentionally restrained: a clinical/research colour
palette, a clear typographic hierarchy, preserved image aspect ratios,
explicit "not available" labels for missing visualisations, and a
research/demo disclaimer that makes no clinical claims.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from PIL import Image, UnidentifiedImageError
from fpdf import FPDF, XPos, YPos


# ------------------------------------------------------------------ layout
PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
MARGIN_L = 15.0
MARGIN_R = 15.0
MARGIN_T = 14.0
MARGIN_B = 20.0
CONTENT_W = PAGE_W_MM - MARGIN_L - MARGIN_R          # 180 mm
CONTENT_BOTTOM = PAGE_H_MM - MARGIN_B               # last usable body y

FOOTER_RULE_Y = PAGE_H_MM - 17.0
FOOTER_TEXT_Y = PAGE_H_MM - 14.0

# Section image box: wide enough for detail, short enough to keep pages
# balanced. Aspect ratio is always preserved; the box is never stretched.
IMAGE_MAX_W = 150.0
IMAGE_MAX_H = 72.0
UNAVAILABLE_BOX_H = 60.0

# ------------------------------------------------------------------ palette
# Restrained clinical / research colours (RGB 0-255).
INK = (30, 41, 59)        # body text
MUTED = (71, 85, 105)     # secondary text
FAINT = (99, 114, 134)    # captions / footers
NAVY = (16, 42, 67)       # brand primary
NAVY_SOFT = (34, 64, 96)  # metadata labels
ACCENT = (37, 99, 235)    # brand accent
PANEL = (246, 248, 250)   # light panel background
BORDER = (209, 216, 224)  # hairlines / frames
UNAVAILABLE_BG = (238, 241, 244)
UNAVAILABLE_FG = (112, 124, 140)

BRAND = "NeuroScan-AI"
FOOTER_LEFT = "NeuroScan-AI · Research Prototype"
DISCLAIMER = (
    "This report was produced by NeuroScan-AI, a research prototype for "
    "academic demonstration and research communication. It has not been "
    "clinically validated, cleared, or approved for diagnostic use.",
    "The MRI, segmentation, and explainability visualisations are "
    "algorithmic outputs. They do not constitute medical advice, a "
    "clinical diagnosis, or a treatment recommendation, and must not be "
    "used for autonomous clinical decision-making. Results should be "
    "reviewed by qualified medical professionals alongside standard "
    "diagnostic workflows.",
)


def _sanitize_text(value) -> str:
    """Coerce dynamic metadata to safe latin-1 text for core fonts.

    Core PDF fonts (Helvetica) cannot encode arbitrary Unicode. Unsafe
    characters are replaced rather than letting the generator crash on
    long or unusual identifiers.
    """
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC"


class _ReportPDF(FPDF):
    """FPDF subclass supplying running header/footer and layout helpers."""

    def footer(self):
        """Consistent footer on every page: hairline, project id, page numbers."""
        self.set_draw_color(*BORDER)
        self.set_line_width(0.4)
        self.line(MARGIN_L, FOOTER_RULE_Y, PAGE_W_MM - MARGIN_R, FOOTER_RULE_Y)
        self.set_y(FOOTER_TEXT_Y)
        self.set_font("helvetica", "", 7)
        self.set_text_color(*FAINT)
        self.cell(0, 5, FOOTER_LEFT, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(PAGE_W_MM - MARGIN_R - 30.0)
        self.cell(0, 5, f"Page {self.page_no()} of {{nb}}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")

    def header(self):
        """Slim running header on every page after the cover."""
        if self.page_no() == 1:
            return  # cover branding is drawn in the body
        self.set_y(MARGIN_T)
        self.set_font("helvetica", "B", 7.5)
        self.set_text_color(*NAVY)
        self.cell(0, 4.5, BRAND.upper() + " - SEGMENTATION DIAGNOSTIC REPORT",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BORDER)
        self.set_line_width(0.4)
        self.line(MARGIN_L, self.get_y(), PAGE_W_MM - MARGIN_R, self.get_y())
        self.ln(3.0)

    # ------------------------------------------------------------- drawing aids
    def _hairline(self, x1, y1, x2, y2):
        self.set_draw_color(*BORDER)
        self.set_line_width(0.4)
        self.line(x1, y1, x2, y2)

    def _box(self, x, y, w, h, fill=PANEL, stroke=BORDER, lw=0.5):
        self.set_fill_color(*fill)
        self.set_draw_color(*stroke)
        self.set_line_width(lw)
        self.rect(x, y, w, h, style="DF")

    def _truncate_to_fit(self, text, max_w, size):
        """Return ``text`` shortened (with ellipsis) so it fits within ``max_w``.

        Uses the PDF library's actual text-width measurement so unbreakable
        long identifiers never overflow the value column.
        """
        self.set_font("helvetica", "", size)
        if self.get_string_width(text) <= max_w:
            return text
        ellipsis = "..."
        ell_w = self.get_string_width(ellipsis)
        if ell_w > max_w:
            return ellipsis
        lo, hi = 0, len(text)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.get_string_width(text[:mid] + ellipsis) <= max_w:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return text[:best] + ellipsis

    def _wrap_text(self, text, max_w, size):
        """Wrap text into lines that each fit within ``max_w`` (exact measurement).

        Space-separated words wrap across lines; a single unbreakable word that
        is too long is ellipsis-truncated rather than allowed to overflow.
        """
        self.set_font("helvetica", "", size)
        words = text.split()
        lines = []
        current = ""
        for word in words:
            candidate = (current + " " + word).strip() if current else word
            if self.get_string_width(candidate) <= max_w:
                current = candidate
            else:
                if current:
                    lines.append(current)
                if self.get_string_width(word) <= max_w:
                    current = word
                else:
                    lines.append(self._truncate_to_fit(word, max_w, size))
                    current = ""
        if current:
            lines.append(current)
        return lines if lines else [""]

    def _wrap_lines(self, text, width, size):
        """Return wrapped line count for text at the given font size."""
        return len(self._wrap_text(text, width, size))

    # ------------------------------------------------------------- cover
    def _cover_brand(self):
        """Vector brand mark and wordmark; advances the cursor."""
        cx = PAGE_W_MM / 2
        y = self.get_y()
        # Abstract head/brain mark: outer cranium, inner fold, two dots.
        self.set_draw_color(*NAVY)
        self.set_line_width(1.2)
        self.circle(cx, y + 8.5, 7.0, style="D")
        self.set_line_width(0.7)
        self.arc(cx, y + 8.5, 4.4, 4.4, 25, 155)
        self.set_fill_color(*NAVY)
        self.circle(cx - 2.0, y + 10.0, 0.75, style="F")
        self.circle(cx + 2.0, y + 10.0, 0.75, style="F")

        # Wordmark: "NeuroScan-" in navy, "AI" in accent.
        self.set_font("helvetica", "B", 17)
        w1 = self.get_string_width("NeuroScan-")
        w2 = self.get_string_width("AI")
        x = cx - (w1 + w2) / 2
        yy = y + 20.0
        self.set_xy(x, yy)
        self.set_text_color(*NAVY)
        self.cell(w1, 7, "NeuroScan-")
        self.set_text_color(*ACCENT)
        self.cell(w2, 7, "AI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self._hairline(MARGIN_L, yy + 9.0, PAGE_W_MM - MARGIN_R, yy + 9.0)
        self.set_y(yy + 16.0)

    def _metadata_block(self, patient_identifier, scan_id, model_version):
        """Examination metadata in a bordered panel; advances the cursor.

        Every value is wrapped or truncated so it stays inside the panel's
        value column; unbreakable long identifiers are ellipsis-truncated
        rather than allowed to overflow the border.
        """
        rows = [
            ("Patient Identifier", _sanitize_text(patient_identifier) or "Not recorded"),
            ("Scan Reference", _sanitize_text(scan_id) or "Not recorded"),
            ("AI Model Version", _sanitize_text(model_version) or "Not recorded"),
            ("Report Generated", _now_utc()),
        ]
        label_w = 46.0
        label_x = MARGIN_L + 8.0
        value_x = label_x + label_w
        value_w = CONTENT_W - 8.0 - label_w
        min_row_h = 6.6
        line_h = 4.35
        row_pad = 2.2
        value_max_w = value_w - 1.0  # small safety margin inside the cell

        # First pass: compute row heights so the panel can be drawn behind
        # the text without covering it.
        row_specs = []  # (label, value_lines, row_h)
        y = self.get_y()
        for label, value in rows:
            value_lines = self._wrap_text(value, value_max_w, 9)
            row_h = max(min_row_h, len(value_lines) * line_h + row_pad)
            row_specs.append((label, value_lines, row_h))
            y += row_h
        block_top = self.get_y()
        block_h = y - block_top + 4.0

        self._box(MARGIN_L, block_top, CONTENT_W, block_h)

        y = block_top + 2.0
        for label, value_lines, row_h in row_specs:
            # Per-row accent mark.
            self.set_fill_color(*ACCENT)
            self.rect(MARGIN_L, y + 2.0, 2.6, 2.6, style="F")

            self.set_font("helvetica", "B", 9)
            self.set_text_color(*NAVY_SOFT)
            self.set_xy(label_x, y + 1.0)
            self.cell(label_w, line_h, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            self.set_font("helvetica", "", 9)
            self.set_text_color(*INK)
            line_y = y + 1.0
            for line in value_lines:
                self.set_xy(value_x, line_y)
                self.cell(value_w, line_h, line)
                line_y += line_h
            y += row_h
        self.set_y(block_top + block_h + 7.0)

    def _segmentation_measurements_block(
        self,
        predicted_mask_pixels=None,
        slice_coverage_percent=None,
    ):
        """Report technical measurements for the selected 2D slice only."""

        pixels_value = (
            f"{predicted_mask_pixels:,} pixels"
            if predicted_mask_pixels is not None
            else "Unavailable"
        )
        coverage_value = (
            f"{slice_coverage_percent:.4f}%"
            if slice_coverage_percent is not None
            else "Unavailable"
        )

        rows = [
            ("Predicted Mask Pixels", pixels_value),
            ("Slice Coverage", coverage_value),
            (
                "Measurement Basis",
                "Raw model probabilities >= 0.5 on the selected 224 x 224 slice",
            ),
        ]

        label_w = 46.0
        label_x = MARGIN_L + 8.0
        value_x = label_x + label_w
        value_w = CONTENT_W - 8.0 - label_w
        line_h = 4.35
        row_pad = 2.2
        min_row_h = 6.6

        row_specs = []
        y = self.get_y()
        for label, value in rows:
            value_lines = self._wrap_text(value, value_w - 1.0, 9)
            row_h = max(min_row_h, len(value_lines) * line_h + row_pad)
            row_specs.append((label, value_lines, row_h))
            y += row_h

        block_top = self.get_y()
        block_h = y - block_top + 4.0

        heading_height = 12.0
        if (
            block_top + heading_height + block_h > CONTENT_BOTTOM
        ):
            self.add_page()
            block_top = self.get_y()

        self._section_heading("", "2D Segmentation Measurements")

        self._box(MARGIN_L, block_top, CONTENT_W, block_h)

        y = block_top + 2.0
        for label, value_lines, row_h in row_specs:
            self.set_fill_color(*ACCENT)
            self.rect(MARGIN_L, y + 2.0, 2.6, 2.6, style="F")

            self.set_font("helvetica", "B", 9)
            self.set_text_color(*NAVY_SOFT)
            self.set_xy(label_x, y + 1.0)
            self.cell(label_w, line_h, label)

            self.set_font("helvetica", "", 9)
            self.set_text_color(*INK)
            line_y = y + 1.0
            for line in value_lines:
                self.set_xy(value_x, line_y)
                self.cell(value_w, line_h, line)
                line_y += line_h

            y += row_h

        self.set_y(block_top + block_h + 2.0)
        self._caption(
            "These are technical measurements of one 2D model output slice. "
            "They are not physical tumor area or volume, diagnostic accuracy, "
            "or evidence of clinical validation."
        )
        self.ln(2.0)

    # ------------------------------------------------------------- sections


    def _classification_grading_block(self):
        """Report classification and grading limits without inventing results."""
        self._section_heading("", "Classification and Grading")

        rows = [
            ("Tumor Type", "Not determined by ARMT-GAN segmentation"),
            (
                "WHO Grade / Stage",
                "Not determined; no validated grading output is available",
            ),
        ]

        label_w = 46.0
        label_x = MARGIN_L + 8.0
        value_x = label_x + label_w
        value_w = CONTENT_W - 8.0 - label_w
        line_h = 4.35
        row_pad = 2.2
        min_row_h = 6.6

        row_specs = []
        y = self.get_y()
        for label, value in rows:
            value_lines = self._wrap_text(value, value_w - 1.0, 9)
            row_h = max(min_row_h, len(value_lines) * line_h + row_pad)
            row_specs.append((label, value_lines, row_h))
            y += row_h

        block_top = self.get_y()
        block_h = y - block_top + 4.0

        if block_top + block_h > CONTENT_BOTTOM:
            self.add_page()
            block_top = self.get_y()

        self._box(MARGIN_L, block_top, CONTENT_W, block_h)

        y = block_top + 2.0
        for label, value_lines, row_h in row_specs:
            self.set_fill_color(*ACCENT)
            self.rect(MARGIN_L, y + 2.0, 2.6, 2.6, style="F")

            self.set_font("helvetica", "B", 9)
            self.set_text_color(*NAVY_SOFT)
            self.set_xy(label_x, y + 1.0)
            self.cell(label_w, line_h, label)

            self.set_font("helvetica", "", 9)
            self.set_text_color(*INK)
            line_y = y + 1.0
            for line in value_lines:
                self.set_xy(value_x, line_y)
                self.cell(value_w, line_h, line)
                line_y += line_h

            y += row_h

        self.set_y(block_top + block_h + 7.0)

    # ------------------------------------------------------------- sections
    def _section_heading(self, number, title):
        self.set_font("helvetica", "B", 11.5)
        self.set_text_color(*NAVY)
        self.set_fill_color(*ACCENT)
        self.rect(MARGIN_L, self.get_y() + 1.5, 3.0, 3.0, style="F")
        prefix = f"{number}. " if number else ""
        self.set_x(MARGIN_L + 8.0)
        self.cell(CONTENT_W - 8.0, 7, prefix + title,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)
        self._hairline(MARGIN_L, self.get_y(), PAGE_W_MM - MARGIN_R, self.get_y())
        self.ln(3.0)

    def _caption(self, text):
        self.set_font("helvetica", "", 8)
        self.set_text_color(*FAINT)
        self.multi_cell(CONTENT_W, 4.1, text)
        self.ln(2.5)

    def _render_image(self, path, max_w, max_h):
        """Draw one visualisation preserving aspect ratio.

        Returns the displayed height (mm) or None when the file is missing,
        unreadable, or unsupported - in which case an explicit unavailable
        placeholder is drawn instead.
        """
        y = self.get_y()
        size_px = None
        try:
            with Image.open(path) as img:
                img.load()
                size_px = img.size
        except (FileNotFoundError, UnidentifiedImageError, TypeError, AttributeError, OSError, ValueError):
            size_px = None

        if size_px is None:
            self._render_unavailable()
            self.set_y(y + UNAVAILABLE_BOX_H + 6.0)
            return None

        iw, ih = size_px
        ratio = ih / iw
        disp_w = max_w
        disp_h = disp_w * ratio
        if disp_h > max_h:
            disp_h = max_h
            disp_w = disp_h / ratio

        x = MARGIN_L + (CONTENT_W - disp_w) / 2
        # Thin frame around the image.
        self.set_draw_color(*BORDER)
        self.set_line_width(0.5)
        self.rect(x, y, disp_w, disp_h, style="D")
        self.image(path, x=x, y=y, w=disp_w, h=disp_h)
        self.set_y(y + disp_h + 6.0)
        return disp_h

    def _render_unavailable(self):
        y = self.get_y()
        box_w = CONTENT_W * 0.60
        x = MARGIN_L + (CONTENT_W - box_w) / 2
        self.set_fill_color(*UNAVAILABLE_BG)
        self.set_draw_color(*BORDER)
        self.rect(x, y, box_w, UNAVAILABLE_BOX_H, style="DF")
        self.set_xy(x, y + UNAVAILABLE_BOX_H / 2.0 - 5.0)
        self.set_font("helvetica", "", 8.5)
        self.set_text_color(*UNAVAILABLE_FG)
        self.multi_cell(box_w, 4.2,
            "Visualisation not available.\n"
            "The source file is missing or could not be read.")


    def _image_section(self, number, title, caption, path):
        """Full section: heading, caption, visualisation (or placeholder)."""
        block_h = 13.0 + 16.0 + max(IMAGE_MAX_H, UNAVAILABLE_BOX_H) + 10.0
        if self.get_y() + block_h > CONTENT_BOTTOM:
            self.add_page()
        self._section_heading(number, title)
        self._caption(caption)
        self._render_image(path, IMAGE_MAX_W, IMAGE_MAX_H)
        self.ln(4.0)

    def _disclaimer_block(self):
        line_h = 4.3
        gap = 3.0
        est_h = 12.0  # heading block
        for paragraph in DISCLAIMER:
            est_h += self._wrap_lines(paragraph, CONTENT_W, 8.5) * line_h + gap
        est_h += 4.0  # padding
        if self.get_y() + est_h > CONTENT_BOTTOM:
            self.add_page()

        self._section_heading("", "Interpretation Boundaries")
        block_top = self.get_y()
        self._box(MARGIN_L, block_top, CONTENT_W, est_h)

        self.set_font("helvetica", "", 8.5)
        self.set_text_color(*INK)
        y = block_top + 2.0
        for paragraph in DISCLAIMER:
            self.set_xy(MARGIN_L, y)
            self.multi_cell(CONTENT_W, line_h, paragraph)
            y += self._wrap_lines(paragraph, CONTENT_W, 8.5) * line_h + gap
        self.set_y(block_top + est_h + 6.0)


def generate_segmentation_report(
    patient_identifier,
    scan_id,
    model_version,
    orig_path,
    mask_path,
    xai_path,
    output_pdf_path,
):
    """Generate a report using the original seven-argument contract."""
    return _generate_segmentation_report_with_measurements(
        patient_identifier,
        scan_id,
        model_version,
        orig_path,
        mask_path,
        xai_path,
        output_pdf_path,
    )


def _generate_segmentation_report_with_measurements(
    patient_identifier,
    scan_id,
    model_version,
    orig_path,
    mask_path,
    xai_path,
    output_pdf_path,
    *,
    predicted_mask_pixels=None,
    slice_coverage_percent=None,
):
    """Generate a report with optional technical 2D measurements."""
    """Compiles the MRI, Mask, and XAI overlay into a research report.

    The signature and call contract are unchanged from the previous
    implementation. Images are embedded as-is; missing or unreadable
    visualisation files are replaced with an explicit unavailable label
    rather than fabricating content or crashing the report.
    """
    output_pdf_path = str(output_pdf_path)
    out_dir = os.path.dirname(os.path.abspath(output_pdf_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pdf = _ReportPDF(unit="mm", format="A4")
    pdf.set_margins(MARGIN_L, MARGIN_T, MARGIN_R)
    pdf.set_auto_page_break(True, margin=MARGIN_B)
    pdf.alias_nb_pages()
    pdf.set_title("NeuroScan-AI - Segmentation Diagnostic Report")
    pdf.set_subject("Brain tumour segmentation - research prototype report")
    pdf.set_author(BRAND + " Research Prototype")
    pdf.set_creator(BRAND)

    pdf.add_page()

    # ---- Cover / header
    pdf._cover_brand()

    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 10, "Brain Tumor Segmentation Diagnostic Report",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(2.0)
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(*FAINT)
    pdf.cell(0, 6, "Research prototype - not for clinical use",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8.0)

    # ---- Examination metadata
    pdf.set_font("helvetica", "B", 11.5)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 7, "Examination Metadata",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.0)
    pdf._metadata_block(patient_identifier, scan_id, model_version)
    pdf._classification_grading_block()
    pdf._segmentation_measurements_block(
        predicted_mask_pixels=predicted_mask_pixels,
        slice_coverage_percent=slice_coverage_percent,
    )

    # ---- Imaging sections
    pdf._image_section("1", "Imaging Overview: Original MRI",
        "Original MRI slice supplied as input to the segmentation model. "
        "Shown unaltered; aspect ratio is preserved.",
        orig_path)
    pdf._image_section("2", "Segmentation Visualisation: Predicted Mask",
        "Segmentation prediction produced by the model. The highlighted "
        "region indicates the predicted tumour area on the source slice.",
        mask_path)
    pdf._image_section("3", "Explainability Visualisation: XAI Attribution Map",
        "Gradient-based input saliency map computed for the model output. "
        "Brighter areas indicate stronger influence on the segmentation "
        "prediction. Research analysis only.",
        xai_path)

    # ---- Interpretation boundaries
    pdf._disclaimer_block()

    pdf.output(output_pdf_path)
