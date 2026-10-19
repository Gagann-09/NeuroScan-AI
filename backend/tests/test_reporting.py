"""
Focused tests for the NeuroScan-AI diagnostic PDF generator
(backend/app/services/reporting.py).

These tests validate PDF generation only: output validity, integration
contract (signature / call compatibility), image handling robustness,
missing metadata, long values, page-layout boundaries, and aspect-ratio
preservation. Synthetic PNG images are used; no real patient data.
"""

from __future__ import annotations

import inspect
import os
import re
import zlib
from pathlib import Path

import pytest
from PIL import Image

from app.services.reporting import generate_segmentation_report

PAGE_W_PT = 210.0 * 2.834645669
PAGE_H_PT = 297.0 * 2.834645669
MM_TO_PT = 2.834645669
MARGIN_L_PT = 15.0 * MM_TO_PT
MARGIN_R_PT = 15.0 * MM_TO_PT


def make_png(tmp_path, name, size=(120, 90), mode="RGB", color=(100, 150, 200)):
    path = tmp_path / name
    Image.new(mode, size, color).save(path, format="PNG")
    return path


def read_pdf(path):
    return Path(path).read_bytes()


def content_streams(data):
    """Yield decompressed (or raw) content-stream bytes from a PDF."""
    for stream in re.findall(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            yield zlib.decompress(stream)
        except Exception:
            yield stream


def stream_text(data):
    return b"\n".join(content_streams(data))


def _pdf_objects(data):
    """Yield the inside of each ``obj ... endobj`` block."""
    for chunk in data.split(b"endobj"):
        if b"obj" in chunk and b"<<" in chunk:
            yield chunk


def page_count(data):
    return sum(1 for body in _pdf_objects(data) if b"/Type /Page" in body)


def image_xobjects(data):
    return [body for body in _pdf_objects(data)
            if b"/Type /XObject" in body and b"/Subtype /Image" in body]


def assert_valid_pdf(path, min_images=0):
    data = read_pdf(path)
    assert data.startswith(b"%PDF"), "output is not a PDF"
    assert len(data) > 1000, "PDF is empty"
    assert page_count(data) >= 1, "PDF has no pages"
    if min_images:
        assert len(image_xobjects(data)) >= min_images, "expected embedded images"
    return data


class TestReportSignature:
    def test_signature_and_argument_order(self):
        sig = inspect.signature(generate_segmentation_report)
        assert list(sig.parameters) == [
            "patient_identifier",
            "scan_id",
            "model_version",
            "orig_path",
            "mask_path",
            "xai_path",
            "output_pdf_path",
        ]

    def test_call_compatibility_with_existing_integration(self, tmp_path):
        # Mirrors the call site in app/services/ai_tasks.py (positional args).
        orig = make_png(tmp_path, "orig.png", (224, 224), "L", 200)
        mask = make_png(tmp_path, "mask.png", (224, 224), "RGB", (150, 255, 50))
        xai = make_png(tmp_path, "xai.png", (224, 224), "RGB", (200, 50, 50))
        out = tmp_path / "nested" / "report.pdf"
        generate_segmentation_report("PT-ANONYMIZED", "scan-001", "model-v1",
                                     orig, mask, xai, out)
        assert_valid_pdf(out, min_images=3)


class TestReportGeneration:
    def test_creates_readable_nonempty_pdf(self, tmp_path):
        out = tmp_path / "report.pdf"
        generate_segmentation_report(
            "PT-ANONYMIZED", "scan-001", "armt-gan-2d-baseline",
            make_png(tmp_path, "o.png", (200, 150), "L", 180),
            make_png(tmp_path, "m.png", (120, 120), "RGB", (20, 200, 20)),
            make_png(tmp_path, "x.png", (160, 100), "RGB", (200, 60, 60)),
            out,
        )
        data = assert_valid_pdf(out, min_images=3)
        text = stream_text(data)
        for probe in [
            b"Brain Tumor Segmentation Diagnostic Report",
            b"Research prototype - not for clinical use",
            b"Examination Metadata",
            b"Patient Identifier",
            b"PT-ANONYMIZED",
            b"Scan Reference",
            b"scan-001",
            b"AI Model Version",
            b"armt-gan-2d-baseline",
            b"Report Generated",
            b"UTC",
            b"Imaging Overview",
            b"Segmentation Visualisation",
            b"Explainability Visualisation",
            b"XAI Attribution Map",
            b"Interpretation Boundaries",
            b"clinically validated",
            b"Page 1 of",
        ]:
            assert probe in text, f"missing content: {probe!r}"

    def test_preserves_image_aspect_ratio_and_margins(self, tmp_path):
        # Non-square images: aspect ratio must be preserved and
        # placements must stay within the horizontal margins / on the page.
        make_png(tmp_path, "o.png", (200, 100), "L", 180)      # wide 2:1
        make_png(tmp_path, "m.png", (100, 100), "RGB", (20, 200, 20))  # square
        make_png(tmp_path, "x.png", (100, 200), "RGB", (200, 60, 60))    # tall 1:2
        out = tmp_path / "report.pdf"
        generate_segmentation_report("PT-1", "s-1", "m-1",
                                     tmp_path / "o.png", tmp_path / "m.png",
                                     tmp_path / "x.png", out)
        data = assert_valid_pdf(out, min_images=3)

        expected = {"I1": (200, 100), "I2": (100, 100), "I3": (100, 200)}
        found = {}
        for stream in content_streams(data):
            for m in re.finditer(
                rb"q\s*([\d.\-]+)\s0\s0\s([\d.\-]+)\s([\d.\-]+)\s([\d.\-]+)\s*cm\s*/(I\d+)\s*Do",
                stream,
            ):
                sx, sy, tx, ty = (float(g) for g in m.groups()[:4])
                found[m.group(5).decode()] = (sx, sy, tx, ty)
        assert set(found) == set(expected), f"expected {set(expected)}, found {set(found)}"
        for name, (sx, sy, tx, ty) in found.items():
            iw, ih = expected[name]
            # Rendered width/height ratio (sx/sy) must equal pixel aspect (iw/ih).
            # cm values are rounded to 2 decimals in the stream, so compare
            # ratios with a small relative tolerance.
            assert sy > 0
            ratio_err = abs(sx / sy - iw / ih) / (iw / ih)
            assert ratio_err < 1e-3, (
                f"aspect ratio distorted for {name}: sx={sx} sy={sy} px={iw}x{ih}")
            rendered_right = tx + sx
            rendered_bottom = ty + sy  # PDF y is bottom-up origin plus height
            assert MARGIN_L_PT - 1 <= tx, f"{name} left of margin: tx={tx}"
            assert rendered_right <= (PAGE_W_PT - MARGIN_R_PT) + 1, \
                f"{name} right of margin: right={rendered_right}"
            assert 0 <= ty, f"{name} above page: ty={ty}"
            assert rendered_bottom <= PAGE_H_PT, f"{name} below page: bottom={rendered_bottom}"

    def test_grayscale_color_and_alpha_images(self, tmp_path):
        out = tmp_path / "report.pdf"
        generate_segmentation_report(
            "PT-1", "s-1", "m-1",
            make_png(tmp_path, "o.png", (80, 60), "L", 100),       # grayscale
            make_png(tmp_path, "m.png", (80, 60), "RGB", (20, 200, 20)),
            make_png(tmp_path, "x.png", (80, 60), "RGBA", (200, 60, 60, 180)),  # alpha
            out,
        )
        assert_valid_pdf(out, min_images=3)


class TestReportRobustness:
    def test_missing_images_do_not_crash(self, tmp_path):
        out = tmp_path / "report.pdf"
        generate_segmentation_report(
            "PT-ANONYMIZED", "scan-001", "model-v1",
            tmp_path / "missing_orig.png",
            tmp_path / "missing_mask.png",
            tmp_path / "missing_xai.png",
            out,
        )
        data = assert_valid_pdf(out, min_images=0)
        text = stream_text(data)
        assert b"Visualisation not available" in text
        assert b"missing or could not be read" in text

    def test_unreadable_image_is_reported_unavailable(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"\x89PNG corrupted garbage not a real image")
        out = tmp_path / "report.pdf"
        generate_segmentation_report(
            "PT-1", "s-1", "m-1",
            make_png(tmp_path, "o.png", (80, 60), "L", 100),
            bad,
            tmp_path / "also_missing.png",
            out,
        )
        data = assert_valid_pdf(out, min_images=1)
        assert b"Visualisation not available" in stream_text(data)

    def test_missing_metadata(self, tmp_path):
        out = tmp_path / "report.pdf"
        generate_segmentation_report(None, "", None,
                                     make_png(tmp_path, "o.png", (80, 60), "L", 100),
                                     make_png(tmp_path, "m.png", (80, 60), "RGB", (20, 200, 20)),
                                     make_png(tmp_path, "x.png", (80, 60), "RGB", (200, 60, 60)),
                                     out)
        data = assert_valid_pdf(out, min_images=3)
        assert b"Not recorded" in stream_text(data)

    def test_long_identifier_wraps_without_error(self, tmp_path):
        out = tmp_path / "report.pdf"
        generate_segmentation_report(
            "PT-" + "X" * 300,
            "scan-" + "Y" * 200,
            "model-" + "Z" * 120,
            make_png(tmp_path, "o.png", (80, 60), "L", 100),
            make_png(tmp_path, "m.png", (80, 60), "RGB", (20, 200, 20)),
            make_png(tmp_path, "x.png", (80, 60), "RGB", (200, 60, 60)),
            out,
        )
        assert_valid_pdf(out, min_images=3)

    def test_extremely_long_unbroken_values_fit_panel(self, tmp_path):
        """Unbreakable long identifiers are ellipsis-truncated inside the panel."""
        out = tmp_path / "report.pdf"
        generate_segmentation_report(
            "PT-" + "X" * 400,
            "scan-" + "Y" * 300,
            "model-" + "Z" * 250,
            make_png(tmp_path, "o.png", (80, 60), "L", 100),
            make_png(tmp_path, "m.png", (80, 60), "RGB", (20, 200, 20)),
            make_png(tmp_path, "x.png", (80, 60), "RGB", (200, 60, 60)),
            out,
        )
        data = assert_valid_pdf(out, min_images=3)
        text = stream_text(data)
        # Full unbroken strings must not appear (they were truncated).
        assert b"X" * 400 not in text
        assert b"Y" * 300 not in text
        assert b"Z" * 250 not in text
        # A readable leading fragment plus ellipsis remains in each value.
        assert b"PT-" in text and b"..." in text
        assert b"scan-" in text
        assert b"model-" in text

    def test_long_values_with_spaces_wrap(self, tmp_path):
        """Long values containing spaces wrap to multiple lines inside the panel."""
        out = tmp_path / "report.pdf"
        spaced_scan = "scan-" + " ".join(["TOKEN"] * 40)
        generate_segmentation_report(
            "PT-ANONYMIZED",
            spaced_scan,
            "armt-gan-2d-baseline",
            make_png(tmp_path, "o.png", (80, 60), "L", 100),
            make_png(tmp_path, "m.png", (80, 60), "RGB", (20, 200, 20)),
            make_png(tmp_path, "x.png", (80, 60), "RGB", (200, 60, 60)),
            out,
        )
        data = assert_valid_pdf(out, min_images=3)
        text = stream_text(data)
        assert b"PT-ANONYMIZED" in text
        assert b"scan-" in text and b"TOKEN" in text

    def test_mixed_metadata_lengths(self, tmp_path):
        """A mix of short and long metadata renders a valid, bounded panel."""
        out = tmp_path / "report.pdf"
        generate_segmentation_report(
            "PT-1",
            "scan-" + "Q" * 500,          # very long unbroken -> truncated
            "short-model",
            make_png(tmp_path, "o.png", (80, 60), "L", 100),
            make_png(tmp_path, "m.png", (80, 60), "RGB", (20, 200, 20)),
            make_png(tmp_path, "x.png", (80, 60), "RGB", (200, 60, 60)),
            out,
        )
        data = assert_valid_pdf(out, min_images=3)
        text = stream_text(data)
        assert b"PT-1" in text
        assert b"short-model" in text
        assert b"Q" * 500 not in text
        assert b"..." in text

    def test_output_directory_is_created(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "report.pdf"
        generate_segmentation_report(
            "PT-1", "s-1", "m-1",
            make_png(tmp_path, "o.png", (80, 60), "L", 100),
            make_png(tmp_path, "m.png", (80, 60), "RGB", (20, 200, 20)),
            make_png(tmp_path, "x.png", (80, 60), "RGB", (200, 60, 60)),
            out,
        )
        assert out.exists() and out.stat().st_size > 0
