from fpdf import FPDF
from datetime import datetime

def generate_segmentation_report(patient_identifier, scan_id, model_version, orig_path, mask_path, xai_path, output_pdf_path):
    """Compiles the MRI, Mask, and XAI overlay into a research prototype segmentation report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Report Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "NeuroScan AI - Research Prototype Segmentation Report", ln=True, align='C')
    pdf.line(10, 20, 200, 20)
    pdf.ln(10)

    # Metadata
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Patient ID: {patient_identifier}", ln=True)
    pdf.cell(0, 8, f"Scan Reference: {scan_id}", ln=True)
    pdf.cell(0, 8, f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC", ln=True)
    pdf.cell(0, 8, f"AI Model: {model_version}", ln=True)
    pdf.ln(10)

    # Disclaimer
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(0, 8, "DISCLAIMER: This is a research prototype. Output is not clinically validated.", ln=True)
    pdf.cell(0, 8, "This report does not constitute medical advice, diagnosis, or treatment recommendation.", ln=True)
    pdf.ln(10)

    # Attach Visuals
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Input MRI Slice", ln=True)
    pdf.image(orig_path, x=15, w=50)
    pdf.ln(5)

    pdf.cell(0, 10, "2. Segmentation Prediction", ln=True)
    pdf.image(mask_path, x=15, w=50)
    pdf.ln(5)

    pdf.cell(0, 10, "3. XAI Attribution Map (Gradient-based Input Saliency)", ln=True)
    pdf.image(xai_path, x=15, w=50)

    pdf.output(output_pdf_path)