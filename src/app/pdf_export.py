# PDF Export
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_pdf_report(kpi_data: dict) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 750, "NeuralRetail — KPI Report")
    c.setFont("Helvetica", 12)
    y = 700
    for key, value in kpi_data.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 25
        if y < 100:
            c.showPage()
            y = 750
    c.save()
    buffer.seek(0)
    return buffer.read()
