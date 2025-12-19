from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Project Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    # Handle encoding to prevent latin-1 errors
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf(source_md, output_pdf):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    if not os.path.exists(source_md):
        print(f"Error: Source file not found at {source_md}")
        return

    with open(source_md, 'r', encoding='utf-8') as f:
        for line in f:
            text = clean_text(line.strip())
            if not text:
                pdf.ln(5)
                continue
                
            if text.startswith('# '):
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, text[2:], 0, 1)
                pdf.set_font("Arial", size=12)
            elif text.startswith('## '):
                pdf.ln(3)
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, text[3:], 0, 1)
                pdf.set_font("Arial", size=12)
            elif text.startswith('### '):
                pdf.ln(2)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, text[4:], 0, 1)
                pdf.set_font("Arial", size=12)
            elif text.startswith('- '):
                pdf.cell(10) # Indent
                pdf.multi_cell(0, 10, chr(149) + " " + text[2:])
            else:
                pdf.multi_cell(0, 10, text)
                
    pdf.output(output_pdf)
    print(f"PDF generated successfully: {output_pdf}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Updated paths to use the docs/ folder
    source = os.path.join(base_dir, 'docs', 'PROJECT_REPORT.md')
    output = os.path.join(base_dir, 'docs', 'Project_Report.pdf')
    create_pdf(source, output)
