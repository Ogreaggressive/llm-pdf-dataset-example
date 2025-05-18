import fitz  # PyMuPDF

def read_pdf(file_path):
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)
        # Iterate through each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            # Extract text from the page
            text = page.get_text()
            print(f"Page {page_num + 1}:")
            print(text)
            print("\n" + "="*80 + "\n")
        pdf_document.close()
    except Exception as e:
        print(f"Error reading PDF file: {e}")

# Example usage
pdf_file_path = 'The-AI-Act_removed.pdf'
read_pdf(pdf_file_path)