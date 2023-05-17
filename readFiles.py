from PyPDF2 import PdfReader

def read_pdf(userFile):
    """
        Reads a PDF file and extracts its text
        :param userFile: PDF file uploaded by user
        :return: text extracted from PDF file
    """
    pdf_reader = PdfReader(userFile)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_csv():
    pass