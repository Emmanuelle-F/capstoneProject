import pdfplumber
import docx
import re
import os


def clean_extracted_text(raw_text: str) -> str:
    """
    Clean extracted resume text by removing unnecessary whitespaces
    and normalizing formatting for better downstream processing.

    Args:
        raw_text (str): The raw extracted text.

    Returns:
        str: A cleaned, readable version of the text.
    """
    if not raw_text:
        return ""

    # Replace multiple spaces or newlines with a single space
    cleaned_text = re.sub(r"\s+", " ", raw_text)

    return cleaned_text.strip()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text content from a PDF resume.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted and cleaned text.
    """
    extracted_text = ""

    try:
        with pdfplumber.open(file_path) as pdf_document:
            for page in pdf_document.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"

    except Exception as error:
        print(f"[ERROR] Failed to extract PDF text from {file_path}: {error}")
        return ""

    return clean_extracted_text(extracted_text)


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text content from a DOCX resume.

    Args:
        file_path (str): Path to the DOCX file.

    Returns:
        str: Extracted and cleaned text.
    """
    extracted_text = ""

    try:
        document = docx.Document(file_path)
        for paragraph in document.paragraphs:
            extracted_text += paragraph.text + "\n"

    except Exception as error:
        print(f"[ERROR] Failed to extract DOCX text from {file_path}: {error}")
        return ""

    return clean_extracted_text(extracted_text)


def extract_resume_text(file_path: str) -> str:
    """
    Detect file type (PDF or DOCX) and extract text accordingly.

    Args:
        file_path (str): Path to the resume file.

    Returns:
        str: Extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"[ERROR] File does not exist: {file_path}")

    file_path_lower = file_path.lower()

    try:
        if file_path_lower.endswith(".pdf"):
            return extract_text_from_pdf(file_path)
        elif file_path_lower.endswith(".docx"):
            return extract_text_from_docx(file_path)
        else:
            raise ValueError("[ERROR] Unsupported file format. Use PDF or DOCX.")
    except Exception as extraction_error:
        print(f"[ERROR] Unexpected extraction failure: {extraction_error}")
        return ""
