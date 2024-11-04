import os
import pandas as pd
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def load_pdfs(folder_path):
    """Load text from all PDFs in the specified folder."""
    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() or ''
                    pdf_texts.append(text)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return pdf_texts


def load_pdfs_to_csv(folder_path, csv_output_path):
    """Extract text from PDFs in the folder and save to a CSV file."""
    pdf_texts = load_pdfs(folder_path)
    df = pd.DataFrame(pdf_texts, columns=['Text'])
    df.to_csv(csv_output_path, index=False)

    print(f"Extracted text from {len(pdf_texts)} PDFs and saved to {csv_output_path}.")

def extract_sentences_from_text(text):
    """Extract sentences from the provided text for more precise querying."""
    sentences = sent_tokenize(text)  # Split text into sentences
    return sentences






