import os
import re
from pathlib import Path
from PyPDF2 import PdfReader
from textblob import TextBlob


CHUNK_SIZE = 500
OVERLAP = 100
INPUT_DIR = Path("data/raw_docs")
OUTPUT_DIR = Path("data/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TECHNICAL_TERMS = {
    "sql", "sqli", "xss", "csrf", "ssrf", "ssh", "ftp", "http", "https",
    "rdp", "mitm", "dns", "jwt", "burp", "nmap", "hydra", "pentest",
    "localhost", "127.0.0.1", "metasploit", "cmd", "ip", "tcp", "udp"
}


def clean_text(text: str) -> str:
    text = re.sub(r"Page \d+", "", text)   # Remove page numbers
    text = re.sub(r"[\r\n]+", "\n", text)  # Normalize line breaks
    text = re.sub(r"•\s*", "- ", text)     # Replace bullet points
    text = re.sub(r"\s{2,}", " ", text)    # Extra spaces
    return text.strip()

def correct_spelling(text: str) -> str:
    words = text.split()
    corrected = []
    for word in words:
        w = re.sub(r"\W+", '', word.lower())
        if w in TECHNICAL_TERMS or len(word) < 3:
            corrected.append(word)
        else:
            blob = TextBlob(word)
            fixed = str(blob.correct())
            corrected.append(fixed if fixed else word)
    return ' '.join(corrected)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_text(file_path: Path) -> str:
    if file_path.suffix == ".pdf":
        reader = PdfReader(file_path)
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
    elif file_path.suffix == ".txt":
        text = file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file: {file_path.name}")
    return clean_text(text)

# Main
def preprocess_documents():
    files = list(INPUT_DIR.glob("*"))
    for file in files:
        try:
            print(f"Processing: {file.name}")
            text = extract_text(file)
            fixed = correct_spelling(text)
            chunks = chunk_text(fixed)
            for i, chunk in enumerate(chunks):
                out_path = OUTPUT_DIR / f"{file.stem}_chunk_{i+1:03}.txt"
                out_path.write_text(chunk, encoding="utf-8")
        except Exception as e:
            print(f"Error on {file.name}: {e}")

if __name__ == "__main__":
    preprocess_documents()
