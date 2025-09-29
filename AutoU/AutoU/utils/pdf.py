import fitz

def pdf_bytes(b: bytes) -> str:
    with fitz.open(stream=b, filetype="pdf") as d:
        t = [p.get_textbox("text") for p in d]
    return "".join(t)
