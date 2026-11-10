from pathlib import Path 
import pymupdf 

def extract_text_from_pdf(pdf_path:str) ->list[dict]:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Please provide a PDF file.")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page" : page_number,
                    "text" : text,
                    "source" : path.name,
                })

        return pages

if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ").strip()

    try:
        pages = extract_text_from_pdf(pdf_path)
        print(f"\nExtracted text from {len(pages)} pages:")

        for page in pages[:2]:
            print(f"\n-- Page {page['page']} --")
            print(page['text'][:1000])

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")